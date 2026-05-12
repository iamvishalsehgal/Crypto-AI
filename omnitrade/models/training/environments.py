"""
OpenAI Gymnasium-compatible trading environment for reinforcement learning.

Simulates a crypto market environment where an agent can take BUY / HOLD /
SELL actions on OHLCV candle data.  Supports multiple simultaneous positions
(up to *max_positions*), transaction fees, slippage, and a risk-adjusted
reward signal with production-grade shaping (log returns, trailing Sharpe
bonus, drawdown penalty, transaction cost penalty, holding bonus).

Usage::

    from omnitrade.models.training.environments import CryptoTradingEnv

    env = CryptoTradingEnv(features_df, initial_balance=10_000)
    obs, info = env.reset()
    for _ in range(1000):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data-classes for bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open long position."""

    entry_price: float
    size: float  # amount in quote currency invested
    entry_step: int

    @property
    def notional(self) -> float:
        """Quote-currency amount originally invested."""
        return self.size


@dataclass
class Trade:
    """Record of a closed trade."""

    entry_price: float
    exit_price: float
    size: float
    entry_step: int
    exit_step: int
    pnl: float
    pnl_pct: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class CryptoTradingEnv(gym.Env):
    """Gymnasium environment for crypto trading with discrete actions.

    Observation space:
        A flat vector combining:
        - A sliding window of the past *window_size* candles (all feature
          columns, z-score normalised).
        - Market regime features: ADX, volume_ratio, log_return_5, OBV_ratio.
        - Portfolio state: ``[normalised_balance, n_positions / max_positions,
          normalised_unrealised_pnl, drawdown]``.

    Action space:
        ``Discrete(3)`` -- 0 = BUY, 1 = HOLD, 2 = SELL.

    Reward (production-grade shaping):
        ``r = log_return + alpha * sharpe_bonus - beta * drawdown_penalty
             - gamma * transaction_cost + delta * holding_bonus``

    Args:
        df: DataFrame of features / OHLCV data.  Must contain a ``close``
            column (case-insensitive).
        initial_balance: Starting cash balance in quote currency.
        transaction_fee: Proportional fee per trade (e.g. 0.001 = 0.1%).
        slippage: Proportional slippage applied to execution price.
        max_positions: Maximum number of simultaneous open positions.
        window_size: Number of past candles included in each observation.
        random_start: If True, pick a random start index for data augmentation.
        min_episode_steps: Minimum steps per episode (for random start bound).
        reward_sharpe_alpha: Weight for trailing Sharpe bonus (default 0.1).
        reward_drawdown_beta: Weight for drawdown penalty (default 3.0).
        reward_txn_cost_gamma: Weight for transaction cost penalty (default 0.001).
        reward_holding_delta: Weight for holding bonus (default 0.0001).
        sharpe_window: Steps for trailing Sharpe computation (default 100).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        max_positions: int = 3,
        window_size: int = 30,
        random_start: bool = True,
        min_episode_steps: int = 200,
        reward_sharpe_alpha: float = 0.1,
        reward_drawdown_beta: float = 3.0,
        reward_txn_cost_gamma: float = 0.001,
        reward_holding_delta: float = 0.0001,
        sharpe_window: int = 100,
    ) -> None:
        super().__init__()

        # Locate the close-price column (case-insensitive).
        close_col = self._find_close_column(df)
        self._close_col = close_col

        self.df = df.reset_index(drop=True)
        self.close_prices: np.ndarray = self.df[close_col].values.astype(np.float64)

        # Drop target / signal columns from feature matrix (if present).
        feature_cols = [
            c for c in self.df.columns if c.lower() not in ("signal", "target", "label")
        ]
        self.features: np.ndarray = self.df[feature_cols].values.astype(np.float64)

        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_positions = max_positions
        self.window_size = window_size

        # Reward shaping parameters.
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps
        self.reward_sharpe_alpha = reward_sharpe_alpha
        self.reward_drawdown_beta = reward_drawdown_beta
        self.reward_txn_cost_gamma = reward_txn_cost_gamma
        self.reward_holding_delta = reward_holding_delta
        self.sharpe_window = sharpe_window

        self.n_features = self.features.shape[1]

        # Pre-compute feature statistics for normalisation (across entire df).
        self._feat_mean = self.features.mean(axis=0)
        self._feat_std = self.features.std(axis=0) + 1e-8

        # Pre-compute market regime features per step (cached arrays).
        self._regime_features = self._compute_regime_features()

        # Gym spaces.
        regime_state_size = 4   # ADX, volume_ratio, log_return_5, OBV_ratio
        portfolio_state_size = 4  # balance, positions ratio, unrealised PnL, drawdown
        obs_size = (
            self.window_size * self.n_features
            + regime_state_size
            + portfolio_state_size
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Episode state -- populated in reset().
        self.balance: float = 0.0
        self.positions: List[Position] = []
        self.trade_history: List[Trade] = []
        self.current_step: int = 0
        self.max_drawdown: float = 0.0
        self._peak_equity: float = 0.0
        self._equity_history: List[float] = []
        self._step_returns: deque = deque(maxlen=sharpe_window)

        logger.info(
            "CryptoTradingEnv created: rows=%d, features=%d, window=%d, "
            "balance=%.2f, max_positions=%d, random_start=%s",
            len(self.df),
            self.n_features,
            self.window_size,
            initial_balance,
            max_positions,
            random_start,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment, optionally to a random start point.

        Random start provides data augmentation — each episode sees a
        different trajectory, preventing the agent from memorising a
        single price path.
        """
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.max_drawdown = 0.0
        self._peak_equity = self.initial_balance
        self._equity_history = [self.initial_balance]
        self._step_returns.clear()

        # Random start point for data augmentation.
        max_start = len(self.df) - self.min_episode_steps - 1
        min_start = self.window_size
        if self.random_start and max_start > min_start:
            self.current_step = random.randint(min_start, max_start)
        else:
            self.current_step = self.window_size

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time-step.

        Args:
            action: 0 = BUY, 1 = HOLD, 2 = SELL.

        Returns:
            ``(observation, reward, terminated, truncated, info)``

        Reward shaping (production-grade):
            ``r = log_return + alpha*sharpe_bonus - beta*drawdown_penalty
                 - gamma*transaction_cost + delta*holding_bonus``
        """
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, self._build_info()

        prev_equity = self._equity_history[-1] if self._equity_history else self.initial_balance
        current_price = self._get_execution_price(action)

        # -- execute action --------------------------------------------------
        txn_cost = 0.0
        trade_taken = False
        realised_pnl = 0.0

        if action == ACTION_BUY:
            trade_taken = self._execute_buy(current_price)
            txn_cost = self.transaction_fee if trade_taken else 0.0
        elif action == ACTION_SELL:
            realised_pnl = self._execute_sell(current_price)
            trade_taken = realised_pnl != 0.0
            txn_cost = self.transaction_fee if trade_taken else 0.0

        # -- advance time ----------------------------------------------------
        self.current_step += 1

        # -- compute equity and drawdown ------------------------------------
        equity = self._compute_equity()
        self._equity_history.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        current_drawdown = (
            (self._peak_equity - equity) / self._peak_equity
            if self._peak_equity > 0 else 0.0
        )
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # -- production reward shaping ---------------------------------------
        # 1. Log return: more stable than raw equity change, handles scale.
        log_return = np.log(equity / prev_equity) if prev_equity > 0 and equity > 0 else 0.0
        self._step_returns.append(log_return)

        # 2. Trailing Sharpe bonus: rewards consistent risk-adjusted returns.
        sharpe_bonus = self._compute_trailing_sharpe()

        # 3. Drawdown penalty: heavily penalise drawdowns (beta=3.0).
        drawdown_penalty = -self.reward_drawdown_beta * current_drawdown

        # 4. Transaction cost penalty: discourage excessive trading.
        txn_penalty = -self.reward_txn_cost_gamma * txn_cost

        # 5. Holding bonus: small incentive to stay invested (delta=0.0001).
        holding_bonus = (
            self.reward_holding_delta if len(self.positions) > 0 and not trade_taken
            else 0.0
        )

        reward = (
            log_return
            + self.reward_sharpe_alpha * sharpe_bonus
            + drawdown_penalty
            + txn_penalty
            + holding_bonus
        )

        # -- termination conditions -----------------------------------------
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
        if equity <= self.initial_balance * 0.5:
            terminated = True

        truncated = False
        obs = self._get_observation()
        info = self._build_info()
        info["reward_components"] = {
            "log_return": round(log_return, 6),
            "sharpe_bonus": round(sharpe_bonus, 6),
            "drawdown_penalty": round(drawdown_penalty, 6),
            "txn_penalty": round(txn_penalty, 6),
            "holding_bonus": round(holding_bonus, 6),
            "total": round(reward, 6),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable snapshot of the current state."""
        equity = self._compute_equity()
        unrealised = self._unrealised_pnl()
        realised = sum(t.pnl for t in self.trade_history)
        current_price = self.close_prices[min(self.current_step, len(self.close_prices) - 1)]

        print(
            f"Step {self.current_step}/{len(self.df) - 1} | "
            f"Price {current_price:.2f} | "
            f"Balance {self.balance:.2f} | "
            f"Equity {equity:.2f} | "
            f"Positions {len(self.positions)}/{self.max_positions} | "
            f"Unrealised PnL {unrealised:+.2f} | "
            f"Realised PnL {realised:+.2f} | "
            f"Max DD {self.max_drawdown:.2%}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build the flat observation vector.

        Includes: sliding feature window + market regime features + portfolio state.
        """
        start = max(0, self.current_step - self.window_size)
        end = self.current_step

        window = self.features[start:end]
        # Pad if window is shorter than expected (beginning of series).
        if window.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - window.shape[0], self.n_features))
            window = np.concatenate([padding, window], axis=0)

        # Z-score normalise.
        window_norm = (window - self._feat_mean) / self._feat_std

        # Market regime features for current step.
        step = min(self.current_step, len(self._regime_features) - 1)
        regime = self._regime_features[step].astype(np.float64)

        # Portfolio state (normalised).
        equity = self._compute_equity()
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
        else:
            dd = 0.0
        portfolio_state = np.array(
            [
                self.balance / self.initial_balance,
                len(self.positions) / max(self.max_positions, 1),
                self._unrealised_pnl() / self.initial_balance,
                dd,
            ],
            dtype=np.float64,
        )

        obs = np.concatenate([window_norm.flatten(), regime, portfolio_state]).astype(np.float32)
        return obs

    def _get_execution_price(self, action: int) -> float:
        """Apply slippage to the current close price."""
        price = self.close_prices[self.current_step]
        if action == ACTION_BUY:
            price *= 1.0 + self.slippage  # buy at slightly higher price
        elif action == ACTION_SELL:
            price *= 1.0 - self.slippage  # sell at slightly lower price
        return float(price)

    def _execute_buy(self, price: float) -> bool:
        """Open a new position if allowed. Returns True on success."""
        if len(self.positions) >= self.max_positions:
            return False

        invest_fraction = 1.0 / self.max_positions
        invest_amount = self.balance * invest_fraction
        if invest_amount < 1.0:
            return False

        fee = invest_amount * self.transaction_fee
        actual_invest = invest_amount - fee
        self.balance -= invest_amount

        self.positions.append(
            Position(entry_price=price, size=actual_invest, entry_step=self.current_step)
        )
        return True

    def _execute_sell(self, price: float) -> float:
        """Close the oldest open position. Returns realised PnL (0 if none)."""
        if not self.positions:
            return 0.0

        pos = self.positions.pop(0)  # FIFO
        price_change_pct = (price - pos.entry_price) / pos.entry_price
        proceeds = pos.size * (1.0 + price_change_pct)
        fee = proceeds * self.transaction_fee
        net_proceeds = proceeds - fee

        pnl = net_proceeds - pos.size
        pnl_pct = pnl / pos.size if pos.size > 0 else 0.0

        self.balance += net_proceeds
        self.trade_history.append(
            Trade(
                entry_price=pos.entry_price,
                exit_price=price,
                size=pos.size,
                entry_step=pos.entry_step,
                exit_step=self.current_step,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
        )
        return pnl / self.initial_balance

    def _unrealised_pnl(self) -> float:
        """Sum of unrealised PnL across all open positions."""
        if not self.positions:
            return 0.0
        current_price = self.close_prices[min(self.current_step, len(self.close_prices) - 1)]
        total = 0.0
        for pos in self.positions:
            price_change_pct = (current_price - pos.entry_price) / pos.entry_price
            total += pos.size * price_change_pct
        return total

    def _compute_equity(self) -> float:
        """Total equity = cash balance + unrealised value of positions."""
        return self.balance + sum(
            pos.size
            + pos.size
            * (
                (self.close_prices[min(self.current_step, len(self.close_prices) - 1)] - pos.entry_price)
                / pos.entry_price
            )
            for pos in self.positions
        )

    def _compute_regime_features(self) -> np.ndarray:
        """Pre-compute market regime features for every time step.

        Returns:
            Array of shape (n_steps, 4) with columns:
            [adx_14, volume_ratio, log_return_5, obv_ratio]
        """
        n = len(self.df)
        close = self.close_prices

        # Volume ratio: current volume / 20-period average volume.
        vol_col = self._find_column_lower(self.df, "volume")
        volume = self.df[vol_col].values.astype(np.float64) if vol_col else np.ones(n)
        vol_ma20 = pd.Series(volume).rolling(20).mean().fillna(volume).values
        volume_ratio = np.divide(volume, vol_ma20, where=vol_ma20 > 0)

        # Log return over 5 periods.
        log_return_5 = np.zeros(n)
        log_return_5[5:] = np.log(close[5:] / close[:-5])

        # OBV ratio: current OBV / 20-period average OBV.
        direction = np.sign(np.diff(close, prepend=close[0]))
        obv = np.cumsum(direction * volume)
        obv_ma20 = pd.Series(obv).rolling(20).mean().fillna(obv).values
        obv_ratio = np.divide(obv, obv_ma20, where=obv_ma20 > 0)

        # ADX: fetch from features if present, else compute.
        adx_col = self._find_column_lower(self.df, "adx")
        if adx_col:
            adx = self.df[adx_col].values.astype(np.float64)
        else:
            adx = np.zeros(n)

        # Z-score normalise each regime feature.
        regime = np.column_stack([adx, volume_ratio, log_return_5, obv_ratio])
        regime_mean = regime.mean(axis=0)
        regime_std = regime.std(axis=0) + 1e-8
        regime_norm = (regime - regime_mean) / regime_std

        return regime_norm.astype(np.float32)

    def _compute_trailing_sharpe(self) -> float:
        """Compute annualised Sharpe ratio over trailing returns window."""
        if len(self._step_returns) < 10:
            return 0.0
        rets = np.array(list(self._step_returns), dtype=np.float64)
        mean = rets.mean()
        std = rets.std()
        if std == 0.0:
            return 0.0
        # Annualise: crypto = 365 * 24 (hourly), scaled to step frequency.
        # Conservative: use sqrt(window) for trailing Sharpe.
        return (mean / std) * np.sqrt(len(rets))

    @staticmethod
    def _find_column_lower(df: pd.DataFrame, target: str) -> Optional[str]:
        """Return column name matching *target* case-insensitively, or None."""
        for col in df.columns:
            if col.lower() == target:
                return col
        return None

    def _build_info(self) -> Dict[str, Any]:
        """Construct the info dict returned by step / reset."""
        return {
            "balance": self.balance,
            "equity": self._compute_equity(),
            "n_positions": len(self.positions),
            "unrealised_pnl": self._unrealised_pnl(),
            "realised_pnl": sum(t.pnl for t in self.trade_history),
            "n_trades": len(self.trade_history),
            "max_drawdown": self.max_drawdown,
            "step": self.current_step,
        }

    @staticmethod
    def _find_close_column(df: pd.DataFrame) -> str:
        """Return the name of the close-price column (case-insensitive)."""
        for col in df.columns:
            if col.lower() == "close":
                return col
        raise ValueError(
            "DataFrame must contain a 'close' column.  "
            f"Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Utility: chronological train/val/test split
# ---------------------------------------------------------------------------

def create_train_val_envs(
    df: pd.DataFrame,
    env_kwargs: Optional[Dict[str, Any]] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[CryptoTradingEnv, CryptoTradingEnv, CryptoTradingEnv]:
    """Create chronologically-split train/val/test environments.

    70/15/15 split with NO shuffling — critical for time-series data to
    prevent look-ahead bias.  Training env uses random_start=True for
    data augmentation; val and test use deterministic starts.

    Args:
        df: Full feature DataFrame (must contain a 'close' column).
        env_kwargs: Kwargs forwarded to ``CryptoTradingEnv``.
        train_frac: Fraction for training (default 0.70).
        val_frac: Fraction for validation (default 0.15).
            Test gets the remainder (0.15).

    Returns:
        Tuple of (train_env, val_env, test_env).
    """
    if env_kwargs is None:
        env_kwargs = {}

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    train_env = CryptoTradingEnv(df_train, random_start=True, **env_kwargs)
    val_env = CryptoTradingEnv(df_val, random_start=False, **env_kwargs)
    test_env = CryptoTradingEnv(df_test, random_start=False, **env_kwargs)

    logger.info(
        "Train/val/test split: %d / %d / %d rows",
        len(df_train), len(df_val), len(df_test),
    )

    return train_env, val_env, test_env
