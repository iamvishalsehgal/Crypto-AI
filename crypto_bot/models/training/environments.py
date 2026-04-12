"""
OpenAI Gymnasium-compatible trading environment for reinforcement learning.

Simulates a crypto market environment where an agent can take BUY / HOLD /
SELL actions on OHLCV candle data.  Supports multiple simultaneous positions
(up to *max_positions*), transaction fees, slippage, and a risk-adjusted
reward signal.

Usage::

    from crypto_bot.models.training.environments import CryptoTradingEnv

    env = CryptoTradingEnv(features_df, initial_balance=10_000)
    obs, info = env.reset()
    for _ in range(1000):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from crypto_bot.utils.logger import get_logger

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
        - Portfolio state: ``[normalised_balance, n_positions / max_positions,
          normalised_unrealised_pnl]``.

    Action space:
        ``Discrete(3)`` -- 0 = BUY, 1 = HOLD, 2 = SELL.

    Args:
        df: DataFrame of features / OHLCV data.  Must contain a ``close``
            column (case-insensitive).
        initial_balance: Starting cash balance in quote currency.
        transaction_fee: Proportional fee per trade (e.g. 0.001 = 0.1%).
        slippage: Proportional slippage applied to execution price.
        max_positions: Maximum number of simultaneous open positions.
        window_size: Number of past candles included in each observation.
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

        self.n_features = self.features.shape[1]

        # Pre-compute feature statistics for normalisation (across entire df).
        self._feat_mean = self.features.mean(axis=0)
        self._feat_std = self.features.std(axis=0) + 1e-8

        # Gym spaces.
        portfolio_state_size = 3  # balance, positions ratio, unrealised PnL
        obs_size = self.window_size * self.n_features + portfolio_state_size
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

        logger.info(
            "CryptoTradingEnv created: rows=%d, features=%d, window=%d, "
            "balance=%.2f, max_positions=%d",
            len(self.df),
            self.n_features,
            self.window_size,
            initial_balance,
            max_positions,
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
        """Reset the environment to the beginning of the dataset."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.current_step = self.window_size
        self.max_drawdown = 0.0
        self._peak_equity = self.initial_balance
        self._equity_history = [self.initial_balance]

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
        """
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, self._build_info()

        current_price = self._get_execution_price(action)
        reward = 0.0

        # -- execute action --------------------------------------------------
        if action == ACTION_BUY:
            reward += self._execute_buy(current_price)
        elif action == ACTION_SELL:
            reward += self._execute_sell(current_price)
        # HOLD: do nothing, reward stays 0 (before risk adjustment)

        # -- advance time ----------------------------------------------------
        self.current_step += 1

        # -- compute equity and drawdown ------------------------------------
        equity = self._compute_equity()
        self._equity_history.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        current_drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # -- risk-adjusted reward component ---------------------------------
        # Penalise drawdown and reward positive equity change.
        equity_change = (equity - self._equity_history[-2]) / self.initial_balance
        drawdown_penalty = -current_drawdown * 0.1
        reward += equity_change + drawdown_penalty

        # -- termination conditions -----------------------------------------
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
        if equity <= self.initial_balance * 0.5:
            # Stop if equity drops below 50 % of starting capital.
            terminated = True

        truncated = False
        obs = self._get_observation()
        info = self._build_info()
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
        """Build the flat observation vector."""
        start = max(0, self.current_step - self.window_size)
        end = self.current_step

        window = self.features[start:end]
        # Pad if window is shorter than expected (beginning of series).
        if window.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - window.shape[0], self.n_features))
            window = np.concatenate([padding, window], axis=0)

        # Z-score normalise.
        window_norm = (window - self._feat_mean) / self._feat_std

        # Portfolio state (normalised).
        equity = self._compute_equity()
        portfolio_state = np.array(
            [
                self.balance / self.initial_balance,
                len(self.positions) / max(self.max_positions, 1),
                self._unrealised_pnl() / self.initial_balance,
            ],
            dtype=np.float64,
        )

        obs = np.concatenate([window_norm.flatten(), portfolio_state]).astype(np.float32)
        return obs

    def _get_execution_price(self, action: int) -> float:
        """Apply slippage to the current close price."""
        price = self.close_prices[self.current_step]
        if action == ACTION_BUY:
            price *= 1.0 + self.slippage  # buy at slightly higher price
        elif action == ACTION_SELL:
            price *= 1.0 - self.slippage  # sell at slightly lower price
        return float(price)

    def _execute_buy(self, price: float) -> float:
        """Open a new position if allowed. Returns immediate reward component."""
        if len(self.positions) >= self.max_positions:
            return -0.001  # small penalty for invalid action

        # Invest a fixed fraction of current balance.
        invest_fraction = 1.0 / self.max_positions
        invest_amount = self.balance * invest_fraction
        if invest_amount < 1.0:
            return -0.001  # not enough capital

        fee = invest_amount * self.transaction_fee
        actual_invest = invest_amount - fee
        self.balance -= invest_amount

        self.positions.append(
            Position(entry_price=price, size=actual_invest, entry_step=self.current_step)
        )
        return 0.0  # neutral immediate reward; PnL accrues via equity tracking

    def _execute_sell(self, price: float) -> float:
        """Close the oldest open position. Returns immediate reward component."""
        if not self.positions:
            return -0.001  # penalty for invalid sell

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
        # Reward is the realised PnL normalised by initial balance.
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
