"""
Betting backtester — simulates value betting on historical odds, computes
CLV (Closing Line Value), and compares Kelly vs flat staking strategies.

The betting P&L model is fundamentally binary (win/loss), so this uses a
dedicated simulation loop rather than wrapping BacktestEngine.

Usage::

    backtester = BettingBacktester(settings)
    result = backtester.run("soccer_epl", days=365)
    backtester.plot(result, save_path="reports/backtest_betting.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BettingBacktestResult:
    """Aggregate betting backtest metrics."""

    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    win_rate: float = 0.0
    total_staked: float = 0.0
    total_payout: float = 0.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    avg_edge: float = 0.0
    avg_clv: float = 0.0
    clv_positive_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    bankroll_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bet_history: List[Dict] = field(default_factory=list)
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class BettingBacktester:
    """Backtest value betting strategies on historical odds data.

    Simulates:
    - Edge detection via ValueBettingModel
    - Kelly criterion stake sizing (fractional)
    - CLV analysis (odds movement from bet time to close)
    - Bankroll evolution with drawdown tracking

    Args:
        settings: Bot configuration.
        initial_bankroll: Starting bankroll in USD.
    """

    def __init__(
        self,
        settings: Settings,
        initial_bankroll: Optional[float] = None,
    ) -> None:
        self._settings = settings
        self._betting = settings.betting
        self._initial_bankroll = initial_bankroll or self._betting.bankroll

        from omnitrade.data.collectors.betting_data import BettingDataCollector
        from omnitrade.features.betting_features import BettingFeatures
        from omnitrade.models.betting_models import ValueBettingModel, KellyStakingModel
        from omnitrade.risk.betting_risk import BettingRiskManager

        self._collector = BettingDataCollector(settings)
        self._features = BettingFeatures(settings)
        self._model = ValueBettingModel(settings)
        self._kelly_model = KellyStakingModel(settings)
        self._risk_manager = BettingRiskManager(settings)

        self._last_result: Optional[BettingBacktestResult] = None

    # ------------------------------------------------------------------
    # Main backtest
    # ------------------------------------------------------------------

    def run(
        self,
        sport_key: str,
        days: int = 365,
        train_split: float = 0.5,
    ) -> BettingBacktestResult:
        """Run a backtest on historical odds for a sport.

        Splits data chronologically: first *train_split* fraction trains
        the model, remaining fraction tests it.

        Args:
            sport_key: e.g. ``"soccer_epl"``, ``"basketball_nba"``.
            days: Lookback period.
            train_split: Fraction of data for training (0-1).

        Returns:
            BettingBacktestResult with full metrics.
        """
        logger.info("Starting betting backtest for %s (%d days)", sport_key, days)

        historical = self._collector.fetch_historical_results(sport_key, days_back=days)
        if historical.empty or len(historical) < 30:
            logger.warning("Insufficient historical data for %s (%d rows)", sport_key, len(historical))
            return BettingBacktestResult()

        odds_df = self._collector.fetch_odds(sport_key)
        if odds_df.empty:
            odds_df = self._collector._mock_odds(sport_key)

        features = self._features.compute_all(odds_df, historical)
        if features.empty or len(features) < 20:
            logger.warning("Insufficient features for %s", sport_key)
            return BettingBacktestResult()

        features = features.sort_values("commence_time") if "commence_time" in features.columns else features

        split_idx = int(len(features) * train_split)
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]

        if len(train_features) < 20 or len(test_features) < 5:
            logger.warning("Not enough data for train/test split in %s", sport_key)
            return BettingBacktestResult()

        # Train the model on in-sample data
        if "home_score" in historical.columns and not historical.empty:
            outcomes = (historical.get("home_score", 0) > historical.get("away_score", 0)).astype(int)
            self._model.train(train_features, outcomes)

        # Run simulation on test data
        result = self._simulate(test_features, sport_key)
        self._last_result = result

        logger.info(
            "Betting backtest complete: %d bets, win=%.1f%%, ROI=%.1f%%, "
            "CLV+=%.1f%%, bankroll $%.2f -> $%.2f",
            result.total_bets, result.win_rate * 100, result.roi_pct,
            result.clv_positive_pct * 100,
            result.initial_bankroll, result.final_bankroll,
        )
        return result

    def run_kelly_comparison(
        self,
        sport_key: str,
        days: int = 365,
    ) -> Dict[str, BettingBacktestResult]:
        """Compare full Kelly, fractional Kelly, and flat staking.

        Returns:
            Dict mapping strategy name to result.
        """
        historical = self._collector.fetch_historical_results(sport_key, days_back=days)
        odds_df = self._collector.fetch_odds(sport_key)
        if odds_df.empty:
            odds_df = self._collector._mock_odds(sport_key)

        features = self._features.compute_all(odds_df, historical)
        if len(features) < 30:
            return {}

        split_idx = int(len(features) * 0.5)
        test_features = features.iloc[split_idx:]

        strategies = {
            "kelly_full": 1.0,
            "kelly_half": 0.5,
            "kelly_quarter": 0.25,
            "flat_2pct": -1.0,  # sentinel for flat staking
        }

        results = {}
        for name, kelly_frac in strategies.items():
            if kelly_frac > 0:
                result = self._simulate(test_features, sport_key, kelly_fraction=kelly_frac)
            else:
                # Flat staking: 2% of bankroll per bet
                result = self._simulate_flat(test_features, sport_key, stake_pct=0.02)
            results[name] = result

        return results

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        features: pd.DataFrame,
        sport_key: str = "",
        kelly_fraction: Optional[float] = None,
    ) -> BettingBacktestResult:
        """Simulate betting through the feature DataFrame chronologically.

        Args:
            kelly_fraction: Override Kelly fraction (None = use configured value).
        """
        bankroll = self._initial_bankroll
        initial = bankroll
        bankroll_history = [(features.index[0] if not features.empty else 0, bankroll)]
        bets: List[Dict] = []
        peak = bankroll

        for i in range(len(features)):
            row_df = features.iloc[[i]]
            try:
                signal = self._model.predict(row_df)
            except Exception:
                continue

            if signal.side not in ("BACK", "LAY"):
                continue

            model_prob = signal.metadata.get("model_prob", 0.5)
            implied_prob = signal.metadata.get("implied_prob", 0.5)
            edge = model_prob - implied_prob
            odds = signal.metadata.get("odds", -110)

            if edge <= self._betting.min_edge_pct:
                continue

            stake = self._kelly_model.compute_stake(
                bankroll, model_prob, implied_prob, odds,
                kelly_fraction=kelly_fraction,
            )
            if stake <= 0:
                continue

            # Determine outcome from the historical data
            outcome = self._determine_outcome(features, i)

            decimal_odds = self._american_to_decimal(odds)
            if outcome == "win":
                payout = stake * decimal_odds
                pnl = payout - stake
            elif outcome == "void":
                payout = stake
                pnl = 0.0
            else:
                payout = 0.0
                pnl = -stake

            bankroll = bankroll - stake + payout
            bankroll_history.append((features.index[i], bankroll))
            peak = max(peak, bankroll)

            # CLV: compare bet-time implied prob to final implied prob
            clv = self._compute_clv(features, i, implied_prob)

            bets.append({
                "idx": i,
                "symbol": signal.symbol,
                "side": str(signal.side),
                "stake": round(stake, 2),
                "odds": odds,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "clv": round(clv, 4),
                "bankroll": round(bankroll, 2),
            })

            if bankroll <= 0:
                logger.warning("Bankroll exhausted at bet %d", i)
                break

        return self._build_result(bets, bankroll, initial, peak, bankroll_history)

    def _simulate_flat(
        self,
        features: pd.DataFrame,
        sport_key: str = "",
        stake_pct: float = 0.02,
    ) -> BettingBacktestResult:
        """Simulate with flat staking (fixed % of initial bankroll per bet)."""
        bankroll = self._initial_bankroll
        initial = bankroll
        bankroll_history = [(features.index[0] if not features.empty else 0, bankroll)]
        bets: List[Dict] = []
        peak = bankroll
        flat_stake = initial * stake_pct

        for i in range(len(features)):
            row_df = features.iloc[[i]]
            try:
                signal = self._model.predict(row_df)
            except Exception:
                continue

            if signal.side not in ("BACK", "LAY"):
                continue

            model_prob = signal.metadata.get("model_prob", 0.5)
            implied_prob = signal.metadata.get("implied_prob", 0.5)
            edge = model_prob - implied_prob
            odds = signal.metadata.get("odds", -110)

            if edge <= self._betting.min_edge_pct:
                continue

            stake = min(flat_stake, bankroll)
            outcome = self._determine_outcome(features, i)

            decimal_odds = self._american_to_decimal(odds)
            if outcome == "win":
                payout = stake * decimal_odds
                pnl = payout - stake
            elif outcome == "void":
                payout = stake
                pnl = 0.0
            else:
                payout = 0.0
                pnl = -stake

            bankroll = bankroll - stake + payout
            bankroll_history.append((features.index[i], bankroll))
            peak = max(peak, bankroll)

            bets.append({
                "idx": i,
                "symbol": signal.symbol,
                "side": str(signal.side),
                "stake": round(stake, 2),
                "odds": odds,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            })

            if bankroll <= 0:
                break

        return self._build_result(bets, bankroll, initial, peak, bankroll_history)

    # ------------------------------------------------------------------
    # Outcome & CLV
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_outcome(features: pd.DataFrame, idx: int) -> str:
        """Determine bet outcome from historical data.

        For mock data, uses the poisson_home_win probability as a proxy
        for the actual outcome. For real data, checks home_score/away_score.
        """
        row = features.iloc[idx]
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if home_score is not None and away_score is not None:
            try:
                hs = float(home_score)
                aes = float(away_score)
                if np.isnan(hs) or np.isnan(aes):
                    return "loss"
                if hs > aes:
                    return "win"
                return "loss"
            except (ValueError, TypeError):
                pass

        poisson_home = row.get("poisson_home_win", 0.5)
        if not np.isnan(poisson_home):
            return "win" if float(poisson_home) >= 0.5 else "loss"

        return "loss"

    @staticmethod
    def _compute_clv(
        features: pd.DataFrame,
        idx: int,
        bet_implied_prob: float,
    ) -> float:
        """Compute Closing Line Value — how much the line moved in our favour.

        Positive CLV = line moved toward our bet (we beat the market).
        CLV = bet_implied_prob - close_implied_prob

        Uses the last row's implied probability as the "closing line" proxy.
        """
        close_prob = features.get("prob_home", pd.Series([bet_implied_prob])).iloc[-1]
        close_prob = float(close_prob) if not np.isnan(close_prob) else bet_implied_prob

        if idx < len(features) - 5:
            future_window = features.iloc[idx + 1:min(idx + 6, len(features))]
            if "prob_home" in future_window.columns:
                future_probs = future_window["prob_home"].dropna()
                if not future_probs.empty:
                    close_prob = float(future_probs.mean())

        return bet_implied_prob - close_prob

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(
        self,
        bets: List[Dict],
        final_bankroll: float,
        initial_bankroll: float,
        peak: float,
        bankroll_history: List[Tuple],
    ) -> BettingBacktestResult:
        """Assemble a BettingBacktestResult from simulation data."""
        if not bets:
            return BettingBacktestResult(
                initial_bankroll=initial_bankroll,
                final_bankroll=final_bankroll,
            )

        wins = [b for b in bets if b.get("outcome") == "win"]
        losses = [b for b in bets if b.get("outcome") == "loss"]
        voids = [b for b in bets if b.get("outcome") == "void"]
        settled = wins + losses

        total_pnl = sum(b.get("pnl", 0) for b in settled)
        total_staked = sum(b.get("stake", 0) for b in bets)
        roi = (total_pnl / initial_bankroll * 100) if initial_bankroll else 0

        clv_positive = [b for b in bets if b.get("clv", 0) > 0]
        edges = [b.get("edge", 0) for b in bets]
        clvs = [b.get("clv", 0) for b in bets]

        # Drawdown
        max_dd = 0.0
        max_dd_pct = 0.0
        if bankroll_history:
            peak_br = initial_bankroll
            for _, br in bankroll_history:
                peak_br = max(peak_br, br)
                dd = peak_br - br
                dd_pct = dd / peak_br if peak_br > 0 else 0
                max_dd = max(max_dd, dd)
                max_dd_pct = max(max_dd_pct, dd_pct)

        # Sharpe from bet returns
        settled_returns = [b.get("pnl", 0) / (b.get("stake", 1) or 1) for b in settled]
        sharpe = 0.0
        if settled_returns and np.std(settled_returns) > 0:
            sharpe = float(np.mean(settled_returns) / np.std(settled_returns) * np.sqrt(len(settled)))

        # Profit factor
        gross_win = sum(b.get("pnl", 0) for b in wins)
        gross_loss = abs(sum(b.get("pnl", 0) for b in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0)

        # Bankroll curve
        if bankroll_history:
            idx, vals = zip(*bankroll_history)
            curve = pd.Series(vals, index=idx, name="bankroll")
        else:
            curve = pd.Series(dtype=float)

        return BettingBacktestResult(
            total_bets=len(bets),
            wins=len(wins),
            losses=len(losses),
            voids=len(voids),
            win_rate=len(wins) / len(settled) if settled else 0,
            total_staked=round(total_staked, 2),
            total_payout=round(total_staked + total_pnl, 2),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(roi, 2),
            avg_edge=round(float(np.mean(edges)), 4) if edges else 0,
            avg_clv=round(float(np.mean(clvs)), 4) if clvs else 0,
            clv_positive_pct=len(clv_positive) / len(clvs) if clvs else 0,
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            sharpe_ratio=round(sharpe, 3),
            profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            bankroll_curve=curve,
            bet_history=bets,
            initial_bankroll=initial_bankroll,
            final_bankroll=round(final_bankroll, 2),
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        result: Optional[BettingBacktestResult] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot bankroll curve, drawdown, and bet PnL distribution."""
        result = result or self._last_result
        if result is None or result.bankroll_curve.empty:
            logger.warning("No backtest result to plot")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        # Bankroll curve
        ax_br = axes[0]
        result.bankroll_curve.plot(ax=ax_br, linewidth=1.2, color="steelblue")
        ax_br.axhline(y=result.initial_bankroll, color="gray", linestyle="--", alpha=0.5, label="Initial")
        ax_br.set_title("Bankroll Curve")
        ax_br.set_ylabel("Bankroll ($)")
        ax_br.legend()
        ax_br.grid(True, alpha=0.3)

        # Drawdown
        ax_dd = axes[1]
        running_max = result.bankroll_curve.cummax()
        drawdown = (result.bankroll_curve - running_max) / running_max
        drawdown.plot(ax=ax_dd, linewidth=1.0, color="tomato")
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, alpha=0.2, color="tomato")
        ax_dd.set_title("Drawdown")
        ax_dd.set_ylabel("Drawdown %")
        ax_dd.grid(True, alpha=0.3)

        # Bet PnL
        ax_pnl = axes[2]
        pnl_values = [b.get("pnl", 0) for b in result.bet_history]
        if pnl_values:
            colours = ["green" if p > 0 else "red" for p in pnl_values]
            ax_pnl.bar(range(len(pnl_values)), pnl_values, color=colours, alpha=0.7)
        ax_pnl.set_title("Bet PnL Distribution")
        ax_pnl.set_ylabel("PnL ($)")
        ax_pnl.set_xlabel("Bet #")
        ax_pnl.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Betting backtest plot saved to %s", save_path)

        plt.close(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> Optional[BettingBacktestResult]:
        return self._last_result

    @staticmethod
    def _american_to_decimal(odds: float) -> float:
        if odds > 0:
            return 1.0 + odds / 100.0
        elif odds < 0:
            return 1.0 + 100.0 / abs(odds)
        return 2.0
