"""
Core backtesting engine for the AI Crypto Trading Bot.

Simulates strategy execution on historical data with realistic transaction
fees, slippage, and support for long-only or long/short strategies.  Produces
comprehensive performance metrics and visualisations.
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Trade = namedtuple("Trade", [
    "timestamp",
    "side",       # "buy" or "sell"
    "price",      # execution price after fees/slippage
    "raw_price",  # market price before adjustments
    "amount",     # position size in base currency
    "fee",        # total fee paid
    "pnl",        # realised PnL (0.0 for opening trades)
    "balance",    # portfolio balance after trade
])


@dataclass
class BacktestResult:
    """Container for all backtesting outputs."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float
    calmar_ratio: float
    trade_history: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulate a trading strategy on historical OHLCV data.

    Parameters
    ----------
    settings:
        Application settings.  ``settings.backtesting.transaction_fee`` and
        ``settings.backtesting.slippage`` drive realistic execution costs.
    initial_balance:
        Starting portfolio value in quote currency (e.g. USDT).
    """

    def __init__(self, settings: Settings, initial_balance: float = 10_000.0) -> None:
        self._settings = settings
        self._initial_balance = initial_balance

        self._transaction_fee: float = settings.backtesting.transaction_fee
        self._slippage: float = settings.backtesting.slippage

        # Mutable state -- reset on every run
        self._balance: float = initial_balance
        self._position: float = 0.0          # signed: +long, -short
        self._entry_price: float = 0.0
        self._trades: List[Trade] = []

        logger.info(
            "BacktestEngine initialised (balance=%.2f, fee=%.4f, slippage=%.4f)",
            initial_balance,
            self._transaction_fee,
            self._slippage,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        strategy_func: Callable[[pd.DataFrame, int], Optional[Dict]],
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Execute a backtest over *data*.

        Parameters
        ----------
        strategy_func:
            ``strategy_func(data, current_index)`` must return a signal dict
            with at least ``{"side": "buy"|"sell"|"close", "amount": float}``
            or ``None`` when no action should be taken.
        data:
            DataFrame with at minimum ``close`` and a datetime index or a
            ``timestamp`` column.

        Returns
        -------
        BacktestResult
            Aggregated performance metrics and full trade history.
        """
        self._reset()

        if data.empty:
            logger.warning("Empty data provided to backtest; returning zero result")
            return self._empty_result()

        # Normalise index to datetime
        if "timestamp" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index("timestamp")

        equity_values: List[float] = []
        equity_index: List = []

        for i in range(len(data)):
            price = float(data["close"].iloc[i])
            timestamp = data.index[i]

            # Ask strategy for a signal
            try:
                signal = strategy_func(data, i)
            except Exception:
                logger.exception("Strategy raised at index %d; skipping", i)
                signal = None

            if signal is not None:
                self._execute_trade(signal, price, timestamp)

            # Mark-to-market portfolio value
            mtm = self._balance
            if self._position != 0.0:
                unrealised = self._position * (price - self._entry_price)
                mtm += unrealised
            equity_values.append(mtm)
            equity_index.append(timestamp)

        equity_curve = pd.Series(equity_values, index=equity_index, name="equity")

        metrics = self._calculate_metrics(self._trades, equity_curve)

        result = BacktestResult(
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=metrics["total_trades"],
            avg_trade_return=metrics["avg_trade_return"],
            profit_factor=metrics["profit_factor"],
            calmar_ratio=metrics["calmar_ratio"],
            trade_history=list(self._trades),
            equity_curve=equity_curve,
        )

        logger.info(
            "Backtest complete: %d trades, return=%.2f%%, sharpe=%.2f, maxDD=%.2f%%",
            result.total_trades,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
        )
        return result

    def plot_results(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot equity curve, drawdown, and trade markers.

        Parameters
        ----------
        result:
            Output from :meth:`run`.
        save_path:
            If given, save the figure to this file path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # --- Equity curve ---
        ax_eq = axes[0]
        result.equity_curve.plot(ax=ax_eq, linewidth=1.2, color="steelblue")
        ax_eq.set_title("Equity Curve")
        ax_eq.set_ylabel("Portfolio Value")
        ax_eq.grid(True, alpha=0.3)

        # Trade markers
        for trade in result.trade_history:
            colour = "green" if trade.side == "buy" else "red"
            marker = "^" if trade.side == "buy" else "v"
            ax_eq.scatter(
                trade.timestamp,
                trade.balance,
                color=colour,
                marker=marker,
                s=40,
                zorder=5,
                alpha=0.7,
            )

        # --- Drawdown ---
        ax_dd = axes[1]
        running_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - running_max) / running_max
        drawdown.plot(ax=ax_dd, linewidth=1.0, color="tomato")
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, alpha=0.2, color="tomato")
        ax_dd.set_title("Drawdown")
        ax_dd.set_ylabel("Drawdown %")
        ax_dd.grid(True, alpha=0.3)

        # --- Trade PnL ---
        ax_pnl = axes[2]
        pnl_values = [t.pnl for t in result.trade_history if t.pnl != 0.0]
        if pnl_values:
            colours = ["green" if p > 0 else "red" for p in pnl_values]
            ax_pnl.bar(range(len(pnl_values)), pnl_values, color=colours, alpha=0.7)
        ax_pnl.set_title("Trade PnL")
        ax_pnl.set_ylabel("PnL")
        ax_pnl.set_xlabel("Trade #")
        ax_pnl.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Backtest plot saved to %s", save_path)

        return fig

    # ------------------------------------------------------------------ #
    # Trade execution
    # ------------------------------------------------------------------ #

    def _execute_trade(
        self,
        signal: Dict,
        price: float,
        timestamp,
    ) -> Optional[Trade]:
        """Execute a single trade based on *signal*.

        Parameters
        ----------
        signal:
            Must contain ``side`` (``"buy"``, ``"sell"``, or ``"close"``)
            and ``amount`` (float, in quote currency for buys, base for sells).
        price:
            Current market price.
        timestamp:
            Bar timestamp.

        Returns
        -------
        Trade or None
        """
        side: str = signal.get("side", "").lower()
        amount: float = float(signal.get("amount", 0.0))

        if side == "close":
            return self._close_position(price, timestamp)

        if side not in ("buy", "sell"):
            return None

        if amount <= 0:
            return None

        adjusted_price = self._apply_fees_and_slippage(price, side)
        fee = abs(amount * self._transaction_fee * price)

        trade: Optional[Trade] = None

        if side == "buy":
            cost = amount * adjusted_price + fee
            if cost > self._balance:
                # Adjust amount to what we can afford
                amount = (self._balance - fee) / adjusted_price
                if amount <= 0:
                    return None
                cost = amount * adjusted_price + fee
                fee = abs(amount * self._transaction_fee * price)

            self._balance -= cost
            # If we were short, this partially/fully closes the short first
            if self._position < 0:
                pnl = abs(min(amount, abs(self._position))) * (self._entry_price - adjusted_price)
                self._balance += pnl
            else:
                pnl = 0.0

            self._position += amount
            if self._position > 0:
                self._entry_price = adjusted_price

            trade = Trade(
                timestamp=timestamp,
                side=side,
                price=adjusted_price,
                raw_price=price,
                amount=amount,
                fee=fee,
                pnl=pnl,
                balance=self._balance + (self._position * adjusted_price if self._position > 0 else 0),
            )

        elif side == "sell":
            if self._position <= 0 and amount > 0:
                # Opening a short position
                self._position -= amount
                self._entry_price = adjusted_price
                pnl = 0.0
            elif self._position > 0:
                # Closing/reducing a long position
                sell_amount = min(amount, self._position)
                pnl = sell_amount * (adjusted_price - self._entry_price)
                self._balance += sell_amount * adjusted_price - fee
                self._position -= sell_amount
                if self._position == 0:
                    self._entry_price = 0.0
            else:
                return None

            trade = Trade(
                timestamp=timestamp,
                side=side,
                price=adjusted_price,
                raw_price=price,
                amount=amount,
                fee=fee,
                pnl=pnl,
                balance=self._balance,
            )

        if trade is not None:
            self._trades.append(trade)

        return trade

    def _close_position(self, price: float, timestamp) -> Optional[Trade]:
        """Flatten the current position completely."""
        if self._position == 0:
            return None

        if self._position > 0:
            side = "sell"
            adjusted_price = self._apply_fees_and_slippage(price, "sell")
            pnl = self._position * (adjusted_price - self._entry_price)
            fee = abs(self._position * self._transaction_fee * price)
            self._balance += self._position * adjusted_price - fee
        else:
            side = "buy"
            adjusted_price = self._apply_fees_and_slippage(price, "buy")
            pnl = abs(self._position) * (self._entry_price - adjusted_price)
            fee = abs(self._position * self._transaction_fee * price)
            self._balance += pnl - fee

        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=adjusted_price,
            raw_price=price,
            amount=abs(self._position),
            fee=fee,
            pnl=pnl,
            balance=self._balance,
        )
        self._trades.append(trade)

        self._position = 0.0
        self._entry_price = 0.0

        return trade

    def _apply_fees_and_slippage(self, price: float, side: str) -> float:
        """Return the execution price after applying slippage.

        For buys the price is *worse* (higher); for sells the price is
        *worse* (lower).  The fee is applied separately in the calling
        method so only slippage shifts the price here.

        Parameters
        ----------
        price:
            Raw market price.
        side:
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        float
            Adjusted execution price.
        """
        if side == "buy":
            return price * (1.0 + self._slippage)
        return price * (1.0 - self._slippage)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
    ) -> Dict[str, float]:
        """Compute all performance metrics from the trade list and equity curve."""
        total_return = (equity_curve.iloc[-1] / self._initial_balance) - 1.0 if len(equity_curve) > 0 else 0.0

        returns = equity_curve.pct_change().dropna()

        sharpe = self._calculate_sharpe(returns)
        max_dd = self._calculate_max_drawdown(equity_curve)
        profit_factor = self._calculate_profit_factor(trades)

        # Win rate
        closed_trades = [t for t in trades if t.pnl != 0.0]
        winning = [t for t in closed_trades if t.pnl > 0]
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0.0

        # Average trade return
        trade_returns = [t.pnl for t in closed_trades]
        avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

        # Calmar ratio: annualised return / max drawdown
        if len(equity_curve) >= 2:
            n_periods = len(equity_curve)
            # Assume daily data if we cannot infer frequency
            try:
                freq = pd.infer_freq(equity_curve.index)
                if freq and "H" in str(freq).upper():
                    annualisation_factor = 365 * 24 / max(n_periods, 1)
                elif freq and "T" in str(freq).upper():
                    annualisation_factor = 365 * 24 * 60 / max(n_periods, 1)
                else:
                    annualisation_factor = 365 / max(n_periods, 1)
            except (TypeError, ValueError):
                annualisation_factor = 365 / max(n_periods, 1)

            annualised_return = (1.0 + total_return) ** annualisation_factor - 1.0
        else:
            annualised_return = 0.0

        calmar = abs(annualised_return / max_dd) if max_dd != 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "avg_trade_return": avg_trade_return,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
        }

    @staticmethod
    def _calculate_sharpe(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Compute the annualised Sharpe ratio.

        Parameters
        ----------
        returns:
            Series of period returns.
        risk_free_rate:
            Annual risk-free rate.

        Returns
        -------
        float
            Annualised Sharpe ratio; 0.0 if insufficient data.
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        # Assume 252 trading days as default annualisation factor
        periods_per_year = 252
        excess = returns.mean() - risk_free_rate / periods_per_year
        return float((excess / returns.std()) * math.sqrt(periods_per_year))

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Return the maximum drawdown as a positive fraction.

        Parameters
        ----------
        equity_curve:
            Portfolio value over time.

        Returns
        -------
        float
            Maximum peak-to-trough decline (e.g. 0.15 means 15%).
        """
        if equity_curve.empty:
            return 0.0

        running_max = equity_curve.cummax()
        drawdowns = (equity_curve - running_max) / running_max
        return float(abs(drawdowns.min()))

    @staticmethod
    def _calculate_profit_factor(trades: List[Trade]) -> float:
        """Gross profit / gross loss.

        Parameters
        ----------
        trades:
            List of executed trades.

        Returns
        -------
        float
            Profit factor; ``float('inf')`` if there are no losing trades
            but there are winners, ``0.0`` if there are no trades.
        """
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reset(self) -> None:
        """Reset all mutable state for a fresh backtest run."""
        self._balance = self._initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._trades = []

    def _empty_result(self) -> BacktestResult:
        """Return a zeroed-out result when no data is available."""
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            trade_history=[],
            equity_curve=pd.Series(dtype=float),
        )
