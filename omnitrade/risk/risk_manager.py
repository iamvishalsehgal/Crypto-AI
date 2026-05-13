"""
Risk management engine.

Enforces position sizing, stop-loss / take-profit rules, drawdown limits,
and trade count caps to prevent the bot from making unchecked, risky trades.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeValidation:
    """Result of a trade validation check."""

    approved: bool
    adjusted_size: float
    reason: str
    risk_score: float  # 0 = no risk concern, 1 = maximum risk


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio for risk checks."""

    balance: float
    equity: float
    open_positions: List[Dict] = field(default_factory=list)
    daily_starting_equity: float = 0.0
    daily_pnl: float = 0.0


class RiskManager:
    """Centralized risk management for the trading bot.

    All trade signals pass through this manager before execution.  It can
    shrink position sizes, reject trades entirely, and trigger an emergency
    shutdown when the daily drawdown limit is breached.

    When *safety* is provided (a :class:`~omnitrade.risk.safety.SafetyGuard`),
    halt state is delegated to it — eliminating the dual-halt problem where
    ``RiskManager`` and ``SafetyGuard`` could independently block trades.

    Args:
        settings: Bot settings (uses ``trading.*`` parameters).
        safety: Optional :class:`SafetyGuard` for shared halt authority.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        safety: Optional[Any] = None,
    ) -> None:
        self._settings = settings or _default_settings
        t = self._settings.trading

        self.max_position_size: float = t.max_position_size
        self.stop_loss: float = t.stop_loss
        self.take_profit: float = t.take_profit
        self.max_daily_drawdown: float = t.max_daily_drawdown
        self.max_open_trades: int = t.max_open_trades

        # Shared halt authority (optional, eliminates dual-halt)
        self._safety = safety

        # Internal state
        self._daily_start_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._own_trading_halted: bool = False  # only used when _safety is None
        self._own_halt_reason: str = ""
        self._trailing_stops: Dict[str, float] = {}

        # Adaptive sizing: rolling 30-period return tracker
        self._return_history: deque[float] = deque(maxlen=30)
        self._adaptive_multiplier: float = 1.0
        self._adaptive_min: float = 0.5
        self._adaptive_max: float = 1.2

    # ------------------------------------------------------------------
    # Halt delegation
    # ------------------------------------------------------------------

    @property
    def _trading_halted(self) -> bool:
        """Halt flag — delegates to SafetyGuard when wired."""
        if self._safety is not None:
            return self._safety.is_halted
        return self._own_trading_halted

    @_trading_halted.setter
    def _trading_halted(self, value: bool) -> None:
        if self._safety is not None:
            # Don't directly set — halt goes through SafetyGuard._halt()
            return
        self._own_trading_halted = value

    @property
    def _halt_reason(self) -> str:
        """Halt reason — delegates to SafetyGuard when wired."""
        if self._safety is not None:
            return self._safety.halt_reason
        return self._own_halt_reason

    @_halt_reason.setter
    def _halt_reason(self, value: str) -> None:
        if self._safety is not None:
            return
        self._own_halt_reason = value

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def check_position_size(
        self, proposed_size: float, portfolio_value: float
    ) -> Tuple[bool, float]:
        """Cap a proposed position size at the maximum allowed fraction.

        Args:
            proposed_size: Requested position value in quote currency.
            portfolio_value: Current total portfolio value.

        Returns:
            Tuple of ``(is_within_limit, adjusted_size)``.
        """
        max_value = portfolio_value * self.max_position_size
        if proposed_size <= max_value:
            return True, proposed_size
        logger.warning(
            "Position size %.2f exceeds limit %.2f (%.0f%% of %.2f), capping",
            proposed_size,
            max_value,
            self.max_position_size * 100,
            portfolio_value,
        )
        return False, max_value

    def calculate_position_size(
        self,
        portfolio_value: float,
        risk_per_trade: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> float:
        """Calculate optimal position size.

        If *entry_price* and *stop_price* are provided the size is computed
        via a fixed-fractional (risk-per-trade) method.  Otherwise, the
        maximum allowed fraction is used directly.

        Args:
            portfolio_value: Current total portfolio value.
            risk_per_trade: Fraction of portfolio to risk (defaults to
                ``stop_loss``).
            entry_price: Planned entry price.
            stop_price: Planned stop-loss price.

        Returns:
            Position size in quote currency.
        """
        risk_frac = risk_per_trade or self.stop_loss

        if entry_price and stop_price and entry_price != stop_price:
            risk_amount = portfolio_value * risk_frac
            price_risk = abs(entry_price - stop_price) / entry_price
            size = risk_amount / price_risk if price_risk > 0 else 0.0
        else:
            size = portfolio_value * self.max_position_size

        # Apply adaptive multiplier (rolling Sharpe-based)
        size *= self._adaptive_multiplier

        # Never exceed the hard cap
        _, capped = self.check_position_size(size, portfolio_value)
        return capped

    # ------------------------------------------------------------------
    # Adaptive position sizing
    # ------------------------------------------------------------------

    def record_return(self, period_return: float) -> None:
        """Record a period return for rolling Sharpe calculation.

        Args:
            period_return: Return for the period (e.g. daily or per-cycle).
        """
        self._return_history.append(period_return)
        self._update_adaptive_multiplier()

    def _update_adaptive_multiplier(self) -> float:
        """Compute rolling Sharpe and map to a position size multiplier.

        Sharpe < 0    → 0.5x  (reduce exposure)
        Sharpe 0-0.5  → 0.5x-0.8x
        Sharpe 0.5-1  → 0.8x-1.0x
        Sharpe 1-2    → 1.0x-1.2x
        Sharpe > 2    → 1.2x  (max confidence)

        Uses at least 5 data points before adjusting.
        """
        if len(self._return_history) < 5:
            self._adaptive_multiplier = 1.0
            return self._adaptive_multiplier

        returns = np.array(self._return_history)
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))

        if std_r == 0:
            sharpe = 0.0
        else:
            sharpe = mean_r / std_r

        if sharpe <= 0:
            self._adaptive_multiplier = self._adaptive_min
        elif sharpe <= 0.5:
            self._adaptive_multiplier = self._adaptive_min + (sharpe / 0.5) * 0.3
        elif sharpe <= 1.0:
            self._adaptive_multiplier = 0.8 + ((sharpe - 0.5) / 0.5) * 0.2
        elif sharpe <= 2.0:
            self._adaptive_multiplier = 1.0 + ((sharpe - 1.0) / 1.0) * 0.2
        else:
            self._adaptive_multiplier = self._adaptive_max

        self._adaptive_multiplier = max(self._adaptive_min, min(self._adaptive_max, self._adaptive_multiplier))
        logger.debug(
            "Adaptive multiplier: %.2fx (Sharpe=%.2f, n=%d)",
            self._adaptive_multiplier, sharpe, len(self._return_history),
        )
        return self._adaptive_multiplier

    @property
    def adaptive_multiplier(self) -> float:
        return self._adaptive_multiplier

    def get_adaptive_sizing(self) -> Dict[str, Any]:
        """Return current adaptive sizing state."""
        if len(self._return_history) < 2:
            sharpe = 0.0
        else:
            arr = np.array(self._return_history)
            mean_r = float(np.mean(arr))
            std_r = float(np.std(arr))
            sharpe = mean_r / std_r if std_r > 0 else 0.0

        return {
            "adaptive_multiplier": round(self._adaptive_multiplier, 3),
            "rolling_sharpe": round(sharpe, 3),
            "return_samples": len(self._return_history),
            "min_multiplier": self._adaptive_min,
            "max_multiplier": self._adaptive_max,
        }

    # ------------------------------------------------------------------
    # Stop-loss / take-profit
    # ------------------------------------------------------------------

    def check_stop_loss(
        self, entry_price: float, current_price: float, side: str = "long"
    ) -> bool:
        """Return ``True`` if the stop-loss has been triggered."""
        if side == "long":
            pct_change = (current_price - entry_price) / entry_price
            return pct_change <= -self.stop_loss
        else:
            pct_change = (entry_price - current_price) / entry_price
            return pct_change <= -self.stop_loss

    def check_take_profit(
        self, entry_price: float, current_price: float, side: str = "long"
    ) -> bool:
        """Return ``True`` if the take-profit target has been reached."""
        if side == "long":
            pct_change = (current_price - entry_price) / entry_price
            return pct_change >= self.take_profit
        else:
            pct_change = (entry_price - current_price) / entry_price
            return pct_change >= self.take_profit

    def update_trailing_stop(
        self,
        position_id: str,
        entry_price: float,
        current_price: float,
        trailing_pct: float = 0.015,
    ) -> float:
        """Update and return the trailing stop price for a position.

        The trailing stop ratchets up (for longs) as the price rises but
        never moves down.

        Args:
            position_id: Unique identifier for the position.
            entry_price: Original entry price.
            current_price: Latest market price.
            trailing_pct: Distance from peak as a fraction.

        Returns:
            Current trailing stop price.
        """
        new_stop = current_price * (1.0 - trailing_pct)
        prev_stop = self._trailing_stops.get(position_id, entry_price * (1.0 - self.stop_loss))
        self._trailing_stops[position_id] = max(prev_stop, new_stop)
        return self._trailing_stops[position_id]

    # ------------------------------------------------------------------
    # Drawdown & trade caps
    # ------------------------------------------------------------------

    def check_daily_drawdown(
        self, daily_pnl: float, portfolio_value: float
    ) -> bool:
        """Return ``True`` if the daily drawdown limit has been breached.

        When breached the manager enters *halt* mode and blocks all new
        trades until :meth:`reset_daily` is called.
        """
        drawdown = -daily_pnl / portfolio_value if portfolio_value > 0 else 0.0
        if drawdown >= self.max_daily_drawdown:
            reason = (
                f"Daily drawdown {drawdown:.2%} exceeds limit "
                f"{self.max_daily_drawdown:.0%}"
            )
            if self._safety is not None:
                self._safety.halt("Daily drawdown", drawdown)
            else:
                self._trading_halted = True
                self._halt_reason = reason
            logger.critical("EMERGENCY HALT: %s", reason)
            return True
        return False

    def check_max_trades(self, current_open_trades: int) -> bool:
        """Return ``True`` if a new trade can be opened."""
        return current_open_trades < self.max_open_trades

    # ------------------------------------------------------------------
    # Trade validation (main entry point)
    # ------------------------------------------------------------------

    def validate_trade(
        self,
        signal: Dict,
        portfolio_state: PortfolioState,
    ) -> TradeValidation:
        """Validate a proposed trade signal against all risk rules.

        Args:
            signal: Dict with at least ``side`` (``"buy"``/``"sell"``),
                ``symbol``, and ``amount`` keys.
            portfolio_state: Current portfolio snapshot.

        Returns:
            :class:`TradeValidation` with the decision and reasoning.
        """
        # 1. Check halt
        if self._trading_halted:
            return TradeValidation(
                approved=False,
                adjusted_size=0.0,
                reason=f"Trading halted: {self._halt_reason}",
                risk_score=1.0,
            )

        # 2. Daily drawdown
        if self.check_daily_drawdown(portfolio_state.daily_pnl, portfolio_state.equity):
            return TradeValidation(
                approved=False,
                adjusted_size=0.0,
                reason=self._halt_reason,
                risk_score=1.0,
            )

        # 3. Max open trades
        open_count = len(portfolio_state.open_positions)
        if not self.check_max_trades(open_count):
            return TradeValidation(
                approved=False,
                adjusted_size=0.0,
                reason=f"Max open trades reached ({open_count}/{self.max_open_trades})",
                risk_score=0.7,
            )

        # 4. Position sizing
        proposed = signal.get("amount", 0.0)
        within_limit, adjusted = self.check_position_size(proposed, portfolio_state.equity)
        risk_score = 0.0 if within_limit else 0.4

        # 5. Concentration check — don't double-up on same symbol
        symbol = signal.get("symbol", "")
        existing = [
            p for p in portfolio_state.open_positions
            if p.get("symbol") == symbol
        ]
        if existing:
            risk_score = min(risk_score + 0.3, 1.0)
            logger.warning("Already have open position in %s, increasing risk score", symbol)

        return TradeValidation(
            approved=True,
            adjusted_size=adjusted,
            reason="Trade approved" + ("" if within_limit else " (size capped)"),
            risk_score=risk_score,
        )

    # ------------------------------------------------------------------
    # Portfolio-level checks for monitoring existing positions
    # ------------------------------------------------------------------

    def check_positions(
        self,
        positions: List[Dict],
        current_prices: Dict[str, float],
    ) -> List[Dict]:
        """Scan open positions for stop-loss / take-profit triggers.

        Args:
            positions: List of position dicts with ``symbol``, ``entry_price``,
                ``side``, and ``amount`` keys.
            current_prices: Mapping of symbol to latest price.

        Returns:
            List of action dicts for positions that need closing, each with
            ``symbol``, ``action`` (``"stop_loss"`` or ``"take_profit"``),
            and ``current_price``.
        """
        actions: List[Dict] = []
        for pos in positions:
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if price is None:
                continue

            entry = pos["entry_price"]
            side = pos.get("side", "long")
            pos_id = pos.get("id", symbol)

            # Trailing stop update
            trailing_stop = self.update_trailing_stop(pos_id, entry, price)
            if side == "long" and price <= trailing_stop:
                actions.append({"symbol": symbol, "action": "trailing_stop", "current_price": price})
                continue

            if self.check_stop_loss(entry, price, side):
                actions.append({"symbol": symbol, "action": "stop_loss", "current_price": price})
            elif self.check_take_profit(entry, price, side):
                actions.append({"symbol": symbol, "action": "take_profit", "current_price": price})

        return actions

    # ------------------------------------------------------------------
    # Reporting & state management
    # ------------------------------------------------------------------

    def get_risk_report(self, portfolio_state: PortfolioState) -> Dict:
        """Generate a summary of current risk metrics."""
        equity = portfolio_state.equity or 1.0
        drawdown = -portfolio_state.daily_pnl / equity if portfolio_state.daily_pnl < 0 else 0.0
        total_exposure = sum(
            p.get("amount", 0) * p.get("entry_price", 0)
            for p in portfolio_state.open_positions
        )

        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "open_positions": len(portfolio_state.open_positions),
            "max_open_trades": self.max_open_trades,
            "daily_pnl": portfolio_state.daily_pnl,
            "daily_drawdown_pct": drawdown,
            "max_daily_drawdown": self.max_daily_drawdown,
            "drawdown_remaining": max(0, self.max_daily_drawdown - drawdown),
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / equity if equity > 0 else 0.0,
            "trailing_stops": dict(self._trailing_stops),
        }

    def reset_daily(self, starting_equity: float) -> None:
        """Reset daily counters (call at the start of each trading day)."""
        self._daily_start_equity = starting_equity
        self._daily_pnl = 0.0
        if self._safety is not None:
            self._safety.reset_daily()
        else:
            self._trading_halted = False
            self._halt_reason = ""
        logger.info("Daily risk counters reset.  Starting equity: %.2f", starting_equity)

    def clear_trailing_stop(self, position_id: str) -> None:
        """Remove the trailing stop for a closed position."""
        self._trailing_stops.pop(position_id, None)
