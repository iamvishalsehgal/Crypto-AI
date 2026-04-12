"""
Comprehensive safety layer — the last gate before any trade executes.

Covers:
  - Fee estimation & fee-to-profit ratio check
  - Daily fee cap
  - Balance reserve (untouchable emergency fund)
  - Minimum balance to keep trading
  - Per-symbol cooldowns (extra cooldown after losses)
  - Hourly and daily trade rate limits
  - Duplicate / same-direction position blocking
  - Stale data rejection
  - Volatility spike filter (ATR-based)
  - Consecutive-loss kill switch
  - Equity peak drawdown kill switch
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crypto_bot.config.settings import Settings, settings as _default_settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyVerdict:
    """Result of a safety pre-flight check."""
    safe: bool
    reason: str
    adjusted_size: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class SafetyGuard:
    """Every trade proposal must pass through this guard before execution.

    Configure via ``settings.safety.*`` or environment variables prefixed
    with ``SAFETY_``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._s = (settings or _default_settings).safety
        self._ts = (settings or _default_settings).trading
        self._bs = (settings or _default_settings).backtesting

        # ── Stateful trackers ──
        self._daily_fees: float = 0.0
        self._daily_trade_count: int = 0
        self._hourly_trades: deque = deque()          # timestamps
        self._last_trade_time: Dict[str, float] = {}  # symbol → epoch
        self._last_trade_was_loss: Dict[str, bool] = {}
        self._consecutive_losses: int = 0
        self._peak_equity: float = 0.0
        self._halted: bool = False
        self._halt_reason: str = ""

    # ==================================================================
    # Main entry point
    # ==================================================================

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        trade_size_usd: float,
        current_balance: float,
        current_equity: float,
        current_price: float,
        open_positions: List[Dict],
        features_row: Optional[Dict] = None,
        data_timestamp: Optional[float] = None,
    ) -> SafetyVerdict:
        """Run every safety check. Returns a verdict with pass/fail and reason."""

        warnings: List[str] = []

        # ── 0. Kill switch ──
        if self._halted:
            return SafetyVerdict(False, f"HALTED: {self._halt_reason}")

        # ── 1. Equity peak drawdown ──
        verdict = self._check_equity_drawdown(current_equity)
        if not verdict.safe:
            return verdict

        # ── 2. Consecutive loss kill switch ──
        if self._consecutive_losses >= self._s.kill_on_consecutive_losses:
            self._halt("Consecutive losses", self._consecutive_losses)
            return SafetyVerdict(
                False,
                f"KILL SWITCH: {self._consecutive_losses} consecutive losses",
            )

        # ── 3. Minimum balance ──
        if current_balance < self._s.min_balance_to_trade:
            return SafetyVerdict(
                False,
                f"Balance ${current_balance:.2f} below minimum ${self._s.min_balance_to_trade:.2f}",
            )

        # ── 4. Balance reserve — never trade the emergency fund ──
        tradeable = current_balance * (1.0 - self._s.reserve_balance_pct)
        if trade_size_usd > tradeable:
            trade_size_usd = tradeable
            warnings.append(
                f"Size capped to ${tradeable:.2f} (reserve={self._s.reserve_balance_pct:.0%})"
            )
        if trade_size_usd < self._s.min_trade_size_usd:
            return SafetyVerdict(
                False,
                f"Trade ${trade_size_usd:.2f} below minimum ${self._s.min_trade_size_usd:.2f} after reserve",
            )

        # ── 5. Fee estimation ──
        verdict = self._check_fees(trade_size_usd)
        if not verdict.safe:
            return verdict
        warnings.extend(verdict.warnings)

        # ── 6. Daily fee cap ──
        estimated_fee = trade_size_usd * self._bs.transaction_fee
        if self._daily_fees + estimated_fee > self._s.max_daily_fees:
            return SafetyVerdict(
                False,
                f"Daily fee cap: ${self._daily_fees:.2f} + ${estimated_fee:.2f} > ${self._s.max_daily_fees:.2f}",
            )

        # ── 7. Rate limits ──
        verdict = self._check_rate_limits()
        if not verdict.safe:
            return verdict

        # ── 8. Per-symbol cooldown ──
        verdict = self._check_cooldown(symbol)
        if not verdict.safe:
            return verdict

        # ── 9. Duplicate position check ──
        verdict = self._check_duplicate_position(symbol, side, open_positions)
        if not verdict.safe:
            return verdict
        warnings.extend(verdict.warnings)

        # ── 10. Stale data ──
        if data_timestamp is not None:
            verdict = self._check_data_freshness(data_timestamp)
            if not verdict.safe:
                return verdict

        # ── 11. Volatility filter ──
        if features_row is not None:
            verdict = self._check_volatility(features_row)
            if not verdict.safe:
                return verdict
            warnings.extend(verdict.warnings)

        # ── 12. Profit vs fees viability ──
        verdict = self._check_profit_viability(trade_size_usd, current_price)
        if not verdict.safe:
            return verdict

        return SafetyVerdict(
            safe=True,
            reason="All safety checks passed",
            adjusted_size=trade_size_usd,
            warnings=warnings,
        )

    # ==================================================================
    # Post-trade hooks
    # ==================================================================

    def record_trade(self, symbol: str, fee: float, is_loss: bool = False) -> None:
        """Call after every executed trade to update safety state."""
        now = time.time()
        self._daily_fees += fee
        self._daily_trade_count += 1
        self._hourly_trades.append(now)
        self._last_trade_time[symbol] = now

        if is_loss:
            self._consecutive_losses += 1
            self._last_trade_was_loss[symbol] = True
            logger.warning(
                "Consecutive losses: %d/%d",
                self._consecutive_losses, self._s.kill_on_consecutive_losses,
            )
        else:
            self._consecutive_losses = 0
            self._last_trade_was_loss[symbol] = False

    def update_equity_peak(self, equity: float) -> None:
        """Track high-water mark for drawdown kill switch."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_fees = 0.0
        self._daily_trade_count = 0
        self._halted = False
        self._halt_reason = ""
        logger.info("Safety daily counters reset")

    # ==================================================================
    # Individual checks
    # ==================================================================

    def _check_equity_drawdown(self, equity: float) -> SafetyVerdict:
        self.update_equity_peak(equity)
        if self._peak_equity <= 0:
            return SafetyVerdict(True, "ok")
        dd = (self._peak_equity - equity) / self._peak_equity
        if dd >= self._s.kill_on_equity_drop_pct:
            self._halt("Equity drawdown", dd)
            return SafetyVerdict(
                False,
                f"KILL SWITCH: equity down {dd:.1%} from peak ${self._peak_equity:.2f} "
                f"(limit {self._s.kill_on_equity_drop_pct:.0%})",
            )
        return SafetyVerdict(True, "ok")

    def _check_fees(self, trade_size: float) -> SafetyVerdict:
        fee_rate = self._bs.transaction_fee
        fee = trade_size * fee_rate
        # Round-trip fees (open + close)
        round_trip_fee = fee * 2
        round_trip_pct = round_trip_fee / trade_size if trade_size > 0 else 0

        warnings = []
        if round_trip_pct > self._s.max_fee_per_trade_pct * 2:
            return SafetyVerdict(
                False,
                f"Round-trip fees {round_trip_pct:.3%} too high "
                f"(limit {self._s.max_fee_per_trade_pct * 2:.3%})",
            )
        if round_trip_pct > self._s.max_fee_per_trade_pct:
            warnings.append(f"Fees elevated: {round_trip_pct:.3%} round-trip")

        return SafetyVerdict(True, "ok", warnings=warnings)

    def _check_rate_limits(self) -> SafetyVerdict:
        now = time.time()

        # Purge old hourly entries
        while self._hourly_trades and (now - self._hourly_trades[0]) > 3600:
            self._hourly_trades.popleft()

        if len(self._hourly_trades) >= self._s.max_trades_per_hour:
            return SafetyVerdict(
                False,
                f"Hourly rate limit: {len(self._hourly_trades)}/{self._s.max_trades_per_hour}",
            )

        if self._daily_trade_count >= self._s.max_trades_per_day:
            return SafetyVerdict(
                False,
                f"Daily rate limit: {self._daily_trade_count}/{self._s.max_trades_per_day}",
            )

        return SafetyVerdict(True, "ok")

    def _check_cooldown(self, symbol: str) -> SafetyVerdict:
        last = self._last_trade_time.get(symbol, 0)
        elapsed = time.time() - last

        # Extra cooldown after a loss
        cooldown = self._s.cooldown_after_trade_sec
        if self._last_trade_was_loss.get(symbol, False):
            cooldown = max(cooldown, self._s.cooldown_after_loss_sec)

        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return SafetyVerdict(
                False,
                f"Cooldown: {symbol} traded {elapsed:.0f}s ago "
                f"(need {cooldown}s, {remaining:.0f}s left)",
            )
        return SafetyVerdict(True, "ok")

    def _check_duplicate_position(
        self, symbol: str, side: str, positions: List[Dict]
    ) -> SafetyVerdict:
        existing = [p for p in positions if p.get("symbol") == symbol]
        warnings = []

        if existing:
            same_side = [p for p in existing if p.get("side") == side]
            if same_side:
                return SafetyVerdict(
                    False,
                    f"Already have {side} position in {symbol} — no stacking",
                )
            warnings.append(f"Opening opposite position in {symbol} (has existing)")

        return SafetyVerdict(True, "ok", warnings=warnings)

    def _check_data_freshness(self, data_ts: float) -> SafetyVerdict:
        age = time.time() - data_ts
        if age > self._s.max_data_age_sec:
            return SafetyVerdict(
                False,
                f"Stale data: {age:.0f}s old (limit {self._s.max_data_age_sec}s)",
            )
        return SafetyVerdict(True, "ok")

    def _check_volatility(self, features: Dict) -> SafetyVerdict:
        atr = features.get("atr")
        close = features.get("close")
        bb_bandwidth = features.get("bb_bandwidth")

        warnings = []

        # ATR spike detection
        if atr is not None and close is not None and close > 0:
            atr_pct = atr / close
            # If ATR > 5% of price, market is extremely volatile
            if atr_pct > 0.05:
                return SafetyVerdict(
                    False,
                    f"Extreme volatility: ATR {atr_pct:.2%} of price — sitting out",
                )
            if atr_pct > 0.03:
                warnings.append(f"High volatility: ATR {atr_pct:.2%}")

        # Bollinger bandwidth spike
        if bb_bandwidth is not None and bb_bandwidth > 0.15:
            warnings.append(f"Wide Bollinger bands: {bb_bandwidth:.3f}")

        return SafetyVerdict(True, "ok", warnings=warnings)

    def _check_profit_viability(
        self, trade_size: float, price: float
    ) -> SafetyVerdict:
        """Reject if expected minimum move can't cover round-trip fees."""
        fee_rate = self._bs.transaction_fee
        slippage = self._bs.slippage
        total_cost_pct = (fee_rate + slippage) * 2  # round trip

        if total_cost_pct >= self._s.min_profit_after_fees:
            return SafetyVerdict(
                False,
                f"Fees+slippage ({total_cost_pct:.3%} round-trip) exceed "
                f"min profit target ({self._s.min_profit_after_fees:.3%})",
            )
        return SafetyVerdict(True, "ok")

    # ==================================================================
    # Internal
    # ==================================================================

    def _halt(self, reason: str, value: Any) -> None:
        self._halted = True
        self._halt_reason = f"{reason}: {value}"
        logger.critical("SAFETY HALT: %s = %s", reason, value)

    @property
    def is_halted(self) -> bool:
        return self._halted

    def get_status(self) -> Dict:
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_fees": round(self._daily_fees, 2),
            "daily_fee_cap": self._s.max_daily_fees,
            "daily_trades": self._daily_trade_count,
            "hourly_trades": len(self._hourly_trades),
            "consecutive_losses": self._consecutive_losses,
            "peak_equity": round(self._peak_equity, 2),
            "last_trade_times": {
                k: datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
                for k, v in self._last_trade_time.items()
            },
        }
