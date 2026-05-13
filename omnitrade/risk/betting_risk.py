"""
Betting-specific risk management: Kelly criterion, stake capping,
consecutive loss kill switches, and responsible gaming controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from typing import Any

from omnitrade.config.settings import BettingSettings, Settings
from omnitrade.utils.logger import get_logger
from omnitrade.utils.odds import american_to_decimal

logger = get_logger(__name__)


@dataclass
class BetValidation:
    """Result of a bet validation check."""

    approved: bool
    adjusted_stake: float
    reason: str
    risk_score: float  # 0 = safe, 1 = blocked


class BettingRiskManager:
    """Risk management for sports betting with Kelly criterion.

    Enforces:
    - Bankroll sufficiency checks
    - Sport concentration limits
    - Max single stake percentage
    - Consecutive loss kill switch
    - Daily stake count limits
    - Fractional Kelly application
    """

    def __init__(self, settings: Settings, safety: Optional[Any] = None) -> None:
        self._settings = settings
        self._betting: BettingSettings = settings.betting
        self._consecutive_losses = 0
        self._daily_stakes_count = 0
        self._daily_stake_total = 0.0

        # Shared halt authority (optional, eliminates dual-halt)
        self._safety = safety

        # Local halt state (only used when _safety is None)
        self._own_halted = False
        self._own_halt_reason = ""

        self._sport_exposure: Dict[str, float] = {}
        self._bankroll_history: List[float] = []
        self._bankroll_history.append(self._betting.bankroll)

    # ------------------------------------------------------------------
    # Halt delegation (ADR 0003 — unified halt authority)
    # ------------------------------------------------------------------

    @property
    def _halted(self) -> bool:
        """Halt flag — delegates to SafetyGuard when wired."""
        if self._safety is not None:
            return self._safety.is_halted
        return self._own_halted

    @_halted.setter
    def _halted(self, value: bool) -> None:
        if self._safety is not None:
            return  # halt goes through SafetyGuard.halt()
        self._own_halted = value

    @property
    def _halt_reason(self) -> str:
        """Halt reason — delegates to SafetyGuard when wired."""
        if self._safety is not None:
            return self._safety.halt_reason
        return self._own_halt_reason

    @_halt_reason.setter
    def _halt_reason(self, value: str) -> None:
        if self._safety is not None:
            return  # halt reason set through SafetyGuard.halt()
        self._own_halt_reason = value

    # ------------------------------------------------------------------
    # Kelly stake computation
    # ------------------------------------------------------------------

    def kelly_stake(
        self,
        bankroll: float,
        model_prob: float,
        odds_decimal: float,
    ) -> float:
        """Full Kelly criterion stake.

        f* = (bp - q) / b

        Args:
            bankroll: Current bankroll.
            model_prob: Model-estimated probability (0-1).
            odds_decimal: Decimal odds.

        Returns:
            Full Kelly stake in USD.
        """
        b = odds_decimal - 1.0
        if b <= 0:
            return 0.0

        p = model_prob
        q = 1.0 - p
        f_star = (b * p - q) / b

        return max(0.0, bankroll * f_star)

    def adjusted_stake(self, full_kelly: float, bankroll: float) -> float:
        """Apply fractional Kelly and max stake cap."""
        adjusted = full_kelly * self._betting.kelly_fraction
        max_stake = bankroll * self._betting.max_stake_pct
        return min(adjusted, max_stake)

    # ------------------------------------------------------------------
    # Bet validation
    # ------------------------------------------------------------------

    def validate_bet(
        self,
        symbol: str,
        model_prob: float,
        implied_prob: float,
        odds_american: float,
        bankroll: float,
        sport_key: str = "",
    ) -> BetValidation:
        """Validate a proposed bet against all risk rules.

        Args:
            symbol: Match identifier.
            model_prob: Model probability (0-1).
            implied_prob: Market-implied probability (0-1).
            odds_american: American odds.
            bankroll: Current bankroll.
            sport_key: Sport identifier for concentration checks.

        Returns:
            BetValidation with decision and reasoning.
        """
        if self._halted:
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason=f"Halted: {self._halt_reason}", risk_score=1.0,
            )

        if bankroll <= 0:
            self._halt("Bankroll exhausted", bankroll)
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason="Bankroll exhausted", risk_score=1.0,
            )

        if bankroll < self._betting.bankroll * 0.10:
            warning = f"Bankroll critically low (${bankroll:.2f})"
            logger.warning(warning)

        if self._daily_stakes_count >= self._betting.max_daily_stakes:
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason=f"Max daily stakes ({self._betting.max_daily_stakes}) reached",
                risk_score=0.9,
            )

        if self._consecutive_losses >= self._betting.max_consecutive_losses:
            self._halt(
                f"{self._consecutive_losses} consecutive losses",
                bankroll,
            )
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason=f"Kill switch: {self._consecutive_losses} consecutive losses",
                risk_score=1.0,
            )

        edge = model_prob - implied_prob
        if edge < self._betting.min_edge_pct:
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason=f"Edge {edge:.3f} < min {self._betting.min_edge_pct}",
                risk_score=0.5,
            )

        odds_decimal = american_to_decimal(odds_american)
        full_kelly = self.kelly_stake(bankroll, model_prob, odds_decimal)
        stake = self.adjusted_stake(full_kelly, bankroll)

        if stake <= 0:
            return BetValidation(
                approved=False, adjusted_stake=0.0,
                reason="Kelly fraction <= 0 (no edge)", risk_score=0.3,
            )

        if sport_key and sport_key in self._sport_exposure:
            current_exposure = self._sport_exposure[sport_key]
            if current_exposure + stake > bankroll * 0.40:
                return BetValidation(
                    approved=False, adjusted_stake=0.0,
                    reason=f"Sport exposure cap reached for {sport_key}",
                    risk_score=0.8,
                )

        return BetValidation(
            approved=True,
            adjusted_stake=stake,
            reason=f"Approved: edge={edge:.3f}, stake=${stake:.2f}",
            risk_score=0.0,
        )

    # ------------------------------------------------------------------
    # Post-trade logging
    # ------------------------------------------------------------------

    def record_bet(self, symbol: str, stake: float, sport_key: str = "") -> None:
        self._daily_stakes_count += 1
        self._daily_stake_total += stake
        if sport_key:
            self._sport_exposure[sport_key] = self._sport_exposure.get(sport_key, 0) + stake

    def record_win(self) -> None:
        self._consecutive_losses = 0

    def record_loss(self) -> int:
        self._consecutive_losses += 1
        loss_count = self._consecutive_losses
        if loss_count >= self._betting.max_consecutive_losses:
            self._halt(f"{loss_count} consecutive losses", 0.0)
        return loss_count

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_daily(self, bankroll: float) -> None:
        self._daily_stakes_count = 0
        self._daily_stake_total = 0.0
        self._consecutive_losses = 0
        if self._safety is not None:
            self._safety.reset_daily()
        else:
            self._own_halted = False
            self._own_halt_reason = ""
        self._bankroll_history.append(bankroll)
        logger.info("Betting risk counters reset — bankroll: $%.2f", bankroll)

    def _halt(self, reason: str, bankroll: float) -> None:
        # Always set local state first (also covers the no-safety path)
        self._own_halted = True
        self._own_halt_reason = reason
        # Propagate to shared halt authority when wired
        if self._safety is not None:
            self._safety.halt(reason, bankroll)
        logger.critical("BETTING HALT: %s (bankroll: $%.2f)", reason, bankroll)

    def get_status(self) -> Dict[str, Any]:
        return {
            "halted": self._halted,  # delegates to SafetyGuard when wired
            "halt_reason": self._halt_reason,  # delegates to SafetyGuard when wired
            "consecutive_losses": self._consecutive_losses,
            "daily_stakes": self._daily_stakes_count,
            "daily_stake_total": round(self._daily_stake_total, 2),
            "max_consecutive_losses": self._betting.max_consecutive_losses,
            "sport_exposure": {k: round(v, 2) for k, v in self._sport_exposure.items()},
            "bankroll_history_len": len(self._bankroll_history),
        }

    @property
    def is_halted(self) -> bool:
        """Delegates to SafetyGuard when wired."""
        if self._safety is not None:
            return self._safety.is_halted
        return self._own_halted

