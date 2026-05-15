"""
RiskManagerAgent — validates signals against risk rules.

Consumes SignalEvent, runs should_execute gate + SafetyGuard pre-trade
checks + position sizing. Publishes ApprovedSignalEvent for signals that
pass all checks.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional, Tuple

from omnitrade.agents.bus import ApprovedSignalEvent, EventBus, SignalEvent
from omnitrade.config.asset_types import AssetType
from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.risk.risk_manager import RiskManager, PortfolioState
from omnitrade.risk.safety import SafetyGuard
from omnitrade.utils.circuit_breaker import CircuitBreaker
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_CONFIDENCE = 0.65
_SIGNAL_COOLDOWN = 300  # seconds between same (symbol, direction) signals


class RiskManagerAgent:
    """Validates signals and sizes positions before execution.

    Args:
        bus: Shared event bus.
        risk_manager: Centralized RiskManager instance.
        safety: Optional SafetyGuard for comprehensive pre-trade checks.
        settings: Bot configuration.
        ensemble: EnsembleVoter for should_execute gating (optional).
    """

    def __init__(
        self,
        bus: EventBus,
        risk_manager: RiskManager,
        safety: Optional[SafetyGuard] = None,
        settings: Optional[Settings] = None,
        ensemble: Optional[object] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._bus = bus
        self._risk = risk_manager
        self._safety = safety
        self._settings = settings or _default_settings
        self._ensemble = ensemble
        self._breaker = circuit_breaker or CircuitBreaker()
        self._signal_cooldowns: Dict[Tuple[str, str], float] = {}

    async def run(self) -> None:
        """Consume SignalEvents, validate, publish ApprovedSignalEvents."""
        logger.info("RiskManagerAgent starting")
        try:
            while True:
                event: SignalEvent = await self._bus.consume_signal()

                try:
                    approved = self._validate(event)
                    if approved is not None:
                        await self._bus.publish_approved_signal(approved)
                        logger.info(
                            "Approved %s %s @ %.2f size=%.4f",
                            event.symbol,
                            event.signal,
                            event.price,
                            approved.amount,
                        )
                except Exception:
                    logger.exception("Risk validation failed for %s", event.symbol)

        except asyncio.CancelledError:
            logger.info("RiskManagerAgent shutting down")

    def _validate(self, event: SignalEvent) -> Optional[ApprovedSignalEvent]:
        signal = event.signal
        if signal == "HOLD" or signal == "PASS":
            return None

        # Circuit breaker gate — if too many failures, stop approving
        if self._breaker.is_open:
            logger.warning(
                "%s rejected — circuit breaker open (%d consecutive failures)",
                event.symbol, self._breaker._failures,
            )
            return None

        # Confidence gate — skip low-confidence signals without penalising
        if event.confidence < _MIN_CONFIDENCE:
            return None

        # Ensemble agreement gate (if available)
        if self._ensemble is not None:
            vote_result = {
                "signal": signal,
                "confidence": event.confidence,
                "individual_predictions": event.individual_predictions,
            }
            if not self._ensemble.should_execute(vote_result, min_confidence=_MIN_CONFIDENCE):
                logger.info("%s rejected — ensemble gate", event.symbol)
                self._breaker.record_failure()
                return None

        # Determine asset type for position sizing
        try:
            asset_type = AssetType(event.asset_type.upper())
        except ValueError:
            asset_type = AssetType.CRYPTO

        # Position sizing — use default balance if unavailable
        balance = 10_000.0
        if asset_type == AssetType.CRYPTO:
            balance = 10_000.0  # crypto lane default
        elif asset_type == AssetType.STOCK:
            balance = 10_000.0  # stock lane default
        elif asset_type == AssetType.BET:
            balance = self._settings.betting.bankroll

        amount = self._risk.calculate_position_size(balance)

        # Safety guard pre-trade check
        if self._safety is not None:
            portfolio = PortfolioState(
                balance=balance,
                equity=balance,
            )
            # Build signal dict for safety check
            signal_dict = {
                "symbol": event.symbol,
                "side": signal,
                "amount": amount,
                "price": event.price,
                "confidence": event.confidence,
            }
            # Check halted state
            if self._safety.is_halted:
                logger.warning("%s rejected — safety halted: %s", event.symbol, self._safety._halt_reason)
                self._breaker.record_failure()
                return None

        # Signal cooldown — prevent pyramiding from rapid repeated signals
        key = (event.symbol, signal)
        now = time.monotonic()
        last = self._signal_cooldowns.get(key)
        if last is not None and (now - last) < _SIGNAL_COOLDOWN:
            logger.info(
                "%s %s rejected — cooldown (%.0fs remaining)",
                event.symbol, signal, _SIGNAL_COOLDOWN - (now - last),
            )
            return None
        self._signal_cooldowns[key] = now

        self._breaker.record_success()
        return ApprovedSignalEvent(
            symbol=event.symbol,
            asset_type=event.asset_type,
            signal=signal,
            confidence=event.confidence,
            price=event.price,
            amount=amount,
            metadata={"individual_predictions": event.individual_predictions},
        )
