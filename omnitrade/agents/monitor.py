"""
MonitorAgent — watches open positions for stop-loss / take-profit triggers.

Periodically scans all open positions across lanes and publishes
ExitSignalEvents when thresholds are breached.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from omnitrade.agents.bus import EventBus, ExitSignalEvent
from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.execution.asset_router import AssetRouter
from omnitrade.risk.risk_manager import RiskManager
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_CHECK_INTERVAL = 30  # seconds between position checks


class MonitorAgent:
    """Watches positions and publishes exit signals on stop/take-profit hits.

    Args:
        bus: Shared event bus.
        router: AssetRouter for position queries.
        risk_manager: RiskManager for stop-loss/take-profit checks.
        settings: Bot configuration.
    """

    def __init__(
        self,
        bus: EventBus,
        router: AssetRouter,
        risk_manager: RiskManager,
        settings: Optional[Settings] = None,
    ) -> None:
        self._bus = bus
        self._router = router
        self._risk = risk_manager
        self._settings = settings or _default_settings

    async def run(self) -> None:
        """Periodically scan positions and publish exit signals."""
        logger.info("MonitorAgent starting (every %ds)", _CHECK_INTERVAL)
        try:
            while True:
                try:
                    await self._check_positions()
                except Exception:
                    logger.exception("Position check failed")

                await asyncio.sleep(_CHECK_INTERVAL)
        except asyncio.CancelledError:
            logger.info("MonitorAgent shutting down")

    async def _check_positions(self) -> None:
        positions = self._router.get_all_positions()
        if not positions:
            logger.debug("No open positions to monitor")
            return
        logger.info("Monitoring %d open position(s)", len(positions))

        # Build a simple price provider from router
        for pos in positions:
            symbol = pos.get("symbol", "")
            if not symbol:
                continue

            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            amount = pos.get("amount", 0)
            side = pos.get("side", "buy")

            if entry_price <= 0 or current_price <= 0 or amount <= 0:
                continue

            # Stop-loss check
            stop_price = entry_price * (1 - self._settings.trading.stop_loss)
            if side == "buy" and current_price <= stop_price:
                await self._bus.publish_exit_signal(
                    ExitSignalEvent(
                        symbol=symbol,
                        asset_type=pos.get("asset_type", "crypto"),
                        reason="stop_loss",
                        price=current_price,
                        amount=amount,
                    )
                )
                logger.warning("%s stop-loss triggered @ %.2f", symbol, current_price)
                continue

            # Take-profit check
            take_profit_price = entry_price * (1 + self._settings.trading.take_profit)
            if side == "buy" and current_price >= take_profit_price:
                await self._bus.publish_exit_signal(
                    ExitSignalEvent(
                        symbol=symbol,
                        asset_type=pos.get("asset_type", "crypto"),
                        reason="take_profit",
                        price=current_price,
                        amount=amount,
                    )
                )
                logger.info("%s take-profit triggered @ %.2f", symbol, current_price)
