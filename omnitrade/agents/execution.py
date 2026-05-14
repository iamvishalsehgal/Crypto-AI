"""
ExecutionAgent — executes approved signals and handles exit signals.

Consumes ApprovedSignalEvent and ExitSignalEvent from the bus. Routes
approved trades through AssetRouter, stores to DB, sends alerts.
Closes positions on exit signals.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from omnitrade.agents.bus import (
    ApprovedSignalEvent,
    EventBus,
    ExitSignalEvent,
    PortfolioEvent,
)
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.data.storage.database import MongoDBStorage
from omnitrade.execution.asset_router import AssetRouter
from omnitrade.monitoring.telegram_bot import TelegramNotifier
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_PORTFOLIO_INTERVAL = 60  # seconds between portfolio snapshots


class ExecutionAgent:
    """Executes approved trades and monitors exits.

    Args:
        bus: Shared event bus.
        router: AssetRouter dispatching to correct lane executor.
        db: MongoDB storage for trade recording.
        notifier: Telegram bot for alerts (optional).
        retrainer: AutoRetrainer for trade outcome recording (optional).
    """

    def __init__(
        self,
        bus: EventBus,
        router: AssetRouter,
        db: MongoDBStorage,
        notifier: Optional[TelegramNotifier] = None,
        retrainer: Optional[object] = None,
    ) -> None:
        self._bus = bus
        self._router = router
        self._db = db
        self._notifier = notifier
        self._retrainer = retrainer

    async def run(self) -> None:
        """Consume approved signals and exit signals, execute trades."""
        logger.info("ExecutionAgent starting")

        async def handle_approved():
            while True:
                event: ApprovedSignalEvent = await self._bus.consume_approved_signal()
                try:
                    await self._execute(event)
                except Exception:
                    logger.exception("Execution failed for %s", event.symbol)

        async def handle_exits():
            while True:
                event: ExitSignalEvent = await self._bus.consume_exit_signal()
                try:
                    await self._close_position(event)
                except Exception:
                    logger.exception("Exit failed for %s", event.symbol)

        async def report_portfolio():
            while True:
                try:
                    unified = self._router.get_unified_balance()
                    await self._bus.publish_portfolio(
                        PortfolioEvent(
                            balance_usd=unified.get("total_usd_equivalent", 0),
                            equity=unified.get("total_usd_equivalent", 0),
                            return_pct=0.0,
                            open_positions=len(unified.get("positions", [])),
                            lanes=unified.get("by_lane", {}),
                        )
                    )
                except Exception:
                    logger.exception("Portfolio report failed")
                await asyncio.sleep(_PORTFOLIO_INTERVAL)

        tasks = [
            asyncio.create_task(handle_approved()),
            asyncio.create_task(handle_exits()),
            asyncio.create_task(report_portfolio()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("ExecutionAgent shutting down")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(self, event: ApprovedSignalEvent) -> None:
        logger.info(
            "Executing %s %s: %.4f @ %.2f",
            event.symbol,
            event.signal,
            event.amount,
            event.price,
        )

        try:
            asset_type = AssetType(event.asset_type.upper())
        except ValueError:
            asset_type = AssetType.CRYPTO

        signal = UnifiedSignal(
            asset_type=asset_type,
            symbol=event.symbol,
            side=event.signal,
            confidence=event.confidence,
            amount=event.amount,
            price=event.price,
            metadata=event.metadata,
        )

        result = self._router.route_signal(signal)

        if result.get("status") == "filled":
            logger.info("Trade EXECUTED: %s %s @ %.2f", event.symbol, event.signal, event.price)

            if self._notifier:
                await self._notifier.send_trade_alert(result)

            try:
                self._db.store_trade(result)
                logger.info("Recorded trade: %s %s @ %s", event.signal, event.symbol, result.get("price", 0))
            except Exception:
                logger.warning("Failed to store trade: %s", result)

            if self._retrainer and hasattr(self._retrainer, "record_trade"):
                self._retrainer.record_trade(result)
        else:
            logger.info("Trade NOT filled: %s — %s", event.symbol, result.get("reason", "unknown"))

    async def _close_position(self, event: ExitSignalEvent) -> None:
        logger.info(
            "Closing %s: %s @ %.2f (%.4f units)",
            event.symbol,
            event.reason,
            event.price,
            event.amount,
        )

        try:
            asset_type = AssetType(event.asset_type.upper())
        except ValueError:
            asset_type = AssetType.CRYPTO

        if asset_type == AssetType.CRYPTO and self._router.crypto_executor:
            self._router.crypto_executor.close_position(event.symbol, event.price)
        elif asset_type == AssetType.STOCK and self._router.stock_executor:
            self._router.stock_executor.close_position(event.symbol, event.price)

        if self._notifier:
            await self._notifier.send_message(
                f"Position closed: {event.symbol} — {event.reason} @ {event.price:.2f}"
            )
