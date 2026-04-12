"""
System health monitoring for the AI Crypto Trading Bot.

Periodically checks exchange connectivity, database health, ML model status,
system resources, and data freshness. When the overall status degrades the
module fires alerts through :class:`TelegramNotifier`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psutil

from crypto_bot.config.settings import Settings
from crypto_bot.monitoring.telegram_bot import TelegramNotifier
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds that determine a "degraded" or "unhealthy" reading.
_CPU_WARN = 85.0
_MEMORY_WARN = 85.0
_DISK_WARN = 90.0
_DEFAULT_STALE_SECONDS = 300  # 5 minutes


class HealthChecker:
    """
    Runs health checks against every subsystem and optionally loops in the
    background, sending Telegram alerts when something goes wrong.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the health checker.

        Args:
            settings: Application settings (used for intervals and to create
                      the :class:`TelegramNotifier`).
        """
        self._settings = settings
        self._interval: int = settings.monitoring.health_check_interval
        self._notifier = TelegramNotifier(settings)
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._last_status: str = "healthy"

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def check_exchange_connection(self) -> Dict[str, Any]:
        """
        Verify that the exchange API is reachable.

        Attempts to import and instantiate a ccxt exchange using the
        configured credentials, then calls ``load_markets()``.

        Returns:
            Dict with ``status`` (``"ok"`` / ``"error"``) and ``latency_ms``.
        """
        try:
            import ccxt  # type: ignore[import-untyped]

            exchange_cls = getattr(ccxt, self._settings.exchange.name, None)
            if exchange_cls is None:
                return {"status": "error", "latency_ms": 0, "error": "Unknown exchange"}

            exchange = exchange_cls(
                {
                    "apiKey": self._settings.exchange.api_key,
                    "secret": self._settings.exchange.api_secret,
                    "enableRateLimit": True,
                }
            )
            if self._settings.exchange.sandbox_mode:
                exchange.set_sandbox_mode(True)

            start = time.monotonic()
            await asyncio.get_event_loop().run_in_executor(None, exchange.load_markets)
            latency = (time.monotonic() - start) * 1000

            return {"status": "ok", "latency_ms": round(latency, 2)}
        except Exception as exc:
            logger.error("Exchange health check failed: %s", exc)
            return {"status": "error", "latency_ms": 0, "error": str(exc)}

    async def check_database_connection(self) -> Dict[str, Any]:
        """
        Ping the MongoDB server.

        Returns:
            Dict with ``status`` and ``latency_ms``.
        """
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            uri = self._settings.database.uri
            timeout = self._settings.database.connection_timeout_ms

            def _ping() -> float:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=timeout,
                    connectTimeoutMS=timeout,
                )
                start = time.monotonic()
                client.admin.command("ping")
                elapsed = (time.monotonic() - start) * 1000
                client.close()
                return elapsed

            latency = await asyncio.get_event_loop().run_in_executor(None, _ping)
            return {"status": "ok", "latency_ms": round(latency, 2)}
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            return {"status": "error", "latency_ms": 0, "error": str(exc)}

    async def check_model_status(self, models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Report readiness of each ML model.

        Args:
            models: Mapping of ``model_name`` to an object/dict exposing
                    ``loaded`` (bool), ``last_trained`` (datetime or str),
                    and ``accuracy`` (float).

        Returns:
            Per-model dict with ``loaded``, ``last_trained``, and ``accuracy``.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, info in models.items():
            if isinstance(info, dict):
                loaded = info.get("loaded", False)
                last_trained = info.get("last_trained")
                accuracy = info.get("accuracy", 0.0)
            else:
                loaded = getattr(info, "loaded", False)
                last_trained = getattr(info, "last_trained", None)
                accuracy = getattr(info, "accuracy", 0.0)

            result[name] = {
                "loaded": bool(loaded),
                "last_trained": str(last_trained) if last_trained else None,
                "accuracy": float(accuracy),
            }
        return result

    async def check_system_resources(self) -> Dict[str, float]:
        """
        Sample CPU, memory, and disk utilisation via :mod:`psutil`.

        Returns:
            Dict with ``cpu_percent``, ``memory_percent``, ``disk_percent``.
        """
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            return {
                "cpu_percent": cpu,
                "memory_percent": memory,
                "disk_percent": disk,
            }
        except Exception as exc:
            logger.error("System resource check failed: %s", exc)
            return {"cpu_percent": -1, "memory_percent": -1, "disk_percent": -1}

    async def check_data_freshness(
        self, last_update_times: Dict[str, datetime]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Determine whether each data source is stale.

        Args:
            last_update_times: Mapping of source name to its last update
                               :class:`datetime` (UTC).

        Returns:
            Per-source dict with ``is_stale`` (bool) and
            ``seconds_since_update`` (float).
        """
        now = datetime.now(timezone.utc)
        result: Dict[str, Dict[str, Any]] = {}
        for source, last_ts in last_update_times.items():
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            delta = (now - last_ts).total_seconds()
            result[source] = {
                "is_stale": delta > _DEFAULT_STALE_SECONDS,
                "seconds_since_update": round(delta, 2),
            }
        return result

    # ------------------------------------------------------------------
    # Aggregate check
    # ------------------------------------------------------------------

    async def run_all_checks(
        self,
        models: Optional[Dict[str, Any]] = None,
        last_update_times: Optional[Dict[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Execute every health check and determine an overall status.

        Args:
            models: Optional model info dict (forwarded to
                    :meth:`check_model_status`).
            last_update_times: Optional data-freshness dict (forwarded to
                               :meth:`check_data_freshness`).

        Returns:
            A comprehensive dict with individual results and an
            ``overall_status`` of ``"healthy"``, ``"degraded"``, or
            ``"unhealthy"``.
        """
        exchange, database, resources = await asyncio.gather(
            self.check_exchange_connection(),
            self.check_database_connection(),
            self.check_system_resources(),
        )

        model_status = await self.check_model_status(models or {})
        data_freshness = await self.check_data_freshness(last_update_times or {})

        # Determine overall status.
        issues: list[str] = []

        if exchange.get("status") != "ok":
            issues.append("exchange_down")
        if database.get("status") != "ok":
            issues.append("database_down")
        if resources.get("cpu_percent", 0) > _CPU_WARN:
            issues.append("cpu_high")
        if resources.get("memory_percent", 0) > _MEMORY_WARN:
            issues.append("memory_high")
        if resources.get("disk_percent", 0) > _DISK_WARN:
            issues.append("disk_high")

        for src, info in data_freshness.items():
            if info.get("is_stale"):
                issues.append(f"stale_data:{src}")

        for model_name, info in model_status.items():
            if not info.get("loaded"):
                issues.append(f"model_not_loaded:{model_name}")

        critical = {"exchange_down", "database_down"}
        if issues and critical & set(issues):
            overall = "unhealthy"
        elif issues:
            overall = "degraded"
        else:
            overall = "healthy"

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall,
            "issues": issues,
            "exchange": exchange,
            "database": database,
            "system_resources": resources,
            "model_status": model_status,
            "data_freshness": data_freshness,
        }

        logger.info("Health check complete: %s (%d issues).", overall, len(issues))
        return report

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    async def start_monitoring(self, interval: Optional[int] = None) -> None:
        """
        Start a background async loop that runs all checks periodically.

        Args:
            interval: Override for the check interval in seconds.  Falls back
                      to ``settings.monitoring.health_check_interval``.
        """
        if self._running:
            logger.warning("Health monitoring is already running.")
            return

        self._running = True
        check_interval = interval if interval is not None else self._interval
        await self._notifier.start()
        self._monitor_task = asyncio.create_task(self._monitor_loop(check_interval))
        logger.info("Health monitoring started (interval=%ds).", check_interval)

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring loop and the Telegram notifier."""
        self._running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await self._notifier.stop()
        logger.info("Health monitoring stopped.")

    async def _monitor_loop(self, interval: int) -> None:
        """Core monitoring loop executed as an asyncio task."""
        while self._running:
            try:
                report = await self.run_all_checks()
                new_status = report["overall_status"]

                # Alert on status transitions (only when things get worse).
                if self._status_worse(new_status, self._last_status):
                    await self._send_status_alert(report)

                self._last_status = new_status
            except Exception as exc:
                logger.exception("Error during health check loop: %s", exc)
                await self._notifier.send_error_alert(
                    f"Health check loop error: {exc}"
                )

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _status_worse(new: str, old: str) -> bool:
        """Return ``True`` if *new* status is strictly worse than *old*."""
        rank = {"healthy": 0, "degraded": 1, "unhealthy": 2}
        return rank.get(new, 0) > rank.get(old, 0)

    async def _send_status_alert(self, report: Dict[str, Any]) -> None:
        """Format and send a status-degradation alert."""
        status = report["overall_status"]
        issues = report.get("issues", [])
        emoji = "\U0001f7e1" if status == "degraded" else "\U0001f534"

        message = (
            f"{emoji} <b>System Status: {status.upper()}</b>\n"
            f"\U0001f552 {report['timestamp']}\n"
            f"\U000026a0 <b>Issues:</b> {', '.join(issues) if issues else 'none'}"
        )
        await self._notifier.send_message(message)
