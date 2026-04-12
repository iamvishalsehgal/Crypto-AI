"""
Telegram notification bot for the AI Crypto Trading Bot.

Sends trade alerts, daily reports, error notifications, and risk warnings
to a configured Telegram chat via the Bot API. All sends are async and
internally queued to respect the Telegram rate limit (30 msg/s).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# Telegram enforces ~30 messages per second per bot.
_MAX_MESSAGES_PER_SECOND = 30


class TelegramNotifier:
    """Async Telegram notification sender with built-in rate-limit queue."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialise the notifier.

        Args:
            settings: Application settings containing ``monitoring.telegram_bot_token``
                      and ``monitoring.telegram_chat_id``.
        """
        self._bot_token: str = settings.monitoring.telegram_bot_token
        self._chat_id: str = settings.monitoring.telegram_chat_id
        self._base_url: str = (
            f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        )

        # Rate-limit bookkeeping.
        self._send_times: list[float] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background queue worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._queue_worker())
        logger.info("Telegram notifier queue worker started.")

    async def stop(self) -> None:
        """Drain the queue and stop the background worker."""
        self._running = False
        if self._worker_task is not None:
            # Send a sentinel so the worker wakes up and exits.
            await self._queue.put("")
            await self._worker_task
            self._worker_task = None
        logger.info("Telegram notifier queue worker stopped.")

    async def send_message(self, text: str) -> bool:
        """
        Send a plain text message to the configured Telegram chat.

        Args:
            text: Message body (supports Telegram MarkdownV2 / HTML).

        Returns:
            ``True`` if the Telegram API returned ``ok``, ``False`` otherwise.
        """
        return await self._post_message(text)

    async def send_trade_alert(self, trade: Dict[str, Any]) -> bool:
        """
        Send a formatted trade notification.

        Args:
            trade: Dict with keys ``symbol``, ``side``, ``price``, ``amount``,
                   and optionally ``pnl``.

        Returns:
            ``True`` on success.
        """
        message = self._format_trade_message(trade)
        return await self._post_message(message)

    async def send_daily_report(self, stats: Dict[str, Any]) -> bool:
        """
        Send a daily performance summary.

        Args:
            stats: Dict with keys ``daily_pnl``, ``total_trades``,
                   ``win_rate``, ``current_balance``.

        Returns:
            ``True`` on success.
        """
        message = self._format_daily_report(stats)
        return await self._post_message(message)

    async def send_error_alert(self, error: str) -> bool:
        """
        Send an error alert with a UTC timestamp.

        Args:
            error: Human-readable error description.

        Returns:
            ``True`` on success.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        message = (
            "\U0001f6a8 <b>ERROR ALERT</b>\n"
            f"\U0001f552 <b>Time:</b> {now}\n"
            f"\U0000274c <b>Error:</b> {error}"
        )
        return await self._post_message(message)

    async def send_risk_alert(self, risk_data: Dict[str, Any]) -> bool:
        """
        Send a risk threshold warning.

        Args:
            risk_data: Dict with keys such as ``metric``, ``current_value``,
                       ``threshold``, and ``message``.

        Returns:
            ``True`` on success.
        """
        metric = risk_data.get("metric", "Unknown")
        current = risk_data.get("current_value", "N/A")
        threshold = risk_data.get("threshold", "N/A")
        detail = risk_data.get("message", "")

        message = (
            "\U000026a0 <b>RISK ALERT</b>\n"
            f"\U0001f4ca <b>Metric:</b> {metric}\n"
            f"\U0001f534 <b>Current:</b> {current}\n"
            f"\U0001f7e1 <b>Threshold:</b> {threshold}\n"
        )
        if detail:
            message += f"\U0001f4dd <b>Details:</b> {detail}"
        return await self._post_message(message)

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_trade_message(trade: Dict[str, Any]) -> str:
        """
        Build a human-friendly trade notification string.

        Args:
            trade: Trade dict.

        Returns:
            Formatted HTML string suitable for Telegram's ``parse_mode=HTML``.
        """
        symbol = trade.get("symbol", "???")
        side = trade.get("side", "???").upper()
        price = trade.get("price", 0)
        amount = trade.get("amount", 0)
        pnl = trade.get("pnl")

        side_emoji = "\U0001f7e2" if side == "BUY" else "\U0001f534"

        lines = [
            f"{side_emoji} <b>Trade Executed</b>",
            f"\U0001f4b1 <b>Symbol:</b> {symbol}",
            f"\U0001f4c8 <b>Side:</b> {side}",
            f"\U0001f4b0 <b>Price:</b> ${price:,.4f}",
            f"\U0001f4e6 <b>Amount:</b> {amount:,.6f}",
        ]

        if pnl is not None:
            pnl_emoji = "\U00002705" if pnl >= 0 else "\U0000274c"
            lines.append(f"{pnl_emoji} <b>PnL:</b> ${pnl:,.2f}")

        return "\n".join(lines)

    @staticmethod
    def _format_daily_report(stats: Dict[str, Any]) -> str:
        """
        Build a daily performance summary string.

        Args:
            stats: Stats dict.

        Returns:
            Formatted HTML string.
        """
        daily_pnl = stats.get("daily_pnl", 0)
        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        balance = stats.get("current_balance", 0)

        pnl_emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return (
            f"\U0001f4ca <b>Daily Report - {today}</b>\n"
            f"{pnl_emoji} <b>Daily P&L:</b> ${daily_pnl:,.2f}\n"
            f"\U0001f504 <b>Total Trades:</b> {total_trades}\n"
            f"\U0001f3af <b>Win Rate:</b> {win_rate:.1%}\n"
            f"\U0001f4b0 <b>Balance:</b> ${balance:,.2f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_message(self, text: str) -> bool:
        """
        POST a message to the Telegram Bot API, respecting rate limits.

        If the background queue worker is running the message is enqueued;
        otherwise it is sent immediately (with inline rate-limit enforcement).

        Args:
            text: HTML-formatted message body.

        Returns:
            ``True`` when the API responds with ``{"ok": true}``.
        """
        if not self._bot_token or not self._chat_id:
            logger.warning("Telegram bot_token or chat_id not configured; skipping send.")
            return False

        if self._running:
            await self._queue.put(text)
            return True  # Assume success; worker will log errors.

        return await self._send_now(text)

    async def _send_now(self, text: str) -> bool:
        """Send a single message immediately, rate-limit aware."""
        await self._enforce_rate_limit()

        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        self._send_times.append(time.monotonic())
                        return True
                    logger.error("Telegram API error: %s", data.get("description", data))
                    return False
        except asyncio.TimeoutError:
            logger.error("Telegram API request timed out.")
            return False
        except aiohttp.ClientError as exc:
            logger.error("Telegram HTTP error: %s", exc)
            return False

    async def _enforce_rate_limit(self) -> None:
        """Sleep if we have hit the 30 msg/s ceiling."""
        now = time.monotonic()
        # Discard timestamps older than 1 second.
        self._send_times = [t for t in self._send_times if now - t < 1.0]
        if len(self._send_times) >= _MAX_MESSAGES_PER_SECOND:
            sleep_for = 1.0 - (now - self._send_times[0])
            if sleep_for > 0:
                logger.debug("Rate-limit reached; sleeping %.2fs.", sleep_for)
                await asyncio.sleep(sleep_for)

    async def _queue_worker(self) -> None:
        """Background loop that drains the message queue."""
        logger.debug("Queue worker running.")
        while self._running:
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if not text:
                # Empty string is our shutdown sentinel.
                break

            success = await self._send_now(text)
            if not success:
                logger.warning("Failed to send queued Telegram message.")

        # Drain any remaining messages before exiting.
        while not self._queue.empty():
            text = self._queue.get_nowait()
            if text:
                await self._send_now(text)

        logger.debug("Queue worker exited.")
