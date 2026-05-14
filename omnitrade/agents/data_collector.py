"""
DataCollectorAgent — real-time market data ingestion via WebSocket + REST fallback.

Subscribes to ccxt.pro watch_ohlcv() for crypto symbols. Falls back to REST
polling if WebSocket is unavailable. Stocks use fast REST polling (no free
real-time stock feed). Publishes MarketDataEvent to the bus.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import pandas as pd

from omnitrade.agents.bus import EventBus, MarketDataEvent
from omnitrade.config.asset_types import AssetType
from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.data.collectors.market_data import MarketDataCollector
from omnitrade.data.collectors.stock_data import StockDataCollector
from omnitrade.data.collectors.betting_data import BettingDataCollector
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_WS_TIMEFRAME = "1m"
_REST_INTERVAL = 15  # seconds between REST polls for fallback/stock
_STOCK_INTERVAL = 15
_OHLCV_LOOKBACK = 500


class DataCollectorAgent:
    """Streams market data for all symbols and publishes to the event bus.

    Crypto symbols use WebSocket (ccxt.pro) when available, falling back
    to REST polling. Stocks use REST polling every 15s. Betting uses
    periodic REST polling.

    Args:
        bus: Shared event bus for publishing MarketDataEvents.
        market_collector: Existing MarketDataCollector instance.
        stock_collector: Existing StockDataCollector (optional).
        betting_collector: Existing BettingDataCollector (optional).
        settings: Bot configuration.
    """

    def __init__(
        self,
        bus: EventBus,
        market_collector: MarketDataCollector,
        stock_collector: Optional[StockDataCollector] = None,
        betting_collector: Optional[BettingDataCollector] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._bus = bus
        self._market = market_collector
        self._stock = stock_collector
        self._betting = betting_collector
        self._settings = settings or _default_settings

        self._crypto_symbols: List[str] = list(
            self._settings.exchange.supported_symbols
        )
        self._stock_tickers: List[str] = (
            list(self._settings.stock.supported_tickers) if self._stock else []
        )
        self._tasks: List[asyncio.Task] = []
        self._candle_buffers: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all data collection streams. Runs until cancelled."""
        logger.info(
            "DataCollectorAgent starting — %d crypto, %d stock",
            len(self._crypto_symbols),
            len(self._stock_tickers),
        )

        tasks: List[asyncio.Task] = []

        # Crypto — try WebSocket first, fall back to REST
        for symbol in self._crypto_symbols:
            tasks.append(asyncio.create_task(self._stream_crypto(symbol)))

        # Stocks — REST polling
        for ticker in self._stock_tickers:
            tasks.append(asyncio.create_task(self._poll_stock(ticker)))

        # Betting — slow REST polling
        if self._betting:
            tasks.append(asyncio.create_task(self._poll_betting()))

        self._tasks = tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("DataCollectorAgent shutting down")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Crypto streaming
    # ------------------------------------------------------------------

    async def _stream_crypto(self, symbol: str) -> None:
        """Stream OHLCV for a crypto symbol. WebSocket preferred, REST fallback."""
        try:
            await self._ws_stream(symbol)
        except Exception:
            logger.warning(
                "WebSocket stream failed for %s — falling back to REST polling", symbol
            )
            await self._rest_poll_crypto(symbol)

    async def _ws_stream(self, symbol: str) -> None:
        """Real-time OHLCV via ccxt.pro WebSocket.

        Seeds the rolling buffer from REST (full history) first, then merges
        WebSocket updates so technical indicators always have enough data.
        """
        logger.info("Starting WebSocket stream for %s", symbol)

        # Seed buffer with REST history so indicators have enough lookback
        if symbol not in self._candle_buffers:
            try:
                rest_df = self._market.fetch_ohlcv(
                    symbol, timeframe="1h", limit=_OHLCV_LOOKBACK
                )
                if not rest_df.empty:
                    self._candle_buffers[symbol] = rest_df
                    logger.info("%s: seeded buffer with %d REST candles", symbol, len(rest_df))
            except Exception:
                logger.exception("%s: REST seed failed, will build from WebSocket", symbol)

        while True:
            try:
                ohlcv = await self._market.watch_ohlcv(
                    symbol, timeframe=_WS_TIMEFRAME, limit=_OHLCV_LOOKBACK
                )
                if ohlcv is None or ohlcv.empty:
                    await asyncio.sleep(1)
                    continue

                # Merge with rolling buffer
                buffered = self._candle_buffers.get(symbol)
                if buffered is not None and not buffered.empty:
                    merged = pd.concat([buffered, ohlcv], ignore_index=True)
                else:
                    merged = ohlcv

                # Deduplicate by timestamp, keep last
                merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
                merged = merged.sort_values("timestamp").tail(_OHLCV_LOOKBACK)
                self._candle_buffers[symbol] = merged

                await self._bus.publish_market_data(
                    MarketDataEvent(
                        symbol=symbol,
                        asset_type="crypto",
                        ohlcv=merged.copy(),
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("WebSocket error for %s — reconnecting in 3s", symbol)
                self._candle_buffers.pop(symbol, None)  # clear on reconnect
                await asyncio.sleep(3)

    async def _rest_poll_crypto(self, symbol: str) -> None:
        """REST polling fallback for crypto symbols."""
        logger.info("Starting REST poll for %s every %ds", symbol, _REST_INTERVAL)
        while True:
            try:
                ohlcv = self._market.fetch_ohlcv(
                    symbol, timeframe="1h", limit=_OHLCV_LOOKBACK
                )
                if ohlcv is None or ohlcv.empty:
                    await asyncio.sleep(_REST_INTERVAL)
                    continue

                await self._bus.publish_market_data(
                    MarketDataEvent(
                        symbol=symbol,
                        asset_type="crypto",
                        ohlcv=ohlcv,
                        order_book=self._market.fetch_order_book(symbol),
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("REST poll error for %s", symbol)

            await asyncio.sleep(_REST_INTERVAL)

    # ------------------------------------------------------------------
    # Stock polling
    # ------------------------------------------------------------------

    async def _poll_stock(self, ticker: str) -> None:
        """Fast REST polling for stock tickers."""
        logger.info("Starting stock poll for %s every %ds", ticker, _STOCK_INTERVAL)
        while True:
            try:
                ohlcv = self._stock.fetch_ohlcv(ticker, interval="1h")
                if ohlcv is None or ohlcv.empty:
                    await asyncio.sleep(_STOCK_INTERVAL)
                    continue

                fundamentals = self._stock.fetch_fundamentals(ticker)

                await self._bus.publish_market_data(
                    MarketDataEvent(
                        symbol=ticker,
                        asset_type="stock",
                        ohlcv=ohlcv,
                        ticker={"fundamentals": fundamentals},
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Stock poll error for %s", ticker)

            await asyncio.sleep(_STOCK_INTERVAL)

    # ------------------------------------------------------------------
    # Betting polling
    # ------------------------------------------------------------------

    async def _poll_betting(self) -> None:
        """Slow REST polling for betting odds."""
        interval = 300  # 5 minutes
        logger.info("Starting betting poll every %ds", interval)
        while True:
            try:
                for sport in self._settings.betting.supported_sports:
                    odds_df = self._betting.fetch_odds(sport)
                    if odds_df is not None and not odds_df.empty:
                        await self._bus.publish_market_data(
                            MarketDataEvent(
                                symbol=f"bet:{sport}",
                                asset_type="bet",
                                ohlcv=pd.DataFrame(),
                                ticker={"sport": sport, "odds": odds_df},
                            )
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Betting poll error")

            await asyncio.sleep(interval)
