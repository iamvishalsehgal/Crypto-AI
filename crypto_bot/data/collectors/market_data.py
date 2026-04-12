"""
Market data collector using the ccxt library.

Provides OHLCV candle data, order book snapshots, ticker info,
and real-time trade streaming for all configured trading symbols.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

import ccxt
import ccxt.pro as ccxtpro
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_S = 1.0
_RATE_LIMIT_BUFFER_MS = 50  # extra breathing room on top of exchange rate limit


class MarketDataCollector:
    """Collect market data from a cryptocurrency exchange via ccxt."""

    # --------------------------------------------------------------------- #
    # Initialisation
    # --------------------------------------------------------------------- #

    def __init__(self, settings: Settings) -> None:
        """Initialise the collector and underlying ccxt exchange instance.

        Parameters
        ----------
        settings:
            Application settings object.  Expected attributes::

                EXCHANGE_ID   - ccxt exchange id, e.g. "binance" (default)
                API_KEY       - exchange API key
                API_SECRET    - exchange API secret
                EXCHANGE_SANDBOX - bool, use sandbox/testnet if True
                TRADING_SYMBOLS  - list[str] of symbols, e.g. ["BTC/USDT"]
        """
        self._settings = settings

        exchange_id: str = getattr(settings, "EXCHANGE_ID", "binance")
        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        self._exchange: ccxt.Exchange = exchange_cls(
            {
                "apiKey": getattr(settings, "API_KEY", ""),
                "secret": getattr(settings, "API_SECRET", ""),
                "enableRateLimit": True,
                "rateLimit": getattr(settings, "RATE_LIMIT_MS", 1200)
                + _RATE_LIMIT_BUFFER_MS,
                "options": {"defaultType": "spot"},
            }
        )

        if getattr(settings, "EXCHANGE_SANDBOX", False):
            self._exchange.set_sandbox_mode(True)
            logger.info("Exchange sandbox mode enabled for %s", exchange_id)

        self._symbols: List[str] = getattr(settings, "TRADING_SYMBOLS", [])
        self._retry_attempts = _DEFAULT_RETRY_ATTEMPTS
        self._retry_delay = _DEFAULT_RETRY_DELAY_S

        # Async (websocket) exchange instance -- created lazily
        self._ws_exchange: Optional[ccxtpro.Exchange] = None
        self._ws_exchange_id = exchange_id

        logger.info(
            "MarketDataCollector initialised for %s (%d symbols)",
            exchange_id,
            len(self._symbols),
        )

    # --------------------------------------------------------------------- #
    # Public methods
    # --------------------------------------------------------------------- #

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data for *symbol*.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe:
            Candlestick interval understood by the exchange (``"1m"``,
            ``"5m"``, ``"1h"``, ``"1d"``, ...).
        since:
            Start timestamp in **milliseconds** (UTC).  ``None`` lets the
            exchange decide.
        limit:
            Maximum number of candles to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, open, high, low, close, volume``.
            ``timestamp`` is a UTC-aware ``datetime64[ms]``.
        """
        raw = self._retry(
            self._exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since=since,
            limit=limit,
        )

        if not raw:
            logger.warning("No OHLCV data returned for %s %s", symbol, timeframe)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = pd.DataFrame(raw, columns=_OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        df = self._validate_ohlcv(df, symbol)
        return df

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Fetch a snapshot of the order book.

        Returns
        -------
        dict
            ``bids``  - list of ``[price, amount]`` pairs (best first)
            ``asks``  - list of ``[price, amount]`` pairs (best first)
            ``spread`` - best ask minus best bid
            ``depth``  - dict with ``bid_depth`` and ``ask_depth`` totals
        """
        book = self._retry(self._exchange.fetch_order_book, symbol, limit)

        bids: list = book.get("bids", [])
        asks: list = book.get("asks", [])

        spread: float = 0.0
        if bids and asks:
            spread = asks[0][0] - bids[0][0]

        bid_depth = sum(amt for _, amt in bids)
        ask_depth = sum(amt for _, amt in asks)

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "spread": spread,
            "depth": {
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
            },
            "timestamp": book.get("timestamp"),
        }

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch the latest ticker for *symbol*.

        Returns
        -------
        dict
            Keys include ``bid``, ``ask``, ``last``, ``volume``,
            ``high``, ``low``, ``open``, ``change``, ``percentage``,
            ``vwap``, ``timestamp``.
        """
        raw = self._retry(self._exchange.fetch_ticker, symbol)

        return {
            "symbol": raw.get("symbol", symbol),
            "bid": raw.get("bid"),
            "ask": raw.get("ask"),
            "last": raw.get("last"),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "volume": raw.get("baseVolume"),
            "quote_volume": raw.get("quoteVolume"),
            "change": raw.get("change"),
            "percentage": raw.get("percentage"),
            "vwap": raw.get("vwap"),
            "timestamp": raw.get("timestamp"),
        }

    async def stream_trades(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Stream real-time trades over a websocket.

        This coroutine runs indefinitely until cancelled.  Each trade
        dict is passed to *callback* which can be a regular function or
        a coroutine.

        Parameters
        ----------
        symbol:
            Trading pair to subscribe to.
        callback:
            Invoked with a single trade dict for every incoming trade.
        """
        exchange = self._get_ws_exchange()
        logger.info("Starting trade stream for %s", symbol)

        try:
            while True:
                trades: list = await exchange.watch_trades(symbol)
                for trade in trades:
                    trade_data = {
                        "symbol": trade.get("symbol", symbol),
                        "id": trade.get("id"),
                        "price": trade.get("price"),
                        "amount": trade.get("amount"),
                        "side": trade.get("side"),
                        "timestamp": trade.get("timestamp"),
                    }
                    result = callback(trade_data)
                    if asyncio.iscoroutine(result):
                        await result
        except asyncio.CancelledError:
            logger.info("Trade stream cancelled for %s", symbol)
        except Exception:
            logger.exception("Trade stream error for %s", symbol)
            raise
        finally:
            await exchange.close()

    def fetch_all_symbols_ohlcv(
        self,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for every symbol in ``TRADING_SYMBOLS``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol to its OHLCV DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}

        for symbol in self._symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                results[symbol] = df
                logger.debug(
                    "Fetched %d candles for %s (%s)", len(df), symbol, timeframe
                )
            except Exception:
                logger.exception("Failed to fetch OHLCV for %s", symbol)
                results[symbol] = pd.DataFrame(columns=_OHLCV_COLUMNS)

        return results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _retry(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call *fn* with exponential back-off on transient errors."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._retry_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
                ccxt.RateLimitExceeded,
            ) as exc:
                last_exc = exc
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error on attempt %d/%d for %s: %s  "
                    "(retrying in %.1fs)",
                    attempt,
                    self._retry_attempts,
                    fn.__name__,
                    exc,
                    wait,
                )
                time.sleep(wait)
            except ccxt.BaseError:
                # Non-transient exchange errors should surface immediately
                raise

        raise last_exc  # type: ignore[misc]

    def _get_ws_exchange(self) -> ccxtpro.Exchange:
        """Lazily create and return the async (websocket) exchange."""
        if self._ws_exchange is None:
            ws_cls = getattr(ccxtpro, self._ws_exchange_id, None)
            if ws_cls is None:
                raise ValueError(
                    f"No ccxt.pro support for exchange: {self._ws_exchange_id}"
                )
            self._ws_exchange = ws_cls(
                {
                    "apiKey": getattr(self._settings, "API_KEY", ""),
                    "secret": getattr(self._settings, "API_SECRET", ""),
                    "enableRateLimit": True,
                }
            )
            if getattr(self._settings, "EXCHANGE_SANDBOX", False):
                self._ws_exchange.set_sandbox_mode(True)
        return self._ws_exchange

    # --------------------------------------------------------------------- #
    # Data validation
    # --------------------------------------------------------------------- #

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean an OHLCV DataFrame.

        Checks for:
        * duplicate timestamps
        * out-of-order timestamps
        * nonsensical price ranges (high < low, negative values)
        * negative volume
        """
        original_len = len(df)

        # --- duplicates ---
        dup_mask = df["timestamp"].duplicated(keep="last")
        n_dups = int(dup_mask.sum())
        if n_dups:
            logger.warning(
                "%s: dropped %d duplicate candle(s)", symbol, n_dups
            )
            df = df[~dup_mask].copy()

        # --- sort order ---
        if not df["timestamp"].is_monotonic_increasing:
            logger.warning("%s: timestamps out of order, sorting", symbol)
            df = df.sort_values("timestamp").reset_index(drop=True)

        # --- price sanity ---
        bad_price = (
            (df["high"] < df["low"])
            | (df["open"] < 0)
            | (df["close"] < 0)
        )
        n_bad = int(bad_price.sum())
        if n_bad:
            logger.warning(
                "%s: dropped %d candle(s) with invalid prices", symbol, n_bad
            )
            df = df[~bad_price].copy()

        # --- negative volume ---
        neg_vol = df["volume"] < 0
        n_neg = int(neg_vol.sum())
        if n_neg:
            logger.warning(
                "%s: zeroed %d negative volume value(s)", symbol, n_neg
            )
            df.loc[neg_vol, "volume"] = 0.0

        # --- gap detection (warn only) ---
        if len(df) >= 2:
            diffs = df["timestamp"].diff().dropna()
            median_diff = diffs.median()
            # A gap wider than 2x the median interval is suspicious
            gaps = diffs[diffs > median_diff * 2]
            if not gaps.empty:
                logger.info(
                    "%s: detected %d potential gap(s) in candle timestamps",
                    symbol,
                    len(gaps),
                )

        if len(df) != original_len:
            logger.info(
                "%s: validation reduced candles from %d to %d",
                symbol,
                original_len,
                len(df),
            )

        return df.reset_index(drop=True)
