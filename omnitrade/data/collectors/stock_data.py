"""
Stock market data collector using yfinance for OHLCV + fundamentals.

Provides the same interface as :class:`MarketDataCollector` so the
existing feature / model pipeline can consume stock data transparently.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from omnitrade.config.settings import StockSettings, Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class StockDataCollector:
    """Collect stock market data (OHLCV + fundamentals) via yfinance."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stock: StockSettings = settings.stock
        self._tickers: List[str] = self._stock.supported_tickers
        self._lookback = f"{self._stock.data_lookback_years}y"
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_maxsize = 200
        logger.info(
            "StockDataCollector initialised for %d tickers (lookback=%s)",
            len(self._tickers),
            self._lookback,
        )

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data for *symbol*.

        Args:
            symbol: Ticker e.g. ``"AAPL"``.
            interval: ``1m``, ``5m``, ``15m``, ``1h``, ``1d``, etc.
            period: yfinance period string (defaults to configured lookback).

        Returns:
            DataFrame with timestamp, open, high, low, close, volume.
        """
        period_str = period or self._lookback

        cache_key = f"{symbol}:{interval}:{period_str}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period_str)
        except Exception as exc:
            logger.error("yfinance fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        if df.empty:
            logger.warning("No OHLCV data for %s (%s, %s)", symbol, interval, period_str)
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # yfinance may return timezone-aware or naive Date column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        keep_cols = [c for c in _OHLCV_COLUMNS if c in df.columns]
        df = df[keep_cols]
        df = df.dropna()

        if len(self._cache) >= self._cache_maxsize:
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[cache_key] = df.copy()
        logger.info("Fetched %d OHLCV candles for %s", len(df), symbol)
        return df

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch key fundamental metrics for a stock.

        Returns:
            Dict with pe_ratio, pb_ratio, market_cap, eps, dividend_yield,
            debt_to_equity, roe, revenue_growth, profit_margins, beta, sector.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except Exception as exc:
            logger.error("Fundamentals fetch failed for %s: %s", symbol, exc)
            return {}

        return {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "beta": info.get("beta"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "fifty_day_avg": info.get("fiftyDayAverage"),
            "two_hundred_day_avg": info.get("twoHundredDayAverage"),
            "short_ratio": info.get("shortRatio"),
        }

    def fetch_all_symbols_ohlcv(
        self,
        interval: str = "1h",
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for all configured tickers."""
        results: Dict[str, pd.DataFrame] = {}
        for symbol in self._tickers:
            try:
                df = self.fetch_ohlcv(symbol, interval=interval, period=period)
                results[symbol] = df
                time.sleep(0.2)  # rate-limit politeness
            except Exception:
                logger.exception("Failed to fetch %s", symbol)
                results[symbol] = pd.DataFrame(columns=_OHLCV_COLUMNS)
        return results

    def fetch_all_fundamentals(self) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamentals for all configured tickers."""
        results: Dict[str, Dict[str, Any]] = {}
        for symbol in self._tickers:
            try:
                results[symbol] = self.fetch_fundamentals(symbol)
                time.sleep(0.2)
            except Exception:
                logger.exception("Fundamentals fetch failed for %s", symbol)
                results[symbol] = {}
        return results

    def filter_by_market_cap(self, min_cap: Optional[float] = None) -> List[str]:
        """Return tickers with market cap above *min_cap* (defaults to config)."""
        threshold = min_cap or self._stock.min_market_cap
        qualified = []
        for symbol in self._tickers:
            info = self.fetch_fundamentals(symbol)
            cap = info.get("market_cap", 0) or 0
            if cap >= threshold:
                qualified.append(symbol)
            else:
                logger.info("Filtered out %s: market cap %.0f < %.0f", symbol, cap, threshold)
        return qualified
