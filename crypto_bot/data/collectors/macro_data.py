"""
Macroeconomic data collector.

Retrieves Federal Reserve interest rates, VIX volatility index,
dollar index (DXY), gold and oil prices from the FRED API and
presents them as a unified DataFrame.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FRED series identifiers
# ---------------------------------------------------------------------------
_SERIES = {
    "fed_rate": "FEDFUNDS",          # Effective Federal Funds Rate
    "vix": "VIXCLS",                 # CBOE Volatility Index (close)
    "dxy": "DTWEXBGS",               # Trade-Weighted US Dollar Index (broad)
    "gold": "GOLDAMGBD228NLBM",      # Gold Fixing Price (London, USD)
    "oil": "DCOILWTICO",             # WTI Crude Oil Price
}

# Cache TTL in seconds -- macro data rarely changes intraday
_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Simple TTL cache (same pattern as onchain_data)
# ---------------------------------------------------------------------------

class _TTLCache:
    """Minimalist in-memory cache with per-key time-to-live."""

    def __init__(self, default_ttl: float = _CACHE_TTL) -> None:
        self._store: Dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        self._store[key] = (time.monotonic() + ttl, value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

_HISTORY_PERIODS = 30  # number of recent observations to include in history


class MacroDataCollector:
    """Collect macroeconomic indicators from the FRED API."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with a FRED API key.

        The key is read from ``settings.FRED_API_KEY`` first, then
        falls back to the ``FRED_API_KEY`` environment variable.
        """
        self._settings = settings

        self._fred_api_key: str = getattr(
            settings, "FRED_API_KEY", ""
        ) or os.environ.get("FRED_API_KEY", "")

        self._fred = None  # lazy init
        self._cache = _TTLCache(default_ttl=_CACHE_TTL)

        logger.info("MacroDataCollector initialised")

    # ------------------------------------------------------------------ #
    # Lazy client
    # ------------------------------------------------------------------ #

    def _get_fred(self):
        """Return a configured ``fredapi.Fred`` instance."""
        if self._fred is not None:
            return self._fred

        if not self._fred_api_key:
            logger.warning("FRED API key not configured")
            return None

        try:
            from fredapi import Fred

            self._fred = Fred(api_key=self._fred_api_key)
            return self._fred
        except Exception:
            logger.exception("Failed to initialise FRED client")
            return None

    # ------------------------------------------------------------------ #
    # Generic series fetcher
    # ------------------------------------------------------------------ #

    def _fetch_series(
        self,
        series_id: str,
        label: str,
    ) -> Dict[str, Any]:
        """Fetch a single FRED series and return current + history.

        Returns
        -------
        dict
            ``current`` -- most recent non-NaN value.
            ``date``    -- date string of the most recent observation.
            ``history`` -- list of ``{"date": ..., "value": ...}`` dicts
                           for the last ``_HISTORY_PERIODS`` observations.
        """
        cache_key = f"fred:{series_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        fred = self._get_fred()
        if fred is None:
            return {"current": None, "date": None, "history": []}

        result: Dict[str, Any] = {
            "current": None,
            "date": None,
            "history": [],
        }

        try:
            series: pd.Series = fred.get_series(series_id)

            # Drop NaN tail so "current" is meaningful
            series = series.dropna()

            if series.empty:
                logger.warning("FRED series %s returned no data", series_id)
                return result

            result["current"] = float(series.iloc[-1])
            result["date"] = str(series.index[-1].date())

            tail = series.tail(_HISTORY_PERIODS)
            result["history"] = [
                {"date": str(idx.date()), "value": float(val)}
                for idx, val in tail.items()
            ]

            logger.debug(
                "Fetched %s (%s): current=%.4f as of %s",
                label,
                series_id,
                result["current"],
                result["date"],
            )

        except Exception:
            logger.exception("Failed to fetch FRED series %s (%s)", series_id, label)

        self._cache.set(cache_key, result)
        return result

    # ------------------------------------------------------------------ #
    # Public API -- individual indicators
    # ------------------------------------------------------------------ #

    def fetch_fed_rate(self) -> Dict[str, Any]:
        """Fetch the effective Federal Funds Rate.

        Returns
        -------
        dict
            ``current`` (float), ``date`` (str), ``history`` (list).
        """
        return self._fetch_series(_SERIES["fed_rate"], "Federal Funds Rate")

    def fetch_vix(self) -> Dict[str, Any]:
        """Fetch the CBOE Volatility Index (VIX).

        Returns
        -------
        dict
            ``current`` (float), ``date`` (str), ``history`` (list).
        """
        return self._fetch_series(_SERIES["vix"], "VIX")

    def fetch_dxy(self) -> Dict[str, Any]:
        """Fetch the US Dollar Index (trade-weighted, broad).

        Returns
        -------
        dict
            ``current`` (float), ``date`` (str), ``history`` (list).
        """
        return self._fetch_series(_SERIES["dxy"], "DXY")

    def fetch_gold_price(self) -> Dict[str, Any]:
        """Fetch the London gold fixing price in USD per troy ounce.

        Returns
        -------
        dict
            ``current`` (float), ``date`` (str), ``history`` (list).
        """
        return self._fetch_series(_SERIES["gold"], "Gold Price")

    def fetch_oil_price(self) -> Dict[str, Any]:
        """Fetch the WTI crude oil price in USD per barrel.

        Returns
        -------
        dict
            ``current`` (float), ``date`` (str), ``history`` (list).
        """
        return self._fetch_series(_SERIES["oil"], "Oil Price (WTI)")

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #

    def fetch_all_macro(self) -> pd.DataFrame:
        """Fetch all macro indicators and return them in a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Indexed by date (datetime) with columns for each indicator:
            ``fed_rate``, ``vix``, ``dxy``, ``gold``, ``oil``.
            Missing dates are forward-filled.
        """
        cache_key = "macro:all"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        fred = self._get_fred()
        if fred is None:
            logger.warning("Cannot fetch macro aggregate -- FRED unavailable")
            return pd.DataFrame()

        frames: Dict[str, pd.Series] = {}

        for label, series_id in _SERIES.items():
            try:
                s = fred.get_series(series_id).dropna()
                if not s.empty:
                    frames[label] = s
            except Exception:
                logger.warning("Skipping %s (%s) in aggregate", label, series_id)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Forward-fill so all indicators share the same date grid
        df = df.ffill()

        # Keep only the most recent window
        df = df.tail(_HISTORY_PERIODS)

        logger.info(
            "Macro aggregate built: %d rows x %d indicators",
            len(df),
            len(df.columns),
        )

        self._cache.set(cache_key, df)
        return df
