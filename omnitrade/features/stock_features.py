"""
Stock-specific feature engineering combining technical indicators with
fundamental metrics for equity trading signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.features.technical import TechnicalFeatures
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class StockFundamentalFeatures:
    """Compute fundamental-derived features for stock scoring."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        fundamentals: Dict[str, Any],
    ) -> pd.DataFrame:
        """Convert fundamental dict into a single-row DataFrame.

        Returns a DataFrame with one row per unique timestamp if ohlcv has a
        datetime index, otherwise a single row. Columns are normalised where
        possible.
        """
        if not fundamentals:
            return pd.DataFrame()

        def _safe(v: Any) -> float:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        features: Dict[str, float] = {
            "pe_ratio": _safe(fundamentals.get("pe_ratio", np.nan)),
            "forward_pe": _safe(fundamentals.get("forward_pe", np.nan)),
            "pb_ratio": _safe(fundamentals.get("pb_ratio", np.nan)),
            "log_market_cap": np.log(_safe(fundamentals.get("market_cap", 1))),
            "eps": _safe(fundamentals.get("eps", np.nan)),
            "dividend_yield": _safe(fundamentals.get("dividend_yield", 0)) or 0.0,
            "debt_to_equity": _safe(fundamentals.get("debt_to_equity", np.nan)),
            "roe": _safe(fundamentals.get("roe", np.nan)),
            "revenue_growth": _safe(fundamentals.get("revenue_growth", np.nan)),
            "profit_margins": _safe(fundamentals.get("profit_margins", np.nan)),
            "beta": _safe(fundamentals.get("beta", 1.0)) or 1.0,
            "short_ratio": _safe(fundamentals.get("short_ratio", np.nan)),
            "fifty_day_avg": _safe(fundamentals.get("fifty_day_avg", np.nan)),
            "two_hundred_day_avg": _safe(fundamentals.get("two_hundred_day_avg", np.nan)),
        }

        # Derived ratios
        if not np.isnan(features["fifty_day_avg"]) and features["fifty_day_avg"] > 0:
            last_close = ohlcv["close"].iloc[-1] if not ohlcv.empty else np.nan
            features["pct_from_50day"] = (last_close - features["fifty_day_avg"]) / features["fifty_day_avg"]
        else:
            features["pct_from_50day"] = np.nan

        if not np.isnan(features["two_hundred_day_avg"]) and features["two_hundred_day_avg"] > 0:
            last_close = ohlcv["close"].iloc[-1] if not ohlcv.empty else np.nan
            features["pct_from_200day"] = (last_close - features["two_hundred_day_avg"]) / features["two_hundred_day_avg"]
        else:
            features["pct_from_200day"] = np.nan

        # PEG ratio (PE / growth rate)
        if not np.isnan(features["pe_ratio"]) and not np.isnan(features["revenue_growth"]) and features["revenue_growth"] != 0:
            features["peg_ratio"] = features["pe_ratio"] / (features["revenue_growth"] * 100)
        else:
            features["peg_ratio"] = np.nan

        # Earnings yield
        if not np.isnan(features["pe_ratio"]) and features["pe_ratio"] > 0:
            features["earnings_yield"] = 1.0 / features["pe_ratio"]
        else:
            features["earnings_yield"] = np.nan

        if ohlcv.empty:
            return pd.DataFrame([features])

        # If ohlcv has a DatetimeIndex, broadcast single row across it
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            idx = ohlcv.index[-1:]
        else:
            idx = [len(ohlcv)]
        return pd.DataFrame([features], index=idx)

    def compute_market_cap_tier(self, market_cap: float) -> int:
        """Return 0=micro, 1=small, 2=mid, 3=large, 4=mega."""
        if market_cap < 300e6:
            return 0
        if market_cap < 2e9:
            return 1
        if market_cap < 10e9:
            return 2
        if market_cap < 200e9:
            return 3
        return 4


class StockFeaturePipeline:
    """Unified feature pipeline for stocks: technical + fundamental."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.technical = TechnicalFeatures(settings)
        self.fundamental = StockFundamentalFeatures(settings)

    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        fundamentals: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compute full feature matrix for a stock.

        Args:
            ohlcv: OHLCV DataFrame with columns open/high/low/close/volume.
            fundamentals: Dict from StockDataCollector.fetch_fundamentals().

        Returns:
            DataFrame with technical + fundamental columns, no NaN rows.
        """
        if ohlcv.empty:
            logger.warning("Empty OHLCV passed to StockFeaturePipeline")
            return pd.DataFrame()

        tech_features = self.technical.compute_all(ohlcv)

        if fundamentals and fundamentals.get("symbol"):
            fund_df = self.fundamental.compute_all(ohlcv, fundamentals)

            if not fund_df.empty and len(fund_df) == 1:
                for col in fund_df.columns:
                    tech_features[col] = fund_df[col].iloc[0]

        tech_features = tech_features.dropna()

        logger.info(
            "Stock feature pipeline: %d rows x %d columns",
            *tech_features.shape,
        )
        return tech_features
