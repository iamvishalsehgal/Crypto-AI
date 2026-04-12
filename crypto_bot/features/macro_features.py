"""
Macroeconomic feature engineering for crypto trading signals.

Transforms macro data (Fed interest rates, VIX, US Dollar Index, commodities)
into numerical features suitable for ML models.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class MacroFeatures:
    """Derive ML-ready features from macroeconomic data sources."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        logger.info("MacroFeatures initialised")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, required_cols: List[str], name: str) -> None:
        """Raise if *df* is empty or lacks *required_cols*."""
        if df is None or df.empty:
            raise ValueError(f"{name} DataFrame is empty")
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} DataFrame missing columns: {missing}")

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a DatetimeIndex (convert if needed)."""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # ------------------------------------------------------------------
    # interest-rate features
    # ------------------------------------------------------------------
    def compute_rate_features(self, fed_data: pd.DataFrame) -> pd.DataFrame:
        """Compute Fed interest-rate features.

        Expected column: ``rate`` (effective federal funds rate or equivalent).

        Parameters
        ----------
        fed_data : pd.DataFrame
            DataFrame with a ``rate`` column.

        Returns
        -------
        pd.DataFrame
            Columns: rate_change, rate_direction, rate_momentum.
        """
        df = self._ensure_datetime_index(fed_data)
        self._validate_dataframe(df, ["rate"], "Fed data")

        df["rate"] = pd.to_numeric(df["rate"], errors="coerce").ffill()

        rate_change = df["rate"].diff().fillna(0)

        # Direction: +1 (hiking), -1 (cutting), 0 (hold)
        rate_direction = np.sign(rate_change).astype(int)

        # Momentum: cumulative sum of direction over trailing 90 days
        rate_momentum = rate_direction.rolling(window=90, min_periods=1).sum()

        result = pd.DataFrame(
            {
                "rate_change": rate_change,
                "rate_direction": rate_direction,
                "rate_momentum": rate_momentum,
            },
            index=df.index,
        )
        logger.info("Rate features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # volatility (VIX) features
    # ------------------------------------------------------------------
    def compute_volatility_features(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """Compute VIX-based volatility regime features.

        Expected column: ``close`` (VIX closing value).

        Parameters
        ----------
        vix_data : pd.DataFrame
            DataFrame with a ``close`` column for the VIX index.

        Returns
        -------
        pd.DataFrame
            Columns: vix_zscore, vix_regime, vix_change.
        """
        df = self._ensure_datetime_index(vix_data)
        self._validate_dataframe(df, ["close"], "VIX data")

        vix = pd.to_numeric(df["close"], errors="coerce").ffill()

        # Z-score over 90-day rolling window
        vix_mean = vix.rolling(window=90, min_periods=1).mean()
        vix_std = vix.rolling(window=90, min_periods=1).std().replace(0, np.nan)
        vix_zscore = ((vix - vix_mean) / vix_std).fillna(0)

        # Regime classification
        # low: VIX < 15, medium: 15 <= VIX < 25, high: VIX >= 25
        conditions = [vix < 15, (vix >= 15) & (vix < 25), vix >= 25]
        choices = ["low", "medium", "high"]
        vix_regime = pd.Series(
            np.select(conditions, choices, default="medium"),
            index=df.index,
            name="vix_regime",
        )

        vix_change = vix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        result = pd.DataFrame(
            {
                "vix_zscore": vix_zscore,
                "vix_regime": vix_regime,
                "vix_change": vix_change,
            },
            index=df.index,
        )
        logger.info("Volatility features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # US Dollar Index features
    # ------------------------------------------------------------------
    def compute_dollar_features(self, dxy_data: pd.DataFrame) -> pd.DataFrame:
        """Compute US Dollar Index features and a crypto-dollar correlation proxy.

        Expected columns: ``close`` (DXY close). Optionally ``crypto_close``
        for the correlation calculation; if absent, the correlation column is
        filled with NaN.

        Parameters
        ----------
        dxy_data : pd.DataFrame
            DataFrame with a ``close`` column for DXY.

        Returns
        -------
        pd.DataFrame
            Columns: dxy_trend, dxy_change, crypto_dollar_correlation.
        """
        df = self._ensure_datetime_index(dxy_data)
        self._validate_dataframe(df, ["close"], "DXY data")

        dxy = pd.to_numeric(df["close"], errors="coerce").ffill()

        # Trend: sign of 20-day SMA slope
        sma_20 = dxy.rolling(window=20, min_periods=1).mean()
        dxy_trend = np.sign(sma_20.diff()).fillna(0)

        dxy_change = dxy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        # Rolling 30-day correlation between DXY and crypto price
        if "crypto_close" in df.columns:
            crypto = pd.to_numeric(df["crypto_close"], errors="coerce").ffill()
            crypto_dollar_correlation = dxy.rolling(window=30, min_periods=5).corr(crypto).fillna(0)
        else:
            crypto_dollar_correlation = pd.Series(np.nan, index=df.index)

        result = pd.DataFrame(
            {
                "dxy_trend": dxy_trend,
                "dxy_change": dxy_change,
                "crypto_dollar_correlation": crypto_dollar_correlation,
            },
            index=df.index,
        )
        logger.info("Dollar features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # commodity features
    # ------------------------------------------------------------------
    def compute_commodity_features(
        self,
        gold: pd.DataFrame,
        oil: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute commodity-related features.

        Expected column in both DataFrames: ``close``.
        *gold* may additionally contain ``crypto_close`` to compute the
        gold-to-crypto price ratio.

        Parameters
        ----------
        gold : pd.DataFrame
            Gold price data with ``close`` column.
        oil : pd.DataFrame
            Oil price data with ``close`` column.

        Returns
        -------
        pd.DataFrame
            Columns: gold_crypto_ratio, oil_change.
        """
        gold_df = self._ensure_datetime_index(gold)
        oil_df = self._ensure_datetime_index(oil)

        self._validate_dataframe(gold_df, ["close"], "Gold data")
        self._validate_dataframe(oil_df, ["close"], "Oil data")

        gold_price = pd.to_numeric(gold_df["close"], errors="coerce").ffill()
        oil_price = pd.to_numeric(oil_df["close"], errors="coerce").ffill()

        # Gold / crypto ratio
        if "crypto_close" in gold_df.columns:
            crypto_price = pd.to_numeric(gold_df["crypto_close"], errors="coerce").ffill()
            gold_crypto_ratio = (gold_price / crypto_price.replace(0, np.nan)).ffill()
        else:
            gold_crypto_ratio = pd.Series(np.nan, index=gold_df.index, name="gold_crypto_ratio")

        oil_change = oil_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        # Align the two time series on their common index
        common_idx = gold_df.index.union(oil_df.index)
        gold_crypto_ratio = gold_crypto_ratio.reindex(common_idx).ffill()
        oil_change = oil_change.reindex(common_idx).fillna(0)

        result = pd.DataFrame(
            {
                "gold_crypto_ratio": gold_crypto_ratio,
                "oil_change": oil_change,
            },
            index=common_idx,
        )
        logger.info("Commodity features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------
    def compute_all(
        self,
        fed: pd.DataFrame,
        vix: pd.DataFrame,
        dxy: pd.DataFrame,
        gold: pd.DataFrame,
        oil: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute and merge all macroeconomic feature groups.

        Parameters
        ----------
        fed : pd.DataFrame
            Federal funds rate data.
        vix : pd.DataFrame
            VIX index data.
        dxy : pd.DataFrame
            US Dollar Index data.
        gold : pd.DataFrame
            Gold price data.
        oil : pd.DataFrame
            Oil price data.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all macro features.
        """
        frames: List[pd.DataFrame] = []

        try:
            frames.append(self.compute_rate_features(fed))
        except Exception as exc:
            logger.warning("Rate feature computation failed: %s", exc)

        try:
            frames.append(self.compute_volatility_features(vix))
        except Exception as exc:
            logger.warning("Volatility feature computation failed: %s", exc)

        try:
            frames.append(self.compute_dollar_features(dxy))
        except Exception as exc:
            logger.warning("Dollar feature computation failed: %s", exc)

        try:
            frames.append(self.compute_commodity_features(gold, oil))
        except Exception as exc:
            logger.warning("Commodity feature computation failed: %s", exc)

        if not frames:
            logger.error("All macro feature computations failed")
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        logger.info(
            "All macro features merged: %d columns, %d rows",
            result.shape[1],
            result.shape[0],
        )
        return result
