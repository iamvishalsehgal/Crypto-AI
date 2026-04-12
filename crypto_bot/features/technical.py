"""
Technical indicator calculations for crypto trading.

Pure pandas/numpy implementations (no ta-lib dependency) for maximum
portability.  Every method expects an OHLCV DataFrame whose columns are
named: open, high, low, close, volume.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalFeatures:
    """Compute a comprehensive suite of technical indicators from OHLCV data."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        logger.info("TechnicalFeatures initialised")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> None:
        """Raise if required columns are missing."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required OHLCV columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every indicator and merge into a single DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            Original data plus all computed indicator columns.
        """
        self._validate_ohlcv(df)
        result = df.copy()

        try:
            result["rsi"] = self.compute_rsi(df)
        except Exception as exc:
            logger.warning("RSI computation failed: %s", exc)

        try:
            macd_df = self.compute_macd(df)
            result = pd.concat([result, macd_df], axis=1)
        except Exception as exc:
            logger.warning("MACD computation failed: %s", exc)

        try:
            bb_df = self.compute_bollinger_bands(df)
            result = pd.concat([result, bb_df], axis=1)
        except Exception as exc:
            logger.warning("Bollinger Bands computation failed: %s", exc)

        try:
            ema_df = self.compute_ema(df)
            result = pd.concat([result, ema_df], axis=1)
        except Exception as exc:
            logger.warning("EMA computation failed: %s", exc)

        try:
            result["atr"] = self.compute_atr(df)
        except Exception as exc:
            logger.warning("ATR computation failed: %s", exc)

        try:
            result["obv"] = self.compute_obv(df)
        except Exception as exc:
            logger.warning("OBV computation failed: %s", exc)

        try:
            result["vwap"] = self.compute_vwap(df)
        except Exception as exc:
            logger.warning("VWAP computation failed: %s", exc)

        try:
            stoch_df = self.compute_stochastic(df)
            result = pd.concat([result, stoch_df], axis=1)
        except Exception as exc:
            logger.warning("Stochastic computation failed: %s", exc)

        try:
            result["adx"] = self.compute_adx(df)
        except Exception as exc:
            logger.warning("ADX computation failed: %s", exc)

        try:
            ichi_df = self.compute_ichimoku(df)
            result = pd.concat([result, ichi_df], axis=1)
        except Exception as exc:
            logger.warning("Ichimoku computation failed: %s", exc)

        try:
            fib_df = self.compute_fibonacci_levels(df)
            result = pd.concat([result, fib_df], axis=1)
        except Exception as exc:
            logger.warning("Fibonacci computation failed: %s", exc)

        logger.info(
            "Technical features computed: %d indicators added",
            result.shape[1] - df.shape[1],
        )
        return result

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------
    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing method.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        period : int
            Look-back period (default 14).

        Returns
        -------
        pd.Series
            RSI values in [0, 100].
        """
        self._validate_ohlcv(df)
        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's exponential moving average
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.rename("rsi")

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    def compute_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        fast : int
            Fast EMA period.
        slow : int
            Slow EMA period.
        signal : int
            Signal line EMA period.

        Returns
        -------
        pd.DataFrame
            Columns: macd, macd_signal, macd_histogram.
        """
        self._validate_ohlcv(df)
        close = df["close"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std: int = 2,
    ) -> pd.DataFrame:
        """Bollinger Bands with bandwidth and %B.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        period : int
            Rolling window size.
        std : int
            Number of standard deviations.

        Returns
        -------
        pd.DataFrame
            Columns: bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b.
        """
        self._validate_ohlcv(df)
        close = df["close"]

        middle = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper = middle + std * rolling_std
        lower = middle - std * rolling_std

        bandwidth = (upper - lower) / middle
        band_range = upper - lower
        percent_b = (close - lower) / band_range.replace(0, np.nan)

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_bandwidth": bandwidth,
                "bb_percent_b": percent_b,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    def compute_ema(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Exponential Moving Averages for multiple periods.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        periods : list[int] | None
            List of EMA periods (default [9, 21, 50, 200]).

        Returns
        -------
        pd.DataFrame
            One column per period, named ``ema_<period>``.
        """
        self._validate_ohlcv(df)
        if periods is None:
            periods = [9, 21, 50, 200]

        close = df["close"]
        result: dict[str, pd.Series] = {}
        for p in periods:
            result[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
        return pd.DataFrame(result, index=df.index)

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        period : int
            Look-back period (default 14).

        Returns
        -------
        pd.Series
            ATR values.
        """
        self._validate_ohlcv(df)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return atr.rename("atr")

    # ------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------
    def compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.Series
            Cumulative OBV.
        """
        self._validate_ohlcv(df)
        close = df["close"]
        volume = df["volume"]

        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * volume).cumsum()
        return obv.rename("obv")

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------
    def compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume-Weighted Average Price (cumulative).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.Series
            Running VWAP from the start of the DataFrame.
        """
        self._validate_ohlcv(df)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        vwap = cum_tp_vol / cum_vol
        return vwap.rename("vwap")

    # ------------------------------------------------------------------
    # Stochastic Oscillator
    # ------------------------------------------------------------------
    def compute_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Stochastic Oscillator (%K and %D).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        k_period : int
            Look-back for %K (default 14).
        d_period : int
            SMA period for %D (default 3).

        Returns
        -------
        pd.DataFrame
            Columns: stoch_k, stoch_d.
        """
        self._validate_ohlcv(df)
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        denom = (high_max - low_min).replace(0, np.nan)
        stoch_k = 100.0 * (df["close"] - low_min) / denom
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return pd.DataFrame(
            {"stoch_k": stoch_k, "stoch_d": stoch_d},
            index=df.index,
        )

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------
    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        period : int
            Look-back period (default 14).

        Returns
        -------
        pd.Series
            ADX values.
        """
        self._validate_ohlcv(df)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional movements
        plus_dm = np.where(
            (high - prev_high) > (prev_low - low),
            np.maximum(high - prev_high, 0),
            0.0,
        )
        minus_dm = np.where(
            (prev_low - low) > (high - prev_high),
            np.maximum(prev_low - low, 0),
            0.0,
        )

        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # Wilder smoothing
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di = 100.0 * plus_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100.0 * minus_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx.rename("adx")

    # ------------------------------------------------------------------
    # Ichimoku Cloud
    # ------------------------------------------------------------------
    def compute_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud (Tenkan, Kijun, Senkou A/B, Chikou).

        Uses the classic 9-26-52 parameter set.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ichi_tenkan, ichi_kijun, ichi_senkou_a, ichi_senkou_b,
            ichi_chikou.
        """
        self._validate_ohlcv(df)
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (conversion line) - 9 period
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0

        # Kijun-sen (base line) - 26 period
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0

        # Senkou Span A (leading span A) - midpoint of tenkan/kijun, shifted 26 forward
        senkou_a = ((tenkan + kijun) / 2.0).shift(26)

        # Senkou Span B (leading span B) - 52-period midpoint, shifted 26 forward
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)

        # Chikou Span (lagging span) - close shifted 26 periods back
        chikou = close.shift(-26)

        return pd.DataFrame(
            {
                "ichi_tenkan": tenkan,
                "ichi_kijun": kijun,
                "ichi_senkou_a": senkou_a,
                "ichi_senkou_b": senkou_b,
                "ichi_chikou": chikou,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------
    # Fibonacci Retracement Levels
    # ------------------------------------------------------------------
    def compute_fibonacci_levels(
        self,
        df: pd.DataFrame,
        period: int = 100,
    ) -> pd.DataFrame:
        """Rolling Fibonacci retracement levels.

        For each row the high and low over the trailing *period* bars are
        identified, then the standard retracement levels (0 %, 23.6 %,
        38.2 %, 50 %, 61.8 %, 78.6 %, 100 %) are computed.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        period : int
            Look-back window (default 100).

        Returns
        -------
        pd.DataFrame
            Columns: fib_0, fib_236, fib_382, fib_500, fib_618, fib_786,
            fib_1000.
        """
        self._validate_ohlcv(df)
        period_high = df["high"].rolling(window=period).max()
        period_low = df["low"].rolling(window=period).min()
        diff = period_high - period_low

        levels = {
            "fib_0": period_low,
            "fib_236": period_low + 0.236 * diff,
            "fib_382": period_low + 0.382 * diff,
            "fib_500": period_low + 0.500 * diff,
            "fib_618": period_low + 0.618 * diff,
            "fib_786": period_low + 0.786 * diff,
            "fib_1000": period_high,
        }
        return pd.DataFrame(levels, index=df.index)
