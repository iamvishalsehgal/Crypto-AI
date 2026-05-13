"""
Unit tests for TechnicalFeatures indicators.
12 indicators, every trading signal depends on them, zero unit tests previously.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from omnitrade.features.technical import TechnicalFeatures
from omnitrade.config.settings import Settings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_ohlcv(
    n_rows: int = 200,
    start_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OHLCV with known statistical properties."""
    np.random.seed(seed)
    returns = np.random.randn(n_rows) * volatility + trend
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n_rows) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_rows) * 0.01))
    open_ = close * (1 + np.random.randn(n_rows) * 0.005)
    volume = np.random.uniform(100, 10_000, n_rows)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


settings = Settings()
tech = TechnicalFeatures(settings)


# ---------------------------------------------------------------------------
# RSI  (14-period Wilder)
# ---------------------------------------------------------------------------

def test_rsi_bounds():
    """RSI in [0, 100] always."""
    df = make_ohlcv(200)
    result = tech.compute_all(df)
    rsi = result["rsi"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), \
        f"RSI out of bounds: min={rsi.min():.2f}, max={rsi.max():.2f}"


def test_rsi_extreme_values():
    """RSI > 60 in strong uptrend, RSI < 40 in strong downtrend.

    Use volatility=0.015 so that a few opposing-move days exist (otherwise
    avg_loss or avg_gain goes to zero and RSI becomes NaN).
    """
    up = make_ohlcv(200, trend=0.01, volatility=0.015, seed=1)
    rsi_up = tech.compute_rsi(up).dropna()
    assert rsi_up.iloc[-1] > 60, \
        f"RSI on uptrend too low: {rsi_up.iloc[-1]:.2f}"

    down = make_ohlcv(200, trend=-0.01, volatility=0.015, seed=2)
    rsi_down = tech.compute_rsi(down).dropna()
    assert rsi_down.iloc[-1] < 40, \
        f"RSI on downtrend too high: {rsi_down.iloc[-1]:.2f}"


def test_rsi_nan_on_zero_volatility():
    """RSI is NaN when every return is identical (no opposing moves)."""
    n = 200
    np.random.seed(0)
    close = pd.Series(np.full(n, 100.0))
    df = pd.DataFrame({
        "open": close,
        "high": close * 1.001,
        "low": close / 1.001,
        "close": close,
        "volume": np.random.uniform(100, 10_000, n),
    })
    rsi = tech.compute_rsi(df).dropna()
    assert rsi.isna().all(), "RSI should be all NaN on zero-volatility flat data"


# ---------------------------------------------------------------------------
# MACD  (12-26-9)
# ---------------------------------------------------------------------------

def test_macd_sign():
    """MACD histogram positive at end of strong sustained uptrend."""
    up = make_ohlcv(200, trend=0.03, seed=1)
    macd = tech.compute_macd(up).dropna()
    assert macd["macd_histogram"].iloc[-20:].mean() > 0, \
        "MACD histogram negative in uptrend"


def test_macd_columns():
    """compute_macd returns exactly macd, macd_signal, macd_histogram."""
    df = make_ohlcv(200)
    macd = tech.compute_macd(df)
    assert list(macd.columns) == ["macd", "macd_signal", "macd_histogram"]


def test_macd_no_nan():
    """MACD uses full ewm so no NaN for any row (first value = close[0])."""
    df = make_ohlcv(200)
    macd = tech.compute_macd(df)
    assert macd.isna().sum().sum() == 0, "MACD has unexpected NaN"


# ---------------------------------------------------------------------------
# Bollinger Bands  (20,2)
# ---------------------------------------------------------------------------

def test_bollinger_bands_bounds():
    """Price inside bands > 80 % of time (2-sigma -> ~95 % expected)."""
    df = make_ohlcv(200)
    bb = tech.compute_bollinger_bands(df).dropna()
    close = df["close"].loc[bb.index]
    inside = (close >= bb["bb_lower"]) & (close <= bb["bb_upper"])
    assert inside.mean() > 0.80, \
        f"Only {inside.mean() * 100:.0f} % of prices inside bands"


def test_bollinger_bands_columns():
    """compute_bollinger_bands returns bb_upper/bb_middle/bb_lower/bb_bandwidth/bb_percent_b."""
    df = make_ohlcv(200)
    bb = tech.compute_bollinger_bands(df)
    expected = ["bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_percent_b"]
    assert list(bb.columns) == expected


def test_bollinger_middle_is_sma():
    """bb_middle equals simple rolling mean."""
    df = make_ohlcv(200)
    bb = tech.compute_bollinger_bands(df, period=20).dropna()
    sma = df["close"].rolling(20).mean().loc[bb.index]
    assert (bb["bb_middle"] - sma).abs().max() < 1e-12, \
        "bb_middle diverges from SMA"


def test_bollinger_percent_b():
    """Percent B is 0.5 when close equals middle band."""
    df = make_ohlcv(200)
    bb = tech.compute_bollinger_bands(df).dropna()
    mid_idx = (bb["bb_percent_b"] - 0.5).abs().idxmin()
    assert abs(bb.loc[mid_idx, "bb_percent_b"] - 0.5) < 0.1, \
        "Percent B not near 0.5 at middle"


# ---------------------------------------------------------------------------
# EMA  (default periods 9, 21, 50, 200)
# ---------------------------------------------------------------------------

def test_ema_ordering():
    """Shorter EMA closer to price than longer EMA."""
    df = make_ohlcv(200, seed=4)
    result = tech.compute_all(df).dropna()
    close = df["close"].loc[result.index]

    diff_9 = (result["ema_9"] - close).abs().mean()
    diff_50 = (result["ema_50"] - close).abs().mean()
    assert diff_9 < diff_50, \
        f"EMA-9 not closer to price: {diff_9:.2f} vs {diff_50:.2f}"


def test_ema_default_periods():
    """compute_ema with no arguments returns ema_9, ema_21, ema_50, ema_200."""
    df = make_ohlcv(200)
    ema = tech.compute_ema(df)
    assert list(ema.columns) == ["ema_9", "ema_21", "ema_50", "ema_200"]


def test_ema_custom_periods():
    """Custom period list."""
    df = make_ohlcv(200)
    ema = tech.compute_ema(df, periods=[5, 10, 30])
    assert list(ema.columns) == ["ema_5", "ema_10", "ema_30"]


def test_ema_no_nan():
    """Full ewm — no NaN rows."""
    df = make_ohlcv(200)
    ema = tech.compute_ema(df)
    assert ema.isna().sum().sum() == 0, "EMA has unexpected NaN"


# ---------------------------------------------------------------------------
# ATR  (14-period)
# ---------------------------------------------------------------------------

def test_atr_positive():
    """ATR always > 0 (true range is always positive)."""
    df = make_ohlcv(200)
    atr = tech.compute_atr(df).dropna()
    assert (atr > 0).all(), f"ATR has non-positive values: min={atr.min():.4f}"


def test_atr_increasing_volatility():
    """ATR higher when volatility is higher."""
    low_vol = make_ohlcv(300, trend=0.0, volatility=0.005, seed=10)
    high_vol = make_ohlcv(300, trend=0.0, volatility=0.05, seed=20)
    atr_low = tech.compute_atr(low_vol).dropna().iloc[-1]
    atr_high = tech.compute_atr(high_vol).dropna().iloc[-1]
    assert atr_high > atr_low, \
        f"ATR not higher with volatility: {atr_low:.2f} vs {atr_high:.2f}"


# ---------------------------------------------------------------------------
# OBV  (On-Balance Volume)
# ---------------------------------------------------------------------------

def test_obv_trend():
    """OBV increases in sustained uptrend (more volume on up days)."""
    up = make_ohlcv(200, trend=0.01, seed=5)
    obv = tech.compute_obv(up)
    assert obv.iloc[-1] > obv.iloc[0], \
        f"OBV did not increase in uptrend: {obv.iloc[0]:.0f} -> {obv.iloc[-1]:.0f}"


def test_obv_zero_on_no_movement():
    """OBV is all zero when price never changes."""
    df = make_ohlcv(200, trend=0.0, volatility=0.0, seed=6)
    obv = tech.compute_obv(df)
    assert (obv == 0).all(), "OBV should be all zeros with zero volatility"


# ---------------------------------------------------------------------------
# VWAP  (cumulative)
# ---------------------------------------------------------------------------

def test_vwap_positive():
    """VWAP strictly positive for positive prices."""
    df = make_ohlcv(200)
    vwap = tech.compute_vwap(df)
    assert (vwap > 0).all()


def test_vwap_between_low_high():
    """Cumulative VWAP lies in [global_min_low, global_max_high]."""
    df = make_ohlcv(200)
    vwap = tech.compute_vwap(df).dropna()
    assert (vwap >= df["low"].min()).all() and (vwap <= df["high"].max()).all(), \
        "VWAP outside global [min_low, max_high]"


# ---------------------------------------------------------------------------
# Stochastic Oscillator  (14,3)
# ---------------------------------------------------------------------------

def test_stochastic_bounds():
    """Stochastic %K and %D bounded in [0, 100]."""
    df = make_ohlcv(200)
    stoch = tech.compute_stochastic(df).dropna()
    assert (stoch["stoch_k"] >= 0).all() and (stoch["stoch_k"] <= 100).all()
    assert (stoch["stoch_d"] >= 0).all() and (stoch["stoch_d"] <= 100).all()


def test_stochastic_columns():
    """Returns stoch_k and stoch_d."""
    df = make_ohlcv(200)
    stoch = tech.compute_stochastic(df)
    assert list(stoch.columns) == ["stoch_k", "stoch_d"]


def test_stochastic_d_smoother_than_k():
    """%D (3-SMA of %K) has lower variance than %K."""
    df = make_ohlcv(200)
    stoch = tech.compute_stochastic(df).dropna()
    assert stoch["stoch_d"].std() < stoch["stoch_k"].std(), \
        "%D should be smoother (lower std) than %K"


# ---------------------------------------------------------------------------
# ADX  (14-period)
# ---------------------------------------------------------------------------

def test_adx_bounds():
    """ADX bounded in [0, 100]."""
    df = make_ohlcv(200)
    adx = tech.compute_adx(df).dropna()
    assert (adx >= 0).all() and (adx <= 100).all(), \
        f"ADX out of [0, 100]: min={adx.min():.2f}, max={adx.max():.2f}"


def test_adx_low_in_no_trend():
    """ADX tends to be lower in random-walk (no-trend) data."""
    df = make_ohlcv(300, trend=0.0, volatility=0.02, seed=7)
    adx = tech.compute_adx(df).dropna()
    assert adx.iloc[-1] < 50, \
        f"ADX too high for no-trend data: {adx.iloc[-1]:.2f}"


# ---------------------------------------------------------------------------
# Ichimoku Cloud  (9-26-52)
# ---------------------------------------------------------------------------

def test_ichimoku_columns():
    """Returns all five ichimoku components."""
    df = make_ohlcv(200)
    ichi = tech.compute_ichimoku(df)
    expected = [
        "ichi_tenkan", "ichi_kijun",
        "ichi_senkou_a", "ichi_senkou_b",
        "ichi_chikou",
    ]
    assert list(ichi.columns) == expected


def test_ichimoku_tenkan_faster_than_kijun():
    """Tenkan (9-period) more volatile than Kijun (26-period)."""
    df = make_ohlcv(200, trend=0.01)
    ichi = tech.compute_ichimoku(df).dropna()
    assert ichi["ichi_tenkan"].std() > ichi["ichi_kijun"].std(), \
        "Tenkan should be more volatile than Kijun"


# ---------------------------------------------------------------------------
# Fibonacci Retracement  (100-period rolling window)
# ---------------------------------------------------------------------------

def test_fibonacci_columns():
    """Returns all seven retracement levels."""
    df = make_ohlcv(500)
    fib = tech.compute_fibonacci_levels(df, period=100).dropna()
    expected = ["fib_0", "fib_236", "fib_382", "fib_500",
                "fib_618", "fib_786", "fib_1000"]
    assert list(fib.columns) == expected


def test_fibonacci_monotonic():
    """Levels are monotonically increasing: fib_0 <= fib_236 <= ... <= fib_1000."""
    df = make_ohlcv(500)
    fib = tech.compute_fibonacci_levels(df, period=100).dropna()
    pairs = [("fib_0", "fib_236"), ("fib_236", "fib_382"),
             ("fib_382", "fib_500"), ("fib_500", "fib_618"),
             ("fib_618", "fib_786"), ("fib_786", "fib_1000")]
    for lo, hi in pairs:
        assert (fib[hi] >= fib[lo] - 1e-12).all(), \
            f"{hi} < {lo} (violates monotonicity)"


def test_fibonacci_bounds():
    """fib_0 == period_low, fib_1000 == period_high."""
    df = make_ohlcv(500)
    fib = tech.compute_fibonacci_levels(df, period=100).dropna()
    high = df["high"].rolling(100).max().loc[fib.index]
    low = df["low"].rolling(100).min().loc[fib.index]
    assert (fib["fib_0"] - low).abs().max() < 1e-12, \
        "fib_0 diverges from period low"
    assert (fib["fib_1000"] - high).abs().max() < 1e-12, \
        "fib_1000 diverges from period high"


# ---------------------------------------------------------------------------
# compute_all — integration
# ---------------------------------------------------------------------------

def test_compute_all_returns_dataframe():
    """compute_all returns a DataFrame with every indicator column."""
    df = make_ohlcv(200)
    result = tech.compute_all(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    indicator_cols = [
        # RSI
        "rsi",
        # MACD
        "macd", "macd_signal", "macd_histogram",
        # Bollinger Bands
        "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_percent_b",
        # EMA
        "ema_9", "ema_21", "ema_50", "ema_200",
        # ATR / OBV / VWAP
        "atr", "obv", "vwap",
        # Stochastic
        "stoch_k", "stoch_d",
        # ADX
        "adx",
        # Ichimoku
        "ichi_tenkan", "ichi_kijun",
        "ichi_senkou_a", "ichi_senkou_b", "ichi_chikou",
        # Fibonacci
        "fib_0", "fib_236", "fib_382", "fib_500",
        "fib_618", "fib_786", "fib_1000",
    ]
    for col in indicator_cols:
        assert col in result.columns, f"Missing indicator column: {col}"


def test_compute_all_preserves_ohlcv():
    """Original OHLCV columns are unmodified."""
    df = make_ohlcv(200)
    result = tech.compute_all(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns
        assert (result[col] == df[col]).all(), \
            f"{col} was modified by compute_all"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_insufficient_data_handling():
    """Short DataFrame (10 rows) should not crash — returns mostly/fully NaN."""
    df = make_ohlcv(10)
    try:
        result = tech.compute_all(df)
        assert isinstance(result, pd.DataFrame)
    except ValueError as e:
        assert "insufficient" in str(e).lower() or "too few" in str(e).lower()


def test_missing_columns_raises():
    """Missing OHLCV column raises ValueError."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    try:
        tech.compute_all(df)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_empty_df_raises():
    """Empty DataFrame raises ValueError."""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        tech.compute_all(df)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_constant_price():
    """Constant price produces NaN for some indicators but no crash."""
    n = 200
    df = pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 5000.0),
    })
    result = tech.compute_all(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_rsi_bounds,
        test_rsi_extreme_values,
        test_rsi_nan_on_zero_volatility,
        test_macd_sign,
        test_macd_columns,
        test_macd_no_nan,
        test_bollinger_bands_bounds,
        test_bollinger_bands_columns,
        test_bollinger_middle_is_sma,
        test_bollinger_percent_b,
        test_ema_ordering,
        test_ema_default_periods,
        test_ema_custom_periods,
        test_ema_no_nan,
        test_atr_positive,
        test_atr_increasing_volatility,
        test_obv_trend,
        test_obv_zero_on_no_movement,
        test_vwap_positive,
        test_vwap_between_low_high,
        test_stochastic_bounds,
        test_stochastic_columns,
        test_stochastic_d_smoother_than_k,
        test_adx_bounds,
        test_adx_low_in_no_trend,
        test_ichimoku_columns,
        test_ichimoku_tenkan_faster_than_kijun,
        test_fibonacci_columns,
        test_fibonacci_monotonic,
        test_fibonacci_bounds,
        test_compute_all_returns_dataframe,
        test_compute_all_preserves_ohlcv,
        test_insufficient_data_handling,
        test_missing_columns_raises,
        test_empty_df_raises,
        test_constant_price,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
