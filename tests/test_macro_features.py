"""
Unit tests for MacroFeatures.
"""

from __future__ import annotations

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.features.macro_features import MacroFeatures

settings = Settings()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_timestamps(n: int, base: str = "2024-01-01") -> pd.DatetimeIndex:
    """Return a DatetimeIndex of *n* daily timestamps starting at *base*."""
    return pd.date_range(start=base, periods=n, freq="D")


# ---------------------------------------------------------------------------
# compute_rate_features
# ---------------------------------------------------------------------------

def test_rate_features_columns_and_direction() -> None:
    """rate_change, rate_direction, rate_momentum exist; direction = sign(change)."""
    dates = make_timestamps(6)
    fed = pd.DataFrame(
        {"rate": [5.0, 5.0, 5.25, 5.5, 5.5, 5.25]},
        index=dates,
    )
    f = MacroFeatures(settings)
    result = f.compute_rate_features(fed)

    assert "rate_change" in result.columns
    assert "rate_direction" in result.columns
    assert "rate_momentum" in result.columns

    # rate_direction = sign of rate_change  (first is NaN → fillna 0 → sign 0)
    expected_dir = [0, 0, 1, 1, 0, -1]
    for i, exp in enumerate(expected_dir):
        assert result["rate_direction"].iloc[i] == exp, f"row {i}: expected {exp}, got {result['rate_direction'].iloc[i]}"

    # rate_change values
    assert result["rate_change"].iloc[0] == 0.0
    assert result["rate_change"].iloc[2] == 0.25


def test_rate_features_empty() -> None:
    """Empty DataFrame raises ValueError."""
    f = MacroFeatures(settings)
    try:
        f.compute_rate_features(pd.DataFrame())
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_volatility_features
# ---------------------------------------------------------------------------

def test_volatility_features_regime() -> None:
    """Regime classification: low < 15, medium 15-25, high >= 25."""
    dates = make_timestamps(6)
    vix = pd.DataFrame(
        {"close": [10.0, 20.0, 30.0, 15.0, 25.0, 14.99]},
        index=dates,
    )
    f = MacroFeatures(settings)
    result = f.compute_volatility_features(vix)

    assert "vix_zscore" in result.columns
    assert "vix_regime" in result.columns
    assert "vix_change" in result.columns

    expected = ["low", "medium", "high", "medium", "high", "low"]
    for i, exp in enumerate(expected):
        assert result["vix_regime"].iloc[i] == exp, f"row {i}: expected {exp}, got {result['vix_regime'].iloc[i]}"


def test_volatility_features_vix_change() -> None:
    """vix_change handles inf correctly."""
    dates = make_timestamps(3)
    vix = pd.DataFrame(
        {"close": [0.0, 15.0, 20.0]},  # first is 0 → pct_change from 0 yields inf
        index=dates,
    )
    f = MacroFeatures(settings)
    result = f.compute_volatility_features(vix)

    assert not result["vix_change"].isin([np.inf, -np.inf]).any()
    assert not result["vix_change"].isna().any()


# ---------------------------------------------------------------------------
# compute_dollar_features
# ---------------------------------------------------------------------------

def test_dollar_features_with_crypto() -> None:
    """With crypto_close, correlation column is populated."""
    dates = make_timestamps(30)
    dxy = pd.DataFrame(
        {"close": list(range(100, 130)), "crypto_close": list(range(40000, 40030))},
        index=dates,
    )
    f = MacroFeatures(settings)
    result = f.compute_dollar_features(dxy)

    assert "dxy_trend" in result.columns
    assert "dxy_change" in result.columns
    assert "crypto_dollar_correlation" in result.columns
    # With 30 days of data and min_periods=5, at least the last row should be non-NaN
    assert not result["crypto_dollar_correlation"].isna().all()


def test_dollar_features_without_crypto() -> None:
    """Without crypto_close, correlation column is all NaN."""
    dates = make_timestamps(5)
    dxy = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )
    f = MacroFeatures(settings)
    result = f.compute_dollar_features(dxy)

    assert "crypto_dollar_correlation" in result.columns
    assert result["crypto_dollar_correlation"].isna().all()


def test_dollar_features_empty() -> None:
    """Empty DataFrame raises ValueError."""
    f = MacroFeatures(settings)
    try:
        f.compute_dollar_features(pd.DataFrame())
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_commodity_features
# ---------------------------------------------------------------------------

def test_commodity_features_union_index() -> None:
    """Result index is the union of gold and oil indices; missing values handled."""
    gold_dates = make_timestamps(5, "2024-01-01")
    oil_dates = make_timestamps(4, "2024-01-02")  # offset by one day

    gold = pd.DataFrame({"close": [2000.0, 2010.0, 2020.0, 2030.0, 2040.0]}, index=gold_dates)
    oil = pd.DataFrame({"close": [70.0, 71.0, 72.0, 73.0]}, index=oil_dates)

    f = MacroFeatures(settings)
    result = f.compute_commodity_features(gold, oil)

    assert "gold_crypto_ratio" in result.columns
    assert "oil_change" in result.columns

    # Index should be union of the two
    union = gold_dates.union(oil_dates)
    assert len(result) == len(union)
    assert (result.index == union).all()


def test_commodity_features_no_crypto_close() -> None:
    """Missing crypto_close → NaN gold_crypto_ratio."""
    dates = make_timestamps(3)
    gold = pd.DataFrame({"close": [2000.0, 2010.0, 2020.0]}, index=dates)
    oil = pd.DataFrame({"close": [70.0, 71.0, 72.0]}, index=dates)

    f = MacroFeatures(settings)
    result = f.compute_commodity_features(gold, oil)

    assert result["gold_crypto_ratio"].isna().all()


def test_commodity_features_with_crypto_close() -> None:
    """With crypto_close in gold, ratio is computed."""
    dates = make_timestamps(3)
    gold = pd.DataFrame(
        {"close": [2000.0, 2010.0, 2020.0], "crypto_close": [40000.0, 41000.0, 42000.0]},
        index=dates,
    )
    oil = pd.DataFrame({"close": [70.0, 71.0, 72.0]}, index=dates)

    f = MacroFeatures(settings)
    result = f.compute_commodity_features(gold, oil)

    # gold_crypto_ratio = gold / crypto
    eps = 1e-9
    assert abs(result["gold_crypto_ratio"].iloc[0] - 2000.0 / 40000.0) < eps
    assert abs(result["gold_crypto_ratio"].iloc[1] - 2010.0 / 41000.0) < eps


def test_commodity_features_empty() -> None:
    """Empty gold DataFrame raises ValueError."""
    f = MacroFeatures(settings)
    dates = make_timestamps(3)
    gold = pd.DataFrame()
    oil = pd.DataFrame({"close": [70.0, 71.0, 72.0]}, index=dates)
    try:
        f.compute_commodity_features(gold, oil)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_merges_all_groups() -> None:
    """All sub-computations merged into one DataFrame."""
    dates = make_timestamps(10)
    fed = pd.DataFrame({"rate": [5.0] * 10}, index=dates)
    vix = pd.DataFrame({"close": [15.0] * 10}, index=dates)
    dxy = pd.DataFrame({"close": [100.0] * 10}, index=dates)
    gold = pd.DataFrame({"close": [2000.0] * 10}, index=dates)
    oil = pd.DataFrame({"close": [70.0] * 10}, index=dates)

    f = MacroFeatures(settings)
    result = f.compute_all(fed, vix, dxy, gold, oil)

    expected = {
        "rate_change", "rate_direction", "rate_momentum",
        "vix_zscore", "vix_regime", "vix_change",
        "dxy_trend", "dxy_change", "crypto_dollar_correlation",
        "gold_crypto_ratio", "oil_change",
    }
    assert expected.issubset(result.columns), f"Missing: {expected - set(result.columns)}"


def test_compute_all_partial_failure() -> None:
    """One failing sub-computation does not crash the whole pipeline."""
    dates = make_timestamps(5)
    fed = pd.DataFrame({"rate": [5.0] * 5}, index=dates)
    vix = pd.DataFrame({"close": [15.0] * 5}, index=dates)
    dxy = pd.DataFrame({"close": [100.0] * 5}, index=dates)
    bad_gold = pd.DataFrame()  # empty → ValueError
    oil = pd.DataFrame({"close": [70.0] * 5}, index=dates)

    f = MacroFeatures(settings)
    result = f.compute_all(fed, vix, dxy, bad_gold, oil)

    assert not result.empty
    assert "rate_change" in result.columns
    assert "vix_zscore" in result.columns
    assert "dxy_trend" in result.columns
    # commodity features should be absent
    assert "gold_crypto_ratio" not in result.columns
    assert "oil_change" not in result.columns


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_rate_features_columns_and_direction,
        test_rate_features_empty,
        test_volatility_features_regime,
        test_volatility_features_vix_change,
        test_dollar_features_with_crypto,
        test_dollar_features_without_crypto,
        test_dollar_features_empty,
        test_commodity_features_union_index,
        test_commodity_features_no_crypto_close,
        test_commodity_features_with_crypto_close,
        test_commodity_features_empty,
        test_compute_all_merges_all_groups,
        test_compute_all_partial_failure,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
