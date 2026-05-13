"""
Unit tests for OnChainFeatures.
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
from omnitrade.features.onchain_features import OnChainFeatures

settings = Settings()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_timestamps(n: int, base: str = "2024-01-01") -> List[pd.Timestamp]:
    """Return *n* daily timestamps starting at *base*."""
    return list(pd.date_range(start=base, periods=n, freq="D"))


# ---------------------------------------------------------------------------
# compute_whale_pressure
# ---------------------------------------------------------------------------

def test_whale_pressure_case_insensitivity() -> None:
    """Direction strings 'buy' / 'sell' handled case-insensitively."""
    ts = make_timestamps(8)
    transfers: List[Dict[str, Any]] = [
        {"timestamp": t, "amount": 100.0, "direction": d}
        for t, d in zip(
            ts,
            ["BUY", "Sell", "buy", "SELL", "Buy", "sell", "bUY", "SELl"],
        )
    ]
    f = OnChainFeatures(settings)
    result = f.compute_whale_pressure(transfers)

    columns = {"whale_buy_pressure", "whale_sell_pressure", "net_whale_flow"}
    assert columns.issubset(result.columns), f"Missing columns: {columns - set(result.columns)}"

    # Net flow = buy - sell
    pd.testing.assert_series_equal(
        result["net_whale_flow"],
        result["whale_buy_pressure"] - result["whale_sell_pressure"],
        check_names=False,
        check_dtype=False,
    )


def test_whale_pressure_net_flow_formula() -> None:
    """Verify net_whale_flow == buy minus sell for a simple case."""
    ts = make_timestamps(2)
    transfers: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "amount": 200.0, "direction": "buy"},
        {"timestamp": ts[0], "amount": 50.0, "direction": "sell"},
        {"timestamp": ts[1], "amount": 100.0, "direction": "buy"},
    ]
    f = OnChainFeatures(settings)
    result = f.compute_whale_pressure(transfers)

    # Day 0: buy=200, sell=50  → net=150  (rolling mean of 1 point)
    # Day 1: buy=100, sell=0   → buy_roll=(200+100)/2=150, sell_roll=(50+0)/2=25, net=125
    eps = 1e-9
    assert abs(result["net_whale_flow"].iloc[0] - 150.0) < eps
    assert abs(result["net_whale_flow"].iloc[-1] - 125.0) < eps


def test_whale_pressure_empty() -> None:
    """Empty input raises ValueError."""
    f = OnChainFeatures(settings)
    try:
        f.compute_whale_pressure([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_exchange_flow_features
# ---------------------------------------------------------------------------

def test_exchange_flow_columns() -> None:
    """Verify expected columns exist."""
    ts = make_timestamps(5)
    flows: List[Dict[str, Any]] = [
        {"timestamp": t, "inflow": 1000.0, "outflow": 500.0} for t in ts
    ]
    f = OnChainFeatures(settings)
    result = f.compute_exchange_flow_features(flows)

    assert "exchange_inflow_ma" in result.columns
    assert "exchange_outflow_ma" in result.columns
    assert "net_flow_change" in result.columns

    # With constant net flow, pct_change is 0 (first row NaN → fillna 0)
    assert (result["net_flow_change"] == 0.0).all()


def test_exchange_flow_inf_handling() -> None:
    """Inf values from pct_change (zero → non-zero) are replaced with 0."""
    ts = make_timestamps(3)
    flows: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "inflow": 0.0, "outflow": 0.0},     # net=0
        {"timestamp": ts[1], "inflow": 100.0, "outflow": 50.0},   # net=50 → inf
        {"timestamp": ts[2], "inflow": 200.0, "outflow": 100.0},  # net=100
    ]
    f = OnChainFeatures(settings)
    result = f.compute_exchange_flow_features(flows)

    assert not result["net_flow_change"].isin([np.inf, -np.inf]).any()
    assert not result["net_flow_change"].isna().any()


def test_exchange_flow_empty() -> None:
    """Empty input raises ValueError."""
    f = OnChainFeatures(settings)
    try:
        f.compute_exchange_flow_features([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_network_activity
# ---------------------------------------------------------------------------

def test_network_activity_columns() -> None:
    """Verify expected columns exist."""
    ts = make_timestamps(5)
    metrics: List[Dict[str, Any]] = [
        {"timestamp": t, "active_addresses": 1000.0, "hash_rate": 100.0, "gas_fee": 50.0}
        for t in ts
    ]
    f = OnChainFeatures(settings)
    result = f.compute_network_activity(metrics)

    assert "active_addr_change" in result.columns
    assert "hash_rate_change" in result.columns
    assert "gas_fee_zscore" in result.columns


def test_network_activity_identical_gas_fees() -> None:
    """Identical gas fees (std = 0) produce z-scores of 0, not a crash."""
    ts = make_timestamps(35)
    metrics: List[Dict[str, Any]] = [
        {"timestamp": t, "active_addresses": float(i), "hash_rate": 100.0, "gas_fee": 50.0}
        for i, t in enumerate(ts)
    ]
    f = OnChainFeatures(settings)
    result = f.compute_network_activity(metrics)

    assert not result["gas_fee_zscore"].isna().any()
    assert (result["gas_fee_zscore"] == 0.0).all()


def test_network_activity_empty() -> None:
    """Empty input raises ValueError."""
    f = OnChainFeatures(settings)
    try:
        f.compute_network_activity([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_merges_all_groups() -> None:
    """All three sub-computations merged into one DataFrame."""
    ts = make_timestamps(5)
    transfers: List[Dict[str, Any]] = [
        {"timestamp": t, "amount": 100.0, "direction": "buy"} for t in ts
    ]
    flows: List[Dict[str, Any]] = [
        {"timestamp": t, "inflow": 1000.0, "outflow": 500.0} for t in ts
    ]
    metrics: List[Dict[str, Any]] = [
        {"timestamp": t, "active_addresses": 1000.0, "hash_rate": 100.0, "gas_fee": 10.0}
        for t in ts
    ]

    f = OnChainFeatures(settings)
    result = f.compute_all(transfers, flows, metrics)

    expected = {
        # whale
        "whale_buy_pressure", "whale_sell_pressure", "net_whale_flow",
        # exchange
        "exchange_inflow_ma", "exchange_outflow_ma", "net_flow_change",
        # network
        "active_addr_change", "hash_rate_change", "gas_fee_zscore",
    }
    assert expected.issubset(result.columns), f"Missing columns: {expected - set(result.columns)}"
    assert len(result) > 0


def test_compute_all_partial_failure() -> None:
    """One failing sub-computation does not crash the whole pipeline."""
    ts = make_timestamps(3)
    valid_transfers: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "amount": 100.0, "direction": "buy"},
    ]
    bad_flows: List[Dict[str, Any]] = []  # empty → ValueError caught by _safe_compute
    valid_metrics: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "active_addresses": 1000.0, "hash_rate": 100.0, "gas_fee": 10.0},
    ]

    f = OnChainFeatures(settings)
    result = f.compute_all(valid_transfers, bad_flows, valid_metrics)

    assert not result.empty
    assert "whale_buy_pressure" in result.columns
    assert "active_addr_change" in result.columns
    # exchange features should be absent
    assert "exchange_inflow_ma" not in result.columns
    assert "exchange_outflow_ma" not in result.columns


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_whale_pressure_case_insensitivity,
        test_whale_pressure_net_flow_formula,
        test_whale_pressure_empty,
        test_exchange_flow_columns,
        test_exchange_flow_inf_handling,
        test_exchange_flow_empty,
        test_network_activity_columns,
        test_network_activity_identical_gas_fees,
        test_network_activity_empty,
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
