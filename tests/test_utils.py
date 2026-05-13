"""
Unit tests for OmniTrade utility modules.

Covers: sentiment_label_to_float, american_to_decimal, CircuitBreaker,
PnLTracker, and TTLCache.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.utils.sentiment import sentiment_label_to_float
from omnitrade.utils.odds import american_to_decimal
from omnitrade.utils.circuit_breaker import CircuitBreaker
from omnitrade.utils.pnl_tracker import PnLTracker
from omnitrade.utils.cache import TTLCache


# ── sentiment_label_to_float ──────────────────────────────────────────


def test_sentiment_float_passthrough():
    """Float values pass through unchanged."""
    assert sentiment_label_to_float(0.65) == 0.65
    assert sentiment_label_to_float(-0.3) == -0.3
    assert sentiment_label_to_float(0.0) == 0.0


def test_sentiment_string_labels():
    """Recognised string labels map to expected floats."""
    assert sentiment_label_to_float("positive") == 0.8
    assert sentiment_label_to_float("very_positive") == 0.95
    assert sentiment_label_to_float("negative") == -0.8
    assert sentiment_label_to_float("very_negative") == -0.95
    assert sentiment_label_to_float("neutral") == 0.0


def test_sentiment_string_case_insensitive():
    """String labels are case-insensitive."""
    assert sentiment_label_to_float("POSITIVE") == 0.8
    assert sentiment_label_to_float("Very_Positive") == 0.95
    assert sentiment_label_to_float("NEGATIVE") == -0.8


def test_sentiment_integers_clamped():
    """Integers are clamped to [-1, 1]."""
    assert sentiment_label_to_float(1) == 1.0
    assert sentiment_label_to_float(-5) == -1.0
    assert sentiment_label_to_float(0) == 0.0


def test_sentiment_unrecognised_string_defaults_to_zero():
    """Unrecognised string returns 0.0."""
    # Capture the warning log
    logger = logging.getLogger("omnitrade.utils.sentiment")
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        result = sentiment_label_to_float("garbage_label")
        assert result == 0.0
    finally:
        logger.removeHandler(handler)


def test_sentiment_numpy_integer():
    """numpy integer types are handled correctly."""
    assert sentiment_label_to_float(np.int64(1)) == 1.0
    assert sentiment_label_to_float(np.int64(-3)) == -1.0
    assert sentiment_label_to_float(np.int64(0)) == 0.0


# ── american_to_decimal ───────────────────────────────────────────────


def test_american_positive():
    """Positive american odds convert correctly."""
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(200) == 3.0


def test_american_negative():
    """Negative american odds convert correctly."""
    result = american_to_decimal(-200)
    assert abs(result - 1.5) < 1e-9
    result = american_to_decimal(-110)
    assert abs(result - 1.90909090909) < 1e-6


def test_american_zero():
    """Zero odds return 2.0."""
    assert american_to_decimal(0) == 2.0


# ── CircuitBreaker ────────────────────────────────────────────────────


def test_circuit_breaker_not_open_initially():
    """Circuit breaker starts closed."""
    cb = CircuitBreaker(threshold=3, cooldown=300)
    assert not cb.is_open


def test_circuit_breaker_opens_after_threshold():
    """After threshold failures, circuit breaker opens."""
    cb = CircuitBreaker(threshold=3, cooldown=300)
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open  # 2 < 3
    cb.record_failure()  # 3 == threshold
    assert cb.is_open


def test_circuit_breaker_record_success_resets():
    """record_success() resets the failure counter."""
    cb = CircuitBreaker(threshold=3, cooldown=300)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # reset to 0
    cb.record_failure()
    assert not cb.is_open  # 1 < 3


def test_circuit_breaker_reset_clears_state():
    """reset() clears tripped_at and failure count."""
    cb = CircuitBreaker(threshold=3, cooldown=300)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()  # tripped
    assert cb.is_open
    cb.reset()
    assert not cb.is_open
    assert cb._failures == 0
    assert cb._tripped_at is None


def test_circuit_breaker_cooldown_closes():
    """After cooldown expires, circuit breaker closes again."""
    cb = CircuitBreaker(threshold=3, cooldown=0.01)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open  # just tripped
    time.sleep(0.02)  # wait for cooldown
    assert not cb.is_open  # cooldown expired → auto-closed


# ── PnLTracker ────────────────────────────────────────────────────────


def test_pnl_long_trade():
    """Buy then sell → PnL = (exit - entry) * amount."""
    pnl = PnLTracker(initial_balance=10_000)
    pnl.record_entry({
        "order_id": "1", "symbol": "BTC/USDT",
        "side": "buy", "price": 50_000, "filled_amount": 0.1,
        "timestamp": "2026-01-01T00:00:00Z",
    })
    closed = pnl.record_exit("BTC/USDT", 55_000)
    assert closed is not None
    expected_pnl = (55_000 - 50_000) * 0.1  # = 500
    assert abs(closed.pnl - expected_pnl) < 1e-6
    assert closed.status == "closed"


def test_pnl_short_trade():
    """Sell then buy back → reverse PnL = (entry - exit) * amount."""
    pnl = PnLTracker(initial_balance=10_000)
    pnl.record_entry({
        "order_id": "2", "symbol": "ETH/USDT",
        "side": "sell", "price": 3_000, "filled_amount": 1.0,
        "timestamp": "2026-01-01T00:00:00Z",
    })
    closed = pnl.record_exit("ETH/USDT", 2_800)
    assert closed is not None
    expected_pnl = (3_000 - 2_800) * 1.0  # = 200
    assert abs(closed.pnl - expected_pnl) < 1e-6


def test_pnl_summary_keys():
    """summary() returns expected keys."""
    pnl = PnLTracker(initial_balance=10_000)
    summary = pnl.summary()
    expected_keys = {
        "total_trades", "closed_trades", "open_trades",
        "total_pnl", "win_rate_pct", "avg_win", "avg_loss",
        "profit_factor", "best_trade", "worst_trade",
    }
    assert set(summary.keys()) == expected_keys


def test_pnl_open_positions():
    """Open positions tracked correctly."""
    pnl = PnLTracker()
    pnl.record_entry({
        "order_id": "3", "symbol": "BTC/USDT",
        "side": "buy", "price": 50_000, "filled_amount": 0.1,
        "timestamp": "2026-01-01T00:00:00Z",
    })
    assert pnl.open_count == 1
    pnl.record_exit("BTC/USDT", 55_000)
    assert pnl.open_count == 0


def test_pnl_win_rate():
    """Win rate calculated correctly."""
    pnl = PnLTracker()
    # Trade 1: win
    pnl.record_entry({
        "order_id": "1", "symbol": "A", "side": "buy",
        "price": 100, "filled_amount": 1, "timestamp": "t1",
    })
    pnl.record_exit("A", 200)  # PnL = 100
    # Trade 2: loss
    pnl.record_entry({
        "order_id": "2", "symbol": "B", "side": "buy",
        "price": 100, "filled_amount": 1, "timestamp": "t2",
    })
    pnl.record_exit("B", 50)  # PnL = -50
    assert pnl.win_rate == 50.0  # 1 win / 2 total


def test_pnl_record_exit_no_open_position():
    """record_exit returns None when no open position matches."""
    pnl = PnLTracker()
    result = pnl.record_exit("NONEXISTENT/SYMBOL", 100)
    assert result is None


# ── TTLCache ──────────────────────────────────────────────────────────


def test_cache_set_get():
    """set then get returns the value."""
    cache = TTLCache(default_ttl=300)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_cache_expired_entry():
    """Expired entry returns None."""
    cache = TTLCache(default_ttl=0.01)
    cache.set("key1", "value1")
    time.sleep(0.02)
    assert cache.get("key1") is None


def test_cache_clear():
    """clear() empties the cache."""
    cache = TTLCache(default_ttl=300)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_invalidate():
    """invalidate() removes a specific key."""
    cache = TTLCache(default_ttl=300)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.invalidate("key1")
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"


def test_cache_nonexistent_key():
    """get on nonexistent key returns None."""
    cache = TTLCache()
    assert cache.get("does_not_exist") is None


def test_cache_custom_ttl():
    """set accepts per-key TTL override."""
    cache = TTLCache(default_ttl=300)
    cache.set("short", "short_value", ttl=0.01)
    cache.set("long", "long_value", ttl=60)
    time.sleep(0.02)
    assert cache.get("short") is None
    assert cache.get("long") == "long_value"


# ── test runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # sentiment_label_to_float
        test_sentiment_float_passthrough,
        test_sentiment_string_labels,
        test_sentiment_string_case_insensitive,
        test_sentiment_integers_clamped,
        test_sentiment_unrecognised_string_defaults_to_zero,
        test_sentiment_numpy_integer,
        # american_to_decimal
        test_american_positive,
        test_american_negative,
        test_american_zero,
        # CircuitBreaker
        test_circuit_breaker_not_open_initially,
        test_circuit_breaker_opens_after_threshold,
        test_circuit_breaker_record_success_resets,
        test_circuit_breaker_reset_clears_state,
        test_circuit_breaker_cooldown_closes,
        # PnLTracker
        test_pnl_long_trade,
        test_pnl_short_trade,
        test_pnl_summary_keys,
        test_pnl_open_positions,
        test_pnl_win_rate,
        test_pnl_record_exit_no_open_position,
        # TTLCache
        test_cache_set_get,
        test_cache_expired_entry,
        test_cache_clear,
        test_cache_invalidate,
        test_cache_nonexistent_key,
        test_cache_custom_ttl,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} — {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
