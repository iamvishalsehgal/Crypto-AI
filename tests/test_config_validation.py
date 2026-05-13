"""Tests for configuration validation in omnitrade.config.settings.

Verifies that Settings.validate() correctly catches invalid values,
emits warnings for missing credentials, and passes for valid config.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.config.settings import Settings


def test_invalid_position_size_raises_error():
    s = Settings()
    s.trading.max_position_size = 1.5
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_MAX_POSITION_SIZE" in str(e)


def test_missing_api_keys_produce_warnings():
    s = Settings()
    s.asset.enabled_assets = ["crypto"]
    s.exchange.api_key = ""
    s.exchange.api_secret = ""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = s.validate()
        warns = result or [str(x.message) for x in w]
        api_key_warns = [msg for msg in warns if "API_KEY" in str(msg) or "API_SECRET" in str(msg)]
        assert len(api_key_warns) >= 2, f"Expected >=2 API-credential warnings, got {api_key_warns}"


def test_valid_config_passes_silently():
    s = Settings()
    try:
        s.validate()
    except ValueError as exc:
        assert False, f"validate() raised ValueError for valid config: {exc}"


def test_stop_loss_out_of_range():
    s = Settings()
    s.trading.stop_loss = 0.75
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_STOP_LOSS" in str(e)


def test_take_profit_out_of_range():
    s = Settings()
    s.trading.take_profit = 2.0
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_TAKE_PROFIT" in str(e)


def test_max_daily_drawdown_out_of_range():
    s = Settings()
    s.trading.max_daily_drawdown = 0.6
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_MAX_DAILY_DRAWDOWN" in str(e)


def test_invalid_asset_type_raises_error():
    s = Settings()
    s.asset.enabled_assets = ["crypto", "magic"]
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "magic" in str(e)


def test_empty_supported_symbols_raises_error():
    s = Settings()
    s.exchange.supported_symbols = []
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "EXCHANGE_SUPPORTED_SYMBOLS" in str(e)


def test_missing_stock_api_key_warns():
    s = Settings()
    s.asset.enabled_assets = ["stock"]
    s.stock.alpaca_api_key = ""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = s.validate()
        warns = result or [str(x.message) for x in w]
        alpaca_warns = [msg for msg in warns if "ALPACA_API_KEY" in str(msg)]
        assert len(alpaca_warns) >= 1, f"Expected >=1 Alpaca-key warning, got {alpaca_warns}"


def test_negative_position_size_raises_error():
    s = Settings()
    s.trading.max_position_size = -0.1
    try:
        s.validate()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_MAX_POSITION_SIZE" in str(e)


if __name__ == "__main__":
    tests = [
        test_invalid_position_size_raises_error,
        test_missing_api_keys_produce_warnings,
        test_valid_config_passes_silently,
        test_stop_loss_out_of_range,
        test_take_profit_out_of_range,
        test_max_daily_drawdown_out_of_range,
        test_invalid_asset_type_raises_error,
        test_empty_supported_symbols_raises_error,
        test_missing_stock_api_key_warns,
        test_negative_position_size_raises_error,
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
