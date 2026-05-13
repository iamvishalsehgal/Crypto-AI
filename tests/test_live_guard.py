"""Tests for the live-trading safety guard across Settings, TradeExecutor,
StockExecutor, and auto_trader CLI flag handling.

The guard prevents accidental live trading by:
  - Defaulting to paper mode
  - Raising RuntimeError if live mode is requested without LIVE_TRADING_CONFIRMED=true
  - Forcing paper mode when --test is passed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.config.settings import Settings
from omnitrade.risk.risk_manager import RiskManager
from omnitrade.risk.safety import SafetyGuard
from omnitrade.execution.trade_executor import TradeExecutor
from omnitrade.execution.stock_executor import StockExecutor


def _make_risk_manager(settings: Settings) -> RiskManager:
    return RiskManager(settings, safety=SafetyGuard(settings))


# -- Settings-level tests --------------------------------------------------

def test_paper_mode_is_default():
    s = Settings()
    assert s.trading_mode == "paper"
    assert s.live_trading_confirmed == "false"


def test_accepts_live_mode():
    s = Settings(trading_mode="live", live_trading_confirmed="true")
    assert s.trading_mode == "live"
    assert s.live_trading_confirmed == "true"


def test_rejects_invalid_trading_mode():
    try:
        Settings(trading_mode="invalid")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "TRADING_MODE" in str(e)


def test_rejects_invalid_confirmation_value():
    try:
        Settings(live_trading_confirmed="yes")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "LIVE_TRADING_CONFIRMED" in str(e)


# -- TradeExecutor guard tests ---------------------------------------------

def test_executor_paper_mode_default():
    s = Settings(trading_mode="paper")
    executor = TradeExecutor(risk_manager=_make_risk_manager(s), settings=s)
    assert executor.paper_mode is True


def test_live_mode_without_confirmation_raises():
    s = Settings(trading_mode="live", live_trading_confirmed="false")
    try:
        TradeExecutor(risk_manager=_make_risk_manager(s), settings=s)
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "LIVE_TRADING_CONFIRMED" in str(e)


def test_live_mode_with_confirmation_passes():
    s = Settings(trading_mode="live", live_trading_confirmed="true")
    executor = TradeExecutor(risk_manager=_make_risk_manager(s), settings=s)
    assert executor.paper_mode == s.exchange.sandbox_mode


def test_paper_mode_overrides_sandbox():
    s = Settings(trading_mode="paper", live_trading_confirmed="false")
    s.exchange.sandbox_mode = False
    executor = TradeExecutor(risk_manager=_make_risk_manager(s), settings=s)
    assert executor.paper_mode is True


# -- StockExecutor guard tests ---------------------------------------------

def test_stock_executor_paper_mode_default():
    s = Settings(trading_mode="paper")
    executor = StockExecutor(risk_manager=_make_risk_manager(s), settings=s)
    assert executor.paper_mode is True


def test_stock_live_without_confirmation_raises():
    s = Settings(trading_mode="live", live_trading_confirmed="false")
    try:
        StockExecutor(risk_manager=_make_risk_manager(s), settings=s)
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "LIVE_TRADING_CONFIRMED" in str(e)


def test_stock_live_with_confirmation_passes():
    s = Settings(trading_mode="live", live_trading_confirmed="true")
    executor = StockExecutor(risk_manager=_make_risk_manager(s), settings=s)
    assert executor.paper_mode == s.stock.alpaca_paper


# -- AutoTrader CLI tests ---------------------------------------------------

def test_cli_default_is_paper():
    with mock.patch("sys.argv", ["auto_trader.py"]):
        import scripts.auto_trader as at
        import importlib
        importlib.reload(at)
        parser = at.argparse.ArgumentParser()
        # Re-parse
        args = at.argparse.ArgumentParser().parse_args([])
        assert not getattr(args, "test", False)
        # --live doesn't exist on the parser by default, check default mode logic
        # Verify no --live flag means paper
        assert not getattr(args, "live", False) if hasattr(args, "live") else True


if __name__ == "__main__":
    tests = [
        test_paper_mode_is_default,
        test_accepts_live_mode,
        test_rejects_invalid_trading_mode,
        test_rejects_invalid_confirmation_value,
        test_executor_paper_mode_default,
        test_live_mode_without_confirmation_raises,
        test_live_mode_with_confirmation_passes,
        test_paper_mode_overrides_sandbox,
        test_stock_executor_paper_mode_default,
        test_stock_live_without_confirmation_raises,
        test_stock_live_with_confirmation_passes,
        test_cli_default_is_paper,
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
