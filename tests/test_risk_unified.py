"""Smoke tests for RiskManager + SafetyGuard unified halt delegation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.config.settings import Settings
from omnitrade.risk.risk_manager import RiskManager, PortfolioState
from omnitrade.risk.safety import SafetyGuard


def test_standalone_risk_manager_halt():
    """Without SafetyGuard, RiskManager uses own halt flag."""
    settings = Settings()
    rm = RiskManager(settings)
    assert not rm._trading_halted
    # Trigger daily drawdown halt
    portfolio = PortfolioState(balance=1000, equity=1000, daily_pnl=-200)
    halted = rm.check_daily_drawdown(-200, 1000)
    assert halted
    assert rm._trading_halted
    assert "20.00%" in rm._halt_reason or "drawdown" in rm._halt_reason.lower()


def test_standalone_risk_manager_reset():
    """reset_daily clears standalone halt."""
    settings = Settings()
    rm = RiskManager(settings)
    rm._trading_halted = True
    rm._halt_reason = "test"
    rm.reset_daily(1000)
    assert not rm._trading_halted


def test_unified_halt_delegation():
    """When SafetyGuard is wired, halt delegates to it."""
    settings = Settings()
    sg = SafetyGuard(settings)
    rm = RiskManager(settings, safety=sg)
    assert not rm._trading_halted
    # Halt via SafetyGuard
    sg.halt("test halt", "reason")
    assert sg.is_halted
    assert rm._trading_halted  # delegates to sg.is_halted
    assert "test halt" in rm._halt_reason


def test_unified_daily_drawdown_halt():
    """Daily drawdown halt goes through SafetyGuard when wired."""
    settings = Settings()
    sg = SafetyGuard(settings)
    rm = RiskManager(settings, safety=sg)
    assert not sg.is_halted
    portfolio = PortfolioState(balance=1000, equity=1000, daily_pnl=-250)
    halted = rm.check_daily_drawdown(-250, 1000)
    assert halted
    assert sg.is_halted  # SafetyGuard was halted
    assert rm._trading_halted  # risk manager reflects it


def test_unified_reset():
    """reset_daily clears SafetyGuard when wired."""
    settings = Settings()
    sg = SafetyGuard(settings)
    rm = RiskManager(settings, safety=sg)
    sg.halt("before reset", None)
    assert sg.is_halted
    rm.reset_daily(1000)
    assert not sg.is_halted
    assert not rm._trading_halted


def test_validate_trade_respects_halt():
    """validate_trade rejects when halted."""
    settings = Settings()
    rm = RiskManager(settings)
    rm._trading_halted = True
    rm._halt_reason = "halt test"
    result = rm.validate_trade(
        {"symbol": "BTC/USDT", "side": "buy", "amount": 100},
        PortfolioState(balance=1000, equity=1000),
    )
    assert not result.approved
    assert "halted" in result.reason.lower()


def test_validate_trade_respects_unified_halt():
    """validate_trade rejects when SafetyGuard (wired) is halted."""
    settings = Settings()
    sg = SafetyGuard(settings)
    rm = RiskManager(settings, safety=sg)
    sg.halt("unified halt", "test")
    result = rm.validate_trade(
        {"symbol": "BTC/USDT", "side": "buy", "amount": 100},
        PortfolioState(balance=1000, equity=1000),
    )
    assert not result.approved
    assert "halted" in result.reason.lower()


if __name__ == "__main__":
    tests = [
        test_standalone_risk_manager_halt,
        test_standalone_risk_manager_reset,
        test_unified_halt_delegation,
        test_unified_daily_drawdown_halt,
        test_unified_reset,
        test_validate_trade_respects_halt,
        test_validate_trade_respects_unified_halt,
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
