"""Smoke tests for PaperWallet — balance math, positions, trade history."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.execution.paper_wallet import PaperWallet


def test_buy_and_close_no_dust():
    w = PaperWallet(initial_balance=10000.0, base_currency="USDT", fee_rate=0.001, slippage=0.0005)
    w.simulate_order("BTC/USDT", "buy", 1000.0, 50000.0)
    bal = w.get_balance()
    assert bal["BTC"]["total"] > 0

    w.close_position("BTC/USDT", 51000.0)
    bal = w.get_balance()
    btc = bal.get("BTC", {}).get("total", 0.0)
    assert btc == 0.0, f"Dust BTC: {btc:.10f}"


def test_close_all_positions():
    w = PaperWallet(initial_balance=10000.0, base_currency="USD")
    w.simulate_order("AAPL", "buy", 2000.0, 150.0)
    w.simulate_order("GOOGL", "buy", 3000.0, 140.0)
    w.close_all_positions(price_provider={"AAPL": 155.0, "GOOGL": 145.0}.get)
    assert len(w.get_positions()) == 0


def test_unrealized_pnl():
    w = PaperWallet(initial_balance=5000.0, base_currency="USD")
    w.simulate_order("TSLA", "buy", 1000.0, 200.0)
    positions = w.get_positions(price_provider={"TSLA": 210.0}.get)
    assert positions[0]["unrealized_pnl"] > 0


def test_trade_history():
    w = PaperWallet(initial_balance=10000.0, base_currency="USDT")
    w.simulate_order("ETH/USDT", "buy", 500.0, 3000.0)
    w.close_position("ETH/USDT", 3100.0)
    hist = w.get_order_history()
    assert len(hist) == 2
    assert hist.iloc[0]["side"] == "buy"
    assert hist.iloc[1]["side"] == "sell"


def test_limit_order():
    w = PaperWallet(initial_balance=10000.0, base_currency="USDT")
    r = w.simulate_order("BTC/USDT", "buy", 500.0, 48000.0, order_type="limit")
    assert r["order_type"] == "limit"
    assert r["status"] == "filled"


def test_add_balance():
    w = PaperWallet(initial_balance=5000.0, base_currency="USD")
    w.add_balance("AAPL", 10.0)
    assert w.get_balance()["AAPL"]["free"] == 10.0
    assert w.total_equity == 5000.0


def test_multiple_orders_same_symbol():
    w = PaperWallet(initial_balance=10000.0, base_currency="USDT")
    w.simulate_order("BTC/USDT", "buy", 2000.0, 50000.0)
    w.simulate_order("BTC/USDT", "buy", 1000.0, 51000.0)
    positions = w.get_positions()
    assert len(positions) == 2  # two separate buys
    # close should liquidate both
    w.close_position("BTC/USDT", 52000.0)
    assert len(w.get_positions()) == 0


if __name__ == "__main__":
    tests = [
        test_buy_and_close_no_dust,
        test_close_all_positions,
        test_unrealized_pnl,
        test_trade_history,
        test_limit_order,
        test_add_balance,
        test_multiple_orders_same_symbol,
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
