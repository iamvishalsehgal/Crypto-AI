"""
Unit tests for AssetRouter.

Tests route_signal dispatch, unified balance, status reporting, and lane
initialisation error handling. All executor classes are mocked to avoid
real exchange connections. Uses the standalone test pattern (no pytest).
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path for imports
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from unittest.mock import MagicMock, patch

from omnitrade.config.settings import Settings
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.execution.asset_router import AssetRouter


# ---------------------------------------------------------------------------
# Helpers — each test manages its own patches via ``with`` blocks.
# The patched import targets are the module paths used inside the
# _init_*_lane lazy-import bodies.
# ---------------------------------------------------------------------------


def _default_settings(enabled: list[str] | None = None) -> Settings:
    """Return a Settings object with the given enabled assets."""
    if enabled is None:
        enabled = ["crypto", "stock", "bet"]
    s = Settings()
    s.asset.enabled_assets = enabled
    return s


def _make_crypto_executor() -> MagicMock:
    """Return a mock crypto executor with a sensible default place_order."""
    ex = MagicMock()
    ex.place_order.return_value = {"order_id": "crypto-1", "status": "filled"}
    ex.get_balance.return_value = {
        "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
        "USD": {"free": 5000.0, "used": 0.0, "total": 5000.0},
    }
    ex.get_positions.return_value = []
    ex.close_all_positions.return_value = []
    return ex


def _make_stock_executor() -> MagicMock:
    """Return a mock stock executor."""
    ex = MagicMock()
    ex.place_order.return_value = {"order_id": "stock-1", "status": "filled"}
    ex.paper_mode = True
    ex.get_balance.return_value = {
        "USD": {"free": 10000.0, "used": 0.0, "total": 10000.0},
    }
    ex.get_positions.return_value = []
    ex.close_all_positions.return_value = []
    return ex


def _make_bet_executor() -> MagicMock:
    """Return a mock bet executor."""
    ex = MagicMock()
    ex.place_order.return_value = {"order_id": "bet-1", "status": "filled"}
    ex.get_bankroll.return_value = 1000.0
    ex.get_balance.return_value = {
        "USD": {"free": 1000.0, "used": 0.0, "total": 1000.0},
    }
    ex.get_positions.return_value = []
    ex.close_all_positions.return_value = []
    return ex


# ===================================================================
# route_signal — dispatch
# ===================================================================

def test_route_crypto_signal() -> None:
    """CRYPTO signal is dispatched to crypto executor (dict-converted)."""
    settings = _default_settings()
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        crypto_ex = _make_crypto_executor()
        MockTE.return_value = crypto_ex
        MockSE.return_value = _make_stock_executor()
        MockBE.return_value = _make_bet_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        signal = UnifiedSignal(
            asset_type=AssetType.CRYPTO,
            symbol="BTC/USDT",
            side="BUY",
            amount=0.01,
        )

        result = router.route_signal(signal)
        assert result["status"] == "filled", f"Got {result}"

        # Crypto routing converts signal to a plain dict
        crypto_ex.place_order.assert_called_once()
        call_args = crypto_ex.place_order.call_args[0][0]
        assert call_args["symbol"] == "BTC/USDT"
        assert call_args["side"] == "buy"
        assert call_args["amount"] == 0.01


def test_route_stock_signal() -> None:
    """STOCK signal is dispatched to stock executor directly."""
    settings = _default_settings()
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        MockTE.return_value = _make_crypto_executor()
        stock_ex = _make_stock_executor()
        MockSE.return_value = stock_ex
        MockBE.return_value = _make_bet_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        signal = UnifiedSignal(
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            side="BUY",
            amount=10,
        )

        result = router.route_signal(signal)
        assert result["status"] == "filled", f"Got {result}"

        # Stock routing passes the signal object directly
        stock_ex.place_order.assert_called_once_with(signal)


def test_route_bet_signal() -> None:
    """BET signal is dispatched to bet executor directly."""
    settings = _default_settings()
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        MockTE.return_value = _make_crypto_executor()
        MockSE.return_value = _make_stock_executor()
        bet_ex = _make_bet_executor()
        MockBE.return_value = bet_ex

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        signal = UnifiedSignal(
            asset_type=AssetType.BET,
            symbol="GAME1",
            side="BACK",
            amount=25,
        )

        result = router.route_signal(signal)
        assert result["status"] == "filled", f"Got {result}"

        # Bet routing passes the signal object directly
        bet_ex.place_order.assert_called_once_with(signal)


def test_route_unknown_asset_type() -> None:
    """Unknown/disabled asset type signal is dropped."""
    # Only enable crypto — stock is not initialised
    settings = _default_settings(enabled=["crypto"])
    mock_rm = MagicMock()
    with patch("omnitrade.execution.trade_executor.TradeExecutor") as MockTE:
        MockTE.return_value = _make_crypto_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        signal = UnifiedSignal(
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            side="BUY",
            amount=10,
        )

        result = router.route_signal(signal)
        assert result["status"] == "dropped", f"Got {result}"
        assert result["order_id"] is None


# ===================================================================
# get_status
# ===================================================================

def test_get_status_returns_enabled_lanes() -> None:
    """get_status returns the list of enabled lanes as strings."""
    settings = _default_settings(enabled=["crypto", "bet"])
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        MockTE.return_value = _make_crypto_executor()
        MockBE.return_value = _make_bet_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        status = router.get_status()
        assert "enabled_lanes" in status
        assert "crypto" in status["enabled_lanes"]
        assert "bet" in status["enabled_lanes"]
        assert "stock" not in status["enabled_lanes"]


def test_get_status_has_correct_booleans() -> None:
    """has_crypto/has_stock/has_bet reflect the active executors."""
    settings = _default_settings(enabled=["crypto"])
    mock_rm = MagicMock()
    with patch("omnitrade.execution.trade_executor.TradeExecutor") as MockTE:
        MockTE.return_value = _make_crypto_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        status = router.get_status()
        assert status["has_crypto"] is True
        assert status["has_stock"] is False
        assert status["has_bet"] is False


# ===================================================================
# get_unified_balance
# ===================================================================

def test_get_unified_balance_aggregates() -> None:
    """get_unified_balance returns total and per-lane balances."""
    settings = _default_settings()
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        MockTE.return_value = _make_crypto_executor()
        MockSE.return_value = _make_stock_executor()
        MockBE.return_value = _make_bet_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        bal = router.get_unified_balance()
        assert "total_usd_equivalent" in bal
        assert "by_lane" in bal

        # crypto: 5000 + stock: 10000 + bet: 1000 = 16000
        assert abs(bal["total_usd_equivalent"] - 16000.0) < 1.0, (
            f"Expected ~16000, got {bal['total_usd_equivalent']}"
        )

        # All three lanes present in by_lane
        assert "crypto" in bal["by_lane"]
        assert "stock" in bal["by_lane"]
        assert "bet" in bal["by_lane"]


def test_get_unified_balance_error_handling() -> None:
    """If a lane's get_balance fails, it's reported as error in by_lane."""
    settings = _default_settings()
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE, patch(
        "omnitrade.execution.betting_executor.BettingExecutor",
    ) as MockBE:
        crypto_ex = _make_crypto_executor()
        crypto_ex.get_balance.side_effect = RuntimeError("API timeout")
        MockTE.return_value = crypto_ex
        MockSE.return_value = _make_stock_executor()
        MockBE.return_value = _make_bet_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        bal = router.get_unified_balance()
        assert "error" in bal["by_lane"]["crypto"]
        assert "API timeout" in bal["by_lane"]["crypto"]["error"]
        # Other lanes should still contribute
        assert "USD" in bal["by_lane"]["stock"]
        assert "USD" in bal["by_lane"]["bet"]


# ===================================================================
# Lane initialisation error handling
# ===================================================================

def test_lane_init_error_sets_lane_status() -> None:
    """If an executor constructor fails, the lane status shows the error."""
    settings = _default_settings(enabled=["crypto", "stock"])
    mock_rm = MagicMock()
    with patch(
        "omnitrade.execution.trade_executor.TradeExecutor",
    ) as MockTE, patch(
        "omnitrade.execution.stock_executor.StockExecutor",
    ) as MockSE:
        # Make crypto lane init fail
        MockTE.side_effect = ValueError("Missing exchange API key")
        MockSE.return_value = _make_stock_executor()

        router = AssetRouter(settings=settings, risk_manager=mock_rm)

        status = router.get_status()
        assert status["lane_status"].get("crypto", "").startswith("error:")
        assert "Missing exchange API key" in status["lane_status"]["crypto"]
        # Stock should still be active
        assert status["lane_status"].get("stock") == "active"
        assert status["has_stock"] is True
        # Crypto executor not in the executors dict
        assert status["has_crypto"] is False


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    tests = [
        test_route_crypto_signal,
        test_route_stock_signal,
        test_route_bet_signal,
        test_route_unknown_asset_type,
        test_get_status_returns_enabled_lanes,
        test_get_status_has_correct_booleans,
        test_get_unified_balance_aggregates,
        test_get_unified_balance_error_handling,
        test_lane_init_error_sets_lane_status,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
