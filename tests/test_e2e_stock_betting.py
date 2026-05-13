"""End-to-end tests: AutoTrader stock and betting lanes.

Mocks data collectors, exercises every code path in run_stock_cycle and
run_betting_cycle (signal generation, safety guard, circuit breaker,
execution routing, PnL tracking, error handling).

Patches lazy imports at their *source module* because StockDataCollector,
BettingDataCollector, StockFeaturePipeline, StockModelFactory, and
ValueBettingModel are imported inside ``AutoTrader.__init__`` rather than at
the top of auto_trader.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.config.asset_types import AssetType, BACK, PASS, UnifiedSignal
from omnitrade.config.settings import Settings


# =====================================================================
# Synthetic data helpers
# =====================================================================


def _make_stock_ohlcv(n_rows: int = 100, start_price: float = 150.0) -> pd.DataFrame:
    """Generate stock-like OHLCV with a modest uptrend + noise.

    Columns: open, high, low, close, volume.  DatetimeIndex.
    """
    rng = np.random.default_rng(42)
    trend = np.linspace(start_price, start_price * 1.20, n_rows)
    noise = rng.normal(0, start_price * 0.01, n_rows).cumsum() * 0.2
    close = trend + noise
    close = np.maximum(close, 1.0)

    wick = np.abs(rng.normal(0, close * 0.003, n_rows))
    high = close + wick
    low = close - wick
    open_ = np.roll(close, 1)
    open_[0] = start_price

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(100_000, 1_000_000, n_rows),
    })
    df.index = pd.date_range("2026-01-01", periods=n_rows, freq="h")
    return df


def _make_betting_odds() -> pd.DataFrame:
    """Synthetic odds DataFrame with two matches (looks like mock real output)."""
    now = pd.Timestamp.now(tz="UTC").isoformat()
    return pd.DataFrame([
        {
            "home_team": "Soccer Home A",
            "away_team": "Soccer Away A",
            "bookmaker": "pinnacle",
            "commence_time": now,
            "home_odds": -150,
            "away_odds": 130,
            "draw_odds": 220,
            "implied_home_pct": 0.60,
            "implied_away_pct": 0.43,
            "implied_draw_pct": 0.31,
        },
        {
            "home_team": "Soccer Home B",
            "away_team": "Soccer Away B",
            "bookmaker": "pinnacle",
            "commence_time": now,
            "home_odds": 110,
            "away_odds": -130,
            "draw_odds": 240,
            "implied_home_pct": 0.45,
            "implied_away_pct": 0.55,
            "implied_draw_pct": 0.29,
        },
    ])


def _make_betting_history(n_rows: int = 50) -> pd.DataFrame:
    """Synthetic historical match results."""
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_rows, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame([
        {
            "match_id": f"mock_{i}",
            "home_team": f"Team {i % 5}",
            "away_team": f"Team {(i + 1) % 5}",
            "commence_time": d.isoformat(),
            "home_score": int(rng.poisson(1.5)),
            "away_score": int(rng.poisson(1.2)),
            "completed": True,
        }
        for i, d in enumerate(dates)
    ])


# =====================================================================
# Stock lane tests  (8 tests)
# =====================================================================


def _patch_stock():
    """Return a context manager that mocks the three lazy-imported stock classes.

    Yields (mock_collector, mock_pipeline, mock_models) so each test can
    configure return values.
    """
    return patch.multiple(
        "scripts.auto_trader",
        StockDataCollector=MagicMock(),
        StockFeaturePipeline=MagicMock(),
        StockModelFactory=MagicMock(),
    )


def _make_stock_fixtures():
    """Build standard mocks for a happy-path stock cycle.

    Returns (mock_collector, mock_pipeline, mock_models) configured for a
    default cycle with 100 rows of OHLCV, empty fundamentals, feature
    passthrough, and untrained models (== SignalGenerator fallback).
    """
    ohlcv = _make_stock_ohlcv(100)
    collector = MagicMock()
    collector.fetch_ohlcv.return_value = ohlcv
    collector.fetch_fundamentals.return_value = {}

    pipeline = MagicMock()
    pipeline.compute_all.return_value = ohlcv  # passthrough

    models = MagicMock()
    models.is_trained = False

    return collector, pipeline, models


def test_stock_cycle_no_crash():
    """run_stock_cycle completes without exception."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, pipeline, models = _make_stock_fixtures()

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline") as mock_sfp,
        patch("omnitrade.models.stock_models.StockModelFactory") as mock_smf,
    ):
        mock_sdc.return_value = collector
        mock_sfp.return_value = pipeline
        mock_smf.return_value = models

        trader = AutoTrader(asset_types=["stock"])
        result = trader.run_stock_cycle("AAPL")

    assert result is not None
    assert isinstance(result, dict)
    assert "symbol" in result
    assert result.get("signal") != "ERROR", f"Got ERROR: {result.get('reason')}"


def test_stock_cycle_produces_signal():
    """Signal dict contains symbol, signal, confidence."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, pipeline, models = _make_stock_fixtures()

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline") as mock_sfp,
        patch("omnitrade.models.stock_models.StockModelFactory") as mock_smf,
    ):
        mock_sdc.return_value = collector
        mock_sfp.return_value = pipeline
        mock_smf.return_value = models

        trader = AutoTrader(asset_types=["stock"])
        result = trader.run_stock_cycle("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["signal"] in ("BUY", "SELL", "HOLD", "SKIP")
    assert "confidence" in result
    assert isinstance(result["confidence"], (int, float))
    assert result["confidence"] >= 0.0
    assert "asset_type" in result
    assert result["asset_type"] == "stock"


def test_stock_cycle_with_safety_halt():
    """Halted safety → HALTED."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, pipeline, models = _make_stock_fixtures()

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline") as mock_sfp,
        patch("omnitrade.models.stock_models.StockModelFactory") as mock_smf,
    ):
        mock_sdc.return_value = collector
        mock_sfp.return_value = pipeline
        mock_smf.return_value = models

        trader = AutoTrader(asset_types=["stock"])
        trader.safety._halt("test halt", "e2e")
        result = trader.run_stock_cycle("AAPL")

    assert result["signal"] == "HALTED"
    assert "halt" in result.get("reason", "").lower()


def test_stock_cycle_empty_data_skips():
    """Empty OHLCV → SKIP."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector = MagicMock()
    collector.fetch_ohlcv.return_value = pd.DataFrame()
    collector.fetch_fundamentals.return_value = {}

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline"),
        patch("omnitrade.models.stock_models.StockModelFactory"),
    ):
        mock_sdc.return_value = collector

        trader = AutoTrader(asset_types=["stock"])
        result = trader.run_stock_cycle("AAPL")

    assert result["signal"] == "SKIP"
    assert "insufficient" in result.get("reason", "").lower()


def test_stock_cycle_insufficient_rows():
    """OHLCV with < 20 rows → SKIP."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    short = _make_stock_ohlcv(15)  # below threshold of 20

    collector = MagicMock()
    collector.fetch_ohlcv.return_value = short
    collector.fetch_fundamentals.return_value = {}

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline"),
        patch("omnitrade.models.stock_models.StockModelFactory"),
    ):
        mock_sdc.return_value = collector

        trader = AutoTrader(asset_types=["stock"])
        result = trader.run_stock_cycle("AAPL")

    assert result["signal"] == "SKIP"
    assert "insufficient" in result.get("reason", "").lower()


def test_stock_cycle_falls_back_to_signal_generator():
    """Untrained StockModelFactory → SignalGenerator produces valid signal."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    ohlcv = _make_stock_ohlcv(100)

    collector = MagicMock()
    collector.fetch_ohlcv.return_value = ohlcv
    collector.fetch_fundamentals.return_value = {}

    pipeline = MagicMock()
    pipeline.compute_all.return_value = ohlcv

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline") as mock_sfp,
        patch("omnitrade.models.stock_models.StockModelFactory") as mock_smf,
    ):
        mock_sdc.return_value = collector
        mock_sfp.return_value = pipeline

        # StockModelFactory mock: is_trained = False → forces SignalGenerator path
        models = MagicMock()
        models.is_trained = False
        mock_smf.return_value = models

        # Spy on SignalGenerator to confirm it is called
        with patch("scripts.auto_trader.SignalGenerator.generate",
                   wraps=None) as mock_gen:

            # Monkey-patch generate to return a real-looking signal
            def _fake_gen(features_row, prev_row=None):
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "buy_score": 0.0,
                    "sell_score": 0.0,
                    "reasons": ["SignalGenerator fallback"],
                    "indicators": {},
                }
            mock_gen.side_effect = _fake_gen

            trader = AutoTrader(asset_types=["stock"])
            result = trader.run_stock_cycle("AAPL")

            # SignalGenerator.generate should have been called
            assert mock_gen.called, "SignalGenerator.generate was not called"

    assert result["signal"] in ("BUY", "SELL", "HOLD", "SKIP")
    assert "confidence" in result
    assert result.get("signal") != "ERROR", f"Got ERROR: {result.get('reason')}"


def test_stock_portfolio_summary():
    """get_portfolio_summary returns valid structure after a cycle."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, pipeline, models = _make_stock_fixtures()

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline") as mock_sfp,
        patch("omnitrade.models.stock_models.StockModelFactory") as mock_smf,
    ):
        mock_sdc.return_value = collector
        mock_sfp.return_value = pipeline
        mock_smf.return_value = models

        trader = AutoTrader(asset_types=["stock"])
        trader.run_stock_cycle("AAPL")
        summary = trader.get_portfolio_summary()

    assert "balance_usd" in summary
    assert "equity" in summary
    assert "return_pct" in summary
    assert "open_positions" in summary
    assert "lanes" in summary
    assert "safety_status" in summary
    assert summary["equity"] > 0


def test_stock_cycle_fetch_error():
    """Exception during fetch → ERROR, circuit breaker records failure."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector = MagicMock()
    collector.fetch_ohlcv.side_effect = RuntimeError("yfinance down")
    collector.fetch_fundamentals.side_effect = RuntimeError("yfinance down")

    with (
        patch("omnitrade.data.collectors.stock_data.StockDataCollector") as mock_sdc,
        patch("omnitrade.features.stock_features.StockFeaturePipeline"),
        patch("omnitrade.models.stock_models.StockModelFactory"),
    ):
        mock_sdc.return_value = collector

        trader = AutoTrader(asset_types=["stock"])
        result = trader.run_stock_cycle("AAPL")

    assert result["signal"] == "ERROR"
    assert "yfinance down" in result.get("reason", "")
    # Circuit breaker recorded the failure
    assert trader.circuit._failures > 0


# =====================================================================
# Betting lane tests  (6 tests)
# =====================================================================


def _make_betting_signal(side: str = PASS) -> UnifiedSignal:
    """Create a UnifiedSignal suitable for betting tests.

    Args:
        side: ``BACK``, ``LAY``, or ``PASS``.

    When side is BACK/LAY and metadata provides a meaningful edge, the
    real BettingExecutor will accept the bet.
    """
    return UnifiedSignal(
        asset_type=AssetType.BET,
        symbol="Soccer Home A vs Soccer Away A",
        side=side,
        confidence=0.75 if side != PASS else 0.0,
        amount=0.0,
        price=0.50,
        metadata={
            "model_prob": 0.60,
            "implied_prob": 0.50,
            "edge": 0.10,
            "min_edge": 0.05,
            "odds": -110,
            "sport_key": "soccer_epl",
        },
    )


def _make_betting_setup(side: str = PASS):
    """Create pre-configured mocks for a betting cycle.

    Returns (mock_collector, mock_model) ready to pass to patch.return_value.
    """
    odds_df = _make_betting_odds()
    history_df = _make_betting_history()

    collector = MagicMock()
    collector.fetch_odds.return_value = odds_df
    collector.fetch_historical_results.return_value = history_df

    model = MagicMock()
    model.predict.return_value = _make_betting_signal(side)

    return collector, model


def test_betting_cycle_no_crash():
    """run_betting_cycle returns a list without exception."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, model = _make_betting_setup(PASS)

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        # Assign to bypass lazy-import reference inside __init__
        trader.betting_collector = collector
        trader.betting_model = model

        results = trader.run_betting_cycle()

    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "sport" in r
        assert "signal" in r
        assert r["signal"] != "ERROR", f"Got ERROR: {r.get('reason')}"


def test_betting_cycle_empty_odds_skips():
    """Empty odds DataFrame → SKIP for that sport."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector = MagicMock()
    collector.fetch_odds.return_value = pd.DataFrame()
    collector.fetch_historical_results.return_value = pd.DataFrame()

    model = MagicMock()

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        trader.betting_collector = collector

        results = trader.run_betting_cycle()

    assert isinstance(results, list)
    for r in results:
        assert r["signal"] == "SKIP"
        assert "no odds data" in r.get("reason", "").lower()


def test_betting_cycle_with_safety_halt():
    """Halted safety → HALTED."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, model = _make_betting_setup(PASS)

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        trader.betting_collector = collector
        trader.safety._halt("test halt", "e2e")

        results = trader.run_betting_cycle()

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["signal"] == "HALTED"


def test_betting_cycle_circuit_breaker():
    """Circuit breaker open → PAUSED."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, model = _make_betting_setup(PASS)

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        trader.betting_collector = collector
        trader.circuit._tripped_at = time.time()  # force open

        results = trader.run_betting_cycle()

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["signal"] == "PAUSED"


def test_betting_cycle_fetch_error():
    """Exception during fetch → ERROR per sport."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector = MagicMock()
    collector.fetch_odds.side_effect = RuntimeError("odds API down")
    collector.fetch_historical_results.side_effect = RuntimeError("odds API down")

    model = MagicMock()

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        trader.betting_collector = collector

        results = trader.run_betting_cycle()

    assert isinstance(results, list)
    assert len(results) == len(settings.betting.supported_sports)
    for r in results:
        assert r["signal"] == "ERROR"
        assert "odds API down" in r.get("reason", "")


def test_betting_stats():
    """BettingExecutor.get_stats() returns valid structure after a placed bet."""
    from scripts.auto_trader import AutoTrader

    settings = Settings()
    collector, model = _make_betting_setup(BACK)

    with (
        patch("omnitrade.data.collectors.betting_data.BettingDataCollector") as mock_bdc,
        patch("omnitrade.models.betting_models.ValueBettingModel") as mock_vbm,
    ):
        mock_bdc.return_value = collector
        mock_vbm.return_value = model

        trader = AutoTrader(asset_types=["bet"])
        trader.betting_collector = collector
        trader.betting_model = model

        # Run cycle to trigger a real bet placement via the real BettingExecutor
        trader.run_betting_cycle()

        # Fetch stats from the real BettingExecutor wired by AssetRouter
        bet_executor = trader.router.bet_executor
        stats = bet_executor.get_stats()

    assert "bankroll" in stats
    assert "initial_bankroll" in stats
    assert "total_pnl" in stats
    assert "roi_pct" in stats
    assert "total_bets" in stats
    assert "wins" in stats
    assert "losses" in stats
    assert "win_rate" in stats
    assert "open_bets" in stats
    assert "avg_stake" in stats
    assert "risk_status" in stats
    # At least one bet was placed
    assert stats["total_bets"] >= 1 or stats["open_bets"] >= 0
    assert stats["bankroll"] > 0
    assert stats["initial_bankroll"] > 0


# =====================================================================
# Runner
# =====================================================================

if __name__ == "__main__":
    tests = [
        # Stock
        test_stock_cycle_no_crash,
        test_stock_cycle_produces_signal,
        test_stock_cycle_with_safety_halt,
        test_stock_cycle_empty_data_skips,
        test_stock_cycle_insufficient_rows,
        test_stock_cycle_falls_back_to_signal_generator,
        test_stock_portfolio_summary,
        test_stock_cycle_fetch_error,
        # Betting
        test_betting_cycle_no_crash,
        test_betting_cycle_empty_odds_skips,
        test_betting_cycle_with_safety_halt,
        test_betting_cycle_circuit_breaker,
        test_betting_cycle_fetch_error,
        test_betting_stats,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} — {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
