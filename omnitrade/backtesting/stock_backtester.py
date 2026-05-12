"""
Stock backtester — wraps BacktestEngine with yfinance data, stock features,
and model-driven signal generation. Handles dividend/split-adjusted prices.

Usage::

    backtester = StockBacktester(settings)
    result = backtester.run("AAPL", days=365)
    backtester.plot(result, save_path="reports/backtest_AAPL.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.backtesting.engine import BacktestEngine, BacktestResult
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class StockBacktester:
    """Backtest stock trading strategies using the existing BacktestEngine.

    Integrates StockDataCollector (yfinance), StockFeaturePipeline, and
    StockModelFactory into a coherent backtesting pipeline.

    Args:
        settings: Bot configuration.
        initial_balance: Starting portfolio value in USD.
    """

    def __init__(
        self,
        settings: Settings,
        initial_balance: float = 10_000.0,
    ) -> None:
        self._settings = settings
        self._initial_balance = initial_balance
        self._engine = BacktestEngine(settings, initial_balance)

        from omnitrade.data.collectors.stock_data import StockDataCollector
        from omnitrade.features.stock_features import StockFeaturePipeline
        from omnitrade.models.stock_models import StockModelFactory

        self._collector = StockDataCollector(settings)
        self._feature_pipeline = StockFeaturePipeline(settings)
        self._model_factory = StockModelFactory(settings)
        self._last_result: Optional[BacktestResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "1d",
        train_split: float = 0.6,
    ) -> BacktestResult:
        """Run a full backtest for a stock ticker.

        Args:
            ticker: Stock symbol (e.g. ``"AAPL"``).
            days: Lookback period in days.
            interval: OHLCV interval (``"1d"``, ``"1h"``).
            train_split: Fraction of data used for training (0-1).

        Returns:
            BacktestResult with metrics, trade history, and equity curve.
        """
        logger.info("Starting stock backtest for %s (%d days, %s)", ticker, days, interval)

        ohlcv = self._collector.fetch_ohlcv(ticker, interval=interval, period=f"{days}d")
        if ohlcv.empty or len(ohlcv) < 50:
            logger.warning("Insufficient data for %s (%d rows)", ticker, len(ohlcv))
            return self._engine._empty_result()

        fundamentals = self._collector.fetch_fundamentals(ticker)

        features = self._feature_pipeline.compute_all(ohlcv, fundamentals)
        if features.empty:
            features = ohlcv

        features = features.dropna()
        if len(features) < 50:
            logger.warning("Insufficient features for %s after dropna (%d rows)", ticker, len(features))
            return self._engine._empty_result()

        split_idx = int(len(features) * train_split)
        train_data = features.iloc[:split_idx]
        test_data = features.iloc[split_idx:]

        # Train on in-sample portion
        train_result = self._model_factory.train_all(train_data)
        logger.info("Training result for %s: %s", ticker, train_result)

        # Create strategy function wrapping model predictions
        strategy = self._make_strategy(test_data)

        self._last_result = self._engine.run(strategy, test_data)
        return self._last_result

    def run_walk_forward(
        self,
        ticker: str,
        days: int = 365,
        train_window: int = 120,
        test_window: int = 20,
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """Walk-forward backtest with rolling train/test windows.

        Returns:
            Dict with aggregate metrics across all windows.
        """
        ohlcv = self._collector.fetch_ohlcv(ticker, interval=interval, period=f"{days}d")
        if ohlcv.empty or len(ohlcv) < train_window + test_window:
            return {"status": "insufficient_data", "ticker": ticker}

        fundamentals = self._collector.fetch_fundamentals(ticker)
        features = self._feature_pipeline.compute_all(ohlcv, fundamentals)
        if features.empty:
            features = ohlcv
        features = features.dropna()

        total_steps = len(features) - train_window
        window_results: List[Dict] = []
        start = 0

        while start + train_window + test_window <= len(features):
            train = features.iloc[start:start + train_window]
            test = features.iloc[start + train_window:start + train_window + test_window]

            self._model_factory.train_all(train)
            strategy = self._make_strategy(test)
            result = self._engine.run(strategy, test)

            window_results.append({
                "window_start": start,
                "return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "trades": result.total_trades,
            })
            start += test_window

        if not window_results:
            return {"status": "no_windows", "ticker": ticker}

        returns = [w["return"] for w in window_results]
        sharpe = float(np.mean([w["sharpe"] for w in window_results]))

        return {
            "status": "complete",
            "ticker": ticker,
            "windows": len(window_results),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "win_rate": sum(1 for r in returns if r > 0) / len(returns),
            "avg_sharpe": round(sharpe, 3),
            "window_details": window_results,
        }

    def plot(self, result: Optional[BacktestResult] = None, save_path: Optional[str] = None) -> None:
        result = result or self._last_result
        if result is None:
            logger.warning("No backtest result to plot")
            return
        self._engine.plot_results(result, save_path=save_path)

    @property
    def last_result(self) -> Optional[BacktestResult]:
        return self._last_result

    # ------------------------------------------------------------------
    # Strategy factory
    # ------------------------------------------------------------------

    def _make_strategy(self, features: pd.DataFrame):
        """Build a strategy function from model predictions.

        The returned callable matches the signature expected by
        BacktestEngine.run(): (DataFrame, int) -> Optional[Dict].
        """
        # Pre-compute signals for all rows
        signals: Dict[int, Optional[Dict]] = {}

        predictions = self._model_factory.predict(features) if self._model_factory.is_trained else None

        for i in range(len(features)):
            row = features.iloc[i]
            price = float(row.get("close", 0))
            if price <= 0:
                signals[i] = None
                continue

            signal_name = predictions.get("signal", "HOLD") if predictions else "HOLD"
            confidence = predictions.get("confidence", 0)

            if signal_name in ("BUY", "SELL") and confidence >= 0.5:
                signals[i] = {
                    "side": signal_name.lower(),
                    "amount": self._initial_balance * 0.1,  # 10% position size
                }
            else:
                signals[i] = None

        def strategy(data: pd.DataFrame, idx: int) -> Optional[Dict]:
            return signals.get(idx)

        return strategy

    # ------------------------------------------------------------------
    # Dividend / split info
    # ------------------------------------------------------------------

    def get_corporate_actions(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """Fetch recent dividend and stock split history.

        yfinance auto-adjusts OHLCV prices (adjust=True by default), so
        backtests run on split/dividend-adjusted data. This method
        surfaces the raw corporate action log for transparency.
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            actions = stock.actions
            if actions is not None and not actions.empty:
                cutoff = pd.Timestamp.now(tz=actions.index.tz) - pd.Timedelta(days=days)
                return actions[actions.index >= cutoff]
            return pd.DataFrame()
        except Exception as exc:
            logger.warning("Failed to fetch corporate actions for %s: %s", ticker, exc)
            return pd.DataFrame()
