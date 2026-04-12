"""
AI Crypto Trading Bot — Main Entry Point.

Orchestrates the full trading pipeline:
1. Collect market, on-chain, sentiment, and macro data
2. Engineer features and select the most predictive ones
3. Load (or train) AI models
4. Generate ensemble trading signals
5. Validate signals through risk management
6. Execute trades (paper or live)
7. Monitor performance and send alerts

Usage::

    python -m crypto_bot.main              # Run the bot
    python -m crypto_bot.main --mode paper # Explicit paper trading
    python -m crypto_bot.main --mode live  # Live trading (requires real API keys)
    python -m crypto_bot.main --backtest   # Run backtesting only
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from crypto_bot.config.settings import Settings, settings
from crypto_bot.utils.logger import get_logger

logger = get_logger("crypto_bot.main")


class TradingBot:
    """Main orchestrator for the AI crypto trading bot.

    Ties together data collection, feature engineering, model inference,
    ensemble voting, risk management, and trade execution into a single
    run-loop.
    """

    def __init__(self, mode: str = "paper") -> None:
        self._mode = mode
        self._running = False
        self._settings = settings

        # Override sandbox based on mode
        if mode == "live":
            self._settings.exchange.sandbox_mode = False
        else:
            self._settings.exchange.sandbox_mode = True

        logger.info("Initialising TradingBot in %s mode", mode.upper())

        # ---- Data collectors ----
        from crypto_bot.data.collectors.market_data import MarketDataCollector
        from crypto_bot.data.collectors.onchain_data import OnChainCollector
        from crypto_bot.data.collectors.sentiment_data import SentimentCollector
        from crypto_bot.data.collectors.macro_data import MacroDataCollector

        self.market_collector = MarketDataCollector(self._settings)
        self.onchain_collector = OnChainCollector(self._settings)
        self.sentiment_collector = SentimentCollector(self._settings)
        self.macro_collector = MacroDataCollector(self._settings)

        # ---- Feature engineering ----
        from crypto_bot.features.technical import TechnicalFeatures
        from crypto_bot.features.onchain_features import OnChainFeatures
        from crypto_bot.features.sentiment_features import SentimentFeatures
        from crypto_bot.features.macro_features import MacroFeatures
        from crypto_bot.features.feature_selector import FeatureSelector

        self.tech_features = TechnicalFeatures(self._settings)
        self.onchain_features = OnChainFeatures(self._settings)
        self.sentiment_features = SentimentFeatures(self._settings)
        self.macro_features = MacroFeatures(self._settings)
        self.feature_selector = FeatureSelector(self._settings)

        # ---- Ensemble ----
        from crypto_bot.ensemble.voting_system import EnsembleVoter

        self.ensemble = EnsembleVoter(self._settings)

        # ---- Risk & Execution ----
        from crypto_bot.risk.risk_manager import RiskManager
        from crypto_bot.execution.trade_executor import TradeExecutor

        self.risk_manager = RiskManager(self._settings)
        self.executor = TradeExecutor(
            risk_manager=self.risk_manager,
            settings=self._settings,
        )

        # ---- Monitoring ----
        from crypto_bot.monitoring.telegram_bot import TelegramNotifier
        from crypto_bot.monitoring.health_check import HealthChecker

        self.notifier = TelegramNotifier(self._settings)
        self.health_checker = HealthChecker(self._settings)

        # ---- Database ----
        from crypto_bot.data.storage.database import MongoDBStorage

        self.db = MongoDBStorage(self._settings)

        logger.info("All components initialised successfully")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Collect all data for a given symbol.

        Returns:
            Dict mapping data source names to DataFrames.
        """
        logger.info("Collecting data for %s", symbol)
        data: Dict[str, pd.DataFrame] = {}

        # Market OHLCV
        try:
            ohlcv = self.market_collector.fetch_ohlcv(symbol, timeframe="1h", limit=500)
            data["ohlcv"] = ohlcv
            logger.info("Fetched %d OHLCV candles", len(ohlcv))
        except Exception as exc:
            logger.error("Market data collection failed: %s", exc)
            data["ohlcv"] = pd.DataFrame()

        # Order book
        try:
            book = self.market_collector.fetch_order_book(symbol)
            data["order_book"] = book
        except Exception as exc:
            logger.warning("Order book fetch failed: %s", exc)

        return data

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build feature matrix from collected data."""
        logger.info("Building features")
        ohlcv = data.get("ohlcv", pd.DataFrame())

        if ohlcv.empty:
            logger.warning("No OHLCV data available, returning empty features")
            return pd.DataFrame()

        # Technical features
        features = self.tech_features.compute_all(ohlcv)

        # Drop NaN rows from indicator warm-up period
        features = features.dropna()
        logger.info("Feature matrix: %d rows x %d columns", *features.shape)
        return features

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, features: pd.DataFrame) -> Dict:
        """Run the ensemble to generate a trading signal.

        If no trained models are registered yet, falls back to a simple
        RSI-based heuristic for initial testing.
        """
        if features.empty:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "No features available"}

        # Check if ensemble has models registered
        if not self.ensemble.get_model_weights():
            return self._fallback_signal(features)

        try:
            vote = self.ensemble.vote({"features": features})
            return vote
        except Exception as exc:
            logger.error("Ensemble voting failed: %s, using fallback", exc)
            return self._fallback_signal(features)

    def _fallback_signal(self, features: pd.DataFrame) -> Dict:
        """Simple RSI-based signal for testing when models aren't trained."""
        latest = features.iloc[-1]
        rsi = latest.get("rsi", 50)

        if rsi < 30:
            signal = "BUY"
            confidence = min((30 - rsi) / 30, 1.0)
        elif rsi > 70:
            signal = "SELL"
            confidence = min((rsi - 70) / 30, 1.0)
        else:
            signal = "HOLD"
            confidence = 0.3

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": f"Fallback RSI={rsi:.1f}",
            "individual_predictions": {"rsi_heuristic": signal},
        }

    # ------------------------------------------------------------------
    # Main trading loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main trading loop.  Runs until interrupted."""
        self._running = True
        logger.info("Starting trading loop in %s mode", self._mode.upper())

        # Send startup notification
        await self.notifier.send_message(
            f"Trading bot started in {self._mode.upper()} mode\n"
            f"Symbols: {', '.join(self._settings.exchange.supported_symbols)}"
        )

        # Reset daily risk counters
        balance = self.executor.get_balance()
        usdt_balance = balance.get("USDT", {}).get("total", 10000.0)
        self.risk_manager.reset_daily(usdt_balance)

        cycle = 0
        while self._running:
            cycle += 1
            logger.info("=== Trading cycle %d ===", cycle)

            for symbol in self._settings.exchange.supported_symbols:
                try:
                    await self._process_symbol(symbol)
                except Exception as exc:
                    logger.error("Error processing %s: %s", symbol, exc)
                    await self.notifier.send_error_alert(
                        f"Error processing {symbol}: {exc}"
                    )

            # Check existing positions for stop-loss / take-profit
            await self._monitor_positions()

            # Health check
            if cycle % 10 == 0:
                health = self.health_checker.check_system_resources()
                logger.info("System health: %s", health)

            # Wait before next cycle (1 minute for hourly candles)
            logger.info("Cycle %d complete. Sleeping 60s...", cycle)
            await asyncio.sleep(60)

    async def _process_symbol(self, symbol: str) -> None:
        """Run one full cycle for a single symbol."""
        # 1. Collect data
        data = self.collect_data(symbol)

        # 2. Build features
        features = self.build_features(data)

        # 3. Generate signal
        signal_result = self.generate_signal(features)
        logger.info(
            "%s signal: %s (confidence: %.2f)",
            symbol,
            signal_result["signal"],
            signal_result.get("confidence", 0),
        )

        # 4. Execute if actionable
        if signal_result["signal"] in ("BUY", "SELL") and signal_result.get("confidence", 0) > 0.5:
            balance = self.executor.get_balance()
            usdt_total = balance.get("USDT", {}).get("total", 0)
            trade_amount = self.risk_manager.calculate_position_size(usdt_total)

            trade_signal = {
                "symbol": symbol,
                "side": signal_result["signal"].lower(),
                "amount": trade_amount,
            }

            result = self.executor.execute_trade(trade_signal)

            if result.get("status") == "filled":
                await self.notifier.send_trade_alert(result)

                # Store in database
                try:
                    self.db.connect()
                    self.db.store_trade(result)
                    self.db.disconnect()
                except Exception as exc:
                    logger.warning("Failed to store trade: %s", exc)

    async def _monitor_positions(self) -> None:
        """Check open positions for exit signals."""
        positions = self.executor.get_positions()
        if not positions:
            return

        # Get current prices
        prices = {}
        for pos in positions:
            symbol = pos.get("symbol", "")
            if symbol and symbol not in prices:
                try:
                    ticker = self.executor._exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker["last"]
                except Exception:
                    pass

        # Check for stop-loss / take-profit triggers
        actions = self.risk_manager.check_positions(positions, prices)
        for action in actions:
            logger.info(
                "Position exit triggered: %s %s @ %.2f (%s)",
                action["symbol"],
                action["action"],
                action["current_price"],
                action["action"],
            )
            result = self.executor.close_position(action["symbol"])
            if result.get("status") == "filled":
                await self.notifier.send_trade_alert({
                    **result,
                    "reason": action["action"],
                })

    def stop(self) -> None:
        """Signal the bot to stop after the current cycle."""
        self._running = False
        logger.info("Stop signal received, shutting down after current cycle")

    # ------------------------------------------------------------------
    # Backtesting mode
    # ------------------------------------------------------------------

    def run_backtest(self) -> None:
        """Run backtesting on historical data."""
        from crypto_bot.backtesting.engine import BacktestEngine
        from crypto_bot.backtesting.walk_forward import WalkForwardValidator
        from crypto_bot.backtesting.overfitting_detector import OverfittingDetector

        logger.info("Starting backtesting mode")
        engine = BacktestEngine(self._settings)
        validator = WalkForwardValidator(self._settings)
        detector = OverfittingDetector(self._settings)

        for symbol in self._settings.exchange.supported_symbols:
            logger.info("Backtesting %s", symbol)

            # Fetch historical data
            data = self.collect_data(symbol)
            features = self.build_features(data)

            if features.empty or len(features) < 100:
                logger.warning("Insufficient data for %s, skipping", symbol)
                continue

            # Simple RSI strategy for backtesting demo
            def rsi_strategy(row):
                rsi = row.get("rsi_14", 50)
                if rsi < 30:
                    return 1   # BUY
                elif rsi > 70:
                    return -1  # SELL
                return 0       # HOLD

            # Run backtest
            result = engine.run(rsi_strategy, features)
            logger.info(
                "%s backtest: return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%, trades=%d",
                symbol,
                result.total_return * 100,
                result.sharpe_ratio,
                result.max_drawdown * 100,
                result.total_trades,
            )

            # Plot results
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            engine.plot_results(result, save_path=str(output_dir / f"backtest_{symbol.replace('/', '_')}.png"))

        logger.info("Backtesting complete. Reports saved to reports/")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtesting instead of live trading",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override trading symbols (e.g. BTC/USDT ETH/USDT)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.symbols:
        settings.exchange.supported_symbols = args.symbols

    bot = TradingBot(mode=args.mode)

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        bot.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.backtest:
        bot.run_backtest()
    else:
        asyncio.run(bot.run())


if __name__ == "__main__":
    main()
