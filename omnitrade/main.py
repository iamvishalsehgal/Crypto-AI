"""
OmniTrade AI — Multi-Asset Autonomous Trading System.

Orchestrates the full trading pipeline across crypto, stocks, and betting:
1. Collect market, on-chain, sentiment, macro, and stock data
2. Engineer features and select the most predictive ones
3. Load (or train) AI models per asset class
4. Generate ensemble trading signals via AssetRouter
5. Validate signals through risk management
6. Execute trades (paper or live) across all lanes
7. Monitor performance and send alerts

Usage::

    python -m omnitrade.main                          # Run all enabled assets
    python -m omnitrade.main --mode paper             # Explicit paper trading
    python -m omnitrade.main --mode live              # Live trading
    python -m omnitrade.main --asset-types crypto      # Crypto only
    python -m omnitrade.main --asset-types crypto stock  # Crypto + stocks
    python -m omnitrade.main --backtest               # Run backtesting only
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

from omnitrade.config.settings import Settings, settings
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.utils.logger import get_logger

logger = get_logger("omnitrade.main")


class TradingBot:
    """Main orchestrator for the AI crypto trading bot.

    Ties together data collection, feature engineering, model inference,
    ensemble voting, risk management, and trade execution into a single
    run-loop.
    """

    def __init__(self, mode: str = "paper", asset_types: Optional[list] = None) -> None:
        self._mode = mode
        self._running = False
        self._settings = settings.model_copy(deep=True)

        # Override sandbox based on mode (on our copy, not the global)
        if mode == "live":
            self._settings.exchange.sandbox_mode = False
        else:
            self._settings.exchange.sandbox_mode = True

        # Override enabled assets if specified
        if asset_types:
            self._settings.asset.enabled_assets = list(asset_types)
            logger.info("Asset types overridden: %s", asset_types)

        self._enabled_assets = self._settings.asset.enabled_assets
        logger.info("Initialising OmniTrade AI in %s mode", mode.upper())

        # ---- Data collectors ----
        from omnitrade.data.collectors.market_data import MarketDataCollector
        from omnitrade.data.collectors.onchain_data import OnChainCollector
        from omnitrade.data.collectors.sentiment_data import SentimentCollector
        from omnitrade.data.collectors.macro_data import MacroDataCollector

        self.market_collector = MarketDataCollector(self._settings)
        self.onchain_collector = OnChainCollector(self._settings)
        self.sentiment_collector = SentimentCollector(self._settings)
        self.macro_collector = MacroDataCollector(self._settings)

        # Stock data collector (if stock lane enabled)
        self.stock_collector = None
        if "stock" in self._enabled_assets:
            from omnitrade.data.collectors.stock_data import StockDataCollector
            self.stock_collector = StockDataCollector(self._settings)

        # ---- Feature engineering ----
        from omnitrade.features.technical import TechnicalFeatures
        from omnitrade.features.onchain_features import OnChainFeatures
        from omnitrade.features.sentiment_features import SentimentFeatures
        from omnitrade.features.macro_features import MacroFeatures
        from omnitrade.features.feature_selector import FeatureSelector

        self.tech_features = TechnicalFeatures(self._settings)
        self.onchain_features = OnChainFeatures(self._settings)
        self.sentiment_features = SentimentFeatures(self._settings)
        self.macro_features = MacroFeatures(self._settings)
        self.feature_selector = FeatureSelector(self._settings)

        # Stock feature pipeline (if stock lane enabled)
        self.stock_feature_pipeline = None
        if "stock" in self._enabled_assets:
            from omnitrade.features.stock_features import StockFeaturePipeline
            self.stock_feature_pipeline = StockFeaturePipeline(self._settings)

        # Betting components (if bet lane enabled)
        self.betting_collector = None
        self.betting_features = None
        self.betting_model = None
        if "bet" in self._enabled_assets:
            from omnitrade.data.collectors.betting_data import BettingDataCollector
            from omnitrade.features.betting_features import BettingFeatures
            from omnitrade.models.betting_models import ValueBettingModel
            self.betting_collector = BettingDataCollector(self._settings)
            self.betting_features = BettingFeatures(self._settings)
            self.betting_model = ValueBettingModel(self._settings)

        # ---- Ensemble ----
        from omnitrade.ensemble.voting_system import EnsembleVoter

        self.ensemble = EnsembleVoter(self._settings)

        # ---- Risk & Execution ----
        from omnitrade.risk.risk_manager import RiskManager

        self.risk_manager = RiskManager(self._settings)

        # AssetRouter (central dispatcher for all lanes)
        from omnitrade.execution.asset_router import AssetRouter

        self.router = AssetRouter(
            settings=self._settings,
            risk_manager=self.risk_manager,
            mode=mode,
        )

        # Direct executor access for backward compatibility
        self.executor = self.router.crypto_executor
        self.stock_executor = self.router.stock_executor
        self.bet_executor = self.router.bet_executor

        # ---- Stock models ----
        self.stock_models = None
        if "stock" in self._enabled_assets and self.stock_executor:
            from omnitrade.models.stock_models import StockModelFactory
            self.stock_models = StockModelFactory(self._settings)

        # ---- Monitoring ----
        from omnitrade.monitoring.telegram_bot import TelegramNotifier
        from omnitrade.monitoring.health_check import HealthChecker

        self.notifier = TelegramNotifier(self._settings)
        self.health_checker = HealthChecker(self._settings)

        # ---- Database ----
        from omnitrade.data.storage.database import MongoDBStorage

        self.db = MongoDBStorage(self._settings)
        self.db.connect()

        logger.info("All components initialised — lanes: %s", self.router.get_status())

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
        logger.info("Starting OmniTrade AI loop in %s mode", self._mode.upper())

        lane_info = self.router.get_status()
        await self.notifier.send_message(
            f"OmniTrade AI started in {self._mode.upper()} mode\n"
            f"Lanes: {', '.join(lane_info['enabled_lanes'])}\n"
            f"Crypto symbols: {', '.join(self._settings.exchange.supported_symbols)}\n"
            f"Stock tickers: {', '.join(self._settings.stock.supported_tickers)}"
        )

        # Reset daily risk counters
        unified_balance = self.router.get_unified_balance()
        total_equity = unified_balance.get("total_usd_equivalent", 10_000)
        self.risk_manager.reset_daily(total_equity)

        cycle = 0
        while self._running:
            cycle += 1
            logger.info("=== Trading cycle %d ===", cycle)

            # Process crypto symbols
            if self.router.get_status()["has_crypto"]:
                for symbol in self._settings.exchange.supported_symbols:
                    try:
                        await self._process_symbol(symbol)
                    except Exception as exc:
                        logger.error("Error processing crypto %s: %s", symbol, exc)
                        await self.notifier.send_error_alert(
                            f"Error processing crypto {symbol}: {exc}"
                        )

            # Process stock tickers
            if self.router.get_status()["has_stock"] and self.stock_collector:
                for ticker in self._settings.stock.supported_tickers:
                    try:
                        await self._process_stock(ticker)
                    except Exception as exc:
                        logger.error("Error processing stock %s: %s", ticker, exc)
                        await self.notifier.send_error_alert(
                            f"Error processing stock {ticker}: {exc}"
                        )

            # Process betting
            if self.router.get_status()["has_bet"] and self.betting_collector:
                try:
                    await self._process_betting()
                except Exception as exc:
                    logger.error("Error processing betting: %s", exc)

            # Check existing positions for stop-loss / take-profit
            await self._monitor_positions()

            # Health check every 10 cycles
            if cycle % 10 == 0:
                health = self.health_checker.check_system_resources()
                logger.info("System health: %s", health)

            # Unified balance report
            unified = self.router.get_unified_balance()
            logger.info(
                "Portfolio: $%.2f across %d lanes",
                unified["total_usd_equivalent"],
                len(unified["by_lane"]),
            )

            # Wait before next cycle
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
                    self.db.store_trade(result)
                except Exception as exc:
                    logger.warning("Failed to store trade: %s", exc)

    async def _process_stock(self, ticker: str) -> None:
        """Run one full cycle for a stock ticker."""
        if self.stock_collector is None:
            return

        # 1. Collect stock data
        ohlcv = self.stock_collector.fetch_ohlcv(ticker, interval="1h")
        fundamentals = self.stock_collector.fetch_fundamentals(ticker)

        if ohlcv.empty:
            logger.warning("No OHLCV data for %s, skipping", ticker)
            return

        # 2. Build stock features
        features = pd.DataFrame()
        if self.stock_feature_pipeline:
            features = self.stock_feature_pipeline.compute_all(ohlcv, fundamentals)

        if features.empty:
            logger.warning("No features for %s, using price-only signal", ticker)
            features = ohlcv

        # 3. Generate signal
        if self.stock_models and self.stock_models.is_trained:
            signal_result = self.stock_models.predict(features)
        else:
            signal_result = self._fallback_signal(features)

        logger.info(
            "%s signal: %s (confidence: %.2f)",
            ticker,
            signal_result["signal"],
            signal_result.get("confidence", 0),
        )

        # 4. Execute if actionable
        if signal_result["signal"] in ("BUY", "SELL") and signal_result.get("confidence", 0) > 0.5:
            last_close = float(ohlcv["close"].iloc[-1])
            balance = self.stock_executor.get_balance() if self.stock_executor else {}
            usd_total = balance.get("USD", {}).get("total", 10_000)
            trade_amount = self.risk_manager.calculate_position_size(usd_total)

            signal = UnifiedSignal(
                asset_type=AssetType.STOCK,
                symbol=ticker,
                side=signal_result["signal"],
                confidence=signal_result.get("confidence", 0),
                amount=trade_amount,
                price=last_close,
                metadata={"fundamentals": fundamentals},
            )

            result = self.router.route_signal(signal)

            if result.get("status") == "filled":
                await self.notifier.send_trade_alert(result)

                try:
                    self.db.store_trade(result)
                except Exception as exc:
                    logger.warning("Failed to store stock trade: %s", exc)

    async def _process_betting(self) -> None:
        """Run one full cycle for sports betting."""
        if self.betting_collector is None or self.betting_model is None:
            return

        for sport in self._settings.betting.supported_sports:
            try:
                odds_df = self.betting_collector.fetch_odds(sport)
                if odds_df.empty:
                    logger.info("No odds data for %s, skipping", sport)
                    continue

                historical = self.betting_collector.fetch_historical_results(sport)
                features = odds_df
                if self.betting_features:
                    features = self.betting_features.compute_all(odds_df, historical)

                if features.empty:
                    logger.warning("No features for %s, skipping", sport)
                    continue

                signal = self.betting_model.predict(features)
                logger.info(
                    "BET signal: %s %s (conf=%.2f, edge=%.3f)",
                    signal.side, signal.symbol, signal.confidence,
                    signal.metadata.get("edge", 0),
                )

                if signal.side not in ("BACK", "LAY"):
                    continue

                result = self.router.route_signal(signal)

                if result.get("status") == "filled":
                    await self.notifier.send_trade_alert(result)
                    try:
                        self.db.connect()
                        self.db.store_trade(result)
                        self.db.disconnect()
                    except Exception as exc:
                        logger.warning("Failed to store bet: %s", exc)

            except Exception as exc:
                logger.error("Error processing betting sport %s: %s", sport, exc)

        # Log betting stats
        if self.bet_executor:
            stats = self.bet_executor.get_stats()
            logger.info(
                "Betting bankroll: $%.2f | Bets: %d | Win: %.1f%% | PnL: $%.2f | ROI: %.1f%%",
                stats["bankroll"], stats["total_bets"],
                stats["win_rate"], stats["total_pnl"], stats["roi_pct"],
            )

    async def _monitor_positions(self) -> None:
        """Check open positions for exit signals across all lanes."""
        positions = self.router.get_all_positions()
        if not positions:
            return

        # Get current prices for crypto positions
        prices = {}
        if self.executor:
            for pos in positions:
                if pos.get("asset_type") != "crypto":
                    continue
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
            asset_type_str = action.get("asset_type", "crypto")
            if asset_type_str == "stock" and self.stock_executor:
                result = self.stock_executor.close_position(action["symbol"])
            elif self.executor:
                result = self.executor.close_position(action["symbol"])
            else:
                continue

            if result.get("status") == "filled":
                await self.notifier.send_trade_alert({
                    **result,
                    "reason": action["action"],
                })

    def stop(self) -> None:
        """Signal the bot to stop after the current cycle."""
        self._running = False
        if self.db is not None:
            self.db.disconnect()
        logger.info("Stop signal received, shutting down after current cycle")

    # ------------------------------------------------------------------
    # Backtesting mode
    # ------------------------------------------------------------------

    def run_backtest(self) -> None:
        """Run backtesting on historical data across all enabled asset lanes."""
        from omnitrade.backtesting.engine import BacktestEngine
        from omnitrade.backtesting.walk_forward import WalkForwardValidator
        from omnitrade.backtesting.overfitting_detector import OverfittingDetector

        logger.info("Starting backtesting mode — lanes: %s", self._enabled_assets)
        engine = BacktestEngine(self._settings)
        validator = WalkForwardValidator(self._settings)
        detector = OverfittingDetector(self._settings)
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        # ----- Crypto backtest -----
        if self.router.get_status()["has_crypto"]:
            for symbol in self._settings.exchange.supported_symbols:
                logger.info("Backtesting CRYPTO %s", symbol)
                data = self.collect_data(symbol)
                features = self.build_features(data)

                if features.empty or len(features) < 100:
                    logger.warning("Insufficient data for %s, skipping", symbol)
                    continue

                def rsi_strategy(row):
                    rsi = row.get("rsi_14", 50)
                    if rsi < 30:
                        return 1
                    elif rsi > 70:
                        return -1
                    return 0

                result = engine.run(rsi_strategy, features)
                logger.info(
                    "%s backtest: return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%, trades=%d",
                    symbol, result.total_return * 100, result.sharpe_ratio,
                    result.max_drawdown * 100, result.total_trades,
                )
                engine.plot_results(result, save_path=str(output_dir / f"backtest_{symbol.replace('/', '_')}.png"))

        # ----- Stock backtest -----
        if self.router.get_status()["has_stock"]:
            from omnitrade.backtesting.stock_backtester import StockBacktester
            stock_bt = StockBacktester(self._settings)
            for ticker in self._settings.stock.supported_tickers:
                logger.info("Backtesting STOCK %s", ticker)
                try:
                    result = stock_bt.run(ticker, days=365)
                    logger.info(
                        "%s backtest: return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%, trades=%d",
                        ticker, result.total_return * 100, result.sharpe_ratio,
                        result.max_drawdown * 100, result.total_trades,
                    )
                    stock_bt.plot(result, save_path=str(output_dir / f"backtest_stock_{ticker}.png"))
                except Exception as exc:
                    logger.error("Stock backtest failed for %s: %s", ticker, exc)

        # ----- Betting backtest -----
        if self.router.get_status()["has_bet"]:
            from omnitrade.backtesting.betting_backtester import BettingBacktester
            bet_bt = BettingBacktester(self._settings)
            for sport in self._settings.betting.supported_sports:
                logger.info("Backtesting BETTING %s", sport)
                try:
                    result = bet_bt.run(sport, days=365)
                    logger.info(
                        "%s backtest: bets=%d, win=%.1f%%, ROI=%.1f%%, PnL=$%.2f, CLV+=%.1f%%",
                        sport, result.total_bets, result.win_rate * 100,
                        result.roi_pct, result.total_pnl, result.clv_positive_pct * 100,
                    )
                    bet_bt.plot(result, save_path=str(output_dir / f"backtest_betting_{sport}.png"))
                except Exception as exc:
                    logger.error("Betting backtest failed for %s: %s", sport, exc)

        logger.info("Backtesting complete. Reports saved to reports/")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniTrade AI — Multi-Asset Autonomous Trading System",
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
        help="Override crypto trading symbols (e.g. BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--asset-types",
        nargs="+",
        choices=["crypto", "stock", "bet"],
        help="Asset classes to trade (default: all enabled in config)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Override stock tickers (e.g. AAPL TSLA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.symbols:
        settings.exchange.supported_symbols = args.symbols
    if args.tickers:
        settings.stock.supported_tickers = args.tickers

    bot = TradingBot(mode=args.mode, asset_types=args.asset_types)

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
