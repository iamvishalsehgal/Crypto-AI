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

import os

# Prevent XGBoost OpenMP → PyTorch threading conflict on Apple Silicon
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings, settings
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.data.collectors.market_data import MarketDataCollector
from omnitrade.data.storage.database import MongoDBStorage
from omnitrade.ensemble.voting_system import EnsembleVoter
from omnitrade.ensemble.janus_blender import JanusBlender
from omnitrade.execution.asset_router import AssetRouter
from omnitrade.features.technical import TechnicalFeatures
from omnitrade.monitoring.telegram_bot import TelegramNotifier
from omnitrade.monitoring.health_check import HealthChecker
from omnitrade.risk.risk_manager import RiskManager
from omnitrade.utils.logger import get_logger
from omnitrade.utils.circuit_breaker import CircuitBreaker
from omnitrade.utils.pnl_tracker import PnLTracker
from omnitrade.learning import AutoRetrainer
from omnitrade.risk.safety import SafetyGuard

# Multi-agent pipeline
from omnitrade.agents.bus import EventBus, PortfolioEvent
from omnitrade.agents.data_collector import DataCollectorAgent
from omnitrade.agents.feature_builder import FeatureBuilderAgent
from omnitrade.agents.signal_generator import SignalGeneratorAgent
from omnitrade.agents.risk_manager import RiskManagerAgent
from omnitrade.agents.execution import ExecutionAgent
from omnitrade.agents.monitor import MonitorAgent

# Late imports for backtesting (only used in run_backtest, loaded here for locality)
from omnitrade.backtesting.engine import BacktestEngine
from omnitrade.backtesting.walk_forward import WalkForwardValidator
from omnitrade.backtesting.overfitting_detector import OverfittingDetector

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

        # Live mode disables sandbox for real orders. Paper mode respects
        # the env-var / config setting so public API works without keys.
        if mode == "live":
            self._settings.exchange.sandbox_mode = False

        # Override enabled assets if specified
        if asset_types:
            self._settings.asset.enabled_assets = list(asset_types)
            logger.info("Asset types overridden: %s", asset_types)

        self._enabled_assets = self._settings.asset.enabled_assets
        logger.info("Initialising OmniTrade AI in %s mode", mode.upper())

        # ---- Data collectors ----
        self.market_collector = MarketDataCollector(self._settings)

        # Stock data collector (if stock lane enabled)
        self.stock_collector = None
        if "stock" in self._enabled_assets:
            from omnitrade.data.collectors.stock_data import StockDataCollector
            self.stock_collector = StockDataCollector(self._settings)

        # ---- Feature engineering ----
        self.tech_features = TechnicalFeatures(self._settings)

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
        self.ensemble = EnsembleVoter(self._settings)

        # JANUS meta-blender: softmax weight calibration with constraints,
        # disagreement penalty, and regime detection
        self.janus = JanusBlender(
            model_types={"xgboost": "tree", "lightgbm": "tree", "lstm": "neural"},
        )

        # ---- Risk & Execution ----
        self.risk_manager = RiskManager(self._settings)
        self.safety = SafetyGuard(self._settings)
        self.pnl_tracker = PnLTracker()
        self.circuit_breaker = CircuitBreaker(
            threshold=self._settings.safety.kill_on_consecutive_losses,
            cooldown=300,
        )

        # AssetRouter (central dispatcher for all lanes)
        self.router = AssetRouter(
            settings=self._settings,
            risk_manager=self.risk_manager,
            mode=mode,
            safety=self.safety,
        )

        # ---- Stock models ----
        self.stock_models = None
        if "stock" in self._enabled_assets and self.router.stock_executor:
            from omnitrade.models.stock_models import StockModelFactory
            self.stock_models = StockModelFactory(self._settings)

        # ---- Load trained models ----
        self._crypto_xgb = None
        self._crypto_lgb = None
        self._load_saved_models()

        # ---- Monitoring ----
        self.notifier = TelegramNotifier(self._settings)
        self.health_checker = HealthChecker(self._settings, notifier=self.notifier)

        # ---- Database ----
        self.db = MongoDBStorage(self._settings)
        self.db.connect()

        # ---- Auto-learning ----
        self.retrainer = AutoRetrainer(self._settings)
        self.retrainer.load_retrain_state()

        logger.info("All components initialised — lanes: %s", self.router.get_status())

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_saved_models(self) -> None:
        """Load trained models from disk for all enabled lanes."""
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models" / "saved"

        # ── Crypto models ──────────────────────────────────────────
        if "crypto" in self._enabled_assets:
            xgb_path = models_dir / "xgboost_model"
            if xgb_path.exists():
                try:
                    from omnitrade.models.xgboost_model import XGBoostTrader
                    self._crypto_xgb = XGBoostTrader(self._settings)
                    self._crypto_xgb.load_model(str(xgb_path))
                    self.ensemble.register_model("xgboost", self._crypto_xgb)
                    logger.info("Loaded crypto XGBoost model from %s", xgb_path)
                except Exception as exc:
                    logger.warning("Failed to load crypto XGBoost: %s", exc)

            lgb_path = models_dir / "lightgbm_model"
            if lgb_path.exists():
                try:
                    from omnitrade.models.lightgbm_model import LightGBMTrader
                    self._crypto_lgb = LightGBMTrader(self._settings)
                    self._crypto_lgb.load_model(str(lgb_path))
                    self.ensemble.register_model("lightgbm", self._crypto_lgb)
                    logger.info("Loaded crypto LightGBM model from %s", lgb_path)
                except Exception as exc:
                    logger.warning("Failed to load crypto LightGBM: %s", exc)

            # Load ensemble weights from config
            weights_path = project_root / "config" / "ensemble_weights.json"
            if weights_path.exists() and self._crypto_xgb is not None:
                try:
                    import json
                    with open(weights_path) as f:
                        config = json.load(f)
                    for name, weight in config.get("weights", {}).items():
                        if name in self.ensemble.get_model_weights():
                            self.ensemble.update_weight(name, weight)
                    logger.info("Loaded ensemble weights from config")
                except Exception as exc:
                    logger.warning("Failed to load ensemble weights: %s", exc)

        # ── Stock models ──────────────────────────────────────────
        # Create & load LSTM BEFORE XGBoost — XGBoost OpenMP conflicts
        # with PyTorch on Apple Silicon if XGBoost is imported first
        if self.stock_models is not None:
            try:
                # Phase 1: LSTM (PyTorch) — must be created & loaded first
                self.stock_models.create_lstm()
                stock_lstm_path = models_dir / "stock_lstm_model"
                if stock_lstm_path.exists():
                    try:
                        self.stock_models._models["lstm"].load_model(str(stock_lstm_path))
                        logger.info("Loaded stock LSTM model")
                    except Exception as exc:
                        logger.warning("Failed to load stock LSTM model: %s", exc)

                # Phase 2: XGBoost (OpenMP) — only after PyTorch is initialised
                self.stock_models.create_xgboost()
                stock_xgb_path = models_dir / "stock_xgboost_model"
                if stock_xgb_path.exists():
                    try:
                        self.stock_models._models["xgboost"].load_model(str(stock_xgb_path))
                        logger.info("Loaded stock XGBoost model")
                    except Exception as exc:
                        logger.warning("Failed to load stock XGBoost model: %s", exc)

                self.stock_models._trained = True
                logger.info("Stock models loaded from disk")
            except Exception as exc:
                logger.warning("Failed to load stock models: %s", exc)

        # ── Betting model ─────────────────────────────────────────
        if self.betting_model is not None:
            bet_path = models_dir / "value_betting_model"
            if bet_path.exists():
                try:
                    self.betting_model.load_model(str(bet_path))
                    logger.info("Loaded betting model")
                except Exception as exc:
                    logger.warning("Failed to load betting model: %s", exc)

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

    @staticmethod
    def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric columns for model input (exclude timestamp, etc.)."""
        exclude = {"signal", "timestamp"}
        cols = [
            c for c in features.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
        ]
        return features[cols]

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
            numeric_features = self._numeric_features(features)
            vote = self.ensemble.vote({"features": numeric_features})
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
        """Launch multi-agent trading pipeline.

        Spawns 6 concurrent agents that communicate via an event bus:
        DataCollector → FeatureBuilder → SignalGenerator → RiskManager → Execution
        Plus MonitorAgent for position watching.
        """
        self._running = True
        logger.info("Starting OmniTrade AI multi-agent pipeline in %s mode", self._mode.upper())

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

        # ── Event bus ────────────────────────────────────────────────
        bus = EventBus()

        # ── Agents ───────────────────────────────────────────────────
        agents: list = []

        # DataCollectorAgent (crypto + stock + betting)
        data_agent = DataCollectorAgent(
            bus=bus,
            market_collector=self.market_collector,
            stock_collector=self.stock_collector,
            betting_collector=self.betting_collector,
            settings=self._settings,
        )
        agents.append(data_agent)

        # FeatureBuilderAgent
        feature_agent = FeatureBuilderAgent(
            bus=bus,
            tech_features=self.tech_features,
            stock_features=self.stock_feature_pipeline,
        )
        agents.append(feature_agent)

        # SignalGeneratorAgent
        signal_agent = SignalGeneratorAgent(
            bus=bus,
            ensemble=self.ensemble,
            stock_models=self.stock_models,
            janus=self.janus,
        )
        agents.append(signal_agent)

        # RiskManagerAgent
        risk_agent = RiskManagerAgent(
            bus=bus,
            risk_manager=self.risk_manager,
            safety=self.safety,
            settings=self._settings,
            ensemble=self.ensemble,
            circuit_breaker=self.circuit_breaker,
        )
        agents.append(risk_agent)

        # ExecutionAgent
        exec_agent = ExecutionAgent(
            bus=bus,
            router=self.router,
            db=self.db,
            notifier=self.notifier,
            retrainer=self.retrainer,
            pnl_tracker=self.pnl_tracker,
            risk_manager=self.risk_manager,
            janus=self.janus,
        )
        agents.append(exec_agent)

        # MonitorAgent (position stop-loss / take-profit watching)
        monitor_agent = MonitorAgent(
            bus=bus,
            router=self.router,
            risk_manager=self.risk_manager,
            settings=self._settings,
        )
        agents.append(monitor_agent)

        # ── Background tasks ─────────────────────────────────────────
        background_tasks: list = []

        # Auto-retrain loop
        async def retrain_loop():
            while self._running:
                try:
                    if self.retrainer.should_retrain():
                        logger.info("Retraining interval elapsed — starting auto-retrain")
                        self.retrainer.retrain_all(
                            market_collector=self.market_collector,
                            stock_collector=self.stock_collector,
                            betting_collector=self.betting_collector,
                            stock_feature_pipeline=self.stock_feature_pipeline,
                            betting_features=self.betting_features,
                            ensemble=self.ensemble,
                            stock_models=self.stock_models,
                            betting_model=self.betting_model,
                        )
                except Exception:
                    logger.exception("Auto-retrain failed")
                await asyncio.sleep(60)

        background_tasks.append(asyncio.create_task(retrain_loop()))

        # JANUS weight sync: recalibrate constrained weights → sync to ensemble
        async def janus_weight_sync():
            while self._running:
                try:
                    model_weights = self.ensemble.get_model_weights()
                    if model_weights and self.janus._outcomes:
                        calibrated = self.janus.calibrate_weights(
                            model_names=list(model_weights.keys())
                        )
                        for name, weight in calibrated.items():
                            if name in model_weights:
                                self.ensemble.update_weight(name, weight)
                        regime = self.janus.detect_regime()
                        if regime["regime"] != "MIXED":
                            logger.info(
                                "JANUS regime: %s (diff=%.3f, conf=%.2f)",
                                regime["regime"],
                                regime["weight_diff"],
                                regime["confidence"],
                            )
                except Exception:
                    logger.exception("JANUS weight sync failed")
                await asyncio.sleep(300)  # every 5 minutes

        background_tasks.append(asyncio.create_task(janus_weight_sync()))

        # Health check loop
        async def health_loop():
            while self._running:
                try:
                    health = await self.health_checker.check_system_resources()
                    if health.get("warnings"):
                        logger.warning("System health: %s", health)
                except Exception:
                    logger.exception("Health check failed")
                await asyncio.sleep(300)

        background_tasks.append(asyncio.create_task(health_loop()))

        # Bus inspector
        async def bus_inspector():
            while self._running:
                sizes = bus.queue_sizes()
                total = sum(sizes.values())
                if total > 0:
                    logger.debug("Bus queues: %s", sizes)
                await asyncio.sleep(30)

        background_tasks.append(asyncio.create_task(bus_inspector()))

        # Portfolio reporter — writes status.json
        async def portfolio_reporter():
            status_path = Path("reports/status.json")
            while self._running:
                try:
                    event: PortfolioEvent = await bus.consume_portfolio()
                    pnl_summary = self.pnl_tracker.summary()
                    snapshot = {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "mode": self._mode,
                        "portfolio": {
                            "balance_usd": event.balance_usd,
                            "equity": event.equity,
                            "return_pct": event.return_pct,
                            "open_positions": event.open_positions,
                            "lanes": event.lanes,
                        },
                        "pnl": pnl_summary,
                        "safety_status": {
                            "halted": self.safety.is_halted,
                            "halt_reason": getattr(self.safety, "_halt_reason", ""),
                            "circuit_breaker_tripped": self.circuit_breaker.is_open,
                        },
                    }
                    status_path.parent.mkdir(parents=True, exist_ok=True)
                    status_path.write_text(json.dumps(snapshot, indent=2))
                except Exception:
                    logger.exception("Portfolio reporter failed")

        background_tasks.append(asyncio.create_task(portfolio_reporter()))

        # ── Launch all agents ────────────────────────────────────────
        agent_tasks = [asyncio.create_task(a.run()) for a in agents]
        all_tasks = agent_tasks + background_tasks

        logger.info(
            "Multi-agent pipeline running — %d agents, %d background tasks",
            len(agent_tasks),
            len(background_tasks),
        )

        try:
            done, pending = await asyncio.wait(
                all_tasks,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("Agent task failed: %s", exc)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down agents...")
            for t in all_tasks:
                t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            logger.info("All agents stopped")

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
        if self.ensemble.should_execute(signal_result, min_confidence=0.65):
            crypto_exec = self.router.crypto_executor
            balance = crypto_exec.get_balance()
            usdt_total = balance.get("USDT", {}).get("total", 0)
            trade_amount = self.risk_manager.calculate_position_size(usdt_total)

            signal = UnifiedSignal(
                asset_type=AssetType.CRYPTO,
                symbol=symbol,
                side=signal_result["signal"],
                confidence=signal_result.get("confidence", 0),
                amount=trade_amount,
                price=0.0,
            )

            result = self.router.route_signal(signal)

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
        if self.ensemble.should_execute(signal_result, min_confidence=0.65):
            last_close = float(ohlcv["close"].iloc[-1])
            balance = self.router.stock_executor.get_balance() if self.router.stock_executor else {}
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
        if self.router.bet_executor:
            stats = self.router.bet_executor.get_stats()
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
        crypto_exec = self.router.crypto_executor
        if crypto_exec:
            for pos in positions:
                if pos.get("asset_type") != "crypto":
                    continue
                symbol = pos.get("symbol", "")
                if symbol and symbol not in prices:
                    try:
                        ticker = crypto_exec._exchange.fetch_ticker(symbol)
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
            if asset_type_str == "stock" and self.router.stock_executor:
                result = self.router.stock_executor.close_position(action["symbol"])
            elif crypto_exec:
                result = crypto_exec.close_position(action["symbol"])
            else:
                continue

            if result.get("status") == "filled":
                await self.notifier.send_trade_alert({
                    **result,
                    "reason": action["action"],
                })
                # Record closed trade for auto-learning feedback
                trade_record = {
                    "model_name": result.get("model", "ensemble"),
                    "pnl": result.get("pnl", result.get("realized_pnl", 0.0)),
                    "side": result.get("side", action.get("action", "exit")),
                    "confidence": result.get("confidence", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": action["symbol"],
                    "asset_type": asset_type_str,
                }
                self.retrainer.record_trade(trade_record)

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
        logger.info("Starting backtesting mode — lanes: %s", self._enabled_assets)

        # Disable sandbox for backtesting — testnet limits historical data
        if self._settings.exchange.sandbox_mode:
            self._settings.exchange.sandbox_mode = False
            self.market_collector._exchange.set_sandbox_mode(False)
            logger.info("Sandbox mode disabled for backtesting (real data, no trades executed)")
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

                # ML ensemble strategy — pre-compute signals for all rows
                numeric_features = self._numeric_features(features)
                n_expected = len(numeric_features.columns)
                signals: Dict[int, Optional[Dict]] = {}
                for i in range(len(features)):
                    price = float(features.iloc[i].get("close", 0))
                    if price <= 0:
                        signals[i] = None
                        continue
                    try:
                        row_df = numeric_features.iloc[[i]]
                        row_df = row_df.replace([np.inf, -np.inf], np.nan)
                        row_df = row_df.dropna(axis=1)
                        if row_df.empty or len(row_df.columns) < n_expected:
                            signals[i] = None
                            continue
                        pred = self.ensemble.vote({"features": row_df})
                    except Exception:
                        pred = {}
                    side = pred.get("signal", "HOLD")
                    conf = pred.get("confidence", 0)
                    if side in ("BUY", "SELL") and conf >= 0.5:
                        signals[i] = {"side": side.lower(), "amount": 500.0}
                    else:
                        signals[i] = None

                # Log signal summary
                buy_signals = sum(1 for s in signals.values() if s and s.get("side") == "buy")
                sell_signals = sum(1 for s in signals.values() if s and s.get("side") == "sell")
                logger.info("Crypto %s signals: BUY=%d, SELL=%d (out of %d rows)", symbol, buy_signals, sell_signals, len(features))

                def ensemble_strategy(data, idx):
                    return signals.get(idx)

                result = engine.run(ensemble_strategy, features)
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

    # ── Configuration validation ───────────────────────────────────
    try:
        warnings_list = settings.validate()
        for msg in warnings_list:
            logger.warning("Config: %s", msg)
    except ValueError as exc:
        logger.critical("Configuration validation failed:\n%s", exc)
        sys.exit(1)

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
