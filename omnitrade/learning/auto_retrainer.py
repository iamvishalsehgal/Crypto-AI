"""
Auto-retraining pipeline for self-learning trading agents.

Handles periodic model retraining across all asset lanes, model
persistence, and ensemble weight adaptation from trade outcomes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_INTERVAL_HOURS = 24
_MIN_TRAINING_ROWS = 100


class AutoRetrainer:
    """Periodically retrains models on fresh data across all lanes.

    Tracks retraining schedule, collects fresh market data, retrains
    models, persists them to disk, and adapts ensemble weights based
    on per-model trade performance.

    Args:
        settings: Bot configuration.
        models_dir: Directory for saved model artifacts.
        retrain_interval_hours: How often to retrain (default 24h).
    """

    def __init__(
        self,
        settings: Settings,
        models_dir: Optional[Path] = None,
        retrain_interval_hours: int = _DEFAULT_INTERVAL_HOURS,
    ) -> None:
        self._settings = settings
        self._interval = retrain_interval_hours
        self._models_dir = models_dir or (
            Path(__file__).resolve().parent.parent.parent / "models" / "saved"
        )
        self._models_dir.mkdir(parents=True, exist_ok=True)

        self._last_retrain: float = 0.0
        self._retrain_count = 0
        self._trade_history: List[Dict[str, Any]] = []
        self._model_pnl: Dict[str, float] = {}  # model_name → cumulative PnL

        logger.info(
            "AutoRetrainer initialised — interval=%dh, dir=%s",
            self._interval,
            self._models_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_retrain_ts(self) -> float:
        return self._last_retrain

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def should_retrain(self) -> bool:
        """Check if retraining interval has elapsed."""
        if self._last_retrain == 0.0:
            return True
        elapsed = time.time() - self._last_retrain
        return elapsed >= self._interval * 3600

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade for feedback learning.

        Expected keys: model_name, pnl, side, confidence, timestamp.
        """
        self._trade_history.append(trade)
        model = trade.get("model_name", "unknown")
        pnl = trade.get("pnl", 0.0)
        self._model_pnl[model] = self._model_pnl.get(model, 0.0) + pnl

        # Keep only last 500 trades
        if len(self._trade_history) > 500:
            self._trade_history = self._trade_history[-500:]

    def retrain_all(
        self,
        market_collector: Any,
        stock_collector: Any = None,
        betting_collector: Any = None,
        stock_feature_pipeline: Any = None,
        betting_features: Any = None,
        ensemble: Any = None,
        stock_models: Any = None,
        betting_model: Any = None,
    ) -> Dict[str, Any]:
        """Retrain all enabled lane models on fresh data.

        Returns dict of per-lane training metrics.
        """
        self._last_retrain = time.time()
        self._retrain_count += 1
        results: Dict[str, Any] = {
            "retrain_id": self._retrain_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lanes": {},
        }

        logger.info("Auto-retrain cycle #%d starting", self._retrain_count)

        assets = self._settings.asset.enabled_assets

        # ── Crypto retraining ──────────────────────────────────────
        if "crypto" in assets and market_collector:
            results["lanes"]["crypto"] = self._retrain_crypto(
                market_collector, ensemble
            )

        # ── Stock retraining ───────────────────────────────────────
        if "stock" in assets and stock_collector and stock_models:
            results["lanes"]["stock"] = self._retrain_stock(
                stock_collector, stock_feature_pipeline, stock_models,
            )

        # ── Betting retraining ─────────────────────────────────────
        if "bet" in assets and betting_collector and betting_model:
            results["lanes"]["betting"] = self._retrain_betting(
                betting_collector, betting_features, betting_model,
            )

        # ── Adapt ensemble weights ─────────────────────────────────
        if ensemble and self._model_pnl:
            results["weights"] = self._adapt_weights(ensemble)

        self._save_retrain_state(results)
        logger.info("Auto-retrain cycle #%d complete: %s", self._retrain_count, results)
        return results

    # ------------------------------------------------------------------
    # Lane-specific retraining
    # ------------------------------------------------------------------

    def _retrain_crypto(
        self,
        collector: Any,
        ensemble: Any,
    ) -> Dict[str, Any]:
        """Retrain crypto XGBoost and LightGBM models."""
        result: Dict[str, Any] = {}
        for symbol in self._settings.exchange.supported_symbols:
            try:
                data = collector.fetch_ohlcv(symbol, limit=500)
                if data is None or len(data) < _MIN_TRAINING_ROWS:
                    logger.warning("Insufficient crypto data for %s (%d rows)", symbol, len(data or []))
                    continue

                from omnitrade.features.technical import TechnicalFeatures
                tf = TechnicalFeatures(self._settings)
                features = tf.compute_all(data)
                if features.empty:
                    continue
                features = features.dropna()

                labels = self._generate_labels(features)
                if len(labels) < _MIN_TRAINING_ROWS:
                    continue

                split = int(len(features) * 0.8)
                X_train, X_val = features.iloc[:split], features.iloc[split:]
                y_train, y_val = labels.iloc[:split], labels.iloc[split:]

                numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

                # XGBoost
                try:
                    from omnitrade.models.xgboost_model import XGBoostTrader
                    xgb = XGBoostTrader(self._settings)
                    xgb.train(X_train[numeric_cols], y_train, X_val[numeric_cols], y_val)
                    xgb.save_model(str(self._models_dir / "xgboost_model"))
                    if ensemble and hasattr(ensemble, "register_model"):
                        ensemble.register_model("xgboost", xgb, weight=1.0)
                    result["xgboost"] = "retrained"
                    logger.info("Crypto XGBoost retrained for %s", symbol)
                except Exception as exc:
                    logger.error("Crypto XGBoost retrain failed: %s", exc)
                    result["xgboost"] = f"error: {exc}"

                # LightGBM
                try:
                    from omnitrade.models.lightgbm_model import LightGBMTrader
                    lgb = LightGBMTrader(self._settings)
                    lgb.train(X_train[numeric_cols], y_train, X_val[numeric_cols], y_val)
                    lgb.save_model(str(self._models_dir / "lightgbm_model"))
                    if ensemble and hasattr(ensemble, "register_model"):
                        ensemble.register_model("lightgbm", lgb, weight=1.0)
                    result["lightgbm"] = "retrained"
                    logger.info("Crypto LightGBM retrained for %s", symbol)
                except Exception as exc:
                    logger.error("Crypto LightGBM retrain failed: %s", exc)
                    result["lightgbm"] = f"error: {exc}"

            except Exception as exc:
                logger.error("Crypto retrain failed for %s: %s", symbol, exc)
                result[symbol] = f"error: {exc}"

        return result

    def _retrain_stock(
        self,
        collector: Any,
        feature_pipeline: Any,
        stock_models: Any,
    ) -> Dict[str, Any]:
        """Retrain stock XGBoost and LSTM models."""
        result: Dict[str, Any] = {}
        for ticker in self._settings.stock.supported_tickers:
            try:
                ohlcv = collector.fetch_ohlcv(ticker, interval="1d", period="365d")
                if ohlcv.empty or len(ohlcv) < _MIN_TRAINING_ROWS:
                    logger.warning("Insufficient stock data for %s", ticker)
                    continue

                fundamentals = collector.fetch_fundamentals(ticker)
                features = feature_pipeline.compute_all(ohlcv, fundamentals) if feature_pipeline else ohlcv
                features = features.dropna()

                labels = self._generate_labels(features)
                split = int(len(features) * 0.6)
                train_data, train_labels = features.iloc[:split], labels.iloc[:split]

                stock_models._trained = False
                stock_models.create_all()
                metrics = stock_models.train_all(train_data, train_labels)

                # Persist trained models
                if "xgboost" in stock_models._models:
                    try:
                        stock_models._models["xgboost"].save_model(
                            str(self._models_dir / "stock_xgboost_model")
                        )
                    except Exception:
                        pass
                if "lstm" in stock_models._models:
                    try:
                        stock_models._models["lstm"].save_model(
                            str(self._models_dir / "stock_lstm_model")
                        )
                    except Exception:
                        pass

                result[ticker] = {"status": "retrained", "metrics": metrics}
                logger.info("Stock models retrained for %s", ticker)
            except Exception as exc:
                logger.error("Stock retrain failed for %s: %s", ticker, exc)
                result[ticker] = f"error: {exc}"

        return result

    def _retrain_betting(
        self,
        collector: Any,
        feature_pipeline: Any,
        model: Any,
    ) -> Dict[str, Any]:
        """Retrain betting value model."""
        result: Dict[str, Any] = {}
        for sport in self._settings.betting.supported_sports:
            try:
                odds_df = collector.fetch_odds(sport)
                if odds_df.empty:
                    logger.warning("No odds data for %s", sport)
                    continue

                history = collector.fetch_historical_results(sport, days_back=365)
                if history.empty:
                    logger.warning("No historical data for %s, skipping retrain", sport)
                    continue

                features = feature_pipeline.compute_all(odds_df) if feature_pipeline else odds_df
                labels = self._generate_betting_labels(history, odds_df)

                if len(features) < 20 or len(labels) < 20:
                    continue

                model.train(features, labels)
                result[sport] = "retrained"
                logger.info("Betting model retrained for %s", sport)
            except Exception as exc:
                logger.error("Betting retrain failed for %s: %s", sport, exc)
                result[sport] = f"error: {exc}"

        return result

    # ------------------------------------------------------------------
    # Ensemble weight adaptation
    # ------------------------------------------------------------------

    def _adapt_weights(self, ensemble: Any) -> Dict[str, float]:
        """Adjust ensemble weights based on per-model PnL performance."""
        if not self._model_pnl:
            return {}

        models = sorted(self._model_pnl.items(), key=lambda x: x[1], reverse=True)
        weights: Dict[str, float] = {}

        if len(models) == 1:
            weights[models[0][0]] = 1.0
            return weights

        # Softmax-style weighting: profitable models get higher weight
        pnls = np.array([pnl for _, pnl in models])
        pnl_range = pnls.max() - pnls.min()
        if pnl_range < 0.01:
            # All models similar — equal weight
            w = 1.0 / len(models)
            for name, _ in models:
                weights[name] = round(w, 3)
        else:
            shifted = pnls - pnls.min()
            exp_pnl = np.exp(shifted / (pnl_range + 1.0))
            total = exp_pnl.sum()
            for (name, _), e in zip(models, exp_pnl):
                weights[name] = round(float(e / total), 3)

        # Apply weights to ensemble
        for name, weight in weights.items():
            try:
                if hasattr(ensemble, "update_weight"):
                    ensemble.update_weight(name, weight)
            except Exception:
                pass

        logger.info("Ensemble weights adapted: %s", weights)
        return weights

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_labels(
        features: pd.DataFrame,
        forward_days: int = 7,
        threshold: float = 0.005,
    ) -> pd.Series:
        """Generate BUY(1)/SELL(-1)/HOLD(0) labels from forward returns."""
        close = features["close"].values
        labels = np.zeros(len(close), dtype=int)
        for i in range(len(close) - forward_days):
            fwd_return = (close[i + forward_days] - close[i]) / close[i]
            if fwd_return > threshold:
                labels[i] = 1
            elif fwd_return < -threshold:
                labels[i] = -1
        return pd.Series(labels, index=features.index)

    @staticmethod
    def _generate_betting_labels(
        history: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> pd.Series:
        """Generate betting labels: 1=home_win, 0=draw, -1=away_win."""
        labels = []
        for _, row in history.iterrows():
            hs = row.get("home_score")
            aw = row.get("away_score")
            if hs is None or aw is None:
                labels.append(0)
            elif hs > aw:
                labels.append(1)
            elif hs < aw:
                labels.append(-1)
            else:
                labels.append(0)
        return pd.Series(labels)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_retrain_state(self, results: Dict[str, Any]) -> None:
        """Persist retraining state for cross-session continuity."""
        state_file = self._models_dir / "retrain_state.json"
        state = {
            "last_retrain": self._last_retrain,
            "retrain_count": self._retrain_count,
            "model_pnl": self._model_pnl,
            "last_result": results,
        }
        try:
            state_file.write_text(json.dumps(state, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save retrain state: %s", exc)

    def load_retrain_state(self) -> bool:
        """Load previous retraining state from disk."""
        state_file = self._models_dir / "retrain_state.json"
        if not state_file.exists():
            return False
        try:
            state = json.loads(state_file.read_text())
            self._last_retrain = state.get("last_retrain", 0.0)
            self._retrain_count = state.get("retrain_count", 0)
            self._model_pnl = state.get("model_pnl", {})
            logger.info(
                "Loaded retrain state: %d cycles, last=%s",
                self._retrain_count,
                datetime.fromtimestamp(self._last_retrain, tz=timezone.utc).isoformat()
                if self._last_retrain else "never",
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load retrain state: %s", exc)
            return False
