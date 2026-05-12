"""
Stock-specific model wrappers reusing existing LSTM, XGBoost, and CNN
architectures trained on stock features (technicals + fundamentals).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.ensemble.voting_system import EnsembleVoter
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class StockModelFactory:
    """Creates and manages stock-specific model instances.

    Wraps the existing LSTM, XGBoost, and CNN architectures. Models trained
    on stock data (technicals + fundamentals) produce BUY/SELL/HOLD signals.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: Dict[str, Any] = {}
        self._trained = False

    def create_xgboost(self) -> Any:
        """Create an XGBoost classifier configured for stock signals."""
        try:
            from omnitrade.models.xgboost_model import XGBoostTrader
            model = XGBoostTrader(self._settings)
            self._models["xgboost"] = model
            logger.info("Stock XGBoost model created")
            return model
        except ImportError as exc:
            logger.warning("Could not create stock XGBoost model: %s", exc)
            return None

    def create_lstm(self) -> Any:
        """Create an LSTM model for stock time-series classification."""
        try:
            from omnitrade.models.lstm_model import LSTMTrainer
            trainer = LSTMTrainer(self._settings)
            self._models["lstm"] = trainer
            logger.info("Stock LSTM model created")
            return trainer
        except ImportError as exc:
            logger.warning("Could not create stock LSTM model: %s", exc)
            return None

    def create_all(self) -> Dict[str, Any]:
        self.create_xgboost()
        self.create_lstm()
        if not self._models:
            logger.warning("No stock models could be instantiated")
        return self._models

    def train_all(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, Any]:
        """Train all registered models on stock data.

        Args:
            features: Feature matrix (N rows x M columns).
            labels: Labels with values in {-1, 0, 1} (SELL, HOLD, BUY).

        Returns:
            Dict mapping model name to training metrics.
        """
        metrics: Dict[str, Any] = {}
        valid_models = 0

        # Train/val split (chronological — no shuffle)
        n = len(features)
        split_idx = int(n * 0.80)
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx] if hasattr(labels, "iloc") else labels[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = labels.iloc[split_idx:] if hasattr(labels, "iloc") else labels[split_idx:]

        if "xgboost" in self._models:
            try:
                model = self._models["xgboost"]
                model.train(X_train, y_train, X_val, y_val)
                acc = self._evaluate(model, features, labels)
                metrics["xgboost"] = {"accuracy": acc}
                valid_models += 1
                logger.info("Stock XGBoost trained — accuracy=%.3f", acc)
            except Exception as exc:
                logger.error("Stock XGBoost training failed: %s", exc)
                metrics["xgboost"] = {"error": str(exc)}

        if "lstm" in self._models:
            try:
                trainer = self._models["lstm"]
                train_results = trainer.train(X_train, y_train, X_val, y_val)
                metrics["lstm"] = train_results
                valid_models += 1
                logger.info("Stock LSTM trained")
            except Exception as exc:
                logger.error("Stock LSTM training failed: %s", exc)
                metrics["lstm"] = {"error": str(exc)}

        self._trained = valid_models > 0
        if self._trained:
            logger.info("Stock models trained: %d/%d succeeded", valid_models, len(self._models))
        else:
            logger.warning("No stock models trained successfully")

        return metrics

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions from all trained models and ensemble them.

        Returns:
            Dict with ``signal``, ``confidence``, and per-model predictions.
        """
        if not self._trained:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Models not trained"}

        predictions: Dict[str, int] = {}
        confidences: Dict[str, float] = {}
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}

        for name, model in self._models.items():
            try:
                if name == "xgboost":
                    pred, proba = self._predict_xgboost(model, features)
                elif name == "lstm":
                    pred, proba = self._predict_lstm(model, features)
                else:
                    continue

                label = {0: "HOLD", 1: "BUY", -1: "SELL"}.get(pred, "HOLD")
                predictions[name] = label
                confidences[name] = proba
                votes[label] += 1
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", name, exc)

        total_votes = sum(votes.values()) or 1
        best = max(votes, key=votes.get)
        confidence = votes[best] / total_votes

        return {
            "signal": best,
            "confidence": round(confidence, 3),
            "individual_predictions": predictions,
            "individual_confidences": confidences,
        }

    @staticmethod
    def _predict_xgboost(model: Any, features: pd.DataFrame) -> tuple:
        row = features.iloc[[-1]] if not features.empty else features
        proba = model.predict_proba(row)
        if hasattr(proba, "iloc"):
            proba = proba.iloc[0].values
        pred = int(np.argmax(proba))
        # XGBoost uses {0,1,2} mapping to {SELL,HOLD,BUY}
        mapped = {0: -1, 1: 0, 2: 1}.get(pred, 0)
        return mapped, float(max(proba))

    @staticmethod
    def _predict_lstm(trainer: Any, features: pd.DataFrame) -> tuple:
        try:
            proba = trainer.predict(features)
            if hasattr(proba, "iloc"):
                proba = proba.iloc[-1].values if len(proba.shape) > 1 else proba.values
            pred = int(np.argmax(proba))
            mapped = {0: -1, 1: 0, 2: 1}.get(pred, 0)
            return mapped, float(max(proba))
        except AttributeError:
            return 0, 0.0

    @staticmethod
    def _evaluate(model: Any, features: pd.DataFrame, labels: pd.Series) -> float:
        try:
            pred = model.predict(features)
            if hasattr(pred, "values"):
                pred = pred.values
            labels_arr = labels.values if hasattr(labels, "values") else labels
            # XGBoost internal mapping: assume accuracy comparison
            return float((pred == labels_arr).mean())
        except Exception:
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._trained
