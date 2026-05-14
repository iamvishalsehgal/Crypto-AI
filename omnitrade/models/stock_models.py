"""
Stock-specific model factory wrapping existing LSTM, XGBoost architectures
trained on stock features (technicals + fundamentals).

Delegates ensemble voting to :class:`EnsembleVoter` — this module is a
pure factory + trainer, not a voting engine.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.ensemble.voting_system import EnsembleVoter
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical integer-to-signal mapping used by model adapters.
_INT_TO_SIGNAL: Dict[int, str] = {-1: "SELL", 0: "HOLD", 1: "BUY"}


class StockModelFactory:
    """Creates and trains stock-specific model instances.

    Wraps existing LSTM and XGBoost architectures. Models produce
    BUY/SELL/HOLD signals via :class:`EnsembleVoter` weighted voting.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: Dict[str, Any] = {}
        self._trained = False
        self._voter = EnsembleVoter(settings)

    def create_xgboost(self) -> Any:
        try:
            from omnitrade.models.xgboost_model import XGBoostTrader
            model = XGBoostTrader(self._settings)
            self._models["xgboost"] = model
            self._voter.register_model("xgboost", _xgboost_adapter(model), weight=1.0)
            logger.info("Stock XGBoost model created")
            return model
        except ImportError as exc:
            logger.warning("Could not create stock XGBoost model: %s", exc)
            return None

    def create_lstm(self) -> Any:
        try:
            from omnitrade.models.lstm_model import LSTMTrainer
            trainer = LSTMTrainer(self._settings)
            self._models["lstm"] = trainer
            self._voter.register_model("lstm", _lstm_adapter(trainer), weight=1.0)
            logger.info("Stock LSTM model created")
            return trainer
        except ImportError as exc:
            logger.warning("Could not create stock LSTM model: %s", exc)
            return None

    def create_all(self) -> Dict[str, Any]:
        # LSTM first: avoids XGBoost OpenMP → PyTorch segfault on Apple Silicon
        self.create_lstm()
        self.create_xgboost()
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

        n = len(features)
        split_idx = int(n * 0.80)
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx] if hasattr(labels, "iloc") else labels[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = labels.iloc[split_idx:] if hasattr(labels, "iloc") else labels[split_idx:]

        if "xgboost" in self._models:
            try:
                model = self._models["xgboost"]
                numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
                X_train_num = X_train[numeric_cols]
                X_val_num = X_val[numeric_cols]
                model.train(X_train_num, y_train, X_val_num, y_val)
                acc = _evaluate(model, X_train_num, y_train)
                metrics["xgboost"] = {"accuracy": acc}
                valid_models += 1
                logger.info("Stock XGBoost trained — accuracy=%.3f", acc)
            except Exception as exc:
                logger.error("Stock XGBoost training failed: %s", exc)
                metrics["xgboost"] = {"error": str(exc)}

        if "lstm" in self._models:
            try:
                trainer = self._models["lstm"]
                combined = features.copy()
                combined["signal"] = (1 - labels.values).astype(np.int64)
                seq_len = min(60, max(10, len(combined) // 4))
                X_train_t, y_train_t, X_val_t, y_val_t = trainer.prepare_sequences(
                    combined, sequence_length=seq_len, target_col="signal",
                )
                train_results = trainer.train(X_train_t, y_train_t, X_val_t, y_val_t)
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
        """Generate ensemble prediction via :class:`EnsembleVoter`.

        Delegates entirely to the voter — no duplicated voting logic.
        """
        if not self._trained:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Models not trained"}

        return self._voter.vote({"features": features})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def voter(self) -> EnsembleVoter:
        """The underlying :class:`EnsembleVoter` used for stock voting."""
        return self._voter


# =====================================================================
# Model adapters — wrap raw model APIs to satisfy EnsembleVoter's
# `predict(features) -> str` interface.
# =====================================================================


class _XGBoostStockAdapter:
    """Wraps XGBoostTrader for the stock ensemble voter.

    Receives a DataFrame directly — the voter unwraps ``{"features": df}``
    before calling ``predict()``.
    """

    __slots__ = ("_model",)

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, df: pd.DataFrame) -> str:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df = df[numeric_cols]
        row = df.iloc[[-1]] if not df.empty else df
        proba = self._model.predict_proba(row)
        if hasattr(proba, "iloc"):
            proba = proba.iloc[0].values
        pred = int(np.argmax(proba))
        mapped = {0: -1, 1: 0, 2: 1}.get(pred, 0)
        return _INT_TO_SIGNAL.get(mapped, "HOLD")


class _LSTMStockAdapter:
    """Wraps LSTMTrainer for the stock ensemble voter.

    Receives a DataFrame directly — the voter unwraps ``{"features": df}``
    before calling ``predict()``.
    """

    __slots__ = ("_trainer",)

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    def predict(self, df: pd.DataFrame) -> str:
        try:
            proba = self._trainer.predict(df)
            if hasattr(proba, "iloc"):
                proba = proba.iloc[-1].values if len(proba.shape) > 1 else proba.values
            pred = int(np.argmax(proba))
            mapped = {0: -1, 1: 0, 2: 1}.get(pred, 0)
            return _INT_TO_SIGNAL.get(mapped, "HOLD")
        except AttributeError:
            return "HOLD"


def _xgboost_adapter(model: Any) -> _XGBoostStockAdapter:
    return _XGBoostStockAdapter(model)


def _lstm_adapter(trainer: Any) -> _LSTMStockAdapter:
    return _LSTMStockAdapter(trainer)


def _evaluate(model: Any, features: pd.DataFrame, labels: pd.Series) -> float:
    try:
        numeric = features.select_dtypes(include=[np.number])
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        if numeric.empty:
            return 0.0
        pred = model.predict(numeric)
        if hasattr(pred, "values"):
            pred = pred.values
        labels_arr = labels.values if hasattr(labels, "values") else labels
        if len(pred) != len(labels_arr):
            return 0.0
        return float((pred == labels_arr).mean())
    except Exception:
        return 0.0
