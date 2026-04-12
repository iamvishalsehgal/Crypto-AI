"""
XGBoost classification model for generating trading signals.

Wraps :class:`xgboost.XGBClassifier` with project-aware configuration,
early-stopping on a validation set, feature importance reporting, and
model persistence.

Usage::

    from crypto_bot.config.settings import Settings
    from crypto_bot.models.xgboost_model import XGBoostTrader

    trader = XGBoostTrader(Settings())
    metrics = trader.train(X_train, y_train, X_val, y_val)
    predictions = trader.predict(X_test)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# Trading signal encoding used throughout the project.
_SIGNAL_MAP: Dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}
_SIGNAL_INT: Dict[str, int] = {"SELL": -1, "HOLD": 0, "BUY": 1}


class XGBoostTrader:
    """XGBoost-based classifier that produces BUY / HOLD / SELL signals.

    The model is a multi-class XGBClassifier trained with log-loss and
    early stopping.  Internally, the three classes are encoded as
    ``{0: SELL, 1: HOLD, 2: BUY}``; public methods map results back to
    the ``{-1, 0, 1}`` convention expected by the rest of the system.

    Parameters
    ----------
    settings:
        Project-wide settings.  ``settings.model.xgboost_n_estimators``
        controls the maximum boosting rounds.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        n_estimators: int = settings.model.xgboost_n_estimators

        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        self._is_fitted: bool = False
        self._feature_names: Optional[list[str]] = None

        logger.info(
            "XGBoostTrader initialised (n_estimators=%d, max_depth=6, lr=0.1)",
            n_estimators,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Train the classifier with early stopping on the validation set.

        Labels in *y_train* / *y_val* should use the project convention
        ``{-1: SELL, 0: HOLD, 1: BUY}``.  They are remapped internally
        to ``{0, 1, 2}`` for XGBoost.

        Parameters
        ----------
        X_train, y_train:
            Training features and labels.
        X_val, y_val:
            Validation features and labels (used for early stopping).

        Returns
        -------
        dict
            Training metrics including ``accuracy``, ``f1_macro``,
            ``val_accuracy``, ``val_f1_macro``, ``best_iteration``, and
            the full ``classification_report`` string.
        """
        # Preserve feature names when a DataFrame is supplied.
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()

        # Remap labels: {-1, 0, 1} -> {0, 1, 2}.
        y_train_mapped = self._remap_labels(y_train)
        y_val_mapped = self._remap_labels(y_val)

        logger.info(
            "Training XGBoost: %d train samples, %d val samples, %d features",
            len(y_train_mapped),
            len(y_val_mapped),
            X_train.shape[1] if hasattr(X_train, "shape") else "?",
        )

        self._model.fit(
            X_train,
            y_train_mapped,
            eval_set=[(X_val, y_val_mapped)],
            verbose=False,
        )
        self._is_fitted = True

        # Compute metrics on training and validation sets.
        train_preds_internal = self._model.predict(X_train)
        val_preds_internal = self._model.predict(X_val)

        train_acc = float(accuracy_score(y_train_mapped, train_preds_internal))
        val_acc = float(accuracy_score(y_val_mapped, val_preds_internal))
        train_f1 = float(
            f1_score(y_train_mapped, train_preds_internal, average="macro", zero_division=0)
        )
        val_f1 = float(
            f1_score(y_val_mapped, val_preds_internal, average="macro", zero_division=0)
        )

        best_iteration = int(self._model.best_iteration) if hasattr(self._model, "best_iteration") and self._model.best_iteration is not None else self._model.n_estimators

        report = classification_report(
            y_val_mapped,
            val_preds_internal,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0,
        )

        metrics: Dict[str, Any] = {
            "accuracy": round(train_acc, 4),
            "f1_macro": round(train_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4),
            "best_iteration": best_iteration,
            "classification_report": report,
        }

        logger.info(
            "Training complete -- val_accuracy=%.4f, val_f1_macro=%.4f, best_iter=%d",
            val_acc,
            val_f1,
            best_iteration,
        )

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Return trading signal predictions.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Integer array with values in ``{-1, 0, 1}`` corresponding to
            SELL, HOLD, BUY respectively.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        self._check_fitted()
        internal_preds = self._model.predict(X)
        return self._unmap_labels(internal_preds)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Return class probabilities for each sample.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, 3)`` with columns ordered as
            ``[SELL, HOLD, BUY]``.
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        self._check_fitted()

        importances = self._model.feature_importances_
        names = (
            self._feature_names
            if self._feature_names
            else [f"feature_{i}" for i in range(len(importances))]
        )

        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: Union[str, Path]) -> None:
        """Persist the trained model to disk.

        Parameters
        ----------
        path:
            Destination file path (typically ``*.json`` or ``*.ubj``).
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("Model saved to %s", path)

    def load_model(self, path: Union[str, Path]) -> None:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            Path to the saved model file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model.load_model(str(path))
        self._is_fitted = True
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been trained or loaded."""
        if not self._is_fitted:
            raise RuntimeError(
                "XGBoostTrader model is not fitted. "
                "Call train() or load_model() first."
            )

    @staticmethod
    def _remap_labels(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Map project labels ``{-1, 0, 1}`` to XGBoost classes ``{0, 1, 2}``."""
        arr = np.asarray(y, dtype=int)
        return arr + 1  # -1 -> 0, 0 -> 1, 1 -> 2

    @staticmethod
    def _unmap_labels(y: np.ndarray) -> np.ndarray:
        """Map XGBoost classes ``{0, 1, 2}`` back to ``{-1, 0, 1}``."""
        return y.astype(int) - 1
