"""
LightGBM classification model for generating trading signals.

Wraps :class:`lightgbm.LGBMClassifier` with project-aware configuration,
early-stopping on a validation set, feature importance reporting, and
model persistence.  Mirrors the XGBoostTrader API for drop-in use in
the ensemble voting system.

Usage::

    from omnitrade.config.settings import Settings
    from omnitrade.models.lightgbm_model import LightGBMTrader

    trader = LightGBMTrader(Settings())
    metrics = trader.train(X_train, y_train, X_val, y_val)
    predictions = trader.predict(X_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_SIGNAL_MAP: Dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}
_SIGNAL_INT: Dict[str, int] = {"SELL": -1, "HOLD": 0, "BUY": 1}


class LightGBMTrader:
    """LightGBM-based classifier that produces BUY / HOLD / SELL signals.

    LightGBM often outperforms XGBoost on tabular financial data due to
    leaf-wise tree growth and native categorical feature support.
    Internally encodes classes as ``{0: SELL, 1: HOLD, 2: BUY}`` and
    maps back to ``{-1, 0, 1}`` for the project convention.

    Parameters
    ----------
    settings:
        Project-wide settings.  Uses ``settings.model.lightgbm_*`` fields.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        n_estimators: int = getattr(
            settings.model, "lightgbm_n_estimators", 500
        )
        max_depth: int = getattr(
            settings.model, "lightgbm_max_depth", 6
        )
        learning_rate: float = getattr(
            settings.model, "lightgbm_learning_rate", 0.1
        )

        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self._is_fitted: bool = False
        self._feature_names: Optional[list[str]] = None

        logger.info(
            "LightGBMTrader initialised (n_estimators=%d, max_depth=%d, lr=%.3f)",
            n_estimators, max_depth, learning_rate,
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
        """Train with early stopping on the validation set.

        Labels in ``{-1, 0, 1}`` are remapped to ``{0, 1, 2}`` internally.

        Returns:
            Dict with accuracy, f1_macro, val_accuracy, val_f1_macro,
            best_iteration, and classification_report.
        """
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()

        y_train_mapped = self._remap_labels(y_train)
        y_val_mapped = self._remap_labels(y_val)

        # Compute balanced sample weights from class distribution
        classes, counts = np.unique(y_train_mapped, return_counts=True)
        class_weight = {c: len(y_train_mapped) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([class_weight[y] for y in y_train_mapped])

        logger.info(
            "Training LightGBM: %d train, %d val, %d features (class weights: %s)",
            len(y_train_mapped), len(y_val_mapped),
            X_train.shape[1] if hasattr(X_train, "shape") else "?",
            {k: round(v, 2) for k, v in class_weight.items()},
        )

        self._model.fit(
            X_train,
            y_train_mapped,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_mapped)],
            eval_metric="multi_logloss",
            callbacks=[],
        )
        self._is_fitted = True

        train_preds = self._model.predict(X_train)
        val_preds = self._model.predict(X_val)

        train_acc = float(accuracy_score(y_train_mapped, train_preds))
        val_acc = float(accuracy_score(y_val_mapped, val_preds))
        train_f1 = float(f1_score(y_train_mapped, train_preds, average="macro", zero_division=0))
        val_f1 = float(f1_score(y_val_mapped, val_preds, average="macro", zero_division=0))

        best_iter = (
            int(self._model.best_iteration_)
            if hasattr(self._model, "best_iteration_") and self._model.best_iteration_ is not None
            else self._model.n_estimators
        )

        report = classification_report(
            y_val_mapped, val_preds,
            target_names=["SELL", "HOLD", "BUY"],
            zero_division=0,
        )

        metrics: Dict[str, Any] = {
            "accuracy": round(train_acc, 4),
            "f1_macro": round(train_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4),
            "best_iteration": best_iter,
            "classification_report": report,
        }

        logger.info(
            "LightGBM training complete -- val_accuracy=%.4f, val_f1=%.4f, best_iter=%d",
            val_acc, val_f1, best_iter,
        )
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Return trading signals in {-1, 0, 1}."""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not np.isfinite(X).all():
            raise RuntimeError("LightGBMTrader.predict received NaN or Inf values")
        internal_preds = self._model.predict(X)
        return self._unmap_labels(internal_preds)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Return class probabilities, columns [SELL, HOLD, BUY]."""
        self._check_fitted()
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
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
        """Save model to *path* using joblib (full sklearn wrapper)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._model, str(path))
        logger.info("LightGBM model saved to %s", path)

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model from *path* (joblib or LightGBM text format)."""
        import joblib
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # SECURITY WARNING: joblib.load uses pickle internally and can execute
        # arbitrary code. Only load models from trusted, hardcoded paths (e.g.,
        # PROJECT_ROOT / "models" / "saved" / "*.pkl"). Never load user-supplied
        # or network-sourced model files.
        # Try joblib first (full sklearn wrapper), fall back to booster text file
        try:
            self._model = joblib.load(str(path))
            self._is_fitted = True
            logger.info("LightGBM model loaded from %s (joblib)", path)
            return
        except Exception:
            logger.debug("Not a joblib file, trying booster text format")

        booster = lgb.Booster(model_file=str(path))
        self._model = LGBMClassifier(verbose=-1, random_state=42, n_jobs=-1)
        self._model._Booster = booster
        self._model._n_features = booster.num_feature()
        self._model._n_features_in = booster.num_feature()
        self._model._n_classes = 3
        self._model._classes = np.array([0, 1, 2])
        self._model._objective = "multiclass"
        le = LabelEncoder()
        le.classes_ = np.array([0, 1, 2])
        self._model._le = le
        self._model.fitted_ = True
        self._is_fitted = True
        logger.info("LightGBM model loaded from %s (booster text)", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LightGBMTrader model is not fitted. Call train() or load_model() first."
            )

    @staticmethod
    def _remap_labels(y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Map {-1, 0, 1} -> {0, 1, 2}."""
        arr = np.asarray(y, dtype=int)
        return arr + 1

    @staticmethod
    def _unmap_labels(y: np.ndarray) -> np.ndarray:
        """Map {0, 1, 2} -> {-1, 0, 1}."""
        return y.astype(int) - 1
