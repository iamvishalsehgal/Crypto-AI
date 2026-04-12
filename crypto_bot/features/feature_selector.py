"""
XGBoost-based feature selection for the crypto trading pipeline.

Trains a lightweight XGBoost classifier to rank features by importance, then
provides several strategies for narrowing the feature set (top-k, threshold,
combined).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Select the most predictive features using XGBoost importance scores."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Configurable defaults -- fall back to sensible values when settings
        # does not carry the attribute.
        self._top_k: int = getattr(settings, "feature_top_k", 20)
        self._threshold: float = getattr(settings, "feature_importance_threshold", 0.01)

        self._model: Optional[XGBClassifier] = None
        self._importance: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._is_fitted: bool = False

        logger.info(
            "FeatureSelector initialised (top_k=%d, threshold=%.4f)",
            self._top_k,
            self._threshold,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def selected_features(self) -> List[str]:
        """Feature names kept after the last ``fit`` call."""
        if self._selected_features is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")
        return list(self._selected_features)

    @property
    def importance_scores(self) -> pd.DataFrame:
        """Feature importance table (sorted descending)."""
        if self._importance is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")
        return self._importance.copy()

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Train an XGBoost classifier and extract feature importances.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (samples x features).
        y : pd.Series
            Binary target variable (0 / 1).

        Returns
        -------
        FeatureSelector
            ``self`` (for method-chaining).
        """
        if X.empty or y.empty:
            raise ValueError("X and y must not be empty")
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )

        # Drop columns that are entirely NaN -- XGBoost cannot learn from them
        X_clean = X.dropna(axis=1, how="all")
        dropped = set(X.columns) - set(X_clean.columns)
        if dropped:
            logger.warning("Dropped all-NaN columns before fitting: %s", dropped)

        # Fill remaining NaN values with column medians
        X_clean = X_clean.fillna(X_clean.median())

        # Determine number of unique classes for the objective
        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError("Target variable must have at least 2 classes")

        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        logger.info(
            "Fitting XGBoost on %d samples x %d features",
            X_clean.shape[0],
            X_clean.shape[1],
        )
        self._model.fit(X_clean, y)

        importances = self._model.feature_importances_
        self._importance = (
            pd.DataFrame(
                {"feature": X_clean.columns, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Default selection uses top_k
        self._selected_features = self.select_top_k(self._top_k)
        self._is_fitted = True

        logger.info(
            "Feature importance computed; %d features selected (top_k=%d)",
            len(self._selected_features),
            self._top_k,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns that passed feature selection.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must contain the selected columns).

        Returns
        -------
        pd.DataFrame
            Subset of *X* with only selected feature columns.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureSelector has not been fitted yet")

        available = [f for f in self._selected_features if f in X.columns]
        missing = set(self._selected_features) - set(available)
        if missing:
            logger.warning(
                "Selected features missing from input: %s", missing
            )
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Convenience: ``fit`` then ``transform`` in a single call.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix.
        """
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # importance inspection
    # ------------------------------------------------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance table sorted by descending importance.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance.
        """
        if self._importance is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")
        return self._importance.copy()

    def plot_feature_importance(self, top_n: int = 20) -> str:
        """Save a horizontal bar chart of the top-N most important features.

        The plot is saved as a PNG file inside the project's ``output``
        directory (created if it does not exist).

        Parameters
        ----------
        top_n : int
            Number of features to include in the chart.

        Returns
        -------
        str
            Absolute path to the saved image.
        """
        if self._importance is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")

        # Lazy import so matplotlib is only required when plotting
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_data = self._importance.head(top_n).iloc[::-1]  # reverse for barh

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(plot_data["feature"], plot_data["importance"], color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()

        output_dir = getattr(self.settings, "output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        logger.info("Feature importance plot saved to %s", filepath)
        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    # selection strategies
    # ------------------------------------------------------------------
    def select_by_threshold(self, threshold: Optional[float] = None) -> List[str]:
        """Return feature names whose importance exceeds *threshold*.

        Parameters
        ----------
        threshold : float | None
            Minimum importance value.  Falls back to the instance default.

        Returns
        -------
        list[str]
            Feature names that pass the threshold filter.
        """
        if self._importance is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")

        if threshold is None:
            threshold = self._threshold

        mask = self._importance["importance"] >= threshold
        selected = self._importance.loc[mask, "feature"].tolist()
        logger.info(
            "%d features above threshold %.4f", len(selected), threshold
        )
        return selected

    def select_top_k(self, k: Optional[int] = None) -> List[str]:
        """Return the *k* most important feature names.

        Parameters
        ----------
        k : int | None
            How many features to keep.  Falls back to the instance default.

        Returns
        -------
        list[str]
            Top-k feature names.
        """
        if self._importance is None:
            raise RuntimeError("FeatureSelector has not been fitted yet")

        if k is None:
            k = self._top_k

        selected = self._importance.head(k)["feature"].tolist()
        logger.info("Top %d features selected", len(selected))
        return selected
