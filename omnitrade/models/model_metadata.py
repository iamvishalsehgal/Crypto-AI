"""
Model metadata persistence and auto-retraining staleness checks.

Tracks training date, metrics, data range, and feature names per model.
Per-asset staleness schedules: crypto = 7d, stock = 14d, betting = 30d.

Usage::

    from omnitrade.models.model_metadata import ModelMetadata

    meta = ModelMetadata()
    meta.record_training("xgboost_crypto", metrics, data_end="2026-05-12")
    if meta.is_stale("xgboost_crypto", asset_type="crypto"):
        retrain()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

# Staleness thresholds per asset class.
_STALENESS_DAYS: Dict[str, int] = {
    "crypto": 7,
    "stock": 14,
    "betting": 30,
}

_DEFAULT_METADATA_PATH = "output/model_metadata.json"


class ModelMetadata:
    """Read/write model training metadata for auto-retraining decisions.

    Args:
        path: JSON file path for persistence.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path or _DEFAULT_METADATA_PATH)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_training(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        asset_type: str = "crypto",
        data_start: Optional[str] = None,
        data_end: Optional[str] = None,
        feature_count: Optional[int] = None,
    ) -> None:
        """Record a training event.

        Args:
            model_name: Unique model identifier (e.g. "xgboost_crypto").
            metrics: Training metrics dict (accuracy, f1, etc.).
            asset_type: "crypto", "stock", or "betting".
            data_start: ISO date string for earliest training data.
            data_end: ISO date string for most recent training data.
            feature_count: Number of features used.
        """
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "last_trained": now,
            "asset_type": asset_type,
            "metrics": _sanitize_metrics(metrics),
            "data_start": data_start,
            "data_end": data_end,
            "feature_count": feature_count,
        }

        self._data[model_name] = entry
        self._save()
        logger.info(
            "Model '%s' training recorded: %s (asset=%s)",
            model_name, now, asset_type,
        )

    # ------------------------------------------------------------------
    # Staleness checking
    # ------------------------------------------------------------------

    def is_stale(
        self,
        model_name: str,
        asset_type: Optional[str] = None,
        override_days: Optional[int] = None,
    ) -> bool:
        """Check if a model needs retraining.

        Args:
            model_name: Model identifier.
            asset_type: Used to select staleness threshold (default: from
                recorded metadata, or "crypto").
            override_days: Override the staleness threshold.

        Returns:
            True if the model has never been trained or exceeds the
            staleness threshold.
        """
        entry = self._data.get(model_name)
        if entry is None:
            logger.info("Model '%s' never trained — stale", model_name)
            return True

        asset = asset_type or entry.get("asset_type", "crypto")
        threshold_days = override_days or _STALENESS_DAYS.get(asset, 7)

        last_trained_str = entry.get("last_trained")
        if not last_trained_str:
            return True

        try:
            last_trained = datetime.fromisoformat(last_trained_str)
        except ValueError:
            logger.warning("Invalid last_trained format for '%s'", model_name)
            return True

        now = datetime.now(timezone.utc)
        if last_trained.tzinfo is None:
            last_trained = last_trained.replace(tzinfo=timezone.utc)

        age_days = (now - last_trained).total_seconds() / 86400
        stale = age_days > threshold_days

        if stale:
            logger.info(
                "Model '%s' is stale: %.1f days old (threshold=%dd)",
                model_name, age_days, threshold_days,
            )

        return stale

    def get_stale_models(
        self,
        asset_types: Optional[list[str]] = None,
    ) -> Dict[str, list[str]]:
        """Return all stale models, grouped by asset type.

        Args:
            asset_types: Filter to specific asset types.

        Returns:
            Dict mapping asset_type → list of model names.
        """
        result: Dict[str, list[str]] = {}
        for name, entry in self._data.items():
            asset = entry.get("asset_type", "crypto")
            if asset_types and asset not in asset_types:
                continue
            if self.is_stale(name, asset_type=asset):
                result.setdefault(asset, []).append(name)
        return result

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return stored metadata for a model, or None."""
        return self._data.get(model_name)

    def list_models(self) -> Dict[str, str]:
        """Return {model_name: last_trained} for all recorded models."""
        return {
            name: entry.get("last_trained", "unknown")
            for name, entry in self._data.items()
        }

    def get_latest_training_date(self) -> Optional[str]:
        """Return the most recent training timestamp across all models."""
        if not self._data:
            return None
        return max(
            (entry.get("last_trained", "") for entry in self._data.values()),
            default=None,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load metadata from JSON file."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
                logger.debug(
                    "Loaded model metadata: %d models from %s",
                    len(self._data), self._path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load model metadata: %s — starting fresh", exc)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Persist metadata to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except OSError as exc:
            logger.error("Failed to save model metadata: %s", exc)


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only JSON-serializable scalar values from metrics."""
    clean: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = [float(x) if isinstance(x, (int, float)) else str(x) for x in v]
        elif hasattr(v, "item"):  # numpy scalars
            clean[k] = float(v.item())
        else:
            clean[k] = str(v)
    return clean
