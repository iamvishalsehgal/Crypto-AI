"""
Weighted ensemble voting system for combining multiple model predictions.

Aggregates trading signals from heterogeneous models (XGBoost, CNN,
sentiment, etc.) via a weighted vote, with configurable confidence
thresholds and automatic weight calibration from backtest results.

Usage::

    from crypto_bot.config.settings import Settings
    from crypto_bot.ensemble.voting_system import EnsembleVoter

    voter = EnsembleVoter(Settings())
    voter.register_model("xgboost", xgb_model, weight=1.5)
    voter.register_model("cnn", cnn_model, weight=1.0)
    result = voter.vote(features)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical signal names.
_BUY = "BUY"
_HOLD = "HOLD"
_SELL = "SELL"

_SIGNAL_CLASSES: List[str] = [_SELL, _HOLD, _BUY]

# Mapping from integer codes to strings and back.
_INT_TO_SIGNAL: Dict[int, str] = {-1: _SELL, 0: _HOLD, 1: _BUY}
_SIGNAL_TO_INT: Dict[str, int] = {_SELL: -1, _HOLD: 0, _BUY: 1}


class _PredictableModel(Protocol):
    """Structural protocol for models that the voter can call."""

    def predict(self, X: Any) -> Any: ...


class EnsembleVoter:
    """Aggregate trading signals from multiple models via weighted voting.

    Parameters
    ----------
    settings:
        Project-wide settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # {model_name: (model_instance, weight)}
        self._models: Dict[str, Tuple[Any, float]] = {}

        logger.info("EnsembleVoter initialised")

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
    ) -> None:
        """Add a model to the ensemble.

        Parameters
        ----------
        name:
            Unique identifier for this model.
        model:
            Any object that exposes a ``predict`` method returning
            integer signals in ``{-1, 0, 1}`` or string signals in
            ``{"SELL", "HOLD", "BUY"}``.  Alternatively, a plain callable
            ``features -> signal`` is also accepted.
        weight:
            Relative weight of this model in the ensemble vote.

        Raises
        ------
        ValueError
            If *weight* is negative.
        """
        if weight < 0:
            raise ValueError(f"Weight must be non-negative; got {weight}")

        self._models[name] = (model, weight)
        logger.info("Registered model '%s' with weight %.2f", name, weight)

    def update_weight(self, name: str, new_weight: float) -> None:
        """Change the voting weight for an already-registered model.

        Parameters
        ----------
        name:
            Model identifier.
        new_weight:
            New weight value (must be >= 0).

        Raises
        ------
        KeyError
            If *name* has not been registered.
        ValueError
            If *new_weight* is negative.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        if new_weight < 0:
            raise ValueError(f"Weight must be non-negative; got {new_weight}")

        model_instance, _ = self._models[name]
        self._models[name] = (model_instance, new_weight)
        logger.info("Updated weight for '%s' to %.2f", name, new_weight)

    def get_model_weights(self) -> Dict[str, float]:
        """Return the current weight of every registered model.

        Returns
        -------
        dict
            ``{model_name: weight}``
        """
        return {name: weight for name, (_, weight) in self._models.items()}

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def vote(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Collect predictions from all models and compute a weighted vote.

        Each model is called with the subset of *features* it needs.
        Models that expose a ``predict`` method receive the full
        *features* dict; plain callables are called with *features*
        directly.

        Parameters
        ----------
        features:
            Dictionary of input features.  The exact keys depend on the
            registered models.

        Returns
        -------
        dict
            Keys:

            * ``signal`` -- winning signal (``"BUY"`` / ``"HOLD"`` / ``"SELL"``).
            * ``confidence`` -- agreement level in ``[0, 1]``.
            * ``individual_predictions`` -- ``{model: signal_str}`` for
              every registered model.
            * ``weighted_scores`` -- ``{signal: weighted_sum}`` for each
              signal class.
        """
        if not self._models:
            logger.warning("No models registered; returning HOLD with 0 confidence")
            return {
                "signal": _HOLD,
                "confidence": 0.0,
                "individual_predictions": {},
                "weighted_scores": {_BUY: 0.0, _HOLD: 0.0, _SELL: 0.0},
            }

        individual_predictions: Dict[str, str] = {}

        for name, (model, _weight) in self._models.items():
            try:
                prediction = self._get_prediction(model, features)
                individual_predictions[name] = prediction
            except Exception:
                logger.exception(
                    "Model '%s' failed during prediction; defaulting to HOLD",
                    name,
                )
                individual_predictions[name] = _HOLD

        weighted_scores = self._aggregate_signals(individual_predictions)

        # The winning signal is the class with the highest weighted score.
        winning_signal = max(weighted_scores, key=weighted_scores.get)  # type: ignore[arg-type]

        # Confidence: proportion of total weight that backed the winner.
        total_weight = sum(weighted_scores.values())
        confidence = (
            weighted_scores[winning_signal] / total_weight
            if total_weight > 0
            else 0.0
        )

        return {
            "signal": winning_signal,
            "confidence": round(confidence, 4),
            "individual_predictions": individual_predictions,
            "weighted_scores": {
                k: round(v, 4) for k, v in weighted_scores.items()
            },
        }

    # ------------------------------------------------------------------
    # Execution gate
    # ------------------------------------------------------------------

    def should_execute(
        self,
        vote_result: Dict[str, Any],
        min_confidence: float = 0.6,
        min_agree: int = 4,
    ) -> bool:
        """Decide whether the ensemble vote is strong enough to execute.

        Parameters
        ----------
        vote_result:
            Output of :meth:`vote`.
        min_confidence:
            Minimum confidence score required.
        min_agree:
            Minimum number of models that must agree on the winning
            signal.

        Returns
        -------
        bool
            ``True`` if the signal is not ``HOLD``, confidence meets the
            threshold, and enough models agree.
        """
        signal = vote_result.get("signal", _HOLD)
        confidence = vote_result.get("confidence", 0.0)
        predictions = vote_result.get("individual_predictions", {})

        if signal == _HOLD:
            return False

        if confidence < min_confidence:
            logger.debug(
                "Execution blocked: confidence %.4f < threshold %.4f",
                confidence,
                min_confidence,
            )
            return False

        agree_count = sum(1 for v in predictions.values() if v == signal)
        if agree_count < min_agree:
            logger.debug(
                "Execution blocked: %d models agree < min_agree %d",
                agree_count,
                min_agree,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Weight calibration
    # ------------------------------------------------------------------

    def calibrate_weights(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """Automatically adjust model weights based on historical accuracy.

        *backtest_results* must contain one column per registered model
        (column name = model name) holding its predicted signal, plus a
        ``"actual"`` column with the realised label.

        Weights are set proportional to each model's accuracy on the
        backtest data.  Models not present in the DataFrame keep their
        current weight.

        Parameters
        ----------
        backtest_results:
            DataFrame with columns for each model and an ``"actual"``
            column.

        Returns
        -------
        dict
            Updated ``{model_name: new_weight}`` mapping.
        """
        if "actual" not in backtest_results.columns:
            raise ValueError(
                "backtest_results must contain an 'actual' column"
            )

        actual = backtest_results["actual"]

        for name in list(self._models.keys()):
            if name not in backtest_results.columns:
                logger.debug(
                    "Model '%s' not in backtest_results; keeping current weight",
                    name,
                )
                continue

            model_preds = backtest_results[name]

            # Compute accuracy.
            correct = (model_preds == actual).sum()
            total = len(actual)
            accuracy = correct / total if total > 0 else 0.0

            # Use accuracy as the new weight (floor at a small epsilon so
            # no model is completely zeroed out).
            new_weight = max(float(accuracy), 0.01)

            model_instance, _ = self._models[name]
            self._models[name] = (model_instance, round(new_weight, 4))

            logger.info(
                "Calibrated '%s': accuracy=%.4f -> weight=%.4f",
                name,
                accuracy,
                new_weight,
            )

        return self.get_model_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_signals(
        self,
        predictions: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute weighted sums for each signal class.

        Parameters
        ----------
        predictions:
            ``{model_name: signal_string}`` mapping.

        Returns
        -------
        dict
            ``{"BUY": float, "HOLD": float, "SELL": float}`` weighted
            score totals.
        """
        scores: Dict[str, float] = {_BUY: 0.0, _HOLD: 0.0, _SELL: 0.0}

        for name, signal in predictions.items():
            if name not in self._models:
                continue

            _, weight = self._models[name]
            normalised_signal = self._normalise_signal(signal)
            scores[normalised_signal] += weight

        return scores

    @staticmethod
    def _normalise_signal(signal: Any) -> str:
        """Convert an arbitrary signal value to a canonical string.

        Accepts integers ``{-1, 0, 1}``, NumPy scalars, or strings.
        Falls back to ``"HOLD"`` for unrecognised values.
        """
        if isinstance(signal, (np.integer, int)):
            return _INT_TO_SIGNAL.get(int(signal), _HOLD)

        if isinstance(signal, str):
            upper = signal.upper().strip()
            if upper in _SIGNAL_CLASSES:
                return upper
            return _HOLD

        # NumPy arrays with a single element, etc.
        try:
            return _INT_TO_SIGNAL.get(int(signal), _HOLD)
        except (TypeError, ValueError):
            return _HOLD

    @staticmethod
    def _get_prediction(model: Any, features: Dict[str, Any]) -> str:
        """Obtain a single string prediction from a model or callable.

        The method tries, in order:

        1.  ``model.predict(features)`` -- if the model has a ``predict``
            attribute.
        2.  ``model(features)`` -- if the model is callable.
        3.  Falls back to ``"HOLD"``.
        """
        raw: Any

        if hasattr(model, "predict") and callable(model.predict):
            raw = model.predict(features)
        elif callable(model):
            raw = model(features)
        else:
            logger.warning(
                "Model %r has no predict method and is not callable; "
                "defaulting to HOLD",
                model,
            )
            return _HOLD

        # If the result is array-like, take the first element.
        if isinstance(raw, (np.ndarray, list, pd.Series)):
            raw = raw[0] if len(raw) > 0 else 0

        return EnsembleVoter._normalise_signal(raw)
