"""
JANUS-style meta-weighting layer for OmniTrade ensemble.

Wraps EnsembleVoter to add:
- Softmax weight calibration with 20%/80% floor/ceiling constraints
- Disagreement penalty when models produce opposing signals
- Emergent regime detection from model-type weight differentials
- Rolling 30-period accuracy tracking per model

Adapted from ATLAS-GIC janus.py (General Intelligence Capital).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_BUY = "BUY"
_HOLD = "HOLD"
_SELL = "SELL"

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class JanusBlender:
    """Meta-weighting layer with constrained softmax blending and regime detection.

    Tracks per-model accuracy over a rolling window, calibrates weights
    via softmax with min/max floors, penalises confidence when models
    disagree, and detects market regime from weight differentials between
    model types (e.g. tree models vs. neural models).

    Args:
        model_types: Optional mapping of model_name -> type_label
            (e.g. ``{"xgboost": "tree", "lstm": "neural"}``).
            Used for regime detection. Defaults to treating all models
            as one cohort.
        rolling_window: Number of periods for accuracy calculation.
        min_weight: Floor constraint (no model drops below this).
        max_weight: Ceiling constraint (no model dominates above this).
        state_file: Path to persistence file.
    """

    MIN_WEIGHT = 0.2
    MAX_WEIGHT = 0.8
    ROLLING_WINDOW = 30
    REGIME_THRESHOLD = 0.15

    def __init__(
        self,
        model_types: Optional[Dict[str, str]] = None,
        rolling_window: int = ROLLING_WINDOW,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
        state_file: Optional[Path] = None,
    ) -> None:
        self._model_types = model_types or {}
        self._rolling_window = rolling_window
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._state_file = state_file or (STATE_DIR / "janus_state.json")

        # model_name -> list of (hit: bool, weighted_return: float)
        self._outcomes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Current calibrated weights
        self._weights: Dict[str, float] = {}

        # Current accuracy metrics per model
        self._accuracy: Dict[str, Dict[str, float]] = {}

        # Regime history for trend detection
        self._regime_history: List[Dict[str, Any]] = []

        self._load_state()
        logger.info(
            "JanusBlender initialised (window=%d, min=%.0f%%, max=%.0f%%)",
            rolling_window,
            min_weight * 100,
            max_weight * 100,
        )

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        model_name: str,
        predicted_signal: str,
        actual_return: float,
        conviction: float = 0.5,
    ) -> None:
        """Record a model prediction outcome for accuracy tracking.

        Args:
            model_name: Model identifier.
            predicted_signal: ``BUY``, ``SELL``, or ``HOLD``.
            actual_return: Realised return (decimal, e.g. 0.02 for +2%).
            conviction: Model's conviction at prediction time (0-1).
        """
        predicted_signal = predicted_signal.upper()

        # Determine if directional prediction was correct
        if predicted_signal == _BUY:
            is_hit = actual_return > 0
        elif predicted_signal == _SELL:
            is_hit = actual_return < 0
        else:
            is_hit = abs(actual_return) < 0.001  # HOLD correct if flat

        # Conviction-weighted return
        if predicted_signal == _BUY:
            weighted_return = conviction * actual_return
        elif predicted_signal == _SELL:
            weighted_return = conviction * (-actual_return)
        else:
            weighted_return = -conviction * abs(actual_return)  # HOLD costs opportunity

        entry = {
            "date": datetime.now(timezone.utc).isoformat(),
            "predicted": predicted_signal,
            "actual_return": actual_return,
            "is_hit": is_hit,
            "weighted_return": weighted_return,
            "conviction": conviction,
        }

        self._outcomes[model_name].append(entry)

        # Trim to rolling window
        if len(self._outcomes[model_name]) > self._rolling_window * 2:
            self._outcomes[model_name] = self._outcomes[model_name][
                -self._rolling_window :
            ]

    # ------------------------------------------------------------------
    # Weight calibration
    # ------------------------------------------------------------------

    def _compute_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Calculate hit rate and Sharpe-like score for a model."""
        outcomes = self._outcomes.get(model_name, [])
        recent = outcomes[-self._rolling_window :]

        if len(recent) < 3:
            return {"hit_rate": 0.5, "score": 0.5}

        hits = sum(1 for o in recent if o.get("is_hit", False))
        hit_rate = hits / len(recent)

        returns = [o.get("weighted_return", 0) for o in recent]
        if len(returns) < 2:
            return {"hit_rate": hit_rate, "score": hit_rate}

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0001

        # Annualised Sharpe-like score (daily frequency assumed)
        sharpe = (mean_ret / std_dev) * math.sqrt(252) if std_dev > 0 else 0.0

        # Normalise Sharpe to 0-1 range
        norm_sharpe = max(0.0, min(1.0, (sharpe + 1.0) / 2.0))

        # Combined score: 50% hit rate, 50% normalised Sharpe
        score = 0.5 * hit_rate + 0.5 * norm_sharpe

        return {"hit_rate": hit_rate, "score": score, "sharpe": sharpe}

    def calibrate_weights(self, model_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Recalibrate model weights using softmax with floor/ceiling constraints.

        Args:
            model_names: Models to include. If None, uses all tracked models.

        Returns:
            ``{model_name: constrained_weight}``
        """
        if model_names is None:
            model_names = list(self._outcomes.keys())

        if not model_names:
            self._weights = {}
            return {}

        # Compute metrics for each model
        raw_scores: Dict[str, float] = {}
        for name in model_names:
            metrics = self._compute_model_metrics(name)
            self._accuracy[name] = metrics
            raw_scores[name] = metrics["score"]

        # Apply softmax with min/max constraints
        self._weights = self._softmax_with_constraints(raw_scores)

        logger.info(
            "JanusBlender calibrated: %s",
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return dict(self._weights)

    def _softmax_with_constraints(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax then clamp to [min_weight, max_weight] with renormalisation."""
        if not scores:
            return {}

        # Softmax
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())

        if total == 0:
            equal = 1.0 / len(scores)
            return {k: equal for k in scores}

        weights = {k: v / total for k, v in exp_scores.items()}

        # Apply floor
        for name in weights:
            if weights[name] < self._min_weight:
                weights[name] = self._min_weight

        # Renormalise after floor
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Apply ceiling
        for name in weights:
            if weights[name] > self._max_weight:
                weights[name] = self._max_weight

        # Final renormalisation
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    # ------------------------------------------------------------------
    # Signal blending with disagreement penalty
    # ------------------------------------------------------------------

    def blend(
        self,
        individual_predictions: Dict[str, str],
        convictions: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Blend individual model predictions with disagreement penalty.

        When models produce opposing directional signals (one BUY, another
        SELL), the winning conviction is reduced by 50% of the opposing
        weighted conviction.

        Args:
            individual_predictions: ``{model_name: signal_str}``
            convictions: ``{model_name: confidence}`` (0-1). Defaults to
                model's recent hit rate if not provided.
            weights: Model weights. Uses calibrated weights if not provided.

        Returns:
            Dict with ``signal``, ``confidence``, ``contested``,
            ``direction_scores``, and ``weight_breakdown``.
        """
        if weights is None:
            weights = self._weights

        # Fallback to equal weights if no calibration yet
        if not weights and individual_predictions:
            n = len(individual_predictions)
            weights = {name: 1.0 / n for name in individual_predictions}

        if convictions is None:
            convictions = {
                name: self._accuracy.get(name, {}).get("hit_rate", 0.5)
                for name in individual_predictions
            }

        # Separate signals by direction
        bull_score = 0.0  # BUY-weighted
        bear_score = 0.0  # SELL-weighted
        weight_breakdown: Dict[str, Dict[str, Any]] = {}

        for model_name, signal in individual_predictions.items():
            w = weights.get(model_name, 0.0)
            c = convictions.get(model_name, 0.5)
            weighted = w * c

            entry = {
                "signal": signal,
                "conviction": round(c, 4),
                "weight": round(w, 4),
                "weighted": round(weighted, 4),
            }
            weight_breakdown[model_name] = entry

            if signal == _BUY:
                bull_score += weighted
            elif signal == _SELL:
                bear_score += weighted
            # HOLD contributes to neither direction

        # Determine winner and disagreement
        contested = bool(bull_score > 0 and bear_score > 0)

        if bull_score >= bear_score:
            direction = _BUY
            base_conviction = bull_score
            opposing = bear_score
        else:
            direction = _SELL
            base_conviction = bear_score
            opposing = bull_score

        # Disagreement penalty: reduce conviction by 50% of opposing score
        if contested and opposing > 0:
            total_dir = bull_score + bear_score
            # Penalty proportional to opposition strength relative to total
            penalty = opposing * 0.5
            final_conviction = max(0.0, base_conviction - penalty)
            # Normalise to 0-1 range based on total possible weight
            max_possible = sum(weights.values()) if weights else 1.0
            final_conviction = min(1.0, final_conviction / max_possible)
        else:
            max_possible = sum(weights.values()) if weights else 1.0
            final_conviction = (
                min(1.0, base_conviction / max_possible) if max_possible > 0 else 0.0
            )

        if bull_score == 0 and bear_score == 0:
            direction = _HOLD
            final_conviction = 0.0

        return {
            "signal": direction,
            "confidence": round(final_conviction, 4),
            "contested": contested,
            "direction_scores": {
                "BUY": round(bull_score, 4),
                "SELL": round(bear_score, 4),
            },
            "weight_breakdown": weight_breakdown,
        }

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self) -> Dict[str, Any]:
        """Detect market regime from model-type weight differentials.

        Groups models by type (e.g. "tree" vs "neural"), compares aggregate
        weights. When one type dominates beyond REGIME_THRESHOLD, signals
        a regime shift.

        Returns:
            Dict with ``regime``, ``type_weights``, ``weight_diff``,
            and ``confidence``.
        """
        if not self._model_types or not self._weights:
            return {
                "regime": "MIXED",
                "type_weights": {},
                "weight_diff": 0.0,
                "confidence": 0.0,
            }

        # Aggregate weights by type
        type_weights: Dict[str, float] = defaultdict(float)
        for model_name, weight in self._weights.items():
            mtype = self._model_types.get(model_name, model_name)
            type_weights[mtype] += weight

        if len(type_weights) < 2:
            return {
                "regime": "MIXED",
                "type_weights": dict(type_weights),
                "weight_diff": 0.0,
                "confidence": 0.0,
            }

        # Sort types by aggregate weight
        sorted_types = sorted(type_weights.items(), key=lambda x: x[1], reverse=True)
        top_type, top_weight = sorted_types[0]
        second_type, second_weight = sorted_types[1]

        weight_diff = top_weight - second_weight

        if weight_diff > self.REGIME_THRESHOLD:
            regime = f"{top_type.upper()}_DOMINANT"
        elif weight_diff < -self.REGIME_THRESHOLD:
            regime = f"{second_type.upper()}_DOMINANT"
        else:
            regime = "MIXED"

        confidence = min(1.0, abs(weight_diff) / (self.REGIME_THRESHOLD * 2))

        entry = {
            "regime": regime,
            "type_weights": {
                k: round(v, 4) for k, v in type_weights.items()
            },
            "weight_diff": round(weight_diff, 4),
            "confidence": round(confidence, 4),
            "date": datetime.now(timezone.utc).isoformat(),
        }

        self._regime_history.append(entry)
        if len(self._regime_history) > 365:
            self._regime_history = self._regime_history[-365:]

        return entry

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        individual_predictions: Optional[Dict[str, str]] = None,
        convictions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run full JanusBlender cycle: calibrate weights, blend, detect regime.

        Args:
            individual_predictions: Current model predictions to blend.
            convictions: Per-model conviction scores.

        Returns:
            Dict with ``weights``, ``blend_result``, ``regime``,
            ``accuracy``, and ``timestamp``.
        """
        # Recalibrate weights from recent outcomes
        model_names = list(self._outcomes.keys())
        if not model_names and individual_predictions:
            model_names = list(individual_predictions.keys())

        weights = self.calibrate_weights(model_names)

        # Blend current predictions if provided
        blend_result = None
        if individual_predictions:
            blend_result = self.blend(individual_predictions, convictions, weights)

        # Regime detection
        regime = self.detect_regime()

        output = {
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "blend_result": blend_result,
            "regime": regime,
            "accuracy": {
                k: {
                    "hit_rate": round(v.get("hit_rate", 0.5), 4),
                    "score": round(v.get("score", 0.5), 4),
                    "sharpe": round(v.get("sharpe", 0.0), 4),
                }
                for k, v in self._accuracy.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._save_state(output)
        return output

    def get_weights(self) -> Dict[str, float]:
        """Return current calibrated weights."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self, output: Dict[str, Any]) -> None:
        """Persist state to disk."""
        try:
            payload = {
                "weights": output["weights"],
                "accuracy": output["accuracy"],
                "regime": output["regime"],
                "outcome_counts": {
                    k: len(v) for k, v in self._outcomes.items()
                },
                "timestamp": output["timestamp"],
            }
            with open(self._state_file, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            logger.exception("Failed to save JanusBlender state")

    def _load_state(self) -> None:
        """Restore state from disk."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)

            self._weights = state.get("weights", {})
            self._accuracy = state.get("accuracy", {})

            logger.info(
                "JanusBlender state loaded: %d models, regime=%s",
                len(self._weights),
                state.get("regime", {}).get("regime", "unknown"),
            )
        except Exception:
            logger.exception("Failed to load JanusBlender state")
