"""
Sentiment normalisation helpers.

Bridges the gap between VADER float scores and FinBERT categorical labels so
that downstream feature engineering always receives numeric float values.
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Default mapping from string labels to normalised float scores.
_LABEL_TO_FLOAT: dict[str, float] = {
    "very_positive": 0.95,
    "positive": 0.8,
    "neutral": 0.0,
    "negative": -0.8,
    "very_negative": -0.95,
}


def sentiment_label_to_float(value: Union[str, int, float]) -> float:
    """Convert a sentiment value of any format to a normalised float in [-1, 1].

    Parameters
    ----------
    value:
        One of:

        - A ``float`` in [-1, 1] (passed through unchanged).
        - An ``int`` clamped to [-1, 1].
        - A ``str`` label recognised by ``_LABEL_TO_FLOAT`` (case-insensitive).

    Returns
    -------
    float
        Normalised sentiment score in [-1, 1].  Unrecognised strings default to
        ``0.0`` and are logged as a warning.

    Examples
    --------
    >>> sentiment_label_to_float(0.65)
    0.65
    >>> sentiment_label_to_float("positive")
    0.8
    >>> sentiment_label_to_float("negative")
    -0.8
    >>> sentiment_label_to_float("neutral")
    0.0
    >>> sentiment_label_to_float(1)
    1.0
    >>> sentiment_label_to_float(-5)
    -1.0
    """
    if isinstance(value, float):
        return value

    if isinstance(value, (int, np.integer)):
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, float(value)))

    if isinstance(value, str):
        key = value.strip().lower()
        if key in _LABEL_TO_FLOAT:
            return _LABEL_TO_FLOAT[key]
        # Unrecognised string – default to neutral.
        import logging

        logging.getLogger(__name__).warning(
            "Unrecognised sentiment label '%s' – defaulting to 0.0", value,
        )
        return 0.0

    # Fallback – attempt numeric coercion then clamp.
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, v))
