"""Base class for feature pipelines — shared try/except + concat pattern."""

from __future__ import annotations

from typing import Callable, List

import pandas as pd

from omnitrade.utils.logger import get_logger

_logger = get_logger(__name__)


class FeaturePipelineBase:
    """Mixin that provides safe compute + frame aggregation for feature modules."""

    @staticmethod
    def _safe_compute(
        frames: List[pd.DataFrame],
        method: Callable[..., pd.DataFrame],
        name: str,
        *args,
        **kwargs,
    ) -> None:
        try:
            result = method(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and not result.empty:
                frames.append(result)
        except Exception as exc:
            _logger.warning("%s computation failed: %s", name, exc)

    @staticmethod
    def _aggregate(frames: List[pd.DataFrame], module_name: str) -> pd.DataFrame:
        if not frames:
            _logger.error("%s: no features could be computed", module_name)
            return pd.DataFrame()
        result = pd.concat(frames, axis=1)
        _logger.info("%s features computed: %d rows x %d cols", module_name, len(result), len(result.columns))
        return result
