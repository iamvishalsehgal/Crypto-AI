"""
Walk-forward validation for the AI Crypto Trading Bot.

Implements time-series-aware walk-forward analysis that trains a model on
successive expanding or rolling windows and tests on the immediately
following out-of-sample period.  This prevents future data leakage and
gives a realistic estimate of live-trading performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Metrics captured for a single walk-forward window."""

    window_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_size: int
    test_size: int
    in_sample_return: float = 0.0
    out_sample_return: float = 0.0
    in_sample_sharpe: float = 0.0
    out_sample_sharpe: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Walk-forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Time-series walk-forward validation engine.

    Parameters
    ----------
    settings:
        Application settings; ``settings.backtesting.walk_forward_windows``
        supplies the default number of windows.
    n_windows:
        Override for the number of walk-forward splits.  Falls back to the
        settings value when ``None``.
    """

    def __init__(self, settings: Settings, n_windows: Optional[int] = None) -> None:
        self._settings = settings
        self._n_windows: int = (
            n_windows
            if n_windows is not None
            else settings.backtesting.walk_forward_windows
        )

        if self._n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self._n_windows}")

        logger.info("WalkForwardValidator initialised with %d windows", self._n_windows)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate(
        self,
        model_trainer_func: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, float]],
        data: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> List[WindowResult]:
        """Run walk-forward validation.

        Parameters
        ----------
        model_trainer_func:
            ``model_trainer_func(train_data, test_data) -> metrics_dict``
            where *metrics_dict* contains at minimum ``"return"`` and
            ``"sharpe"`` keys for the **test** period.  It may also
            contain ``"in_sample_return"`` and ``"in_sample_sharpe"``
            for the training period evaluation.
        data:
            Full historical dataset.  Must have a datetime index or a
            ``timestamp`` column.
        train_ratio:
            Fraction of each window devoted to training (the rest is
            the test set).

        Returns
        -------
        list[WindowResult]
            One result per window.
        """
        if data.empty:
            logger.warning("Empty data provided; returning no results")
            return []

        # Ensure datetime index
        if "timestamp" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index("timestamp")

        windows = self._create_windows(data, self._n_windows, train_ratio)
        results: List[WindowResult] = []

        for wid, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            if train_data.empty or test_data.empty:
                logger.warning(
                    "Window %d has empty train (%d) or test (%d) set; skipping",
                    wid, len(train_data), len(test_data),
                )
                continue

            logger.info(
                "Window %d/%d  train=[%s..%s] (%d rows)  test=[%s..%s] (%d rows)",
                wid + 1,
                self._n_windows,
                train_start,
                train_end,
                len(train_data),
                test_start,
                test_end,
                len(test_data),
            )

            try:
                metrics = model_trainer_func(train_data, test_data)
            except Exception:
                logger.exception("model_trainer_func failed on window %d", wid)
                metrics = {}

            wr = WindowResult(
                window_id=wid,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_data),
                test_size=len(test_data),
                in_sample_return=float(metrics.get("in_sample_return", 0.0)),
                out_sample_return=float(metrics.get("return", 0.0)),
                in_sample_sharpe=float(metrics.get("in_sample_sharpe", 0.0)),
                out_sample_sharpe=float(metrics.get("sharpe", 0.0)),
                extra_metrics={
                    k: float(v)
                    for k, v in metrics.items()
                    if k not in ("return", "sharpe", "in_sample_return", "in_sample_sharpe")
                },
            )
            results.append(wr)

        logger.info("Walk-forward validation complete: %d/%d windows succeeded", len(results), self._n_windows)
        return results

    def aggregate_results(self, window_results: List[WindowResult]) -> Dict[str, Any]:
        """Compute aggregate statistics across all walk-forward windows.

        Parameters
        ----------
        window_results:
            Output of :meth:`validate`.

        Returns
        -------
        dict
            Average and standard deviation of key metrics across windows,
            plus a *consistency_score* (fraction of profitable windows).
        """
        if not window_results:
            return {
                "avg_out_sample_return": 0.0,
                "std_out_sample_return": 0.0,
                "avg_out_sample_sharpe": 0.0,
                "std_out_sample_sharpe": 0.0,
                "avg_in_sample_return": 0.0,
                "std_in_sample_return": 0.0,
                "consistency_score": 0.0,
                "n_windows": 0,
                "n_profitable_windows": 0,
            }

        os_returns = np.array([wr.out_sample_return for wr in window_results])
        os_sharpes = np.array([wr.out_sample_sharpe for wr in window_results])
        is_returns = np.array([wr.in_sample_return for wr in window_results])

        n_profitable = int(np.sum(os_returns > 0))
        consistency = n_profitable / len(window_results)

        return {
            "avg_out_sample_return": float(np.mean(os_returns)),
            "std_out_sample_return": float(np.std(os_returns, ddof=1)) if len(os_returns) > 1 else 0.0,
            "avg_out_sample_sharpe": float(np.mean(os_sharpes)),
            "std_out_sample_sharpe": float(np.std(os_sharpes, ddof=1)) if len(os_sharpes) > 1 else 0.0,
            "avg_in_sample_return": float(np.mean(is_returns)),
            "std_in_sample_return": float(np.std(is_returns, ddof=1)) if len(is_returns) > 1 else 0.0,
            "consistency_score": consistency,
            "n_windows": len(window_results),
            "n_profitable_windows": n_profitable,
        }

    def is_consistent(self, results: List[WindowResult]) -> bool:
        """Check whether the strategy is profitable in a majority of windows.

        A strategy is deemed *consistent* when more than half of the
        out-of-sample windows are profitable.

        Parameters
        ----------
        results:
            Output of :meth:`validate`.

        Returns
        -------
        bool
        """
        if not results:
            return False
        n_profitable = sum(1 for wr in results if wr.out_sample_return > 0)
        return n_profitable > len(results) / 2

    def plot_walk_forward(
        self,
        results: List[WindowResult],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualise walk-forward results.

        Produces a multi-panel figure with:
        * In-sample vs out-of-sample returns per window.
        * Sharpe ratios per window.
        * Cumulative out-of-sample return.

        Parameters
        ----------
        results:
            Output of :meth:`validate`.
        save_path:
            If given, save the figure to this path.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not results:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No results to plot", ha="center", va="center")
            return fig

        window_ids = [wr.window_id for wr in results]
        is_rets = [wr.in_sample_return for wr in results]
        os_rets = [wr.out_sample_return for wr in results]
        is_sharpes = [wr.in_sample_sharpe for wr in results]
        os_sharpes = [wr.out_sample_sharpe for wr in results]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # --- Returns comparison ---
        ax0 = axes[0]
        x = np.arange(len(window_ids))
        width = 0.35
        ax0.bar(x - width / 2, is_rets, width, label="In-Sample", color="steelblue", alpha=0.8)
        ax0.bar(x + width / 2, os_rets, width, label="Out-of-Sample", color="coral", alpha=0.8)
        ax0.set_xticks(x)
        ax0.set_xticklabels([f"W{wid}" for wid in window_ids])
        ax0.set_ylabel("Return")
        ax0.set_title("In-Sample vs Out-of-Sample Returns")
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        ax0.axhline(y=0, color="black", linewidth=0.8)

        # --- Sharpe comparison ---
        ax1 = axes[1]
        ax1.bar(x - width / 2, is_sharpes, width, label="In-Sample", color="steelblue", alpha=0.8)
        ax1.bar(x + width / 2, os_sharpes, width, label="Out-of-Sample", color="coral", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"W{wid}" for wid in window_ids])
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title("In-Sample vs Out-of-Sample Sharpe Ratios")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="black", linewidth=0.8)

        # --- Cumulative OOS return ---
        ax2 = axes[2]
        cum_return = np.cumprod(1.0 + np.array(os_rets)) - 1.0
        ax2.plot(x, cum_return, marker="o", color="darkgreen", linewidth=1.5)
        ax2.fill_between(x, cum_return, 0, alpha=0.15, color="green")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"W{wid}" for wid in window_ids])
        ax2.set_ylabel("Cumulative Return")
        ax2.set_title("Cumulative Out-of-Sample Return")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linewidth=0.8)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Walk-forward plot saved to %s", save_path)

        return fig

    # ------------------------------------------------------------------ #
    # Window creation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_windows(
        data: pd.DataFrame,
        n_windows: int,
        train_ratio: float,
    ) -> List[Tuple[Any, Any, Any, Any]]:
        """Create non-overlapping walk-forward (train, test) windows.

        The dataset is partitioned into *n_windows* consecutive segments.
        Within each segment the first ``train_ratio`` fraction of rows is
        the training set and the remainder is the test set.

        This is *anchored* walk-forward: each window is independent,
        ensuring no future data leakage.

        Parameters
        ----------
        data:
            DataFrame with a datetime index.
        n_windows:
            Number of walk-forward splits.
        train_ratio:
            Fraction of each window used for training.

        Returns
        -------
        list of (train_start, train_end, test_start, test_end) tuples
            Index values (timestamps) delineating each window.
        """
        if train_ratio <= 0.0 or train_ratio >= 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

        n = len(data)
        if n < n_windows * 2:
            raise ValueError(
                f"Not enough data ({n} rows) for {n_windows} windows; "
                f"need at least {n_windows * 2}"
            )

        window_size = n // n_windows
        windows: List[Tuple[Any, Any, Any, Any]] = []

        for i in range(n_windows):
            start_idx = i * window_size
            # Last window absorbs any remainder rows
            end_idx = (i + 1) * window_size if i < n_windows - 1 else n

            split_idx = start_idx + int((end_idx - start_idx) * train_ratio)
            # Ensure at least one row in test
            if split_idx >= end_idx:
                split_idx = end_idx - 1

            train_start = data.index[start_idx]
            train_end = data.index[split_idx - 1] if split_idx > start_idx else data.index[start_idx]
            test_start = data.index[split_idx]
            test_end = data.index[end_idx - 1]

            windows.append((train_start, train_end, test_start, test_end))

        return windows
