"""
Backtest overfitting detection.

Implements statistical tests to detect whether a trading strategy's backtest
performance is likely due to overfitting rather than genuine alpha.  Based on
the combinatorial purged cross-validation (CPCV) framework and deflated Sharpe
ratio methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from crypto_bot.config.settings import Settings, settings as _default_settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OverfittingReport:
    """Container for overfitting detection results."""

    overfitting_probability: float
    deflated_sharpe: float
    is_overfitted: bool
    in_sample_sharpe: float
    out_sample_sharpe: float
    performance_degradation: float
    recommendation: str


class OverfittingDetector:
    """Detect backtest overfitting using statistical hypothesis testing.

    The core idea: if a strategy was found by searching over many
    configurations, its reported Sharpe ratio is inflated.  We estimate how
    likely it is that the observed out-of-sample performance is no better than
    chance after accounting for the search process.

    Args:
        settings: Bot settings (uses ``backtesting.overfitting_threshold``).
        threshold: Override for the maximum acceptable overfitting probability.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self._settings = settings or _default_settings
        self.threshold = (
            threshold
            if threshold is not None
            else self._settings.backtesting.overfitting_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_overfitting_probability(
        self,
        in_sample_returns: np.ndarray,
        out_sample_returns: np.ndarray,
        n_trials: int = 1000,
    ) -> float:
        """Estimate the probability that backtest performance is due to overfitting.

        Uses a permutation-based approach: we randomly shuffle the combined
        return series many times and measure how often a random split yields
        an in-sample vs out-of-sample gap as large as the one observed.

        Args:
            in_sample_returns: Returns achieved on the training period.
            out_sample_returns: Returns achieved on the held-out period.
            n_trials: Number of permutation trials.

        Returns:
            Probability of overfitting in ``[0, 1]``.
        """
        is_sharpe = self._annualised_sharpe(in_sample_returns)
        oos_sharpe = self._annualised_sharpe(out_sample_returns)
        observed_gap = is_sharpe - oos_sharpe

        combined = np.concatenate([in_sample_returns, out_sample_returns])
        split_idx = len(in_sample_returns)
        count_worse = 0

        rng = np.random.default_rng(42)
        for _ in range(n_trials):
            perm = rng.permutation(combined)
            perm_is = self._annualised_sharpe(perm[:split_idx])
            perm_oos = self._annualised_sharpe(perm[split_idx:])
            if (perm_is - perm_oos) >= observed_gap:
                count_worse += 1

        probability = count_worse / n_trials
        logger.info(
            "Overfitting probability: %.3f (IS Sharpe=%.2f, OOS Sharpe=%.2f, gap=%.2f)",
            probability,
            is_sharpe,
            oos_sharpe,
            observed_gap,
        )
        return probability

    def combinatorial_purged_cv(
        self,
        data: np.ndarray,
        n_splits: int = 10,
        embargo_pct: float = 0.01,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged cross-validation splits.

        Each fold is used once as a test set while the rest form the training
        set.  An *embargo* region is removed between training and test folds to
        prevent information leakage due to serial correlation.

        Args:
            data: 1-D array (e.g. returns or feature matrix row count).
            n_splits: Number of folds.
            embargo_pct: Fraction of data to purge between train/test.

        Returns:
            List of ``(train_indices, test_indices)`` tuples.
        """
        n = len(data)
        embargo_size = max(1, int(n * embargo_pct))
        fold_size = n // n_splits
        indices = np.arange(n)
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min(test_start + fold_size, n)
            test_idx = indices[test_start:test_end]

            # Purge: remove embargo region around test fold
            purge_start = max(0, test_start - embargo_size)
            purge_end = min(n, test_end + embargo_size)
            train_idx = np.concatenate([
                indices[:purge_start],
                indices[purge_end:],
            ])

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        logger.debug("Created %d purged CV splits (embargo=%d rows)", len(splits), embargo_size)
        return splits

    def deflated_sharpe_ratio(
        self,
        sharpe_observed: float,
        sharpe_std: float,
        n_trials: int,
        skew: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Compute the deflated Sharpe ratio.

        Adjusts the observed Sharpe ratio for the number of strategy variants
        tested (multiple-testing bias) and non-normality of returns.

        Reference: Bailey & Lopez de Prado (2014).

        Args:
            sharpe_observed: Best observed Sharpe ratio.
            sharpe_std: Standard deviation of Sharpe ratios across trials.
            n_trials: Number of strategy configurations tested.
            skew: Skewness of the returns distribution.
            kurtosis: Kurtosis of the returns distribution.

        Returns:
            Deflated Sharpe ratio (probability that the observed Sharpe is
            genuine given the search process).
        """
        if sharpe_std <= 0 or n_trials <= 0:
            return 0.0

        # Expected maximum Sharpe under the null (Euler-Mascheroni correction)
        euler_mascheroni = 0.5772156649
        e_max_sharpe = sharpe_std * (
            (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )

        # Non-normality correction
        correction = 1.0 - skew * sharpe_observed + ((kurtosis - 3.0) / 4.0) * sharpe_observed ** 2
        if correction <= 0:
            correction = 1e-6

        test_stat = (sharpe_observed - e_max_sharpe) / (sharpe_std * np.sqrt(correction))
        deflated = float(stats.norm.cdf(test_stat))

        logger.info(
            "Deflated Sharpe: %.4f (observed=%.2f, E[max]=%.2f, trials=%d)",
            deflated,
            sharpe_observed,
            e_max_sharpe,
            n_trials,
        )
        return deflated

    def is_overfitted(self, probability: float) -> bool:
        """Return ``True`` if the overfitting probability exceeds the threshold."""
        return probability > self.threshold

    def detect_and_report(
        self,
        in_sample_returns: np.ndarray,
        out_sample_returns: np.ndarray,
        n_trials: int = 1000,
        n_strategy_variants: int = 1,
    ) -> OverfittingReport:
        """Run the full overfitting detection pipeline.

        Args:
            in_sample_returns: In-sample return series.
            out_sample_returns: Out-of-sample return series.
            n_trials: Permutation trials for probability estimation.
            n_strategy_variants: Number of strategy variants tried during
                the model selection process.

        Returns:
            :class:`OverfittingReport` with all metrics and a recommendation.
        """
        probability = self.estimate_overfitting_probability(
            in_sample_returns, out_sample_returns, n_trials
        )

        is_sharpe = self._annualised_sharpe(in_sample_returns)
        oos_sharpe = self._annualised_sharpe(out_sample_returns)

        sharpe_std = np.std([is_sharpe, oos_sharpe]) if is_sharpe != oos_sharpe else 0.5
        deflated = self.deflated_sharpe_ratio(
            sharpe_observed=is_sharpe,
            sharpe_std=sharpe_std,
            n_trials=max(n_strategy_variants, 1),
            skew=float(stats.skew(in_sample_returns)) if len(in_sample_returns) > 2 else 0.0,
            kurtosis=float(stats.kurtosis(in_sample_returns, fisher=False)) if len(in_sample_returns) > 3 else 3.0,
        )

        degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 1e-6)
        overfitted = self.is_overfitted(probability)

        if overfitted:
            recommendation = (
                f"REJECT: overfitting probability {probability:.1%} exceeds "
                f"threshold {self.threshold:.0%}.  Performance degradation "
                f"{degradation:.1%}.  Retrain with regularisation or reduce "
                f"strategy complexity."
            )
        elif probability > self.threshold * 0.7:
            recommendation = (
                f"CAUTION: overfitting probability {probability:.1%} is near "
                f"threshold.  Monitor out-of-sample performance closely."
            )
        else:
            recommendation = (
                f"ACCEPT: overfitting probability {probability:.1%} is within "
                f"acceptable range.  Strategy appears robust."
            )

        report = OverfittingReport(
            overfitting_probability=probability,
            deflated_sharpe=deflated,
            is_overfitted=overfitted,
            in_sample_sharpe=is_sharpe,
            out_sample_sharpe=oos_sharpe,
            performance_degradation=degradation,
            recommendation=recommendation,
        )

        logger.info("Overfitting report: %s", recommendation)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _annualised_sharpe(
        returns: np.ndarray,
        periods_per_year: int = 365,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Compute annualised Sharpe ratio from a return series."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / periods_per_year
        mean = np.mean(excess)
        std = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(mean / std * np.sqrt(periods_per_year))
