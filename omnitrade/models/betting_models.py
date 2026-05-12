"""
Betting-specific models: value detection, Poisson goal prediction, Kelly staking.

- ValueBettingModel: logistic regression comparing model probability to
  implied probability. Signal = BACK when edge > threshold.
- PoissonModel: expected goals → win/draw/loss probability distribution.
- KellyStakingModel: optimal stake sizing using Kelly criterion.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from omnitrade.config.asset_types import BACK, LAY, PASS, UnifiedSignal, AssetType
from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class ValueBettingModel:
    """Detects value bets by comparing model probability to implied odds.

    Uses logistic regression on historical features vs outcomes. A value
    bet exists when model_probability > implied_probability + min_edge.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._betting = settings.betting
        self._model: Any = None
        self._trained = False
        self._feature_names: list = []

    def train(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
    ) -> Dict[str, Any]:
        """Train the probability estimator on historical data.

        Args:
            features: Feature matrix from BettingFeatures.compute_all().
            outcomes: Binary outcomes (1 = home win, 0 = otherwise).

        Returns:
            Dict with training metrics.
        """
        if features.empty or outcomes.empty or len(features) < 20:
            logger.warning("Insufficient data for betting model training (%d rows)", len(features))
            return {"status": "insufficient_data", "rows": len(features)}

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        numeric = features.select_dtypes(include=[np.number])
        self._feature_names = list(numeric.columns)

        if len(numeric) < 20:
            return {"status": "insufficient_numeric_features", "rows": len(numeric)}

        self._model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        try:
            self._model.fit(numeric, outcomes)
            self._trained = True

            cv_scores = cross_val_score(self._model, numeric, outcomes, cv=3, scoring="roc_auc")
            train_score = self._model.score(numeric, outcomes)

            logger.info(
                "Betting model trained — AUC=%.3f (±%.3f), accuracy=%.3f",
                cv_scores.mean(), cv_scores.std(), train_score,
            )

            return {
                "status": "trained",
                "rows": len(features),
                "cv_auc_mean": float(cv_scores.mean()),
                "cv_auc_std": float(cv_scores.std()),
                "train_accuracy": float(train_score),
                "feature_count": len(self._feature_names),
            }
        except Exception as exc:
            logger.error("Betting model training failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def predict(self, features: pd.DataFrame) -> UnifiedSignal:
        """Generate a betting signal from features.

        Returns:
            UnifiedSignal with side=BACK/LAY/PASS based on edge detection.
        """
        if not self._trained or self._model is None:
            return UnifiedSignal(
                asset_type=AssetType.BET,
                symbol="unknown",
                side=PASS,
                confidence=0.0,
            )

        row = features.iloc[[-1]] if len(features) > 1 else features
        numeric = row.select_dtypes(include=[np.number])

        missing = set(self._feature_names) - set(numeric.columns)
        for col in missing:
            numeric[col] = 0.0
        numeric = numeric[self._feature_names]

        if not np.isfinite(numeric.values).all():
            logger.error("Betting model received NaN/Inf features — returning PASS")
            return UnifiedSignal(
                asset_type=AssetType.BET, symbol="unknown", side=PASS, confidence=0.0,
                metadata={"error": "NaN/Inf in input features"},
            )

        try:
            proba = self._model.predict_proba(numeric)
            if hasattr(proba, "values"):
                proba = proba.values
            model_prob = float(proba[0][1])  # probability of home win (class 1)
        except Exception as exc:
            logger.error("Betting model prediction failed: %s — returning PASS", exc)
            return UnifiedSignal(
                asset_type=AssetType.BET, symbol="unknown", side=PASS, confidence=0.0,
                metadata={"error": str(exc)},
            )

        implied_prob = features.get("prob_home", pd.Series([0.5])).iloc[-1]
        implied_prob = float(implied_prob) if not np.isnan(implied_prob) else 0.5

        edge = model_prob - implied_prob
        min_edge = self._betting.min_edge_pct

        if edge > min_edge:
            side = BACK
            confidence = min(edge / (1.0 - implied_prob), 1.0) if implied_prob < 1.0 else 0.0
        elif -edge > min_edge:
            side = LAY
            confidence = min(-edge / implied_prob, 1.0) if implied_prob > 0 else 0.0
        else:
            side = PASS
            confidence = 0.0

        symbol = features.get("home_team", pd.Series(["unknown"])).iloc[-1] if "home_team" in features.columns else "unknown"
        away = features.get("away_team", pd.Series([""])).iloc[-1] if "away_team" in features.columns else ""

        return UnifiedSignal(
            asset_type=AssetType.BET,
            symbol=f"{symbol} vs {away}",
            side=side,
            confidence=round(confidence, 3),
            amount=0.0,
            price=float(implied_prob),
            metadata={
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "min_edge": min_edge,
            },
        )

    @property
    def is_trained(self) -> bool:
        return self._trained


class PoissonModel:
    """Poisson goal expectation model for soccer match prediction."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._home_attack: Dict[str, float] = {}
        self._away_defense: Dict[str, float] = {}
        self._league_avg: float = 1.4
        self._trained = False

    def train(self, historical_df: pd.DataFrame) -> Dict[str, Any]:
        if historical_df.empty or "home_score" not in historical_df.columns:
            return {"status": "no_data"}

        scores = historical_df[["home_team", "away_team", "home_score", "away_score"]].dropna()
        if len(scores) < 10:
            return {"status": "insufficient_data"}

        self._league_avg = float(scores["home_score"].mean() + scores["away_score"].mean()) / 2.0

        teams: set = set(scores["home_team"].unique()) | set(scores["away_team"].unique())
        for team in teams:
            home_matches = scores[scores["home_team"] == team]
            away_matches = scores[scores["away_team"] == team]

            home_goals = home_matches["home_score"].mean() if not home_matches.empty else self._league_avg
            away_conceded = away_matches["home_score"].mean() if not away_matches.empty else self._league_avg

            self._home_attack[team] = home_goals / self._league_avg if self._league_avg > 0 else 1.0
            self._away_defense[team] = away_conceded / self._league_avg if self._league_avg > 0 else 1.0

        self._trained = True
        logger.info(
            "Poisson model trained — %d teams, league_avg=%.2f",
            len(teams), self._league_avg,
        )
        return {"status": "trained", "teams": len(teams), "league_avg": self._league_avg}

    def predict(
        self,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float, float, float]:
        """Predict match outcome probabilities.

        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob, home_expected_goals).
        """
        h_attack = self._home_attack.get(home_team, 1.0)
        a_defense = self._away_defense.get(away_team, 1.0)
        a_attack = self._home_attack.get(away_team, 1.0)
        h_defense = self._away_defense.get(home_team, 1.0)

        home_exp = self._league_avg * h_attack * a_defense
        away_exp = self._league_avg * a_attack * h_defense

        max_goals = 10
        from scipy.stats import poisson
        home_p = [poisson.pmf(i, home_exp) for i in range(max_goals + 1)]
        away_p = [poisson.pmf(i, away_exp) for i in range(max_goals + 1)]

        home_win = sum(hp * sum(ap for j, ap in enumerate(away_p) if i > j) for i, hp in enumerate(home_p))
        draw = sum(hp * away_p[i] for i, hp in enumerate(home_p) if i <= max_goals)
        away_win = 1.0 - home_win - draw

        return home_win, draw, away_win, home_exp

    @property
    def is_trained(self) -> bool:
        return self._trained


class KellyStakingModel:
    """Kelly criterion stake sizing for betting.

    f* = (bp - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._betting = settings.betting

    def compute_stake(
        self,
        bankroll: float,
        model_prob: float,
        implied_prob: float,
        odds_american: float = -110,
        kelly_fraction: Optional[float] = None,
    ) -> float:
        """Compute optimal stake using fractional Kelly.

        Args:
            bankroll: Current total bankroll.
            model_prob: Model-estimated true probability (0-1).
            implied_prob: Market-implied probability (0-1).
            odds_american: American odds offered.
            kelly_fraction: Override the configured Kelly fraction
                (None uses the instance default).

        Returns:
            Recommended stake in USD.
        """
        decimal_odds = self._american_to_decimal(odds_american)
        b = decimal_odds - 1.0

        if b <= 0:
            return 0.0

        p = model_prob
        q = 1.0 - p

        f_star = (b * p - q) / b

        if f_star <= 0:
            return 0.0

        frac = kelly_fraction if kelly_fraction is not None else self._betting.kelly_fraction
        stake = bankroll * f_star * frac

        max_stake = bankroll * self._betting.max_stake_pct
        return min(stake, max_stake)

    def validate_stake(
        self,
        suggested_stake: float,
        bankroll: float,
        daily_stakes_count: int,
    ) -> Tuple[float, Optional[str]]:
        """Validate and cap a suggested stake.

        Returns:
            Tuple of (adjusted_stake, rejection_reason_or_None).
        """
        if suggested_stake <= 0:
            return 0.0, "Kelly fraction <= 0"

        if bankroll <= 0:
            return 0.0, "Bankroll exhausted"

        max_pct = self._betting.max_stake_pct
        if suggested_stake > bankroll * max_pct:
            suggested_stake = bankroll * max_pct

        if daily_stakes_count >= self._betting.max_daily_stakes:
            return 0.0, f"Max daily stakes ({self._betting.max_daily_stakes}) reached"

        if suggested_stake > bankroll:
            return 0.0, f"Stake (${suggested_stake:.2f}) exceeds bankroll (${bankroll:.2f})"

        return suggested_stake, None

    @staticmethod
    def _american_to_decimal(odds: float) -> float:
        if odds > 0:
            return 1.0 + odds / 100.0
        elif odds < 0:
            return 1.0 + 100.0 / abs(odds)
        return 2.0
