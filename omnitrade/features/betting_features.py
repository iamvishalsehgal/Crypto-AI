"""
Betting-specific feature engineering.

Converts raw odds and historical data into predictive features:
- Implied probability features (home/away/draw, margin-removed)
- Odds movement / steam move detection
- ELO rating system per team
- Poisson expected goals model
- Recent form features
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class BettingFeatures:
    """Compute features from odds data for betting signal generation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._elo_ratings: Dict[str, float] = {}
        self._initial_elo = 1500.0
        self._k_factor = 32.0

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def compute_all(
        self,
        odds_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute full feature matrix from odds and historical data.

        Args:
            odds_df: From BettingDataCollector.fetch_odds().
            historical_df: From BettingDataCollector.fetch_historical_results().

        Returns:
            DataFrame with one row per match, columns of betting features.
        """
        if odds_df.empty:
            logger.warning("Empty odds DataFrame — no features computed")
            return pd.DataFrame()

        df = odds_df.copy()

        df = self._compute_implied_features(df)
        df = self._compute_odds_movement(df)
        df = self._compute_elo_features(df, historical_df)
        df = self._compute_poisson_features(df, historical_df)
        df = self._compute_form_features(df, historical_df)
        df = self._compute_time_features(df)

        df = df.dropna()

        logger.info("Betting features computed: %d rows x %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Implied probability features
    # ------------------------------------------------------------------

    def _compute_implied_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probability-based features from odds."""
        df = df.copy()

        if "implied_home_pct" in df.columns:
            df["prob_home"] = df["implied_home_pct"]
        else:
            df["prob_home"] = 0.33

        if "implied_away_pct" in df.columns:
            df["prob_away"] = df["implied_away_pct"]
        else:
            df["prob_away"] = 0.33

        if "implied_draw_pct" in df.columns:
            df["prob_draw"] = df["implied_draw_pct"]
        else:
            df["prob_draw"] = 0.0

        df["prob_spread"] = df["prob_home"] - df["prob_away"]
        df["prob_favorite"] = df[["prob_home", "prob_away", "prob_draw"]].max(axis=1)
        df["prob_underdog"] = df[["prob_home", "prob_away", "prob_draw"]].min(axis=1)
        df["is_close_match"] = (df["prob_spread"].abs() < 0.10).astype(float)

        return df

    # ------------------------------------------------------------------
    # Odds movement
    # ------------------------------------------------------------------

    def _compute_odds_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect steam moves — sharp odds shifts across sportsbooks.

        When multiple books are available, compare each book's implied
        probability to the mean. Large deviations suggest sharp action.
        """
        df = df.copy()

        prob_cols = ["prob_home", "prob_away", "prob_draw"]
        for col in prob_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                if mean_val > 0:
                    df[f"{col}_deviation"] = (df[col] - mean_val) / mean_val
                else:
                    df[f"{col}_deviation"] = 0.0

        df["steam_signal"] = 0.0
        for col in prob_cols:
            dev_col = f"{col}_deviation"
            if dev_col in df.columns:
                steam = (df[dev_col].abs() > 0.10).astype(float)
                df["steam_signal"] = df["steam_signal"] + steam

        df["steam_signal"] = (df["steam_signal"] > 0).astype(float)
        return df

    # ------------------------------------------------------------------
    # ELO ratings
    # ------------------------------------------------------------------

    def _compute_elo_features(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if historical_df is None or historical_df.empty:
            df["elo_home"] = self._initial_elo
            df["elo_away"] = self._initial_elo
            df["elo_delta"] = 0.0
            df["elo_home_win_prob"] = 0.5
            return df

        self._compute_elo_ratings(historical_df)

        home_elos = []
        away_elos = []
        for _, row in df.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            home_elos.append(self._elo_ratings.get(home, self._initial_elo))
            away_elos.append(self._elo_ratings.get(away, self._initial_elo))

        df["elo_home"] = home_elos
        df["elo_away"] = away_elos
        df["elo_delta"] = df["elo_home"] - df["elo_away"]
        df["elo_home_win_prob"] = 1.0 / (
            1.0 + 10.0 ** (-df["elo_delta"] / 400.0)
        )
        return df

    def _compute_elo_ratings(self, historical_df: pd.DataFrame) -> Dict[str, float]:
        """Iterative ELO calculation from historical match results."""
        if historical_df.empty or "home_score" not in historical_df.columns:
            return self._elo_ratings

        elo = self._elo_ratings.copy() if self._elo_ratings else {}

        for _, row in historical_df.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            home_score = row.get("home_score")
            away_score = row.get("away_score")

            if not home or not away:
                continue
            if home_score is None or away_score is None:
                continue

            r_home = elo.get(home, self._initial_elo)
            r_away = elo.get(away, self._initial_elo)

            e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))

            if home_score > away_score:
                s_home = 1.0
            elif home_score < away_score:
                s_home = 0.0
            else:
                s_home = 0.5

            delta = self._k_factor * (s_home - e_home)
            elo[home] = r_home + delta
            elo[away] = r_away - delta

        self._elo_ratings = elo
        return elo

    # ------------------------------------------------------------------
    # Poisson expected goals (soccer)
    # ------------------------------------------------------------------

    def _compute_poisson_features(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if historical_df is None or historical_df.empty:
            df["poisson_home_exp"] = 1.5
            df["poisson_away_exp"] = 1.2
            df["poisson_home_win"] = 0.40
            df["poisson_draw"] = 0.27
            df["poisson_away_win"] = 0.33
            return df

        league_avg_home = 1.5
        league_avg_away = 1.2

        if not historical_df.empty and "home_score" in historical_df.columns:
            home_scores = historical_df["home_score"].dropna()
            away_scores = historical_df["away_score"].dropna()
            if not home_scores.empty:
                league_avg_home = float(home_scores.mean())
            if not away_scores.empty:
                league_avg_away = float(away_scores.mean())

        poisson_rows = []
        for _, _row in df.iterrows():
            home_exp = league_avg_home
            away_exp = league_avg_away

            if "prob_home" in df.columns and "prob_away" in df.columns:
                fav_bias = 1.0 + max(0, _row.get("prob_home", 0.5) - 0.5)
                home_exp = league_avg_home * fav_bias
                away_exp = league_avg_away / fav_bias

            probs = self._poisson_match_probs(home_exp, away_exp)
            poisson_rows.append(probs)

        poisson_df = pd.DataFrame(poisson_rows, index=df.index)
        return pd.concat([df, poisson_df], axis=1)

    @staticmethod
    def _poisson_match_probs(
        home_exp: float,
        away_exp: float,
        max_goals: int = 10,
    ) -> Dict[str, float]:
        """Compute home win / draw / away win probabilities via Poisson."""
        from scipy.stats import poisson

        home_probs = [poisson.pmf(i, home_exp) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, away_exp) for i in range(max_goals + 1)]

        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i, hp in enumerate(home_probs):
            for j, ap in enumerate(away_probs):
                p = hp * ap
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

        return {
            "poisson_home_exp": home_exp,
            "poisson_away_exp": away_exp,
            "poisson_home_win": round(home_win, 4),
            "poisson_draw": round(draw, 4),
            "poisson_away_win": round(away_win, 4),
        }

    # ------------------------------------------------------------------
    # Form features
    # ------------------------------------------------------------------

    def _compute_form_features(
        self,
        df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if historical_df is None or historical_df.empty:
            df["home_form_pct"] = 0.5
            df["away_form_pct"] = 0.5
            return df

        last_n = 5
        home_forms: Dict[str, float] = {}
        away_forms: Dict[str, float] = {}

        if not historical_df.empty and "home_score" in historical_df.columns:
            historical_df = historical_df.sort_values("commence_time")
            for _, row in historical_df.iterrows():
                home = row.get("home_team", "")
                away = row.get("away_team", "")
                hs = row.get("home_score")
                aes = row.get("away_score")

                if hs is None or aes is None:
                    continue

                home_results = historical_df[
                    (historical_df["home_team"] == home) | (historical_df["away_team"] == home)
                ].tail(last_n)

                if len(home_results) > 0:
                    pts = 0
                    for _, r in home_results.iterrows():
                        if r["home_team"] == home and r["home_score"] > r["away_score"]:
                            pts += 1
                        elif r["away_team"] == home and r["away_score"] > r["home_score"]:
                            pts += 1
                    home_forms[home] = pts / len(home_results)

        df["home_form_pct"] = df["home_team"].map(home_forms).fillna(0.5)
        df["away_form_pct"] = df["away_team"].map(away_forms).fillna(0.5)
        df["form_delta"] = df["home_form_pct"] - df["away_form_pct"]
        return df

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "commence_time" in df.columns:
            try:
                times = pd.to_datetime(df["commence_time"], utc=True)
                df["hours_to_match"] = (
                    (times - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 3600.0
                )
                df["match_hour"] = times.dt.hour.astype(float)
                df["match_dayofweek"] = times.dt.dayofweek.astype(float)
            except Exception:
                df["hours_to_match"] = 24.0
                df["match_hour"] = 15.0
                df["match_dayofweek"] = 3.0
        else:
            df["hours_to_match"] = 24.0
            df["match_hour"] = 15.0
            df["match_dayofweek"] = 3.0
        return df
