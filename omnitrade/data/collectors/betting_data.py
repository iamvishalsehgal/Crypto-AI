"""
Sports betting data collector using The Odds API.

Fetches real-time odds, historical results, and live scores for
configured sports. Converts American odds to implied probabilities.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from omnitrade.config.settings import BettingSettings, Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"


class BettingDataCollector:
    """Collect sports betting odds and historical results via The Odds API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._betting: BettingSettings = settings.betting
        self._api_key = self._betting.odds_api_key
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 min TTL for odds
        logger.info(
            "BettingDataCollector initialised — %d sports, %d books",
            len(self._betting.supported_sports),
            len(self._betting.sportsbooks),
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key and self._api_key != "")

    # ------------------------------------------------------------------
    # Odds fetching
    # ------------------------------------------------------------------

    def fetch_odds(
        self,
        sport_key: str,
        regions: str = "us",
        markets: str = "h2h",
    ) -> pd.DataFrame:
        """Fetch current odds for a sport.

        Args:
            sport_key: e.g. ``"soccer_epl"``, ``"basketball_nba"``.
            regions: ``"us"``, ``"uk"``, ``"eu"``, ``"au"``.
            markets: ``"h2h"`` (moneyline), ``"spreads"``, ``"totals"``.

        Returns:
            DataFrame with home_team, away_team, bookmaker, home_odds,
            away_odds, draw_odds, implied_home_pct, implied_away_pct,
            implied_draw_pct, commence_time.
        """
        cache_key = f"odds:{sport_key}:{regions}:{markets}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached["ts"] < self._cache_ttl:
            return cached["data"].copy()

        if not self.has_api_key:
            return self._mock_odds(sport_key)

        try:
            url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds"
            resp = requests.get(
                url,
                params={
                    "apiKey": self._api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "american",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info("Fetched odds for %s (%s requests remaining)", sport_key, remaining)
        except requests.RequestException as exc:
            logger.error("Odds API failed for %s: %s", sport_key, exc)
            return pd.DataFrame()

        if not data:
            logger.warning("No odds data returned for %s", sport_key)
            return pd.DataFrame()

        rows = []
        for match in data:
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            commence = match.get("commence_time", "")

            for book in match.get("bookmakers", []):
                book_name = book.get("title", "")
                for market_data in book.get("markets", []):
                    if market_data.get("key") != "h2h":
                        continue
                    outcomes = market_data.get("outcomes", [])
                    row = {
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": book_name,
                        "commence_time": commence,
                        "home_odds": 0,
                        "away_odds": 0,
                        "draw_odds": 0,
                    }
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if name == home:
                            row["home_odds"] = price
                        elif name == away:
                            row["away_odds"] = price
                        elif name.lower() == "draw":
                            row["draw_odds"] = price
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = self._add_implied_probabilities(df)
        self._cache[cache_key] = {"ts": time.time(), "data": df}
        return df

    def fetch_all_sports_odds(self, regions: str = "us") -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for sport in self._betting.supported_sports:
            try:
                df = self.fetch_odds(sport, regions=regions)
                results[sport] = df
                time.sleep(1.0)  # Rate limit: Odds API allows ~1 req/sec on free tier
            except Exception:
                logger.exception("Failed to fetch odds for %s", sport)
                results[sport] = pd.DataFrame()
        return results

    # ------------------------------------------------------------------
    # Historical results
    # ------------------------------------------------------------------

    def fetch_historical_results(
        self,
        sport_key: str,
        days_back: int = 365,
    ) -> pd.DataFrame:
        """Fetch historical match results for backtesting.

        Uses The Odds API historical endpoint (paid tier). Falls back to
        mock data when API key is missing or on free tier.
        """
        cache_key = f"history:{sport_key}:{days_back}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached["ts"] < 3600:
            return cached["data"].copy()

        if not self.has_api_key:
            return self._mock_history(sport_key, days_back)

        try:
            url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds-history"
            resp = requests.get(
                url,
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "dateFormat": "iso",
                    "daysBack": str(days_back),
                },
                timeout=30,
            )
            if resp.status_code == 422:
                logger.warning("Historical odds require paid API tier — using mock data")
                return self._mock_history(sport_key, days_back)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except requests.RequestException as exc:
            logger.error("Historical odds failed for %s: %s", sport_key, exc)
            return self._mock_history(sport_key, days_back)

        rows = []
        for entry in data:
            rows.append({
                "match_id": entry.get("id", ""),
                "home_team": entry.get("home_team", ""),
                "away_team": entry.get("away_team", ""),
                "commence_time": entry.get("commence_time", ""),
                "home_score": entry.get("scores", {}).get("home", None),
                "away_score": entry.get("scores", {}).get("away", None),
                "completed": entry.get("completed", False),
            })

        df = pd.DataFrame(rows)
        self._cache[cache_key] = {"ts": time.time(), "data": df}
        return df

    # ------------------------------------------------------------------
    # Live scores
    # ------------------------------------------------------------------

    def fetch_live_scores(self, sport_key: str) -> pd.DataFrame:
        if not self.has_api_key:
            return pd.DataFrame()
        try:
            url = f"{_ODDS_API_BASE}/sports/{sport_key}/scores"
            resp = requests.get(
                url,
                params={"apiKey": self._api_key, "daysFrom": "1"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Live scores failed for %s: %s", sport_key, exc)
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        rows = []
        for match in data:
            scores = match.get("scores", {})
            rows.append({
                "match_id": match.get("id", ""),
                "home_team": match.get("home_team", ""),
                "away_team": match.get("away_team", ""),
                "home_score": scores.get("home"),
                "away_score": scores.get("away"),
                "completed": match.get("completed", False),
                "last_update": match.get("last_update", ""),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Implied probability
    # ------------------------------------------------------------------

    @staticmethod
    def american_to_implied(odds: float) -> float:
        """Convert American odds to implied probability (0-1)."""
        if odds == 0:
            return 0.0
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["implied_home_pct"] = df["home_odds"].apply(
            BettingDataCollector.american_to_implied
        )
        df["implied_away_pct"] = df["away_odds"].apply(
            BettingDataCollector.american_to_implied
        )
        df["implied_draw_pct"] = df["draw_odds"].apply(
            BettingDataCollector.american_to_implied
        )
        return df

    def compute_implied_probabilities(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Convert odds to probabilities and remove the overround margin."""
        df = self._add_implied_probabilities(odds_df)
        if df.empty:
            return df

        total = df["implied_home_pct"] + df["implied_away_pct"]
        has_draw = (df["draw_odds"] > 0).any()
        if has_draw:
            total += df["implied_draw_pct"]

        margin = total.where(total > 0, 1.0)
        df["implied_home_pct"] = df["implied_home_pct"] / margin
        df["implied_away_pct"] = df["implied_away_pct"] / margin
        if has_draw:
            df["implied_draw_pct"] = df["implied_draw_pct"] / margin
        return df

    # ------------------------------------------------------------------
    # Mock data (for paper trading without API key)
    # ------------------------------------------------------------------

    def _mock_odds(self, sport_key: str) -> pd.DataFrame:
        sport_name = sport_key.replace("_", " ").title()
        now = datetime.now(timezone.utc).isoformat()
        return pd.DataFrame([
            {
                "home_team": f"{sport_name} Home A",
                "away_team": f"{sport_name} Away A",
                "bookmaker": "pinnacle",
                "commence_time": now,
                "home_odds": -150,
                "away_odds": 130,
                "draw_odds": 220,
                "implied_home_pct": 0.58,
                "implied_away_pct": 0.42,
                "implied_draw_pct": 0.31,
            },
            {
                "home_team": f"{sport_name} Home B",
                "away_team": f"{sport_name} Away B",
                "bookmaker": "pinnacle",
                "commence_time": now,
                "home_odds": 110,
                "away_odds": -130,
                "draw_odds": 240,
                "implied_home_pct": 0.45,
                "implied_away_pct": 0.55,
                "implied_draw_pct": 0.29,
            },
        ]).pipe(self._add_implied_probabilities)

    def _mock_history(self, sport_key: str, days_back: int) -> pd.DataFrame:
        sport_name = sport_key.replace("_", " ").title()
        import numpy as np
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=min(days_back, 200), freq="D")
        np.random.seed(hash(sport_key) % 2**31)
        return pd.DataFrame([
            {
                "match_id": f"mock_{sport_key}_{i}",
                "home_team": f"{sport_name} Home {i % 10}",
                "away_team": f"{sport_name} Away {i % 10}",
                "commence_time": d.isoformat(),
                "home_score": np.random.poisson(1.5),
                "away_score": np.random.poisson(1.2),
                "completed": True,
            }
            for i, d in enumerate(dates)
        ])
