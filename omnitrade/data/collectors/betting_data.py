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
_ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# sport_key → ESPN sport/league path for scoreboard endpoint
_ESPN_SPORT_PATH: Dict[str, str] = {
    "basketball_nba": "basketball/nba",
    "americanfootball_nfl": "football/nfl",
    "soccer_epl": "soccer/eng.1",
}


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
            logger.warning("No Odds API key — cannot fetch live odds for %s. Set BETTING_ODDS_API_KEY.", sport_key)
            return pd.DataFrame()

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
        """Fetch historical match results for backtesting and training.

        Primary: The Odds API historical endpoint (paid tier).
        Fallback: ESPN public API (free, no key — scores only).
        """
        cache_key = f"history:{sport_key}:{days_back}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached["ts"] < 3600:
            return cached["data"].copy()

        # ── Try The Odds API first ────────────────────────────────
        if self.has_api_key:
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
                    logger.info("Odds API history requires paid tier — using ESPN fallback")
                else:
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    if data:
                        df = self._parse_odds_history(data)
                        self._cache[cache_key] = {"ts": time.time(), "data": df}
                        return df
            except requests.RequestException as exc:
                logger.warning("Odds API history failed: %s — using ESPN fallback", exc)

        # ── ESPN fallback (real scores, free, no API key) ─────────
        df = self._fetch_historical_espn(sport_key, days_back)
        self._cache[cache_key] = {"ts": time.time(), "data": df}
        return df

    @staticmethod
    def _parse_odds_history(data: list) -> pd.DataFrame:
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
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # ESPN public API (free — scores + outcomes, no odds)
    # ------------------------------------------------------------------

    def _fetch_historical_espn(
        self, sport_key: str, days_back: int = 365
    ) -> pd.DataFrame:
        """Fetch real historical scores from ESPN public API.

        Samples ~1 date per week over the lookback window. ESPN does not
        require an API key. Returns completed games with scores.
        """
        espn_path = _ESPN_SPORT_PATH.get(sport_key)
        if not espn_path:
            logger.warning("No ESPN mapping for sport key %s", sport_key)
            return pd.DataFrame()

        today = datetime.now(timezone.utc).date()
        interval = max(days_back // 40, 3)  # ~40 samples, min 3-day spacing

        all_games: List[Dict[str, Any]] = []
        seen_matches: set = set()

        for offset in range(0, days_back, interval):
            date_str = pd.Timestamp(today - pd.Timedelta(days=offset)).strftime("%Y%m%d")
            try:
                url = f"{_ESPN_API_BASE}/{espn_path}/scoreboard"
                resp = requests.get(
                    url, params={"dates": date_str}, timeout=10,
                )
                time.sleep(0.05)  # Be polite to ESPN's servers
                if resp.status_code != 200:
                    continue
                data = resp.json()
                day_games = self._parse_espn_scoreboard(data, date_str)
                for g in day_games:
                    mid = g.get("match_id", "")
                    if mid and mid not in seen_matches:
                        seen_matches.add(mid)
                        all_games.append(g)
            except Exception:
                continue

        if not all_games:
            logger.warning("ESPN returned no historical games for %s (%d day lookback)", sport_key, days_back)
            return pd.DataFrame()

        df = pd.DataFrame(all_games)
        df["commence_time"] = pd.to_datetime(df["commence_time"], errors="coerce")
        df = df.sort_values("commence_time", ascending=False).reset_index(drop=True)
        logger.info(
            "ESPN: %d historical games for %s over %d days",
            len(df), sport_key, days_back,
        )
        return df

    @staticmethod
    def _parse_espn_scoreboard(data: dict, date_str: str) -> List[Dict[str, Any]]:
        games = []
        for event in data.get("events", []):
            comps = event.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            status = comps.get("status", {})

            # Only include completed games (soccer uses STATUS_FULL_TIME)
            status_name = status.get("type", {}).get("name", "")
            if status_name not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                continue

            home = away = None
            home_score = away_score = None
            for t in competitors:
                if t.get("homeAway") == "home":
                    home = t.get("team", {}).get("displayName", "")
                    home_score = int(t.get("score", 0)) if t.get("score") else None
                else:
                    away = t.get("team", {}).get("displayName", "")
                    away_score = int(t.get("score", 0)) if t.get("score") else None

            if not home or not away:
                continue

            commence = comps.get("date", date_str)
            games.append({
                "match_id": event.get("id", ""),
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "home_score": home_score,
                "away_score": away_score,
                "completed": True,
                "source": "espn",
            })
        return games

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
