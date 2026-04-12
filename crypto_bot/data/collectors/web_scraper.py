"""
Web scraper for non-API crypto data sources.

Scrapes the Fear & Greed Index, coin rankings, and upcoming
blockchain events from public web pages using aiohttp and
BeautifulSoup.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# User-Agent rotation pool
# ---------------------------------------------------------------------------
_USER_AGENTS: List[str] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
]

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
_FEAR_GREED_API = "https://api.alternative.me/fng/"
_FEAR_GREED_URL = "https://alternative.me/crypto/fear-and-greed-index/"
_COINGECKO_URL = "https://www.coingecko.com"
_COINMARKETCAL_URL = "https://coinmarketcal.com/en/"

# ---------------------------------------------------------------------------
# Rate-limit config
# ---------------------------------------------------------------------------
_MIN_DELAY_S = 1.0
_MAX_DELAY_S = 2.0


class WebScraper:
    """Scrape crypto data from web pages that lack formal APIs."""

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, settings: Settings) -> None:
        """Configure the scraper.

        Parameters
        ----------
        settings:
            Application settings.  Optionally provides::

                SCRAPER_MIN_DELAY  -- minimum delay between requests (s).
                SCRAPER_MAX_DELAY  -- maximum delay between requests (s).
                SCRAPER_TIMEOUT    -- HTTP timeout in seconds.
        """
        self._settings = settings
        self._min_delay = float(
            getattr(settings, "SCRAPER_MIN_DELAY", _MIN_DELAY_S)
        )
        self._max_delay = float(
            getattr(settings, "SCRAPER_MAX_DELAY", _MAX_DELAY_S)
        )
        self._timeout = float(getattr(settings, "SCRAPER_TIMEOUT", 30))

        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0.0

        logger.info("WebScraper initialised (delay %.1f-%.1fs)", self._min_delay, self._max_delay)

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ #
    # Core HTTP helper
    # ------------------------------------------------------------------ #

    async def _get_page(self, url: str) -> str:
        """Fetch *url* and return the response body as a string.

        Applies rate limiting (random delay between requests) and
        rotates the User-Agent header on every call.
        """
        # --- rate limit ---
        now = time.monotonic()
        elapsed = now - self._last_request_time
        delay = random.uniform(self._min_delay, self._max_delay)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

        session = await self._get_session()
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                text = await resp.text()
                self._last_request_time = time.monotonic()
                return text
        except aiohttp.ClientResponseError as exc:
            logger.error("HTTP %s fetching %s: %s", exc.status, url, exc.message)
            raise
        except aiohttp.ClientError as exc:
            logger.error("Request error for %s: %s", url, exc)
            raise

    async def _get_json(self, url: str, params: Optional[dict] = None) -> Any:
        """Fetch *url* and return decoded JSON."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        delay = random.uniform(self._min_delay, self._max_delay)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

        session = await self._get_session()
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "application/json",
        }

        try:
            async with session.get(url, headers=headers, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._last_request_time = time.monotonic()
                return data
        except aiohttp.ClientResponseError as exc:
            logger.error("HTTP %s fetching JSON %s: %s", exc.status, url, exc.message)
            raise
        except aiohttp.ClientError as exc:
            logger.error("Request error for %s: %s", url, exc)
            raise

    # ------------------------------------------------------------------ #
    # Fear & Greed Index
    # ------------------------------------------------------------------ #

    async def scrape_fear_greed_index(self) -> Dict[str, Any]:
        """Return the current Crypto Fear & Greed Index.

        Tries the alternative.me JSON API first, then falls back to
        scraping the HTML page.

        Returns
        -------
        dict
            ``value`` (int 0-100), ``classification`` (str),
            ``timestamp`` (str ISO-8601).
        """
        result: Dict[str, Any] = {
            "value": None,
            "classification": None,
            "timestamp": None,
        }

        # --- Try JSON API ---
        try:
            data = await self._get_json(_FEAR_GREED_API, params={"limit": "1"})
            entry = data.get("data", [{}])[0]
            result["value"] = int(entry.get("value", 0))
            result["classification"] = entry.get("value_classification", "")
            result["timestamp"] = entry.get("timestamp", "")
            logger.info(
                "Fear & Greed Index: %s (%s)",
                result["value"],
                result["classification"],
            )
            return result
        except Exception:
            logger.warning("Fear & Greed JSON API failed, trying HTML fallback")

        # --- HTML fallback ---
        try:
            html = await self._get_page(_FEAR_GREED_URL)
            soup = BeautifulSoup(html, "html.parser")

            # The index page displays the value in a prominent div
            value_div = soup.find("div", class_="fng-circle")
            if value_div:
                result["value"] = int(value_div.get_text(strip=True))

            # Classification is usually in a nearby element
            class_el = soup.find("div", class_="status")
            if class_el:
                result["classification"] = class_el.get_text(strip=True)

            if result["value"] is not None and not result["classification"]:
                result["classification"] = self._classify_fear_greed(
                    result["value"]
                )

        except Exception:
            logger.exception("Failed to scrape Fear & Greed Index")

        return result

    # ------------------------------------------------------------------ #
    # Crypto rankings
    # ------------------------------------------------------------------ #

    async def scrape_crypto_rankings(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Scrape top cryptocurrency rankings from CoinGecko.

        Parameters
        ----------
        limit:
            Number of top coins to return.

        Returns
        -------
        list[dict]
            Each dict contains ``rank``, ``name``, ``symbol``,
            ``price``, ``market_cap``, ``volume``, ``change_24h``.
        """
        rankings: List[Dict[str, Any]] = []

        try:
            # Use CoinGecko's public API endpoint (no key required)
            api_url = f"{_COINGECKO_URL}/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": str(min(limit, 250)),
                "page": "1",
                "sparkline": "false",
            }

            data = await self._get_json(api_url, params=params)

            if isinstance(data, list):
                for idx, coin in enumerate(data[:limit], start=1):
                    rankings.append(
                        {
                            "rank": idx,
                            "name": coin.get("name", ""),
                            "symbol": coin.get("symbol", "").upper(),
                            "price": coin.get("current_price"),
                            "market_cap": coin.get("market_cap"),
                            "volume": coin.get("total_volume"),
                            "change_24h": coin.get(
                                "price_change_percentage_24h"
                            ),
                        }
                    )

            logger.info("Fetched rankings for top %d coins", len(rankings))
            return rankings

        except Exception:
            logger.warning("CoinGecko API failed, falling back to HTML scrape")

        # --- HTML fallback ---
        try:
            html = await self._get_page(_COINGECKO_URL)
            soup = BeautifulSoup(html, "html.parser")

            table = soup.find("table")
            if table is None:
                logger.warning("Could not locate rankings table on page")
                return rankings

            rows = table.find("tbody")
            if rows is None:
                return rankings

            for row in rows.find_all("tr")[:limit]:
                cells = row.find_all("td")
                if len(cells) < 6:
                    continue

                rank_text = cells[0].get_text(strip=True)
                name_text = cells[1].get_text(strip=True)
                price_text = cells[2].get_text(strip=True)
                change_text = cells[3].get_text(strip=True)
                volume_text = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                mcap_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                rankings.append(
                    {
                        "rank": self._parse_int(rank_text),
                        "name": name_text,
                        "symbol": "",
                        "price": self._parse_money(price_text),
                        "market_cap": self._parse_money(mcap_text),
                        "volume": self._parse_money(volume_text),
                        "change_24h": self._parse_pct(change_text),
                    }
                )

        except Exception:
            logger.exception("Failed to scrape crypto rankings")

        return rankings

    # ------------------------------------------------------------------ #
    # Upcoming events
    # ------------------------------------------------------------------ #

    async def scrape_upcoming_events(self) -> List[Dict[str, Any]]:
        """Scrape upcoming blockchain / crypto events.

        Returns
        -------
        list[dict]
            Each dict contains ``event``, ``date``, ``description``,
            ``coins`` (related tickers if available).
        """
        events: List[Dict[str, Any]] = []

        try:
            html = await self._get_page(_COINMARKETCAL_URL)
            soup = BeautifulSoup(html, "html.parser")

            event_cards = soup.find_all("article", class_="card")
            if not event_cards:
                # Try alternative selector
                event_cards = soup.find_all("div", class_="card")

            for card in event_cards:
                title_el = (
                    card.find("h5")
                    or card.find("h4")
                    or card.find("h3")
                    or card.find(class_="card__title")
                )
                date_el = card.find(class_="card__date") or card.find("time")
                desc_el = card.find(class_="card__description") or card.find("p")
                coins_el = card.find(class_="card__coins")

                event_title = title_el.get_text(strip=True) if title_el else ""
                event_date = (
                    date_el.get_text(strip=True)
                    if date_el
                    else date_el.get("datetime", "") if date_el else ""
                )
                description = desc_el.get_text(strip=True) if desc_el else ""
                coins = coins_el.get_text(strip=True) if coins_el else ""

                if event_title:
                    events.append(
                        {
                            "event": event_title,
                            "date": event_date,
                            "description": description,
                            "coins": coins,
                        }
                    )

            logger.info("Scraped %d upcoming events", len(events))

        except Exception:
            logger.exception("Failed to scrape upcoming events")

        return events

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_fear_greed(value: int) -> str:
        """Map a 0-100 fear/greed score to a human-readable label."""
        if value <= 20:
            return "Extreme Fear"
        if value <= 40:
            return "Fear"
        if value <= 60:
            return "Neutral"
        if value <= 80:
            return "Greed"
        return "Extreme Greed"

    @staticmethod
    def _parse_money(text: str) -> Optional[float]:
        """Best-effort parse of currency strings like ``$1,234.56``."""
        cleaned = text.replace("$", "").replace(",", "").strip()
        # Handle suffixes (B, M, K)
        multiplier = 1.0
        if cleaned.endswith("B"):
            multiplier = 1e9
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1e6
            cleaned = cleaned[:-1]
        elif cleaned.endswith("K"):
            multiplier = 1e3
            cleaned = cleaned[:-1]
        try:
            return float(cleaned) * multiplier
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_pct(text: str) -> Optional[float]:
        """Best-effort parse of percentage strings like ``-2.3%``."""
        cleaned = text.replace("%", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        """Best-effort parse of integer strings."""
        cleaned = text.replace("#", "").replace(",", "").strip()
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return None
