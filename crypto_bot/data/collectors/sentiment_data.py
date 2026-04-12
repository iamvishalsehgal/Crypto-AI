"""
Social media and news sentiment collector.

Aggregates posts from Reddit, Twitter / X, and news outlets,
then provides a simple sentiment summary for downstream ML models.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists used by the placeholder sentiment scorer
# ---------------------------------------------------------------------------
_BULLISH_KEYWORDS = frozenset(
    {
        "bull",
        "bullish",
        "moon",
        "pump",
        "long",
        "buy",
        "rally",
        "breakout",
        "green",
        "ath",
        "hodl",
        "accumulate",
        "undervalued",
        "uptrend",
    }
)

_BEARISH_KEYWORDS = frozenset(
    {
        "bear",
        "bearish",
        "dump",
        "short",
        "sell",
        "crash",
        "red",
        "overvalued",
        "downtrend",
        "rug",
        "scam",
        "liquidat",
        "capitulat",
    }
)


class SentimentCollector:
    """Collect and score social / news sentiment for crypto assets."""

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, settings: Settings) -> None:
        """Set up API clients for Reddit, Twitter, and NewsAPI.

        Credentials are read from *settings* attributes first, then
        from environment variables as a fallback::

            REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT
            TWITTER_BEARER_TOKEN
            NEWS_API_KEY
        """
        self._settings = settings

        # --- Reddit (praw) ---
        self._reddit_client_id: str = getattr(
            settings, "REDDIT_CLIENT_ID", ""
        ) or os.environ.get("REDDIT_CLIENT_ID", "")

        self._reddit_secret: str = getattr(
            settings, "REDDIT_SECRET", ""
        ) or os.environ.get("REDDIT_SECRET", "")

        self._reddit_user_agent: str = getattr(
            settings, "REDDIT_USER_AGENT", ""
        ) or os.environ.get("REDDIT_USER_AGENT", "crypto_bot:v1.0 (by /u/crypto_bot)")

        self._reddit = None  # lazy init

        # --- Twitter / X (tweepy) ---
        self._twitter_bearer: str = getattr(
            settings, "TWITTER_BEARER_TOKEN", ""
        ) or os.environ.get("TWITTER_BEARER_TOKEN", "")

        self._twitter_client = None  # lazy init

        # --- NewsAPI ---
        self._news_api_key: str = getattr(
            settings, "NEWS_API_KEY", ""
        ) or os.environ.get("NEWS_API_KEY", "")

        self._news_client = None  # lazy init

        logger.info("SentimentCollector initialised")

    # ------------------------------------------------------------------ #
    # Lazy client construction
    # ------------------------------------------------------------------ #

    def _get_reddit(self):  # -> praw.Reddit
        """Return a configured praw Reddit instance (created on first call)."""
        if self._reddit is not None:
            return self._reddit

        if not self._reddit_client_id or not self._reddit_secret:
            logger.warning("Reddit credentials not configured")
            return None

        try:
            import praw

            self._reddit = praw.Reddit(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_secret,
                user_agent=self._reddit_user_agent,
            )
            return self._reddit
        except Exception:
            logger.exception("Failed to initialise Reddit client")
            return None

    def _get_twitter(self):  # -> tweepy.Client
        """Return a configured tweepy v2 client (created on first call)."""
        if self._twitter_client is not None:
            return self._twitter_client

        if not self._twitter_bearer:
            logger.warning("Twitter bearer token not configured")
            return None

        try:
            import tweepy

            self._twitter_client = tweepy.Client(
                bearer_token=self._twitter_bearer,
                wait_on_rate_limit=True,
            )
            return self._twitter_client
        except Exception:
            logger.exception("Failed to initialise Twitter client")
            return None

    def _get_newsapi(self):  # -> newsapi.NewsApiClient
        """Return a configured NewsAPI client (created on first call)."""
        if self._news_client is not None:
            return self._news_client

        if not self._news_api_key:
            logger.warning("NewsAPI key not configured")
            return None

        try:
            from newsapi import NewsApiClient

            self._news_client = NewsApiClient(api_key=self._news_api_key)
            return self._news_client
        except Exception:
            logger.exception("Failed to initialise NewsAPI client")
            return None

    # ------------------------------------------------------------------ #
    # Reddit
    # ------------------------------------------------------------------ #

    def fetch_reddit_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch recent posts from crypto-related subreddits.

        Parameters
        ----------
        subreddits:
            List of subreddit names (without ``r/``).  Defaults to a
            curated list of popular crypto subreddits.
        limit:
            Maximum number of posts to return **per subreddit**.

        Returns
        -------
        list[dict]
            Each dict contains ``title``, ``text``, ``score``,
            ``created``, ``subreddit``, ``num_comments``, ``url``.
        """
        if subreddits is None:
            subreddits = [
                "CryptoCurrency",
                "Bitcoin",
                "ethereum",
                "CryptoMarkets",
            ]

        reddit = self._get_reddit()
        if reddit is None:
            return []

        posts: List[Dict[str, Any]] = []

        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)

                for submission in subreddit.hot(limit=limit):
                    posts.append(
                        {
                            "title": submission.title,
                            "text": submission.selftext[:2000] if submission.selftext else "",
                            "score": submission.score,
                            "created": datetime.fromtimestamp(
                                submission.created_utc, tz=timezone.utc
                            ).isoformat(),
                            "subreddit": sub_name,
                            "num_comments": submission.num_comments,
                            "url": submission.url,
                        }
                    )

                logger.debug("Fetched %d posts from r/%s", limit, sub_name)

            except Exception:
                logger.exception("Failed to fetch posts from r/%s", sub_name)

        logger.info("Total Reddit posts collected: %d", len(posts))
        return posts

    # ------------------------------------------------------------------ #
    # Twitter / X
    # ------------------------------------------------------------------ #

    def fetch_twitter_mentions(
        self,
        query: str = "bitcoin OR crypto",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search recent tweets matching *query*.

        Parameters
        ----------
        query:
            Twitter search query string.
        limit:
            Maximum number of tweets to return (capped at 100 per
            request by the v2 API).

        Returns
        -------
        list[dict]
            Each dict contains ``text``, ``created_at``,
            ``author_id``, ``retweet_count``, ``like_count``, ``id``.
        """
        client = self._get_twitter()
        if client is None:
            return []

        tweets: List[Dict[str, Any]] = []

        try:
            # Exclude retweets for cleaner signal
            full_query = f"({query}) -is:retweet lang:en"

            response = client.search_recent_tweets(
                query=full_query,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
            )

            if response.data:
                for tweet in response.data:
                    metrics = tweet.public_metrics or {}
                    tweets.append(
                        {
                            "id": str(tweet.id),
                            "text": tweet.text,
                            "created_at": (
                                tweet.created_at.isoformat()
                                if tweet.created_at
                                else None
                            ),
                            "author_id": str(tweet.author_id),
                            "retweet_count": metrics.get("retweet_count", 0),
                            "like_count": metrics.get("like_count", 0),
                        }
                    )

            logger.info("Fetched %d tweets for query '%s'", len(tweets), query)

        except Exception:
            logger.exception("Failed to fetch tweets for query '%s'", query)

        return tweets

    # ------------------------------------------------------------------ #
    # News
    # ------------------------------------------------------------------ #

    def fetch_news_articles(
        self,
        query: str = "cryptocurrency",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch recent news articles matching *query*.

        Parameters
        ----------
        query:
            Search query for NewsAPI.
        limit:
            Maximum number of articles.

        Returns
        -------
        list[dict]
            Each dict contains ``title``, ``description``, ``url``,
            ``published``, ``source``.
        """
        client = self._get_newsapi()
        if client is None:
            return []

        articles: List[Dict[str, Any]] = []

        try:
            response = client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=min(limit, 100),
            )

            for item in response.get("articles", []):
                articles.append(
                    {
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "url": item.get("url", ""),
                        "published": item.get("publishedAt", ""),
                        "source": (item.get("source") or {}).get("name", ""),
                    }
                )

            logger.info(
                "Fetched %d news articles for query '%s'",
                len(articles),
                query,
            )

        except Exception:
            logger.exception("Failed to fetch news articles for '%s'", query)

        return articles

    # ------------------------------------------------------------------ #
    # Sentiment aggregation
    # ------------------------------------------------------------------ #

    def aggregate_sentiment(
        self,
        posts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute a simple sentiment summary from a list of posts/articles.

        This is a keyword-based *placeholder* scorer.  Replace the
        inner scoring logic with a proper NLP model (e.g. FinBERT,
        VADER, or a fine-tuned transformer) for production use.

        Parameters
        ----------
        posts:
            Each item should have at least a ``"text"`` or ``"title"``
            key containing natural-language content.

        Returns
        -------
        dict
            ``avg_score``      -- mean sentiment score (--1 to +1).
            ``volume``         -- total number of posts scored.
            ``bullish_ratio``  -- fraction classified as bullish.
            ``bearish_ratio``  -- fraction classified as bearish.
            ``neutral_ratio``  -- fraction classified as neutral.
        """
        if not posts:
            return {
                "avg_score": 0.0,
                "volume": 0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
                "neutral_ratio": 0.0,
            }

        scores: List[float] = []
        bullish = 0
        bearish = 0
        neutral = 0

        for post in posts:
            text = " ".join(
                filter(
                    None,
                    [
                        post.get("title", ""),
                        post.get("text", ""),
                        post.get("description", ""),
                    ],
                )
            ).lower()

            score = self._score_text(text)
            scores.append(score)

            if score > 0.1:
                bullish += 1
            elif score < -0.1:
                bearish += 1
            else:
                neutral += 1

        total = len(scores)
        avg = sum(scores) / total if total else 0.0

        return {
            "avg_score": round(avg, 4),
            "volume": total,
            "bullish_ratio": round(bullish / total, 4) if total else 0.0,
            "bearish_ratio": round(bearish / total, 4) if total else 0.0,
            "neutral_ratio": round(neutral / total, 4) if total else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Internal scoring (placeholder)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_text(text: str) -> float:
        """Return a naive sentiment score in [-1.0, 1.0].

        Counts bullish vs. bearish keyword matches in *text*.  This is
        intentionally simple -- swap in a real NLP model for production.
        """
        words = set(text.split())

        bull_hits = sum(1 for w in words if w in _BULLISH_KEYWORDS)
        bear_hits = sum(1 for w in words if w in _BEARISH_KEYWORDS)

        total_hits = bull_hits + bear_hits
        if total_hits == 0:
            return 0.0

        # Score normalised to [-1, 1]
        return (bull_hits - bear_hits) / total_hits
