"""
Sentiment feature engineering for crypto trading signals.

Converts raw social-media posts, news articles, and Fear & Greed data into
numerical features that can feed directly into ML models.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentFeatures:
    """Derive ML-ready features from sentiment data sources."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        logger.info("SentimentFeatures initialised")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_dataframe(records: List[Dict[str, Any]], time_col: str = "timestamp") -> pd.DataFrame:
        """Convert a list of dicts to a time-indexed DataFrame.

        Raises
        ------
        ValueError
            If *records* is empty or *time_col* is absent.
        """
        if not records:
            raise ValueError("records list is empty")
        df = pd.DataFrame(records)
        if time_col not in df.columns:
            raise ValueError(f"Expected column '{time_col}' not found in records")
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).set_index(time_col)
        return df

    # ------------------------------------------------------------------
    # Reddit
    # ------------------------------------------------------------------
    def compute_reddit_features(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute Reddit sentiment features.

        Expected keys per post dict:
            timestamp, sentiment (-1.0 to 1.0), score (int)

        Parameters
        ----------
        posts : list[dict]
            Raw Reddit post records.

        Returns
        -------
        pd.DataFrame
            Columns: reddit_sentiment_ma, reddit_volume, reddit_bullish_ratio.
        """
        df = self._to_dataframe(posts)

        for col in ("sentiment", "score"):
            if col not in df.columns:
                raise ValueError(f"Reddit posts missing required column: {col}")

        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

        # Daily aggregation
        daily = df.resample("1D").agg(
            sentiment_mean=("sentiment", "mean"),
            post_count=("sentiment", "count"),
            bullish_count=("sentiment", lambda s: (s > 0).sum()),
        )
        daily = daily.fillna(0)

        reddit_sentiment_ma = daily["sentiment_mean"].rolling(window=7, min_periods=1).mean()
        reddit_volume = daily["post_count"].rolling(window=7, min_periods=1).mean()

        total = daily["post_count"].replace(0, np.nan)
        reddit_bullish_ratio = (daily["bullish_count"] / total).fillna(0)

        result = pd.DataFrame(
            {
                "reddit_sentiment_ma": reddit_sentiment_ma,
                "reddit_volume": reddit_volume,
                "reddit_bullish_ratio": reddit_bullish_ratio,
            },
            index=daily.index,
        )
        logger.info("Reddit features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # Twitter / X
    # ------------------------------------------------------------------
    def compute_twitter_features(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute Twitter/X sentiment features.

        Expected keys per tweet dict:
            timestamp, sentiment (-1.0 to 1.0), likes (int), retweets (int)

        Parameters
        ----------
        tweets : list[dict]
            Raw tweet records.

        Returns
        -------
        pd.DataFrame
            Columns: twitter_sentiment_ma, twitter_volume, twitter_engagement.
        """
        df = self._to_dataframe(tweets)

        for col in ("sentiment", "likes", "retweets"):
            if col not in df.columns:
                raise ValueError(f"Tweets missing required column: {col}")

        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
        df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce").fillna(0)
        df["engagement"] = df["likes"] + df["retweets"]

        daily = df.resample("1D").agg(
            sentiment_mean=("sentiment", "mean"),
            tweet_count=("sentiment", "count"),
            engagement_sum=("engagement", "sum"),
        )
        daily = daily.fillna(0)

        twitter_sentiment_ma = daily["sentiment_mean"].rolling(window=7, min_periods=1).mean()
        twitter_volume = daily["tweet_count"].rolling(window=7, min_periods=1).mean()
        twitter_engagement = daily["engagement_sum"].rolling(window=7, min_periods=1).mean()

        result = pd.DataFrame(
            {
                "twitter_sentiment_ma": twitter_sentiment_ma,
                "twitter_volume": twitter_volume,
                "twitter_engagement": twitter_engagement,
            },
            index=daily.index,
        )
        logger.info("Twitter features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------
    def compute_news_features(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute news sentiment features.

        Expected keys per article dict:
            timestamp, sentiment (-1.0 to 1.0), relevance (0.0 to 1.0)

        Parameters
        ----------
        articles : list[dict]
            Raw news article records.

        Returns
        -------
        pd.DataFrame
            Columns: news_sentiment_ma, news_volume, news_impact_score.
        """
        df = self._to_dataframe(articles)

        for col in ("sentiment", "relevance"):
            if col not in df.columns:
                raise ValueError(f"News articles missing required column: {col}")

        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0)
        df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(0)

        # Impact score weights sentiment by relevance
        df["impact"] = df["sentiment"] * df["relevance"]

        daily = df.resample("1D").agg(
            sentiment_mean=("sentiment", "mean"),
            article_count=("sentiment", "count"),
            impact_mean=("impact", "mean"),
        )
        daily = daily.fillna(0)

        news_sentiment_ma = daily["sentiment_mean"].rolling(window=7, min_periods=1).mean()
        news_volume = daily["article_count"].rolling(window=7, min_periods=1).mean()
        news_impact_score = daily["impact_mean"].rolling(window=7, min_periods=1).mean()

        result = pd.DataFrame(
            {
                "news_sentiment_ma": news_sentiment_ma,
                "news_volume": news_volume,
                "news_impact_score": news_impact_score,
            },
            index=daily.index,
        )
        logger.info("News features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------
    def compute_fear_greed_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compute Fear & Greed Index features.

        Expected keys per record:
            timestamp, value (0-100)

        Parameters
        ----------
        data : list[dict]
            Raw Fear & Greed index records.

        Returns
        -------
        pd.DataFrame
            Columns: fear_greed_value, fear_greed_change, fear_greed_ma.
        """
        df = self._to_dataframe(data)

        if "value" not in df.columns:
            raise ValueError("Fear & Greed data missing required column: value")

        df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill()

        fear_greed_value = df["value"]
        fear_greed_change = df["value"].diff().fillna(0)
        fear_greed_ma = df["value"].rolling(window=7, min_periods=1).mean()

        result = pd.DataFrame(
            {
                "fear_greed_value": fear_greed_value,
                "fear_greed_change": fear_greed_change,
                "fear_greed_ma": fear_greed_ma,
            },
            index=df.index,
        )
        logger.info("Fear & Greed features computed (%d rows)", len(result))
        return result

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------
    def compute_all(
        self,
        reddit: List[Dict[str, Any]],
        twitter: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        fear_greed: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Compute and merge all sentiment feature groups.

        Parameters
        ----------
        reddit : list[dict]
            Reddit post records.
        twitter : list[dict]
            Tweet records.
        news : list[dict]
            News article records.
        fear_greed : list[dict]
            Fear & Greed index records.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all sentiment features.
        """
        frames: List[pd.DataFrame] = []

        try:
            frames.append(self.compute_reddit_features(reddit))
        except Exception as exc:
            logger.warning("Reddit feature computation failed: %s", exc)

        try:
            frames.append(self.compute_twitter_features(twitter))
        except Exception as exc:
            logger.warning("Twitter feature computation failed: %s", exc)

        try:
            frames.append(self.compute_news_features(news))
        except Exception as exc:
            logger.warning("News feature computation failed: %s", exc)

        try:
            frames.append(self.compute_fear_greed_features(fear_greed))
        except Exception as exc:
            logger.warning("Fear & Greed feature computation failed: %s", exc)

        if not frames:
            logger.error("All sentiment feature computations failed")
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        logger.info(
            "All sentiment features merged: %d columns, %d rows",
            result.shape[1],
            result.shape[0],
        )
        return result
