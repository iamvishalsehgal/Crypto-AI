"""
Unit tests for SentimentFeatures.
"""

from __future__ import annotations

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.features.sentiment_features import SentimentFeatures

settings = Settings()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_timestamps(n: int, base: str = "2024-01-01") -> List[pd.Timestamp]:
    """Return *n* daily timestamps starting at *base*."""
    return list(pd.date_range(start=base, periods=n, freq="D"))


# ---------------------------------------------------------------------------
# normalize_sentiment
# ---------------------------------------------------------------------------

def test_normalize_sentiment_float_unchanged() -> None:
    """Float values pass through unchanged."""
    f = SentimentFeatures(settings)
    assert f.normalize_sentiment(0.65) == 0.65
    assert f.normalize_sentiment(-0.3) == -0.3
    assert f.normalize_sentiment(0.0) == 0.0


def test_normalize_sentiment_string_labels() -> None:
    """String labels map to correct floats."""
    f = SentimentFeatures(settings)
    assert f.normalize_sentiment("positive") == 0.8
    assert f.normalize_sentiment("negative") == -0.8
    assert f.normalize_sentiment("neutral") == 0.0
    assert f.normalize_sentiment("very_positive") == 0.95
    assert f.normalize_sentiment("very_negative") == -0.95


# ---------------------------------------------------------------------------
# compute_reddit_features
# ---------------------------------------------------------------------------

def test_reddit_features_columns() -> None:
    """Expected columns exist; bullish_ratio in [0, 1]."""
    ts = make_timestamps(14)
    posts: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.5, "score": 10}
        for t in ts[:7]
    ] + [
        {"timestamp": t, "sentiment": -0.3, "score": 5}
        for t in ts[7:]
    ]

    f = SentimentFeatures(settings)
    result = f.compute_reddit_features(posts)

    assert "reddit_sentiment_ma" in result.columns
    assert "reddit_volume" in result.columns
    assert "reddit_bullish_ratio" in result.columns

    # Bullish ratio bounded [0, 1]
    assert (result["reddit_bullish_ratio"] >= 0).all()
    assert (result["reddit_bullish_ratio"] <= 1).all()


def test_reddit_features_bullish_ratio_value() -> None:
    """Bullish ratio reflects proportion of positive-sentiment posts."""
    ts = make_timestamps(3)
    posts: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "sentiment": 0.5, "score": 10},   # bullish
        {"timestamp": ts[0], "sentiment": 0.1, "score": 5},    # bullish
        {"timestamp": ts[0], "sentiment": -0.2, "score": 3},   # not bullish
    ]

    f = SentimentFeatures(settings)
    result = f.compute_reddit_features(posts)

    # Day 0: 2 bullish out of 3 → ratio ≈ 0.667
    eps = 1e-9
    assert abs(result["reddit_bullish_ratio"].iloc[0] - 2.0 / 3.0) < eps


def test_reddit_features_empty() -> None:
    """Empty input raises ValueError."""
    f = SentimentFeatures(settings)
    try:
        f.compute_reddit_features([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_twitter_features
# ---------------------------------------------------------------------------

def test_twitter_features_columns_and_engagement() -> None:
    """Expected columns exist; engagement = likes + retweets."""
    ts = make_timestamps(7)
    tweets: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.2, "likes": 100, "retweets": 50}
        for t in ts
    ]

    f = SentimentFeatures(settings)
    result = f.compute_twitter_features(tweets)

    assert "twitter_sentiment_ma" in result.columns
    assert "twitter_volume" in result.columns
    assert "twitter_engagement" in result.columns

    # With 1 tweet/day, daily engagement_sum = 100 + 50 = 150
    # twitter_engagement = rolling 7-day mean of engagement_sum → 150 for first row
    eps = 1e-9
    assert abs(result["twitter_engagement"].iloc[0] - 150.0) < eps


def test_twitter_features_empty() -> None:
    """Empty input raises ValueError."""
    f = SentimentFeatures(settings)
    try:
        f.compute_twitter_features([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_news_features
# ---------------------------------------------------------------------------

def test_news_features_columns_and_impact() -> None:
    """Expected columns exist; impact = sentiment × relevance."""
    ts = make_timestamps(7)
    articles: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.5, "relevance": 0.8}
        for t in ts
    ]

    f = SentimentFeatures(settings)
    result = f.compute_news_features(articles)

    assert "news_sentiment_ma" in result.columns
    assert "news_volume" in result.columns
    assert "news_impact_score" in result.columns

    # Impact = 0.5 * 0.8 = 0.4 → impact_mean per day = 0.4
    # news_impact_score = rolling 7-day mean → first value = 0.4
    eps = 1e-9
    assert abs(result["news_impact_score"].iloc[0] - 0.4) < eps


def test_news_features_empty() -> None:
    """Empty input raises ValueError."""
    f = SentimentFeatures(settings)
    try:
        f.compute_news_features([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_fear_greed_features
# ---------------------------------------------------------------------------

def test_fear_greed_features_columns() -> None:
    """Expected columns exist; change = diff of value."""
    ts = make_timestamps(10)
    data: List[Dict[str, Any]] = [
        {"timestamp": t, "value": float(v)}
        for v, t in enumerate(ts)
    ]

    f = SentimentFeatures(settings)
    result = f.compute_fear_greed_features(data)

    assert "fear_greed_value" in result.columns
    assert "fear_greed_change" in result.columns
    assert "fear_greed_ma" in result.columns

    # fear_greed_change = diff (first NaN → fillna 0)
    assert result["fear_greed_change"].iloc[0] == 0.0
    assert result["fear_greed_change"].iloc[1] == 1.0
    assert result["fear_greed_change"].iloc[2] == 1.0

    # fear_greed_ma = rolling 7-day mean (min_periods=1)
    # First 7 values: 0..6 → mean = 3.0
    eps = 1e-9
    assert abs(result["fear_greed_ma"].iloc[6] - 3.0) < eps


def test_fear_greed_features_empty() -> None:
    """Empty input raises ValueError."""
    f = SentimentFeatures(settings)
    try:
        f.compute_fear_greed_features([])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_merges_all_groups() -> None:
    """All sub-computations merged into one DataFrame."""
    ts = make_timestamps(10)

    reddit: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.5, "score": 10} for t in ts
    ]
    twitter: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.2, "likes": 100, "retweets": 50} for t in ts
    ]
    news: List[Dict[str, Any]] = [
        {"timestamp": t, "sentiment": 0.5, "relevance": 0.8} for t in ts
    ]
    fear_greed: List[Dict[str, Any]] = [
        {"timestamp": t, "value": 50.0} for t in ts
    ]

    f = SentimentFeatures(settings)
    result = f.compute_all(reddit, twitter, news, fear_greed)

    expected = {
        "reddit_sentiment_ma", "reddit_volume", "reddit_bullish_ratio",
        "twitter_sentiment_ma", "twitter_volume", "twitter_engagement",
        "news_sentiment_ma", "news_volume", "news_impact_score",
        "fear_greed_value", "fear_greed_change", "fear_greed_ma",
    }
    assert expected.issubset(result.columns), f"Missing: {expected - set(result.columns)}"


def test_compute_all_partial_failure() -> None:
    """One failing sub-computation does not crash the whole pipeline."""
    ts = make_timestamps(5)

    reddit: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "sentiment": 0.5, "score": 10},
    ]
    bad_twitter: List[Dict[str, Any]] = []  # empty → ValueError
    news: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "sentiment": 0.5, "relevance": 0.8},
    ]
    fear_greed: List[Dict[str, Any]] = [
        {"timestamp": ts[0], "value": 50.0},
    ]

    f = SentimentFeatures(settings)
    result = f.compute_all(reddit, bad_twitter, news, fear_greed)

    assert not result.empty
    assert "reddit_sentiment_ma" in result.columns
    assert "news_sentiment_ma" in result.columns
    assert "fear_greed_value" in result.columns
    # twitter should be absent
    assert "twitter_sentiment_ma" not in result.columns


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_normalize_sentiment_float_unchanged,
        test_normalize_sentiment_string_labels,
        test_reddit_features_columns,
        test_reddit_features_bullish_ratio_value,
        test_reddit_features_empty,
        test_twitter_features_columns_and_engagement,
        test_twitter_features_empty,
        test_news_features_columns_and_impact,
        test_news_features_empty,
        test_fear_greed_features_columns,
        test_fear_greed_features_empty,
        test_compute_all_merges_all_groups,
        test_compute_all_partial_failure,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
