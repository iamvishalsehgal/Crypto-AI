"""
FinBERT-based sentiment analysis for cryptocurrency news and social media.

Uses the ProsusAI/finbert model from HuggingFace to classify financial text
as positive, negative, or neutral.  Supports single-text inference, batched
processing, and aggregate sentiment scoring for trading signal generation.

Usage::

    from crypto_bot.config.settings import Settings
    from crypto_bot.models.sentiment_model import SentimentAnalyzer

    analyzer = SentimentAnalyzer(Settings())
    result = analyzer.analyze_text("Bitcoin surges past all-time high")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# FinBERT label mapping -- the model's output order is positive, negative, neutral.
_LABEL_MAP: Dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


class SentimentAnalyzer:
    """Analyse financial text sentiment with a pre-trained FinBERT model.

    Parameters
    ----------
    settings:
        Project-wide settings; ``settings.sentiment.finbert_model`` selects
        the HuggingFace model identifier.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_name: str = settings.sentiment.finbert_model

        # Device selection: prefer CUDA, then MPS (Apple Silicon), else CPU.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(
            "Loading FinBERT model '%s' on device '%s'",
            self._model_name,
            self._device,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
            ).to(self._device)
            self._model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception:
            logger.exception("Failed to load FinBERT model '%s'", self._model_name)
            raise

    # ------------------------------------------------------------------
    # Single-text inference
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Return sentiment prediction for a single piece of text.

        Parameters
        ----------
        text:
            Input string (e.g. a headline or Reddit post title).

        Returns
        -------
        dict
            ``{"label": str, "score": float, "confidence": float}``
            where *label* is one of ``"positive"``, ``"negative"``,
            ``"neutral"``; *score* is a normalised sentiment score in
            [-1, 1] (positive => toward +1, negative => toward -1); and
            *confidence* is the softmax probability of the winning class.
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}

        results = self.analyze_batch([text], batch_size=1)
        return results[0]

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Analyse a list of texts in batches for efficiency.

        Parameters
        ----------
        texts:
            Strings to classify.
        batch_size:
            Number of texts per forward pass.

        Returns
        -------
        list[dict]
            One result dict per input text, each containing ``label``,
            ``score``, and ``confidence`` keys.
        """
        if not texts:
            return []

        all_results: List[Dict[str, Any]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            # Tokenize the batch.
            encodings = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**encodings)
                probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1,
                )

            # Move to CPU for numpy conversion.
            probs_np: np.ndarray = probabilities.cpu().numpy()

            for prob_row in probs_np:
                predicted_idx = int(np.argmax(prob_row))
                label = _LABEL_MAP[predicted_idx]
                confidence = float(prob_row[predicted_idx])

                # Composite sentiment score in [-1, 1]:
                #   positive contribution minus negative contribution.
                score = float(prob_row[0] - prob_row[1])

                all_results.append(
                    {
                        "label": label,
                        "score": round(score, 4),
                        "confidence": round(confidence, 4),
                    }
                )

        return all_results

    # ------------------------------------------------------------------
    # Reddit-specific analysis
    # ------------------------------------------------------------------

    def analyze_reddit_posts(self, posts: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyse a list of Reddit post dicts.

        Each element in *posts* is expected to have at least a ``"text"``
        key.  An optional ``"timestamp"`` key (ISO-format string or
        :class:`datetime`) is preserved; if absent, the current UTC time
        is used.

        Parameters
        ----------
        posts:
            Sequence of dicts with ``"text"`` and optional ``"timestamp"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``text``, ``sentiment``, ``score``, ``timestamp``.
        """
        if not posts:
            return pd.DataFrame(columns=["text", "sentiment", "score", "timestamp"])

        texts = [p.get("text", "") for p in posts]
        results = self.analyze_batch(texts)

        rows: List[Dict[str, Any]] = []
        for post, result in zip(posts, results):
            timestamp = post.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(tz=timezone.utc)
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now(tz=timezone.utc)

            rows.append(
                {
                    "text": post.get("text", ""),
                    "sentiment": result["label"],
                    "score": result["score"],
                    "timestamp": timestamp,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # News-specific analysis
    # ------------------------------------------------------------------

    def analyze_news(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyse a list of news article dicts.

        Each element is expected to have a ``"title"`` key.  Optional keys:
        ``"timestamp"`` and ``"impact"`` (default ``"medium"``).

        Parameters
        ----------
        articles:
            Sequence of dicts with ``"title"`` and optional ``"timestamp"``
            and ``"impact"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``title``, ``sentiment``, ``score``, ``impact``,
            ``timestamp``.
        """
        if not articles:
            return pd.DataFrame(
                columns=["title", "sentiment", "score", "impact", "timestamp"]
            )

        titles = [a.get("title", "") for a in articles]
        results = self.analyze_batch(titles)

        rows: List[Dict[str, Any]] = []
        for article, result in zip(articles, results):
            timestamp = article.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(tz=timezone.utc)
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now(tz=timezone.utc)

            impact = article.get("impact", "medium")

            rows.append(
                {
                    "title": article.get("title", ""),
                    "sentiment": result["label"],
                    "score": result["score"],
                    "impact": impact,
                    "timestamp": timestamp,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Aggregate sentiment
    # ------------------------------------------------------------------

    def get_aggregate_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Compute aggregate sentiment statistics across multiple texts.

        Parameters
        ----------
        texts:
            Strings to classify.

        Returns
        -------
        dict
            Keys: ``overall_sentiment``, ``avg_score``, ``bullish_count``,
            ``bearish_count``, ``neutral_count``, ``bullish_ratio``.
        """
        if not texts:
            return {
                "overall_sentiment": "neutral",
                "avg_score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "bullish_ratio": 0.0,
            }

        results = self.analyze_batch(texts)

        bullish_count = sum(1 for r in results if r["label"] == "positive")
        bearish_count = sum(1 for r in results if r["label"] == "negative")
        neutral_count = sum(1 for r in results if r["label"] == "neutral")
        avg_score = float(np.mean([r["score"] for r in results]))

        total = len(results)
        bullish_ratio = bullish_count / total if total > 0 else 0.0

        # Determine overall sentiment from the average composite score.
        if avg_score > 0.05:
            overall = "bullish"
        elif avg_score < -0.05:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "overall_sentiment": overall,
            "avg_score": round(avg_score, 4),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bullish_ratio": round(bullish_ratio, 4),
        }
