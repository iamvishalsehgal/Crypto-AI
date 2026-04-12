"""
MongoDB storage layer for the AI Crypto Trading Bot.

Provides a single :class:`MongoDBStorage` class that handles every collection
the bot requires: OHLCV candles, sentiment scores, on-chain metrics, executed
trades, and model performance metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError

from crypto_bot.config.settings import settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Collection names (single source of truth)
# ---------------------------------------------------------------------------
_COLL_OHLCV = "ohlcv"
_COLL_SENTIMENT = "sentiment"
_COLL_ONCHAIN = "onchain"
_COLL_TRADES = "trades"
_COLL_MODEL_METRICS = "model_metrics"


class MongoDBStorage:
    """
    Thin wrapper around PyMongo that exposes typed helpers for each domain
    collection.

    Supports both explicit ``connect``/``disconnect`` and the context-manager
    protocol (``with MongoDBStorage() as db: ...``).
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """
        Initialise the storage layer.

        Args:
            uri: MongoDB connection string.  Falls back to ``settings.database.uri``.
            db_name: Database name.  Falls back to ``settings.database.name``.
        """
        self._uri = uri or settings.database.uri
        self._db_name = db_name or settings.database.name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Open the MongoDB connection and ensure indexes exist.

        Raises:
            ConnectionFailure: When the server is unreachable.
        """
        if self._client is not None:
            logger.debug("Already connected to MongoDB.")
            return

        logger.info("Connecting to MongoDB at %s ...", self._uri)
        self._client = MongoClient(
            self._uri,
            serverSelectionTimeoutMS=settings.database.server_selection_timeout_ms,
            connectTimeoutMS=settings.database.connection_timeout_ms,
        )

        # Verify connectivity.
        try:
            self._client.admin.command("ping")
        except ConnectionFailure:
            logger.error("MongoDB server is not available at %s", self._uri)
            self._client = None
            raise

        self._db = self._client[self._db_name]
        self._ensure_indexes()
        logger.info("Connected to MongoDB database '%s'.", self._db_name)

    def disconnect(self) -> None:
        """Close the MongoDB connection gracefully."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("Disconnected from MongoDB.")

    # Context-manager protocol

    def __enter__(self) -> "MongoDBStorage":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection(self, name: str) -> Collection:
        """Return a collection handle, raising if not connected."""
        if self._db is None:
            raise RuntimeError(
                "Not connected to MongoDB. Call connect() or use the context manager."
            )
        return self._db[name]

    def _ensure_indexes(self) -> None:
        """Create indexes that accelerate the most common queries."""
        try:
            # OHLCV -- unique compound index for deduplication.
            self._collection(_COLL_OHLCV).create_index(
                [("symbol", pymongo.ASCENDING),
                 ("timeframe", pymongo.ASCENDING),
                 ("timestamp", pymongo.ASCENDING)],
                unique=True,
                name="idx_ohlcv_sym_tf_ts",
            )

            # Sentiment -- query by source + time range.
            self._collection(_COLL_SENTIMENT).create_index(
                [("source", pymongo.ASCENDING),
                 ("timestamp", pymongo.ASCENDING)],
                name="idx_sentiment_src_ts",
            )

            # On-chain -- query by time range.
            self._collection(_COLL_ONCHAIN).create_index(
                [("timestamp", pymongo.ASCENDING)],
                name="idx_onchain_ts",
            )

            # Trades -- query by time range.
            self._collection(_COLL_TRADES).create_index(
                [("timestamp", pymongo.ASCENDING)],
                name="idx_trades_ts",
            )

            # Model metrics -- query by model + time.
            self._collection(_COLL_MODEL_METRICS).create_index(
                [("model_name", pymongo.ASCENDING),
                 ("timestamp", pymongo.ASCENDING)],
                name="idx_model_metrics_name_ts",
            )

            logger.debug("Database indexes verified / created.")
        except PyMongoError as exc:
            logger.warning("Index creation encountered an issue: %s", exc)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def store_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
    ) -> int:
        """
        Upsert OHLCV candle data.

        Expected DataFrame columns: ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.  The ``timestamp`` column should
        contain Unix millisecond values or :class:`datetime` objects.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            timeframe: Candlestick interval, e.g. ``"1h"``.
            data: DataFrame of OHLCV rows.

        Returns:
            Number of upserted documents.
        """
        coll = self._collection(_COLL_OHLCV)
        upserted = 0

        for _, row in data.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)

            doc = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

            result = coll.update_one(
                {"symbol": symbol, "timeframe": timeframe, "timestamp": ts},
                {"$set": doc},
                upsert=True,
            )
            if result.upserted_id or result.modified_count:
                upserted += 1

        logger.info(
            "Stored %d OHLCV records for %s [%s].", upserted, symbol, timeframe
        )
        return upserted

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for a symbol/timeframe within a time window.

        Args:
            symbol: Trading pair.
            timeframe: Candlestick interval.
            start: Inclusive lower bound (UTC).
            end: Inclusive upper bound (UTC).

        Returns:
            A DataFrame sorted by timestamp with columns ``timestamp``,
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        coll = self._collection(_COLL_OHLCV)
        cursor = coll.find(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": {"$gte": start, "$lte": end},
            },
            {"_id": 0, "symbol": 0, "timeframe": 0},
        ).sort("timestamp", pymongo.ASCENDING)

        df = pd.DataFrame(list(cursor))
        if df.empty:
            logger.debug(
                "No OHLCV data for %s [%s] between %s and %s.",
                symbol, timeframe, start, end,
            )
        return df

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def store_sentiment(self, source: str, data: Dict[str, Any]) -> None:
        """
        Insert a sentiment record.

        Args:
            source: Origin of the data (e.g. ``"reddit"``, ``"finbert"``).
            data: Arbitrary dict; a ``timestamp`` key is added automatically
                  if not already present.
        """
        coll = self._collection(_COLL_SENTIMENT)
        doc = {**data, "source": source}
        doc.setdefault("timestamp", datetime.now(timezone.utc))
        coll.insert_one(doc)
        logger.debug("Stored sentiment record from '%s'.", source)

    def get_sentiment(
        self,
        source: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve sentiment records for a source within a time window.

        Args:
            source: Data origin filter.
            start: Inclusive lower bound (UTC).
            end: Inclusive upper bound (UTC).

        Returns:
            A list of dicts (``_id`` field excluded).
        """
        coll = self._collection(_COLL_SENTIMENT)
        cursor = coll.find(
            {
                "source": source,
                "timestamp": {"$gte": start, "$lte": end},
            },
            {"_id": 0},
        ).sort("timestamp", pymongo.ASCENDING)
        return list(cursor)

    # ------------------------------------------------------------------
    # On-chain metrics
    # ------------------------------------------------------------------

    def store_onchain(self, data: Dict[str, Any]) -> None:
        """
        Insert an on-chain metrics record.

        Args:
            data: Arbitrary dict; a ``timestamp`` key is added automatically
                  if not already present.
        """
        coll = self._collection(_COLL_ONCHAIN)
        doc = dict(data)
        doc.setdefault("timestamp", datetime.now(timezone.utc))
        coll.insert_one(doc)
        logger.debug("Stored on-chain metrics record.")

    def get_onchain(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve on-chain records within a time window.

        Args:
            start: Inclusive lower bound (UTC).
            end: Inclusive upper bound (UTC).

        Returns:
            A list of dicts (``_id`` field excluded).
        """
        coll = self._collection(_COLL_ONCHAIN)
        cursor = coll.find(
            {"timestamp": {"$gte": start, "$lte": end}},
            {"_id": 0},
        ).sort("timestamp", pymongo.ASCENDING)
        return list(cursor)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def store_trade(self, trade: Dict[str, Any]) -> None:
        """
        Insert an executed-trade record.

        Args:
            trade: Trade details (symbol, side, price, quantity, etc.).
                   A ``timestamp`` key is added if missing.
        """
        coll = self._collection(_COLL_TRADES)
        doc = dict(trade)
        doc.setdefault("timestamp", datetime.now(timezone.utc))
        coll.insert_one(doc)
        logger.info(
            "Recorded trade: %s %s @ %s.",
            doc.get("side", "?"),
            doc.get("symbol", "?"),
            doc.get("price", "?"),
        )

    def get_trades(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trades within a time window.

        Args:
            start: Inclusive lower bound (UTC).
            end: Inclusive upper bound (UTC).

        Returns:
            A list of trade dicts (``_id`` excluded).
        """
        coll = self._collection(_COLL_TRADES)
        cursor = coll.find(
            {"timestamp": {"$gte": start, "$lte": end}},
            {"_id": 0},
        ).sort("timestamp", pymongo.ASCENDING)
        return list(cursor)

    # ------------------------------------------------------------------
    # Model metrics
    # ------------------------------------------------------------------

    def store_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Store model performance metrics.

        Args:
            model_name: Identifier for the model (e.g. ``"lstm_v2"``).
            metrics: Dict of metric names to values (accuracy, loss, etc.).
        """
        coll = self._collection(_COLL_MODEL_METRICS)
        doc = {
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc),
            **metrics,
        }
        coll.insert_one(doc)
        logger.info("Stored metrics for model '%s'.", model_name)
