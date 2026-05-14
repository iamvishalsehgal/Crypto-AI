"""
Event bus and typed messages for multi-agent trading pipeline.

Agents communicate exclusively through this bus — no direct coupling.
Each message type has its own asyncio.Queue so consumers pull only what
they care about.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Message types ──────────────────────────────────────────────────────────


@dataclass
class MarketDataEvent:
    """Raw or lightly-buffered market data for one symbol."""

    symbol: str
    asset_type: str  # "crypto", "stock", "bet"
    ohlcv: pd.DataFrame  # rolling window of candles
    order_book: Optional[Dict[str, Any]] = None
    ticker: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureEvent:
    """Enriched feature matrix computed from market data."""

    symbol: str
    asset_type: str
    features: pd.DataFrame  # feature matrix (last row = current)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SignalEvent:
    """Trading signal from ensemble voting."""

    symbol: str
    asset_type: str
    signal: str  # BUY, SELL, HOLD, PASS
    confidence: float
    price: float
    individual_predictions: Dict[str, str] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ApprovedSignalEvent:
    """Signal that passed risk checks, ready for execution."""

    symbol: str
    asset_type: str
    signal: str  # BUY, SELL, HOLD, PASS
    confidence: float
    price: float
    amount: float  # position size from risk manager
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExitSignalEvent:
    """Stop-loss / take-profit / trailing stop triggered."""

    symbol: str
    asset_type: str
    reason: str  # "stop_loss", "take_profit", "trailing_stop", "manual"
    price: float
    amount: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PortfolioEvent:
    """Periodic portfolio snapshot."""

    balance_usd: float
    equity: float
    return_pct: float
    open_positions: int
    lanes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Event bus ──────────────────────────────────────────────────────────────


class EventBus:
    """Typed pub/sub message bus for agent communication.

    Each message type gets its own ``asyncio.Queue``. Agents pull from
    the queues they care about. Default queue size is 256 to prevent
    unbounded growth if a consumer is slow.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._queues: Dict[str, asyncio.Queue] = {
            "market_data": asyncio.Queue(maxsize=maxsize),
            "feature": asyncio.Queue(maxsize=maxsize),
            "signal": asyncio.Queue(maxsize=maxsize),
            "approved_signal": asyncio.Queue(maxsize=maxsize),
            "exit_signal": asyncio.Queue(maxsize=maxsize),
            "portfolio": asyncio.Queue(maxsize=maxsize),
        }
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    # -- publish ----------------------------------------------------------

    async def publish_market_data(self, event: MarketDataEvent) -> None:
        await self._queues["market_data"].put(event)

    async def publish_feature(self, event: FeatureEvent) -> None:
        await self._queues["feature"].put(event)

    async def publish_signal(self, event: SignalEvent) -> None:
        await self._queues["signal"].put(event)

    async def publish_approved_signal(self, event: ApprovedSignalEvent) -> None:
        await self._queues["approved_signal"].put(event)

    async def publish_exit_signal(self, event: ExitSignalEvent) -> None:
        await self._queues["exit_signal"].put(event)

    async def publish_portfolio(self, event: PortfolioEvent) -> None:
        await self._queues["portfolio"].put(event)

    # -- consume ----------------------------------------------------------

    async def consume_market_data(self) -> MarketDataEvent:
        return await self._queues["market_data"].get()

    async def consume_feature(self) -> FeatureEvent:
        return await self._queues["feature"].get()

    async def consume_signal(self) -> SignalEvent:
        return await self._queues["signal"].get()

    async def consume_approved_signal(self) -> ApprovedSignalEvent:
        return await self._queues["approved_signal"].get()

    async def consume_exit_signal(self) -> ExitSignalEvent:
        return await self._queues["exit_signal"].get()

    async def consume_portfolio(self) -> PortfolioEvent:
        return await self._queues["portfolio"].get()

    # -- queue inspection -------------------------------------------------

    def queue_sizes(self) -> Dict[str, int]:
        return {name: q.qsize() for name, q in self._queues.items()}
