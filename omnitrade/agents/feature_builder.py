"""
FeatureBuilderAgent — consumes raw market data, produces feature matrices.

Pulls MarketDataEvent from the bus, runs TechnicalFeatures for crypto
or StockFeaturePipeline for stocks, publishes FeatureEvent.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd

from omnitrade.agents.bus import EventBus, FeatureEvent, MarketDataEvent
from omnitrade.features.technical import TechnicalFeatures
from omnitrade.features.stock_features import StockFeaturePipeline
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilderAgent:
    """Computes features from market data and publishes FeatureEvents.

    Args:
        bus: Shared event bus.
        tech_features: TechnicalFeatures instance for crypto indicators.
        stock_features: StockFeaturePipeline instance (optional, for stocks).
    """

    def __init__(
        self,
        bus: EventBus,
        tech_features: TechnicalFeatures,
        stock_features: Optional[StockFeaturePipeline] = None,
    ) -> None:
        self._bus = bus
        self._tech = tech_features
        self._stock_features = stock_features

    async def run(self) -> None:
        """Consume MarketDataEvents, compute features, publish FeatureEvents."""
        logger.info("FeatureBuilderAgent starting")
        try:
            while True:
                event: MarketDataEvent = await self._bus.consume_market_data()

                try:
                    feature_event = self._build(event)
                    if feature_event is not None:
                        await self._bus.publish_feature(feature_event)
                except Exception:
                    logger.exception("Feature build failed for %s", event.symbol)

        except asyncio.CancelledError:
            logger.info("FeatureBuilderAgent shutting down")

    def _build(self, event: MarketDataEvent) -> Optional[FeatureEvent]:
        ohlcv = event.ohlcv
        if ohlcv is None or ohlcv.empty:
            return None

        if event.asset_type == "crypto":
            features = self._tech.compute_all(ohlcv)
        elif event.asset_type == "stock" and self._stock_features is not None:
            fundamentals = event.ticker.get("fundamentals", {}) if event.ticker else {}
            features = self._stock_features.compute_all(ohlcv, fundamentals)
        else:
            features = ohlcv.copy()

        # Drop rows where close is NaN, then forward-fill indicator NaNs
        features = features.dropna(subset=["close"])
        features = features.ffill().fillna(0)
        if features.empty or len(features) < 20:
            logger.info("%s: after clean: empty=%s rows=%d", event.symbol, features.empty, len(features))
            return None

        logger.info("%s: publishing feature event (%d rows x %d cols)", event.symbol, len(features), len(features.columns))

        return FeatureEvent(
            symbol=event.symbol,
            asset_type=event.asset_type,
            features=features,
            metadata={"rows": len(features), "cols": len(features.columns)},
        )
