"""
SignalGeneratorAgent — consumes feature matrices, produces trading signals.

Pulls FeatureEvent from the bus, runs ensemble voting (XGBoost + LightGBM
+ LSTM), publishes SignalEvent.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd

from omnitrade.agents.bus import EventBus, FeatureEvent, SignalEvent
from omnitrade.ensemble.voting_system import EnsembleVoter
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class SignalGeneratorAgent:
    """Generates trade signals from features using the ensemble voter.

    Args:
        bus: Shared event bus.
        ensemble: EnsembleVoter with registered models for crypto.
        stock_models: StockModelFactory for stock tickers (optional).
        janus: JanusBlender for disagreement-penalised blending (optional).
    """

    def __init__(
        self,
        bus: EventBus,
        ensemble: EnsembleVoter,
        stock_models: Optional[object] = None,
        janus: Optional[object] = None,
    ) -> None:
        self._bus = bus
        self._ensemble = ensemble
        self._stock_models = stock_models
        self._janus = janus

    async def run(self) -> None:
        """Consume FeatureEvents, generate signals, publish SignalEvents."""
        logger.info("SignalGeneratorAgent starting")
        try:
            while True:
                event: FeatureEvent = await self._bus.consume_feature()

                try:
                    signal_event = self._generate(event)
                    if signal_event is not None:
                        await self._bus.publish_signal(signal_event)
                        logger.info(
                            "%s → %s (%.2f)",
                            event.symbol,
                            signal_event.signal,
                            signal_event.confidence,
                        )
                except Exception:
                    logger.exception("Signal generation failed for %s", event.symbol)

        except asyncio.CancelledError:
            logger.info("SignalGeneratorAgent shutting down")

    def _generate(self, event: FeatureEvent) -> Optional[SignalEvent]:
        features = event.features
        if features.empty:
            logger.info("%s: features empty, skipping", event.symbol)
            return None

        price = float(features["close"].iloc[-1]) if "close" in features.columns else 0.0

        if event.asset_type == "stock" and self._stock_models is not None:
            if not self._stock_models.is_trained:
                return self._fallback(event.symbol, event.asset_type, features, price)
            try:
                result = self._stock_models.predict(features)
            except Exception:
                logger.exception("Stock model predict failed for %s", event.symbol)
                result = self._fallback(event.symbol, event.asset_type, features, price)
        else:
            # Crypto / other — use ensemble
            numeric = self._numeric_features(features)
            if numeric.empty:
                logger.info("%s: numeric features empty, cols=%s", event.symbol, list(features.columns)[:5])
                return None

            model_weights = self._ensemble.get_model_weights()
            if not model_weights:
                logger.info("%s: no model weights, using RSI fallback", event.symbol)
                return self._fallback(event.symbol, event.asset_type, features, price)

            logger.info("%s: voting with %d models, %d features", event.symbol, len(model_weights), len(numeric.columns))
            result = self._ensemble.vote({"features": numeric})

            # Apply JANUS-style disagreement penalty if blender is configured
            if self._janus is not None:
                janus_blend = self._janus.blend(
                    individual_predictions=result.get("individual_predictions", {}),
                    convictions={
                        k: v / max(sum(result.get("weighted_scores", {}).values()), 1)
                        for k, v in result.get("weighted_scores", {}).items()
                    },
                    weights=self._janus.get_weights() or model_weights,
                )
                if janus_blend.get("contested"):
                    logger.info(
                        "%s: CONTESTED signal — penalty applied (%.4f → %.4f)",
                        event.symbol,
                        result.get("confidence", 0.0),
                        janus_blend.get("confidence", 0.0),
                    )
                # Merge JANUS blend with ensemble result
                result["signal"] = janus_blend.get("signal", result.get("signal"))
                result["confidence"] = janus_blend.get("confidence", result.get("confidence"))
                result["contested"] = janus_blend.get("contested", False)
                result["direction_scores"] = janus_blend.get("direction_scores", {})

        return SignalEvent(
            symbol=event.symbol,
            asset_type=event.asset_type,
            signal=result.get("signal", "HOLD"),
            confidence=result.get("confidence", 0.0),
            price=price,
            individual_predictions=result.get("individual_predictions", {}),
            weighted_scores=result.get("weighted_scores", {}),
        )

    def _fallback(
        self, symbol: str, asset_type: str, features: pd.DataFrame, price: float
    ) -> SignalEvent:
        """RSI heuristic when no models are available."""
        latest = features.iloc[-1]
        rsi = latest.get("rsi", 50)

        if rsi < 30:
            signal, confidence = "BUY", min((30 - rsi) / 30, 1.0)
        elif rsi > 70:
            signal, confidence = "SELL", min((rsi - 70) / 30, 1.0)
        else:
            signal, confidence = "HOLD", 0.3

        return SignalEvent(
            symbol=symbol,
            asset_type=asset_type,
            signal=signal,
            confidence=confidence,
            price=price,
            reasons=[f"Fallback RSI={rsi:.1f}"],
            individual_predictions={"rsi_heuristic": signal},
        )

    @staticmethod
    def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
        exclude = {"signal", "timestamp"}
        cols = [
            c
            for c in features.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
        ]
        return features[cols]
