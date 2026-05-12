"""
Unified asset type system for multi-asset trading (crypto, stocks, betting).

Defines the canonical :class:`AssetType` enum and :class:`UnifiedSignal`
dataclass that all pipelines consume and produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AssetType(str, Enum):
    """Supported asset classes for the trading bot."""

    CRYPTO = "crypto"
    STOCK = "stock"
    BET = "bet"


# Canonical signal labels shared across all asset types.
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
BACK = "BACK"  # betting: place bet
LAY = "LAY"  # betting: lay bet (act as bookmaker)
PASS = "PASS"  # betting: no bet


@dataclass
class UnifiedSignal:
    """Normalised trading signal that every pipeline stage understands.

    Attributes:
        asset_type: Which asset class this signal belongs to.
        symbol: Trading pair / ticker / event identifier.
        side: ``BUY`` / ``SELL`` / ``HOLD`` (crypto & stock) or
              ``BACK`` / ``LAY`` / ``PASS`` (betting).
        confidence: Model confidence in [0, 1].
        amount: Suggested position size in quote currency (crypto/stock)
                or stake amount (betting).
        price: Reference price / odds at signal generation time.
        metadata: Arbitrary pipeline-specific extra data.
    """

    asset_type: AssetType
    symbol: str
    side: str = HOLD
    confidence: float = 0.0
    amount: float = 0.0
    price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return ``True`` if the signal warrants execution."""
        if self.side in (HOLD, PASS):
            return False
        return self.confidence >= 0.5


@dataclass
class AssetConfig:
    """Per-asset-type configuration container."""

    enabled: bool = False
    symbols: list[str] = field(default_factory=list)
    paper_mode: bool = True
