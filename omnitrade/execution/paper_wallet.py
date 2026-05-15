"""
Paper-trading wallet — shared state for simulated orders.

Extracted from TradeExecutor and StockExecutor to eliminate duplicated
balance tracking, position management, and trade history recording.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


class PaperWallet:
    """Simulated account: multi-asset balances, positions, trade history.

    Pure state + math. Price fetching is the caller's responsibility.
    Provides :meth:`simulate_order` for market / limit fills and
    :meth:`close_position` for liquidating an existing holding.

    Args:
        initial_balance: Starting quote-currency balance.
        base_currency: Quote currency symbol (``"USDT"`` for crypto,
            ``"USD"`` for stocks).
        fee_rate: Default transaction fee as a fraction (0.001 = 10 bps).
        slippage: Default slippage fraction applied to fills.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        base_currency: str = "USD",
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.base_currency = base_currency
        self.fee_rate = fee_rate
        self.slippage = slippage

        self._balances: Dict[str, float] = {base_currency: initial_balance}
        self._positions: List[Dict] = []
        self._order_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core order simulation
    # ------------------------------------------------------------------

    def simulate_order(
        self,
        symbol: str,
        side: str,
        amount_quote: float,
        price: float,
        *,
        order_type: str = "market",
        fee_rate: Optional[float] = None,
        slippage: Optional[float] = None,
        base_amount: Optional[float] = None,
    ) -> Dict:
        """Simulate a filled order, updating balances and positions.

        Args:
            symbol: Trading pair or ticker.
            side: ``"buy"`` or ``"sell"``.
            amount_quote: Order size in quote currency.
            price: Fill price (before slippage).
            order_type: ``"market"`` or ``"limit"``.
            fee_rate: Override default fee rate.
            slippage: Override default slippage.
            base_amount: Exact base units to fill (sell closes use this to
                avoid dust from slippage recalculation).

        Returns:
            Standardized order result dict.
        """
        fee = fee_rate if fee_rate is not None else self.fee_rate
        slip = slippage if slippage is not None else self.slippage

        # Apply slippage: worsens fill for buys, worsens fill for sells
        if side == "buy":
            exec_price = price * (1.0 + slip)
        else:
            exec_price = price * (1.0 - slip)

        if base_amount is not None and side == "sell":
            filled_base = base_amount
            # Recalculate amount_quote to match actual base for fee calc
            amount_quote = filled_base * exec_price
        else:
            filled_base = amount_quote / exec_price if exec_price > 0 else 0.0
        fee_cost = amount_quote * fee
        base_asset = symbol.split("/")[0]

        if side == "buy":
            self._balances[self.base_currency] = (
                self._balances.get(self.base_currency, 0.0) - amount_quote - fee_cost
            )
            self._balances[base_asset] = (
                self._balances.get(base_asset, 0.0) + filled_base
            )
            self._positions.append({
                "id": str(uuid.uuid4())[:8],
                "symbol": symbol,
                "side": "buy",
                "entry_price": exec_price,
                "base_amount": filled_base,
                "quote_amount": amount_quote,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            held = self._balances.get(base_asset, 0.0)
            if held < filled_base:
                raise ValueError(
                    f"Naked short rejected: {symbol} sell {filled_base:.6f} {base_asset} "
                    f"but only {held:.6f} held"
                )
            self._balances[self.base_currency] = (
                self._balances.get(self.base_currency, 0.0) + amount_quote - fee_cost
            )
            self._balances[base_asset] = held - filled_base
            self._positions = [
                p for p in self._positions if p["symbol"] != symbol
            ]

        result = {
            "order_id": str(uuid.uuid4())[:12],
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "requested_amount": amount_quote,
            "filled_amount": filled_base,
            "price": exec_price,
            "fee": fee_cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "filled",
            "paper": True,
        }
        self._order_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, price: float) -> Optional[Dict]:
        """Close an open position at *price*.

        Returns the order result, or ``None`` if no position exists.
        """
        positions = [p for p in self._positions if p["symbol"] == symbol]
        if not positions:
            return None

        total_base = sum(p.get("base_amount", 0) for p in positions)
        quote_amount = total_base * price
        return self.simulate_order(
            symbol, "sell", quote_amount, price, base_amount=total_base,
        )

    def close_all_positions(self, price_provider) -> List[Dict]:
        """Close every open position.

        Args:
            price_provider: Callable ``(symbol) -> float`` returning the
                current price used for the closing fill.
        """
        results = []
        for pos in list(self._positions):
            try:
                price = price_provider(pos["symbol"])
            except Exception:
                price = pos["entry_price"]
            result = self.close_position(pos["symbol"], price)
            if result:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Balance & positions
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Dict[str, float]]:
        """Return balances keyed by asset with free/used/total breakdown."""
        return {
            asset: {"free": bal, "used": 0.0, "total": bal}
            for asset, bal in self._balances.items()
        }

    def get_positions(self, price_provider=None) -> List[Dict]:
        """Return open positions, optionally with unrealized PnL.

        Args:
            price_provider: Optional callable ``(symbol) -> float``. When
                provided, each position is annotated with ``current_price``
                and ``unrealized_pnl``.
        """
        positions = []
        for pos in self._positions:
            entry = {**pos}
            if price_provider is not None:
                try:
                    current = price_provider(pos["symbol"])
                except Exception:
                    current = pos["entry_price"]
                entry["current_price"] = current
                pnl = (current - pos["entry_price"]) * pos.get("base_amount", 0)
                if pos.get("side") == "sell":
                    pnl = -pnl
                entry["unrealized_pnl"] = pnl
            positions.append(entry)
        return positions

    def get_order_history(self) -> pd.DataFrame:
        """Return executed trade history as a DataFrame."""
        if not self._order_history:
            return pd.DataFrame()
        return pd.DataFrame(self._order_history)

    def replace_order_history(self, history: pd.DataFrame) -> None:
        """Replace in-memory history (e.g., after loading from storage)."""
        self._order_history = history.to_dict("records") if not history.empty else []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def total_equity(self) -> float:
        """Quote-currency balance (excluding unrealized PnL)."""
        return self._balances.get(self.base_currency, 0.0)

    def add_balance(self, asset: str, amount: float) -> None:
        """Credit *amount* of *asset* to the paper wallet."""
        self._balances[asset] = self._balances.get(asset, 0.0) + amount
