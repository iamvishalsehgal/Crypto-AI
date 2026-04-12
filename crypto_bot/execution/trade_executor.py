"""
Trade execution engine.

Handles order placement, position management, and paper-trading simulation.
All trades pass through the :class:`RiskManager` before execution.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd

from crypto_bot.config.settings import Settings, settings as _default_settings
from crypto_bot.risk.risk_manager import PortfolioState, RiskManager
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class TradeExecutor:
    """Execute trades against a live exchange or in paper-trading mode.

    In paper-trading mode (``settings.exchange.sandbox_mode == True``), all
    orders are simulated locally without hitting the exchange API.

    Args:
        settings: Bot configuration.
        risk_manager: Shared :class:`RiskManager` instance.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or _default_settings
        self.risk_manager = risk_manager
        self.paper_mode: bool = self._settings.exchange.sandbox_mode

        # Exchange connection (real or sandbox)
        exchange_class = getattr(ccxt, self._settings.exchange.name, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {self._settings.exchange.name}")

        self._exchange: ccxt.Exchange = exchange_class({
            "apiKey": self._settings.exchange.api_key,
            "secret": self._settings.exchange.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if self.paper_mode:
            self._exchange.set_sandbox_mode(True)
            logger.info("Trade executor initialised in PAPER TRADING mode")
        else:
            logger.info("Trade executor initialised in LIVE mode on %s", self._settings.exchange.name)

        # Paper trading state
        self._paper_balance: Dict[str, float] = {"USDT": 10_000.0}
        self._paper_positions: List[Dict] = []
        self._trade_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade signal after risk validation.

        Args:
            signal: Must contain ``symbol``, ``side`` (``"buy"``/``"sell"``),
                and ``amount`` (in quote currency).  Optionally ``price`` for
                limit orders and ``order_type`` (``"market"``/``"limit"``).

        Returns:
            Order result dict with ``order_id``, ``symbol``, ``side``,
            ``amount``, ``price``, ``fee``, ``timestamp``, and ``status``.
        """
        # Build portfolio snapshot for risk check
        portfolio = self._get_portfolio_state()
        validation = self.risk_manager.validate_trade(signal, portfolio)

        if not validation.approved:
            logger.warning("Trade REJECTED: %s", validation.reason)
            return {
                "order_id": None,
                "status": "rejected",
                "reason": validation.reason,
                "risk_score": validation.risk_score,
            }

        # Adjust size if risk manager capped it
        adjusted_signal = {**signal, "amount": validation.adjusted_size}
        order_type = adjusted_signal.get("order_type", "market")

        if order_type == "limit":
            result = self._place_limit_order(
                adjusted_signal["symbol"],
                adjusted_signal["side"],
                adjusted_signal["amount"],
                adjusted_signal["price"],
            )
        else:
            result = self._place_market_order(
                adjusted_signal["symbol"],
                adjusted_signal["side"],
                adjusted_signal["amount"],
            )

        self._trade_history.append(result)
        logger.info(
            "Trade EXECUTED: %s %s %.6f %s @ %.2f  [%s]",
            result["side"],
            result["symbol"],
            result["filled_amount"],
            result["symbol"].split("/")[0],
            result["price"],
            result["status"],
        )
        return result

    def _place_market_order(
        self, symbol: str, side: str, amount_quote: float
    ) -> Dict:
        """Place a market order.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            amount_quote: Order size in quote currency.
        """
        if self.paper_mode:
            return self._simulate_order(symbol, side, amount_quote, order_type="market")

        try:
            ticker = self._exchange.fetch_ticker(symbol)
            price = ticker["ask"] if side == "buy" else ticker["bid"]
            base_amount = amount_quote / price

            order = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=base_amount,
            )

            return self._normalise_order(order)
        except ccxt.BaseError as exc:
            logger.error("Market order failed: %s", exc)
            return self._error_result(symbol, side, str(exc))

    def _place_limit_order(
        self, symbol: str, side: str, amount_quote: float, price: float
    ) -> Dict:
        """Place a limit order."""
        if self.paper_mode:
            return self._simulate_order(symbol, side, amount_quote, price=price, order_type="limit")

        try:
            base_amount = amount_quote / price
            order = self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=base_amount,
                price=price,
            )
            return self._normalise_order(order)
        except ccxt.BaseError as exc:
            logger.error("Limit order failed: %s", exc)
            return self._error_result(symbol, side, str(exc))

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.  Returns ``True`` on success."""
        if self.paper_mode:
            logger.info("Paper mode: cancelled order %s", order_id)
            return True
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.BaseError as exc:
            logger.error("Cancel failed for %s: %s", order_id, exc)
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch open orders from the exchange."""
        if self.paper_mode:
            return []
        try:
            orders = self._exchange.fetch_open_orders(symbol)
            return [self._normalise_order(o) for o in orders]
        except ccxt.BaseError as exc:
            logger.error("Failed to fetch open orders: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Balance & positions
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Dict[str, float]]:
        """Return account balances keyed by asset."""
        if self.paper_mode:
            return {
                asset: {"free": bal, "used": 0.0, "total": bal}
                for asset, bal in self._paper_balance.items()
            }
        try:
            raw = self._exchange.fetch_balance()
            return {
                asset: {
                    "free": float(raw.get(asset, {}).get("free", 0)),
                    "used": float(raw.get(asset, {}).get("used", 0)),
                    "total": float(raw.get(asset, {}).get("total", 0)),
                }
                for asset in raw.get("info", {})
                if raw.get(asset, {}).get("total", 0)
            }
        except ccxt.BaseError as exc:
            logger.error("Failed to fetch balance: %s", exc)
            return {}

    def get_positions(self) -> List[Dict]:
        """Return open positions with unrealised PnL."""
        if self.paper_mode:
            positions = []
            for pos in self._paper_positions:
                try:
                    ticker = self._exchange.fetch_ticker(pos["symbol"])
                    current = ticker["last"]
                except Exception:
                    current = pos["entry_price"]

                pnl = (current - pos["entry_price"]) * pos["base_amount"]
                if pos["side"] == "sell":
                    pnl = -pnl

                positions.append({**pos, "current_price": current, "unrealized_pnl": pnl})
            return positions
        try:
            return self._exchange.fetch_positions()
        except (ccxt.BaseError, AttributeError):
            return []

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> Dict:
        """Close all exposure on *symbol*."""
        if self.paper_mode:
            return self._paper_close(symbol)

        positions = self.get_positions()
        target = [p for p in positions if p.get("symbol") == symbol]
        if not target:
            return {"status": "no_position", "symbol": symbol}

        side = "sell" if target[0].get("side") in ("buy", "long") else "buy"
        amount = sum(abs(p.get("base_amount", p.get("contracts", 0))) for p in target)
        try:
            order = self._exchange.create_order(symbol, "market", side, amount)
            return self._normalise_order(order)
        except ccxt.BaseError as exc:
            logger.error("Close position failed for %s: %s", symbol, exc)
            return self._error_result(symbol, side, str(exc))

    def close_all_positions(self) -> List[Dict]:
        """Close every open position."""
        results = []
        if self.paper_mode:
            for pos in list(self._paper_positions):
                results.append(self._paper_close(pos["symbol"]))
            return results

        for pos in self.get_positions():
            results.append(self.close_position(pos["symbol"]))
        return results

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trade_history(self) -> pd.DataFrame:
        """Return executed trade history as a DataFrame."""
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_history)

    # ------------------------------------------------------------------
    # Paper trading helpers
    # ------------------------------------------------------------------

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        amount_quote: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict:
        """Simulate order execution for paper trading."""
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            exec_price = price or (ticker["ask"] if side == "buy" else ticker["bid"])
        except Exception:
            exec_price = price or 0.0

        fee_rate = self._settings.backtesting.transaction_fee
        slippage = self._settings.backtesting.slippage

        if side == "buy":
            exec_price *= 1 + slippage  # Worse fill for buys
        else:
            exec_price *= 1 - slippage  # Worse fill for sells

        base_amount = amount_quote / exec_price if exec_price > 0 else 0.0
        fee = amount_quote * fee_rate
        base_asset = symbol.split("/")[0]

        if side == "buy":
            self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) - amount_quote - fee
            self._paper_balance[base_asset] = self._paper_balance.get(base_asset, 0) + base_amount
            self._paper_positions.append({
                "id": str(uuid.uuid4())[:8],
                "symbol": symbol,
                "side": "buy",
                "entry_price": exec_price,
                "base_amount": base_amount,
                "quote_amount": amount_quote,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + amount_quote - fee
            self._paper_balance[base_asset] = self._paper_balance.get(base_asset, 0) - base_amount
            # Remove from positions
            self._paper_positions = [
                p for p in self._paper_positions if p["symbol"] != symbol
            ]

        return {
            "order_id": str(uuid.uuid4())[:12],
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "requested_amount": amount_quote,
            "filled_amount": base_amount,
            "price": exec_price,
            "fee": fee,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "filled",
            "paper": True,
        }

    def _paper_close(self, symbol: str) -> Dict:
        """Close a paper position."""
        target = [p for p in self._paper_positions if p["symbol"] == symbol]
        if not target:
            return {"status": "no_position", "symbol": symbol}

        total_base = sum(p["base_amount"] for p in target)
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            price = ticker["bid"]
        except Exception:
            price = target[0]["entry_price"]

        quote_amount = total_base * price
        return self._simulate_order(symbol, "sell", quote_amount, price=price)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_order(self, order: Dict) -> Dict:
        """Normalise a ccxt order response to a consistent format."""
        return {
            "order_id": order.get("id", ""),
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "order_type": order.get("type", ""),
            "requested_amount": order.get("amount", 0),
            "filled_amount": order.get("filled", 0),
            "price": order.get("average", order.get("price", 0)),
            "fee": order.get("fee", {}).get("cost", 0) if isinstance(order.get("fee"), dict) else 0,
            "timestamp": order.get("datetime", datetime.now(timezone.utc).isoformat()),
            "status": order.get("status", "unknown"),
            "paper": False,
        }

    @staticmethod
    def _error_result(symbol: str, side: str, error: str) -> Dict:
        return {
            "order_id": None,
            "symbol": symbol,
            "side": side,
            "status": "error",
            "reason": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _get_portfolio_state(self) -> PortfolioState:
        """Build a portfolio state snapshot for risk validation."""
        balance = self.get_balance()
        usdt = balance.get("USDT", {}).get("total", 0)
        positions = self.get_positions() if self.paper_mode else []
        equity = usdt + sum(p.get("unrealized_pnl", 0) for p in positions)

        return PortfolioState(
            balance=usdt,
            equity=equity,
            open_positions=positions if self.paper_mode else self._paper_positions,
            daily_pnl=self.risk_manager._daily_pnl,
        )
