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

from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.execution.paper_wallet import PaperWallet
from omnitrade.risk.risk_manager import PortfolioState, RiskManager
from omnitrade.utils.logger import get_logger

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

        # ── Safety lock: enforce trading_mode guard ──────────────────────
        mode = self._settings.trading_mode
        if mode == "paper":
            self.paper_mode = True
        elif mode == "live":
            if self._settings.live_trading_confirmed != "true":
                raise RuntimeError(
                    "LIVE TRADING MODE requires LIVE_TRADING_CONFIRMED=true. "
                    "Set this environment variable to acknowledge that you want "
                    "real orders placed on the exchange."
                )
            logger.critical(
                "!!!!!!!!!! TRADE EXECUTOR INITIALISED IN LIVE MODE — "
                "REAL ORDERS WILL BE PLACED !!!!!!!!!!"
            )
        else:
            raise RuntimeError(f"Invalid trading_mode: {mode!r}")

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
        self._wallet = PaperWallet(
            initial_balance=10_000.0,
            base_currency="USDT",
            fee_rate=self._settings.backtesting.transaction_fee,
            slippage=self._settings.backtesting.slippage,
        )
        self._order_history: List[Dict] = []  # live-mode only

    # ------------------------------------------------------------------
    # Price helpers for paper mode
    # ------------------------------------------------------------------

    def _paper_price(self, symbol: str) -> float:
        """Fetch live ticker price for paper position valuation."""
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, signal: Dict) -> Dict:
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

        if not self.paper_mode:
            self._order_history.append(result)
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
            try:
                ticker = self._exchange.fetch_ticker(symbol)
                price = ticker["ask"] if side == "buy" else ticker["bid"]
            except Exception:
                price = 0.0
            return self._wallet.simulate_order(symbol, side, amount_quote, price, order_type="market")

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
            return self._wallet.simulate_order(symbol, side, amount_quote, price, order_type="limit")

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
            return self._wallet.get_balance()
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
            return self._wallet.get_positions(price_provider=self._paper_price)
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
            try:
                price = self._paper_price(symbol)
            except Exception:
                price = 0.0
            result = self._wallet.close_position(symbol, price)
            return result or {"status": "no_position", "symbol": symbol}

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
        if self.paper_mode:
            return self._wallet.close_all_positions(price_provider=self._paper_price)

        results = []
        for pos in self.get_positions():
            results.append(self.close_position(pos["symbol"]))
        return results

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trade_history(self) -> pd.DataFrame:
        """Return executed trade history as a DataFrame."""
        if self.paper_mode:
            return self._wallet.get_order_history()
        if not self._order_history:
            return pd.DataFrame()
        return pd.DataFrame(self._order_history)

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
            open_positions=positions,
            daily_pnl=getattr(self.risk_manager, "_daily_pnl", 0.0),
        )
