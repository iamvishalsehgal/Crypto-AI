"""
Stock trade execution via Alpaca Markets with paper-trading simulation.

Provides the same interface as :class:`TradeExecutor` so the AssetRouter
can dispatch stock signals transparently.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.config.asset_types import UnifiedSignal
from omnitrade.risk.risk_manager import PortfolioState, RiskManager
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


class StockExecutor:
    """Execute stock trades via Alpaca (paper or live).

    In paper mode (default), orders are simulated locally against real-time
    prices from yfinance. Live mode requires valid Alpaca API keys.

    Args:
        settings: Bot configuration.
        risk_manager: Shared :class:`RiskManager` for position sizing/validation.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or _default_settings
        self.risk_manager = risk_manager
        self._stock = self._settings.stock
        self.paper_mode = self._stock.alpaca_paper

        self._paper_balance = 10_000.0
        self._paper_positions: List[Dict] = []
        self._trade_history: List[Dict] = []
        self._day_trades: List[float] = []  # timestamps for PDT tracking
        self._alpaca = None

        if not self.paper_mode and self._stock.alpaca_api_key:
            self._init_alpaca()

        logger.info(
            "StockExecutor initialised in %s mode for %d tickers",
            "PAPER" if self.paper_mode else "LIVE",
            len(self._stock.supported_tickers),
        )

    def _init_alpaca(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            endpoint = "https://paper-api.alpaca.markets" if self.paper_mode else "https://api.alpaca.markets"
            self._alpaca = TradingClient(
                api_key=self._stock.alpaca_api_key,
                secret_key=self._stock.alpaca_api_secret,
                paper=self.paper_mode,
            )
            logger.info("Alpaca trading client connected (%s)", "paper" if self.paper_mode else "live")
        except ImportError:
            logger.warning("alpaca-py not installed — stock execution in paper-only mode")
            self.paper_mode = True

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_trade(self, signal: UnifiedSignal) -> Dict:
        """Execute a stock trade signal after risk validation.

        Args:
            signal: UnifiedSignal with symbol, side, amount, confidence.

        Returns:
            Order result dict with standard keys.
        """
        portfolio = self._get_portfolio_state()
        trade_signal = {
            "symbol": signal.symbol,
            "side": signal.side.lower(),
            "amount": signal.amount,
        }
        validation = self.risk_manager.validate_trade(trade_signal, portfolio)

        if not validation.approved:
            logger.warning("Stock trade REJECTED: %s", validation.reason)
            return {
                "order_id": None,
                "status": "rejected",
                "reason": validation.reason,
                "risk_score": validation.risk_score,
            }

        adjusted_amount = validation.adjusted_size

        if self.paper_mode:
            return self._simulate_market_order(signal.symbol, signal.side.lower(), adjusted_amount)

        return self._place_alpaca_market_order(signal.symbol, signal.side.lower(), adjusted_amount, signal.price)

    def _simulate_market_order(self, symbol: str, side: str, amount_usd: float) -> Dict:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                raise RuntimeError(f"No price data available for {symbol}")
            price = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.error("Cannot fetch price for %s: %s", symbol, exc)
            return {
                "order_id": None, "status": "error",
                "symbol": symbol, "side": side,
                "reason": f"Price fetch failed: {exc}",
            }

        fee = amount_usd * 0.001  # 10 bps estimated
        shares = amount_usd / price if price > 0 else 0

        order_id = str(uuid.uuid4())[:12]

        if side == "buy":
            self._paper_balance -= amount_usd + fee
            self._paper_positions.append({
                "id": order_id,
                "symbol": symbol,
                "side": "buy",
                "entry_price": price,
                "shares": shares,
                "amount_usd": amount_usd,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._day_trades.append(time.time())
        else:
            self._paper_balance += amount_usd - fee
            self._paper_positions = [p for p in self._paper_positions if p["symbol"] != symbol]

        result = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": "market",
            "requested_amount": amount_usd,
            "filled_amount": shares,
            "price": price,
            "fee": fee,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "filled",
            "paper": True,
        }
        self._trade_history.append(result)
        logger.info("Paper stock trade: %s %s %s shares @ $%.2f", side, symbol, shares, price)
        return result

    def _place_alpaca_market_order(self, symbol: str, side: str, amount_usd: float, price: float) -> Dict:
        if self._alpaca is None:
            return {"order_id": None, "status": "error", "reason": "Alpaca client not initialised"}

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            notional = round(amount_usd, 2)
            order_req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._alpaca.submit_order(order_req)

            result = {
                "order_id": str(order.id),
                "symbol": symbol,
                "side": side,
                "order_type": "market",
                "requested_amount": amount_usd,
                "filled_amount": float(order.filled_qty or 0),
                "price": float(order.filled_avg_price or price),
                "fee": 0.0,
                "timestamp": order.submitted_at.isoformat() if order.submitted_at else "",
                "status": order.status,
                "paper": self.paper_mode,
            }
            self._trade_history.append(result)
            return result
        except Exception as exc:
            logger.error("Alpaca order failed: %s", exc)
            return {"order_id": None, "symbol": symbol, "side": side, "status": "error", "reason": str(exc)}

    # ------------------------------------------------------------------
    # Balance & positions
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Dict[str, float]]:
        if self.paper_mode:
            return {"USD": {"free": self._paper_balance, "used": 0.0, "total": self._paper_balance}}

        if self._alpaca is None:
            return {"USD": {"free": 0, "used": 0, "total": 0}}

        try:
            account = self._alpaca.get_account()
            return {
                "USD": {
                    "free": float(account.cash) if account.cash else 0,
                    "used": float(account.long_market_value or 0),
                    "total": float(account.equity or account.cash or 0),
                }
            }
        except Exception as exc:
            logger.error("Failed to fetch Alpaca balance: %s", exc)
            return {}

    def get_positions(self) -> List[Dict]:
        if self.paper_mode:
            positions = []
            for pos in self._paper_positions:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(pos["symbol"])
                    hist = ticker.history(period="1d", interval="1m")
                    current = float(hist["Close"].iloc[-1]) if not hist.empty else pos["entry_price"]
                except Exception:
                    current = pos["entry_price"]

                pnl = (current - pos["entry_price"]) * pos["shares"]
                positions.append({**pos, "current_price": current, "unrealized_pnl": pnl})
            return positions

        if self._alpaca is None:
            return []

        try:
            alpaca_positions = self._alpaca.get_all_positions()
            return [
                {
                    "id": p.asset_id or "",
                    "symbol": p.symbol,
                    "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                    "entry_price": float(p.avg_entry_price or 0),
                    "shares": float(p.qty or 0),
                    "current_price": float(p.current_price or 0),
                    "unrealized_pnl": float(p.unrealized_pl or 0),
                }
                for p in alpaca_positions
            ]
        except Exception as exc:
            logger.error("Failed to fetch Alpaca positions: %s", exc)
            return []

    def close_position(self, symbol: str) -> Dict:
        if self.paper_mode:
            target = [p for p in self._paper_positions if p["symbol"] == symbol]
            if not target:
                return {"status": "no_position", "symbol": symbol}
            pos = target[0]
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else pos["entry_price"]
            except Exception:
                price = pos["entry_price"]
            amount = pos["shares"] * price
            return self._simulate_market_order(symbol, "sell", amount)

        if self._alpaca is None:
            return {"status": "error", "reason": "Alpaca not connected", "symbol": symbol}

        try:
            self._alpaca.close_position(symbol)
            return {"status": "filled", "symbol": symbol}
        except Exception as exc:
            logger.error("Close position failed for %s: %s", symbol, exc)
            return {"status": "error", "symbol": symbol, "reason": str(exc)}

    def close_all_positions(self) -> List[Dict]:
        if self.paper_mode:
            results = []
            for pos in list(self._paper_positions):
                results.append(self.close_position(pos["symbol"]))
            return results

        if self._alpaca is None:
            return []

        try:
            self._alpaca.close_all_positions()
            return [{"status": "liquidated", "reason": "close_all"}]
        except Exception as exc:
            logger.error("Close all positions failed: %s", exc)
            return [{"status": "error", "reason": str(exc)}]

    # ------------------------------------------------------------------
    # PDT check
    # ------------------------------------------------------------------

    def _check_pdt_rule(self) -> bool:
        """Check if Pattern Day Trader rule would block new trades.

        PDT triggers when account < $25K and >3 day trades in 5 rolling days.
        """
        now = time.time()
        five_days_ago = now - 5 * 24 * 3600
        recent_day_trades = [t for t in self._day_trades if t > five_days_ago]
        if len(recent_day_trades) >= 3:
            balance = self.get_balance()
            equity = balance.get("USD", {}).get("total", 0)
            if equity < 25_000:
                logger.warning("PDT rule would block: equity=$%.0f, day_trades=%d", equity, len(recent_day_trades))
                return True
        return False

    def get_trade_history(self) -> pd.DataFrame:
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_history)

    def _get_portfolio_state(self) -> PortfolioState:
        balance = self.get_balance()
        usd = balance.get("USD", {}).get("total", 0)
        positions = self.get_positions()
        equity = usd + sum(p.get("unrealized_pnl", 0) for p in positions)

        return PortfolioState(
            balance=usd,
            equity=equity,
            open_positions=positions,
            daily_pnl=0.0,
        )
