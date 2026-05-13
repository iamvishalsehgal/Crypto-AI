"""P&L Tracker -- tracks trades from open to close with real P&L computation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    amount: float
    timestamp: str
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"


class PnLTracker:
    """Tracks every trade from open to close, computes real P&L."""

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.initial_balance = initial_balance
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self._peak_equity: float = initial_balance

    def record_entry(self, trade_result: dict) -> None:
        t = Trade(
            trade_id=trade_result.get("order_id", "?"),
            symbol=trade_result["symbol"],
            side=trade_result["side"],
            entry_price=trade_result.get("price", 0),
            amount=trade_result.get("filled_amount", 0),
            timestamp=trade_result.get("timestamp", ""),
        )
        self.trades.append(t)

    def record_exit(self, symbol: str, exit_price: float) -> Optional[Trade]:
        for t in reversed(self.trades):
            if t.symbol == symbol and t.status == "open":
                t.exit_price = exit_price
                t.exit_timestamp = datetime.now(timezone.utc).isoformat()
                if t.side == "buy":
                    t.pnl = (exit_price - t.entry_price) * t.amount
                else:
                    t.pnl = (t.entry_price - exit_price) * t.amount
                t.pnl_pct = t.pnl / (t.entry_price * t.amount) * 100 if t.entry_price else 0
                t.status = "closed"
                self.closed_trades.append(t)
                return t
        return None

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        return wins / len(self.closed_trades) * 100

    @property
    def open_count(self) -> int:
        return sum(1 for t in self.trades if t.status == "open")

    def summary(self) -> dict:
        total_closed = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        return {
            "total_trades": len(self.trades),
            "closed_trades": total_closed,
            "open_trades": self.open_count,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate_pct": round(self.win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(
                abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
                if losses and sum(t.pnl for t in losses) != 0 else 0, 2
            ),
            "best_trade": round(max((t.pnl for t in self.closed_trades), default=0), 2),
            "worst_trade": round(min((t.pnl for t in self.closed_trades), default=0), 2),
        }
