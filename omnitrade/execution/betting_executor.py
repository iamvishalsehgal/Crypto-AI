"""
Betting execution engine — paper bankroll management and bet settlement.

Provides the same interface as :class:`TradeExecutor` and
:class:`StockExecutor` so the AssetRouter can dispatch bets transparently.
"""

from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from omnitrade.config.settings import Settings, settings as _default_settings
from omnitrade.config.asset_types import BACK, LAY, PASS, UnifiedSignal
from omnitrade.risk.risk_manager import PortfolioState
from omnitrade.utils.logger import get_logger
from omnitrade.utils.odds import american_to_decimal

logger = get_logger(__name__)

BET_STATUS_OPEN = "open"
BET_STATUS_WON = "won"
BET_STATUS_LOST = "lost"
BET_STATUS_VOID = "void"


class BettingExecutor:
    """Execute (paper) bets and manage a simulated bankroll.

    In paper mode, all bets are tracked in-memory. Live sportsbook API
    integration is deferred to a future phase.

    Args:
        risk_manager: :class:`BettingRiskManager` instance.
        settings: Bot configuration.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        self._settings = settings or _default_settings
        self._betting = self._settings.betting
        self.paper_mode = True

        self._bankroll = self._betting.bankroll
        self._initial_bankroll = self._bankroll
        self._bets: List[Dict] = []
        self._bet_history: List[Dict] = []

        from omnitrade.risk.betting_risk import BettingRiskManager
        self.risk_manager = risk_manager or BettingRiskManager(self._settings)

        logger.info(
            "BettingExecutor initialised — paper bankroll: $%.2f, Kelly=%.0f%%",
            self._bankroll,
            self._betting.kelly_fraction * 100,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def place_order(self, signal: UnifiedSignal) -> Dict:
        """Place a bet from a UnifiedSignal.

        Args:
            signal: Must have asset_type=BET, side=BACK/LAY, confidence,
                    and metadata with model_prob, implied_prob, odds.

        Returns:
            Bet result dict.
        """
        if signal.asset_type.value != "bet":
            return {
                "order_id": None, "status": "error",
                "reason": f"Wrong asset type: {signal.asset_type.value}",
                "symbol": signal.symbol,
            }

        if signal.side not in (BACK, LAY):
            return {
                "order_id": None, "status": "skipped",
                "reason": f"Signal is {signal.side}, not actionable",
                "symbol": signal.symbol,
            }

        model_prob = signal.metadata.get("model_prob", signal.confidence)
        implied_prob = signal.metadata.get("implied_prob", 0.5)
        odds = signal.metadata.get("odds", signal.price if signal.price > 0 else -110)
        sport_key = signal.metadata.get("sport_key", "")

        validation = self.risk_manager.validate_bet(
            symbol=signal.symbol,
            model_prob=model_prob,
            implied_prob=implied_prob,
            odds_american=odds,
            bankroll=self._bankroll,
            sport_key=sport_key,
        )

        if not validation.approved:
            logger.info("Bet REJECTED: %s — %s", signal.symbol, validation.reason)
            return {
                "order_id": None, "status": "rejected",
                "reason": validation.reason,
                "symbol": signal.symbol,
                "risk_score": validation.risk_score,
            }

        stake = validation.adjusted_stake
        bet_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()

        self._bankroll -= stake
        self.risk_manager.record_bet(signal.symbol, stake, sport_key)

        bet = {
            "order_id": bet_id,
            "symbol": signal.symbol,
            "side": signal.side,
            "stake": round(stake, 2),
            "odds": odds,
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "edge": round(model_prob - implied_prob, 4),
            "status": BET_STATUS_OPEN,
            "timestamp": now,
            "paper": True,
            "payout": 0.0,
            "pnl": 0.0,
        }
        self._bets.append(bet)
        self._bet_history.append(bet.copy())

        logger.info(
            "Bet PLACED: %s %s $%.2f @ %+.0f (edge=%.1f%%, bankroll=$%.2f)",
            signal.side, signal.symbol, stake, odds,
            (model_prob - implied_prob) * 100, self._bankroll,
        )
        return {**bet, "status": "filled"}

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle_bet(self, bet_id: str, outcome: str) -> Dict:
        """Settle a bet after the match resolves.

        Args:
            bet_id: The bet's order_id.
            outcome: ``"win"``, ``"loss"``, ``"void"``.

        Returns:
            Updated bet dict with pnl.
        """
        for bet in self._bets:
            if bet["order_id"] == bet_id:
                break
        else:
            return {"order_id": bet_id, "status": "error", "reason": "Bet not found"}

        odds = bet["odds"]
        stake = bet["stake"]
        decimal_odds = american_to_decimal(odds)

        if outcome == "win":
            payout = stake * decimal_odds
            pnl = payout - stake
            bet["status"] = BET_STATUS_WON
            self.risk_manager.record_win()
        elif outcome == "loss":
            payout = 0.0
            pnl = -stake
            bet["status"] = BET_STATUS_LOST
            self.risk_manager.record_loss()
        else:
            payout = stake  # void = stake returned
            pnl = 0.0
            bet["status"] = BET_STATUS_VOID

        self._bankroll += payout
        bet["payout"] = round(payout, 2)
        bet["pnl"] = round(pnl, 2)
        bet["settled_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Bet SETTLED: %s %s $%.2f → %s, PnL=$%.2f, bankroll=$%.2f",
            bet["symbol"], bet_id, bet["stake"], bet["status"], pnl, self._bankroll,
        )
        return bet.copy()

    def settle_all_mock(self, win_rate: float = 0.45) -> List[Dict]:
        """Mock-settle all open bets using a simple win rate.

        Useful for testing and dry-run simulations.
        """
        results = []
        for bet in list(self._bets):
            if bet["status"] != BET_STATUS_OPEN:
                continue
            outcome = "win" if random.random() < win_rate else "loss"
            result = self.settle_bet(bet["order_id"], outcome)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Balance & positions
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Dict[str, float]]:
        return {
            "USD": {
                "free": self._bankroll,
                "used": sum(b.get("stake", 0) for b in self._bets if b.get("status") == BET_STATUS_OPEN),
                "total": self._bankroll,
            }
        }

    def get_positions(self) -> List[Dict]:
        return [
            {
                "id": b.get("order_id", ""),
                "symbol": b.get("symbol", ""),
                "side": b.get("side", ""),
                "entry_price": b.get("odds", 0),
                "shares": b.get("stake", 0),
                "current_price": b.get("odds", 0),
                "unrealized_pnl": 0.0,
                "status": b.get("status", ""),
            }
            for b in self._bets if b.get("status") == BET_STATUS_OPEN
        ]

    def close_position(self, symbol: str) -> Dict:
        # Cash out not applicable for betting — void any open bet on symbol
        for bet in self._bets:
            if bet["symbol"] == symbol and bet["status"] == BET_STATUS_OPEN:
                return self.settle_bet(bet["order_id"], "void")
        return {"status": "no_position", "symbol": symbol}

    def close_all_positions(self) -> List[Dict]:
        return [self.settle_bet(b["order_id"], "void")
                for b in self._bets if b["status"] == BET_STATUS_OPEN]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_bankroll(self) -> float:
        return self._bankroll

    def get_bet_history(self) -> pd.DataFrame:
        if not self._bet_history:
            return pd.DataFrame()
        return pd.DataFrame(self._bet_history)

    def get_stats(self) -> Dict[str, Any]:
        settled = [b for b in self._bet_history if b.get("status") in (BET_STATUS_WON, BET_STATUS_LOST, BET_STATUS_VOID)]
        wins = [b for b in settled if b.get("status") == BET_STATUS_WON]
        losses = [b for b in settled if b.get("status") == BET_STATUS_LOST]
        total_pnl = sum(b.get("pnl", 0) for b in settled)
        roi = (total_pnl / self._initial_bankroll * 100) if self._initial_bankroll else 0

        return {
            "bankroll": round(self._bankroll, 2),
            "initial_bankroll": round(self._initial_bankroll, 2),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(roi, 2),
            "total_bets": len(settled),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(settled) * 100, 1) if settled else 0,
            "open_bets": len([b for b in self._bets if b.get("status") == BET_STATUS_OPEN]),
            "avg_stake": round(sum(b.get("stake", 0) for b in settled) / len(settled), 2) if settled else 0,
            "risk_status": self.risk_manager.get_status(),
        }

    def _get_portfolio_state(self) -> PortfolioState:
        open_bets = self.get_positions()
        return PortfolioState(
            balance=self._bankroll,
            equity=self._bankroll + sum(b.get("unrealized_pnl", 0) for b in open_bets),
            open_positions=open_bets,
            daily_pnl=0.0,
        )

