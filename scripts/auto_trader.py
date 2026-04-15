#!/usr/bin/env python3
"""
Fully autonomous trading bot — production-hardened.

Runs the trading pipeline in a continuous loop with:
- Multi-indicator signal generation (real indicator values, not defaults)
- Retry/reconnection on exchange failures
- Circuit breaker: halts on repeated errors or drawdown limits
- Full P&L tracking with entry/exit matching
- --test mode for fast dry-run validation

Usage:
    python scripts/auto_trader.py                         # Run forever (paper)
    python scripts/auto_trader.py --test                  # Quick 3-cycle test
    python scripts/auto_trader.py --cycles 50             # Run 50 cycles
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from crypto_bot.config.settings import settings
from crypto_bot.utils.logger import get_logger
from crypto_bot.data.collectors.market_data import MarketDataCollector
from crypto_bot.features.technical import TechnicalFeatures
from crypto_bot.risk.risk_manager import RiskManager
from crypto_bot.risk.safety import SafetyGuard
from crypto_bot.execution.trade_executor import TradeExecutor

logger = get_logger("crypto_bot.auto_trader")

# ── Directories ──────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
TRADES_DIR = PROJECT_ROOT / "reports" / "trades"
LOGS_DIR = PROJECT_ROOT / "reports" / "logs"


def ensure_dirs() -> None:
    for d in [REPORTS_DIR, TRADES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Retry decorator
# =====================================================================

def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Decorator that retries a function on exception with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
                        wait *= backoff
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# =====================================================================
# Circuit breaker
# =====================================================================

class CircuitBreaker:
    """Trips after *threshold* consecutive failures within *window* seconds."""

    def __init__(self, threshold: int = 5, cooldown: int = 300) -> None:
        self.threshold = threshold
        self.cooldown = cooldown  # seconds to wait after tripping
        self._failures: int = 0
        self._tripped_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._tripped_at is None:
            return False
        if time.time() - self._tripped_at > self.cooldown:
            self.reset()
            logger.info("Circuit breaker reset after cooldown")
            return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._tripped_at = time.time()
            logger.critical(
                "CIRCUIT BREAKER TRIPPED after %d failures — pausing for %ds",
                self._failures, self.cooldown,
            )

    def record_success(self) -> None:
        self._failures = 0

    def reset(self) -> None:
        self._failures = 0
        self._tripped_at = None


# =====================================================================
# P&L Tracker
# =====================================================================

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


# =====================================================================
# Report writers
# =====================================================================

TRADE_CSV_FIELDS = [
    "timestamp", "symbol", "side", "price", "filled_amount",
    "fee", "status", "pnl", "balance_after",
]


def append_trade_csv(trade: dict) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = TRADES_DIR / f"trades_{today}.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(trade)


def write_daily_summary(stats: dict) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"summary_{today}.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def write_status_file(cycle: int, portfolio: dict, signals: dict, pnl: dict) -> None:
    path = REPORTS_DIR / "status.json"
    data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "mode": "paper",
        "portfolio": portfolio,
        "pnl_summary": pnl,
        "last_signals": signals,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =====================================================================
# Signal generator
# =====================================================================

SIGNAL_CONFIG_PATH = PROJECT_ROOT / "config" / "signal_config.json"

# Default signal parameters — auto-tuner can override these via signal_config.json
DEFAULT_SIGNAL_PARAMS = {
    # RSI thresholds
    "rsi_oversold": 30,
    "rsi_weak_buy": 40,
    "rsi_overbought": 70,
    "rsi_weak_sell": 60,
    # RSI weights
    "rsi_strong_weight": 0.25,
    "rsi_weak_weight": 0.10,
    # MACD weights
    "macd_crossover_weight": 0.20,
    "macd_trend_weight": 0.10,
    # Bollinger Band thresholds
    "bb_buy_strong": 0.15,
    "bb_buy_weak": 0.30,
    "bb_sell_strong": 0.85,
    "bb_sell_weak": 0.70,
    "bb_strong_weight": 0.20,
    "bb_weak_weight": 0.08,
    # Stochastic thresholds
    "stoch_oversold": 20,
    "stoch_overbought": 80,
    "stoch_weight": 0.15,
    # EMA weight
    "ema_weight": 0.15,
    # ADX filter
    "adx_weak_threshold": 20,
    "adx_dampen_factor": 0.5,
    # Decision threshold
    "min_confidence": 0.45,
    # Position sizing (fraction of balance)
    "position_size_pct": 0.10,
}


def load_signal_params() -> dict:
    """Load signal parameters from config file, falling back to defaults."""
    params = DEFAULT_SIGNAL_PARAMS.copy()
    if SIGNAL_CONFIG_PATH.exists():
        try:
            with open(SIGNAL_CONFIG_PATH) as f:
                overrides = json.load(f)
            params.update(overrides)
            logger.info("Loaded signal config from %s", SIGNAL_CONFIG_PATH)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load signal config: %s — using defaults", exc)
    return params


class SignalGenerator:
    """Multi-indicator confluence signal using REAL column names from TechnicalFeatures.

    Parameters are loaded from config/signal_config.json if it exists,
    otherwise defaults are used. The auto-tuner can write improved
    parameters to that file.

    Columns used (from crypto_bot.features.technical):
      rsi, macd, macd_signal, macd_histogram,
      bb_percent_b, bb_bandwidth,
      stoch_k, stoch_d, adx, atr,
      ema_9, ema_21, ema_50, ema_200
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        self.p = params if params is not None else load_signal_params()

    def generate(self, features_row: dict, prev_row: Optional[dict] = None) -> dict:
        """Score a single row of features and return a signal dict."""
        p = self.p

        rsi = features_row.get("rsi")
        macd = features_row.get("macd")
        macd_sig = features_row.get("macd_signal")
        macd_hist = features_row.get("macd_histogram")
        bb_pct = features_row.get("bb_percent_b")
        stoch_k = features_row.get("stoch_k")
        stoch_d = features_row.get("stoch_d")
        adx = features_row.get("adx")
        ema_9 = features_row.get("ema_9")
        ema_21 = features_row.get("ema_21")
        ema_50 = features_row.get("ema_50")
        close = features_row.get("close")

        # Safely check None (indicators may be NaN during warm-up)
        def ok(*vals) -> bool:
            return all(v is not None and v == v for v in vals)  # v == v filters NaN

        buy_score = 0.0
        sell_score = 0.0
        reasons_buy: list[str] = []
        reasons_sell: list[str] = []

        # 1. RSI
        if ok(rsi):
            if rsi < p["rsi_oversold"]:
                buy_score += p["rsi_strong_weight"]
                reasons_buy.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < p["rsi_weak_buy"]:
                buy_score += p["rsi_weak_weight"]
            elif rsi > p["rsi_overbought"]:
                sell_score += p["rsi_strong_weight"]
                reasons_sell.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > p["rsi_weak_sell"]:
                sell_score += p["rsi_weak_weight"]

        # 2. MACD histogram direction + crossover
        if ok(macd, macd_sig, macd_hist):
            if macd_hist > 0 and macd < 0:
                buy_score += p["macd_crossover_weight"]
                reasons_buy.append("MACD bullish crossover")
            elif macd_hist > 0:
                buy_score += p["macd_trend_weight"]
            if macd_hist < 0 and macd > 0:
                sell_score += p["macd_crossover_weight"]
                reasons_sell.append("MACD bearish crossover")
            elif macd_hist < 0:
                sell_score += p["macd_trend_weight"]

        # 3. Bollinger %B
        if ok(bb_pct):
            if bb_pct < p["bb_buy_strong"]:
                buy_score += p["bb_strong_weight"]
                reasons_buy.append(f"BB squeeze low ({bb_pct:.2f})")
            elif bb_pct < p["bb_buy_weak"]:
                buy_score += p["bb_weak_weight"]
            if bb_pct > p["bb_sell_strong"]:
                sell_score += p["bb_strong_weight"]
                reasons_sell.append(f"BB overbought ({bb_pct:.2f})")
            elif bb_pct > p["bb_sell_weak"]:
                sell_score += p["bb_weak_weight"]

        # 4. Stochastic
        if ok(stoch_k, stoch_d):
            if stoch_k < p["stoch_oversold"] and stoch_d < p["stoch_oversold"]:
                buy_score += p["stoch_weight"]
                reasons_buy.append(f"Stoch oversold (K={stoch_k:.0f})")
            if stoch_k > p["stoch_overbought"] and stoch_d > p["stoch_overbought"]:
                sell_score += p["stoch_weight"]
                reasons_sell.append(f"Stoch overbought (K={stoch_k:.0f})")

        # 5. EMA trend alignment
        if ok(ema_9, ema_21, ema_50, close):
            if close > ema_9 > ema_21 > ema_50:
                buy_score += p["ema_weight"]
                reasons_buy.append("EMAs bullish aligned")
            elif close < ema_9 < ema_21 < ema_50:
                sell_score += p["ema_weight"]
                reasons_sell.append("EMAs bearish aligned")

        # 6. ADX trend strength filter — only trade in strong trends
        if ok(adx):
            if adx < p["adx_weak_threshold"]:
                # Weak trend, dampen signals
                buy_score *= p["adx_dampen_factor"]
                sell_score *= p["adx_dampen_factor"]

        # ── Decision ──
        min_conf = p["min_confidence"]
        confidence = max(buy_score, sell_score)
        if buy_score >= min_conf and buy_score > sell_score:
            signal = "BUY"
            reasons = reasons_buy
        elif sell_score >= min_conf and sell_score > buy_score:
            signal = "SELL"
            reasons = reasons_sell
        else:
            signal = "HOLD"
            reasons = ["No strong confluence"]
            confidence = max(buy_score, sell_score)

        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "buy_score": round(buy_score, 3),
            "sell_score": round(sell_score, 3),
            "reasons": reasons,
            "indicators": {
                "rsi": round(rsi, 2) if ok(rsi) else None,
                "macd_hist": round(macd_hist, 4) if ok(macd_hist) else None,
                "bb_pct": round(bb_pct, 4) if ok(bb_pct) else None,
                "stoch_k": round(stoch_k, 1) if ok(stoch_k) else None,
                "adx": round(adx, 1) if ok(adx) else None,
            },
        }


# =====================================================================
# Auto Trader (main class)
# =====================================================================

class AutoTrader:
    def __init__(self) -> None:
        self.market = MarketDataCollector(settings)
        self.tech = TechnicalFeatures(settings)
        self.risk = RiskManager(settings)
        self.safety = SafetyGuard(settings)
        self.executor = TradeExecutor(risk_manager=self.risk, settings=settings)
        self.signal_gen = SignalGenerator()
        self.pnl = PnLTracker(
            initial_balance=self.executor.get_balance().get(
                "USDT", {}
            ).get("total", 10_000.0),
        )
        self.circuit = CircuitBreaker(threshold=5, cooldown=300)

        # Reset daily counters
        balance = self.executor.get_balance()
        usdt = balance.get("USDT", {}).get("total", 10_000)
        self.risk.reset_daily(usdt)
        self.safety.update_equity_peak(usdt)

        logger.info(
            "Safety config: reserve=%.0f%% | min_balance=$%.0f | "
            "daily_fee_cap=$%.0f | max_trades/hr=%d | max_trades/day=%d | "
            "cooldown=%ds | kill_on_%d_losses | kill_on_%.0f%%_drop",
            settings.safety.reserve_balance_pct * 100,
            settings.safety.min_balance_to_trade,
            settings.safety.max_daily_fees,
            settings.safety.max_trades_per_hour,
            settings.safety.max_trades_per_day,
            settings.safety.cooldown_after_trade_sec,
            settings.safety.kill_on_consecutive_losses,
            settings.safety.kill_on_equity_drop_pct * 100,
        )

    @retry(max_attempts=3, delay=2.0)
    def _fetch_data(self, symbol: str):
        return self.market.fetch_ohlcv(symbol, timeframe="1h", limit=500)

    def run_cycle(self, symbol: str) -> dict:
        """One full collect → analyse → trade cycle with full safety checks."""

        # ── 0. Circuit breaker ──
        if self.circuit.is_open:
            return {"symbol": symbol, "signal": "PAUSED", "reason": "circuit breaker open"}

        # ── 0b. Safety halt check ──
        if self.safety.is_halted:
            return {"symbol": symbol, "signal": "HALTED", "reason": self.safety._halt_reason}

        now_ts = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        # ── 1. Fetch data with retry ──
        try:
            ohlcv = self._fetch_data(symbol)
        except Exception as exc:
            self.circuit.record_failure()
            return {"symbol": symbol, "signal": "ERROR", "reason": f"data fetch: {exc}"}

        if ohlcv.empty or len(ohlcv) < 60:
            return {"symbol": symbol, "signal": "SKIP", "reason": f"insufficient data ({len(ohlcv)} rows)"}

        # ── 2. Compute features ──
        try:
            features = self.tech.compute_all(ohlcv).dropna()
        except Exception as exc:
            self.circuit.record_failure()
            return {"symbol": symbol, "signal": "ERROR", "reason": f"features: {exc}"}

        if features.empty or len(features) < 10:
            return {"symbol": symbol, "signal": "SKIP", "reason": "too few rows after dropna"}

        latest = features.iloc[-1].to_dict()
        prev = features.iloc[-2].to_dict() if len(features) > 1 else None
        price = float(ohlcv["close"].iloc[-1])

        # Data timestamp from the last candle
        data_ts = now_ts  # live data = fresh

        # ── 3. Generate signal ──
        sig = self.signal_gen.generate(latest, prev)
        sig["symbol"] = symbol
        sig["price"] = round(price, 2)
        sig["timestamp"] = now_iso

        # ── 4. Execute if actionable ──
        if sig["signal"] in ("BUY", "SELL") and sig["confidence"] >= 0.45:
            balance = self.executor.get_balance()
            usdt = balance.get("USDT", {}).get("total", 0)
            positions = self.executor.get_positions()
            equity = usdt + sum(p.get("unrealized_pnl", 0) for p in positions)
            trade_size = self.risk.calculate_position_size(usdt)

            # ── SAFETY GATE ──
            verdict = self.safety.pre_trade_check(
                symbol=symbol,
                side=sig["signal"].lower(),
                trade_size_usd=trade_size,
                current_balance=usdt,
                current_equity=equity,
                current_price=price,
                open_positions=positions,
                features_row=latest,
                data_timestamp=data_ts,
            )

            if not verdict.safe:
                sig["trade_blocked"] = verdict.reason
                logger.info("  BLOCKED %s %s: %s", sig["signal"], symbol, verdict.reason)
            else:
                final_size = verdict.adjusted_size or trade_size
                if verdict.warnings:
                    sig["safety_warnings"] = verdict.warnings
                    for w in verdict.warnings:
                        logger.warning("  SAFETY WARNING: %s", w)

                trade = self.executor.execute_trade({
                    "symbol": symbol,
                    "side": sig["signal"].lower(),
                    "amount": final_size,
                })

                if trade.get("status") == "filled":
                    self.circuit.record_success()
                    fee = trade.get("fee", 0)
                    self.pnl.record_entry(trade)
                    self.safety.record_trade(symbol, fee, is_loss=False)
                    trade["pnl"] = ""
                    trade["balance_after"] = self.executor.get_balance().get("USDT", {}).get("total", 0)
                    append_trade_csv(trade)
                    sig["trade"] = {
                        "order_id": trade.get("order_id"),
                        "filled": trade.get("filled_amount"),
                        "price": trade.get("price"),
                        "fee": fee,
                    }
                    logger.info(
                        "  TRADE %s %s %.6f @ $%.2f (fee=$%.2f, conf=%.2f) [%s]",
                        sig["signal"], symbol,
                        trade.get("filled_amount", 0),
                        trade.get("price", 0),
                        fee,
                        sig["confidence"],
                        "; ".join(sig["reasons"]),
                    )
                elif trade.get("status") == "rejected":
                    sig["trade_rejected"] = trade.get("reason", "unknown")
                else:
                    self.circuit.record_failure()
                    sig["trade_error"] = trade.get("reason", "unknown")

        # ── 5. Check positions for stop-loss / take-profit ──
        positions = self.executor.get_positions()
        if positions:
            prices = {symbol: price}
            actions = self.risk.check_positions(positions, prices)
            for action in actions:
                close_result = self.executor.close_position(action["symbol"])
                if close_result.get("status") == "filled":
                    exit_price = close_result.get("price", price)
                    closed_trade = self.pnl.record_exit(action["symbol"], exit_price)
                    pnl_val = closed_trade.pnl if closed_trade else 0
                    is_loss = pnl_val < 0
                    fee = close_result.get("fee", 0)
                    self.safety.record_trade(action["symbol"], fee, is_loss=is_loss)
                    close_result["pnl"] = round(pnl_val, 2)
                    close_result["balance_after"] = self.executor.get_balance().get("USDT", {}).get("total", 0)
                    append_trade_csv(close_result)
                    logger.info(
                        "  EXIT %s %s → P&L $%.2f (fee=$%.2f) [%s]",
                        action["symbol"], action["action"], pnl_val, fee, action["action"],
                    )

        # ── 6. Update equity peak for safety tracking ──
        bal = self.executor.get_balance().get("USDT", {}).get("total", 0)
        pos = self.executor.get_positions()
        eq = bal + sum(p.get("unrealized_pnl", 0) for p in pos)
        self.safety.update_equity_peak(eq)

        return sig

    def get_portfolio_summary(self) -> dict:
        balance = self.executor.get_balance()
        positions = self.executor.get_positions()
        usdt = balance.get("USDT", {}).get("total", 0)
        unrealised = sum(p.get("unrealized_pnl", 0) for p in positions)
        equity = usdt + unrealised
        return_pct = ((equity - self.pnl.initial_balance) / self.pnl.initial_balance * 100
                      if self.pnl.initial_balance else 0)
        return {
            "balance_usdt": round(usdt, 2),
            "equity": round(equity, 2),
            "return_pct": round(return_pct, 2),
            "open_positions": len(positions),
            "unrealized_pnl": round(unrealised, 2),
            "safety_status": self.safety.get_status(),
            "positions": [
                {
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "entry": p.get("entry_price"),
                    "current": p.get("current_price"),
                    "pnl": round(p.get("unrealized_pnl", 0), 2),
                }
                for p in positions
            ],
        }


# =====================================================================
# Main loop
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous AI Crypto Trader")
    parser.add_argument("--test", action="store_true",
                        help="Quick 3-cycle test with 10s intervals, no git push")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0 = infinite)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between cycles")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override trading symbols")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.cycles = 3
        args.interval = 10

    ensure_dirs()
    symbols = args.symbols or settings.exchange.supported_symbols

    logger.info("=" * 60)
    logger.info("  AUTONOMOUS TRADER%s", " (TEST MODE)" if args.test else "")
    logger.info("  Symbols : %s", ", ".join(symbols))
    logger.info("  Interval: %ds", args.interval)
    logger.info("  Mode    : PAPER (sandbox)")
    logger.info("=" * 60)

    trader = AutoTrader()
    cycle = 0
    errors_this_session = 0

    try:
        while True:
            cycle += 1
            if args.cycles and cycle > args.cycles:
                break

            cycle_start = time.time()
            logger.info("── Cycle %d ──────────────────────────────────", cycle)
            signals: Dict[str, Any] = {}

            for symbol in symbols:
                try:
                    result = trader.run_cycle(symbol)
                    signals[symbol] = result

                    ind = result.get("indicators", {})
                    logger.info(
                        "  %s → %-4s (conf=%.2f) RSI=%-5s MACD=%-8s BB=%-6s price=$%s",
                        symbol,
                        result.get("signal", "?"),
                        result.get("confidence", 0),
                        ind.get("rsi", "—"),
                        ind.get("macd_hist", "—"),
                        ind.get("bb_pct", "—"),
                        result.get("price", "?"),
                    )

                    if result.get("reasons"):
                        logger.info("    reasons: %s", " | ".join(result["reasons"]))

                except Exception as exc:
                    errors_this_session += 1
                    logger.error("  %s ERROR: %s", symbol, exc)
                    logger.debug(traceback.format_exc())
                    signals[symbol] = {"signal": "ERROR", "error": str(exc)}

            # Portfolio summary
            portfolio = trader.get_portfolio_summary()
            pnl_summary = trader.pnl.summary()

            logger.info(
                "  Portfolio: $%.2f (%.1f%%) | Trades: %d | Win: %.0f%% | P&L: $%.2f",
                portfolio["equity"],
                portfolio["return_pct"],
                pnl_summary["total_trades"],
                pnl_summary["win_rate_pct"],
                pnl_summary["total_pnl"],
            )

            # Write reports
            write_status_file(cycle, portfolio, signals, pnl_summary)
            write_daily_summary({
                "cycle": cycle,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio": portfolio,
                "pnl": pnl_summary,
                "signals": signals,
            })

            # Cycle timing
            cycle_duration = time.time() - cycle_start
            sleep_time = max(0, args.interval - cycle_duration)
            if args.cycles == 0 or cycle < args.cycles:
                logger.info("  Cycle took %.1fs — sleeping %.0fs", cycle_duration, sleep_time)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    # ── Shutdown ──
    logger.info("=" * 60)
    final = trader.pnl.summary()
    port = trader.get_portfolio_summary()
    logger.info("  FINAL RESULTS after %d cycles", cycle)
    logger.info("  Equity   : $%.2f (%.1f%%)", port["equity"], port["return_pct"])
    logger.info("  Trades   : %d total, %.0f%% win rate", final["total_trades"], final["win_rate_pct"])
    logger.info("  P&L      : $%.2f", final["total_pnl"])
    logger.info("  Best     : $%.2f | Worst: $%.2f", final["best_trade"], final["worst_trade"])
    logger.info("  Errors   : %d", errors_this_session)
    logger.info("=" * 60)

    logger.info("Autonomous trader stopped.")


if __name__ == "__main__":
    main()
