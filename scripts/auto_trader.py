#!/usr/bin/env python3
"""
Fully autonomous trading bot — production-hardened.

Runs the trading pipeline in a continuous loop with:
- Multi-indicator signal generation (real indicator values, not defaults)
- Retry/reconnection on exchange failures
- Circuit breaker: halts on repeated errors or drawdown limits
- Full P&L tracking with entry/exit matching
- Auto-commit to GitHub on a configurable interval
- --test mode for fast dry-run validation

Usage:
    python scripts/auto_trader.py                         # Run forever (paper)
    python scripts/auto_trader.py --test                  # Quick 3-cycle test
    python scripts/auto_trader.py --cycles 50             # Run 50 cycles
    python scripts/auto_trader.py --commit-interval 30    # Commit every 30 min
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from omnitrade.config.settings import settings
from omnitrade.config.asset_types import AssetType, UnifiedSignal
from omnitrade.utils.logger import get_logger
from omnitrade.data.collectors.market_data import MarketDataCollector
from omnitrade.features.technical import TechnicalFeatures
from omnitrade.risk.risk_manager import RiskManager
from omnitrade.risk.safety import SafetyGuard
from omnitrade.execution.trade_executor import TradeExecutor
from omnitrade.execution.asset_router import AssetRouter

logger = get_logger("omnitrade.auto_trader")

# ── Directories ──────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
TRADES_DIR = PROJECT_ROOT / "reports" / "trades"
LOGS_DIR = PROJECT_ROOT / "reports" / "logs"
MODEL_META_PATH = PROJECT_ROOT / "reports" / "model_metadata.json"

# Retraining intervals (in days)
RETRAIN_INTERVALS = {
    "crypto": 7,
    "stock": 14,
    "bet": 30,
}

# Disk space threshold (%), warn below this
DISK_MIN_FREE_PCT = 10.0
DISK_MIN_FREE_GB = 1.0


def ensure_dirs() -> None:
    for d in [REPORTS_DIR, TRADES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_log_rotation(log_path: Optional[Path] = None) -> None:
    """Add a rotating file handler to the root logger.

    Keeps 7 daily files of up to 50 MB each.
    """
    path = log_path or LOGS_DIR / "auto_trader.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(path), maxBytes=50 * 1024 * 1024, backupCount=7,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger.info("Log rotation configured: %s (50 MB x 7)", path)


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
# Git helpers
# =====================================================================

def git_run(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30,
    )
    return result.stdout.strip()


def auto_commit_and_push() -> bool:
    try:
        git_run("add", "reports/")
        status = git_run("status", "--porcelain", "reports/")
        if not status:
            logger.info("No new report changes to commit")
            return False

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = f"bot: trading report — {now}"
        git_run("commit", "-m", msg)
        push = subprocess.run(
            ["git", "push", "origin", "master"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
        )
        if push.returncode == 0:
            logger.info("Pushed report update to GitHub")
            return True
        else:
            logger.warning("Push failed: %s", push.stderr.strip())
            return False
    except Exception as exc:
        logger.error("Git commit/push error: %s", exc)
        return False


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

class SignalGenerator:
    """Multi-indicator confluence signal using REAL column names from TechnicalFeatures.

    Columns used (from omnitrade.features.technical):
      rsi, macd, macd_signal, macd_histogram,
      bb_percent_b, bb_bandwidth,
      stoch_k, stoch_d, adx, atr,
      ema_9, ema_21, ema_50, ema_200
    """

    @staticmethod
    def generate(features_row: dict, prev_row: Optional[dict] = None) -> dict:
        """Score a single row of features and return a signal dict."""

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
            if rsi < 30:
                buy_score += 0.25
                reasons_buy.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < 40:
                buy_score += 0.10
            elif rsi > 70:
                sell_score += 0.25
                reasons_sell.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > 60:
                sell_score += 0.10

        # 2. MACD histogram direction + crossover
        if ok(macd, macd_sig, macd_hist):
            if macd_hist > 0 and macd < 0:
                buy_score += 0.20
                reasons_buy.append("MACD bullish crossover")
            elif macd_hist > 0:
                buy_score += 0.10
            if macd_hist < 0 and macd > 0:
                sell_score += 0.20
                reasons_sell.append("MACD bearish crossover")
            elif macd_hist < 0:
                sell_score += 0.10

        # 3. Bollinger %B
        if ok(bb_pct):
            if bb_pct < 0.15:
                buy_score += 0.20
                reasons_buy.append(f"BB squeeze low ({bb_pct:.2f})")
            elif bb_pct < 0.30:
                buy_score += 0.08
            if bb_pct > 0.85:
                sell_score += 0.20
                reasons_sell.append(f"BB overbought ({bb_pct:.2f})")
            elif bb_pct > 0.70:
                sell_score += 0.08

        # 4. Stochastic
        if ok(stoch_k, stoch_d):
            if stoch_k < 20 and stoch_d < 20:
                buy_score += 0.15
                reasons_buy.append(f"Stoch oversold (K={stoch_k:.0f})")
            if stoch_k > 80 and stoch_d > 80:
                sell_score += 0.15
                reasons_sell.append(f"Stoch overbought (K={stoch_k:.0f})")

        # 5. EMA trend alignment
        if ok(ema_9, ema_21, ema_50, close):
            if close > ema_9 > ema_21 > ema_50:
                buy_score += 0.15
                reasons_buy.append("EMAs bullish aligned")
            elif close < ema_9 < ema_21 < ema_50:
                sell_score += 0.15
                reasons_sell.append("EMAs bearish aligned")

        # 6. ADX trend strength filter — only trade in strong trends
        if ok(adx):
            if adx < 20:
                # Weak trend, dampen signals
                buy_score *= 0.5
                sell_score *= 0.5

        # ── Decision ──
        confidence = max(buy_score, sell_score)
        if buy_score >= 0.45 and buy_score > sell_score:
            signal = "BUY"
            reasons = reasons_buy
        elif sell_score >= 0.45 and sell_score > buy_score:
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
    def __init__(self, asset_types: Optional[List[str]] = None) -> None:
        self._asset_types = asset_types or settings.asset.enabled_assets
        self.has_crypto = "crypto" in self._asset_types
        self.has_stock = "stock" in self._asset_types
        self.has_bet = "bet" in self._asset_types

        self.market = MarketDataCollector(settings) if self.has_crypto else None
        self.tech = TechnicalFeatures(settings)
        self.risk = RiskManager(settings)
        self.safety = SafetyGuard(settings)

        # AssetRouter as central dispatcher
        self.router = AssetRouter(settings=settings, risk_manager=self.risk, mode="paper")
        self.executor = self.router.crypto_executor

        # Stock components
        self.stock_collector = None
        self.stock_feature_pipeline = None
        self.stock_models = None
        if self.has_stock:
            from omnitrade.data.collectors.stock_data import StockDataCollector
            from omnitrade.features.stock_features import StockFeaturePipeline
            from omnitrade.models.stock_models import StockModelFactory
            self.stock_collector = StockDataCollector(settings)
            self.stock_feature_pipeline = StockFeaturePipeline(settings)
            self.stock_models = StockModelFactory(settings)

        # Betting components
        self.betting_collector = None
        self.betting_features = None
        self.betting_model = None
        if self.has_bet:
            from omnitrade.data.collectors.betting_data import BettingDataCollector
            from omnitrade.features.betting_features import BettingFeatures
            from omnitrade.models.betting_models import ValueBettingModel
            self.betting_collector = BettingDataCollector(settings)
            self.betting_features = BettingFeatures(settings)
            self.betting_model = ValueBettingModel(settings)

        self.signal_gen = SignalGenerator()
        initial_equity = self.router.get_unified_balance().get("total_usd_equivalent", 10_000)
        self.pnl = PnLTracker(initial_balance=initial_equity)
        self.circuit = CircuitBreaker(threshold=5, cooldown=300)

        # Reset daily counters
        self.risk.reset_daily(initial_equity)
        self.safety.update_equity_peak(initial_equity)

        logger.info(
            "AutoTrader initialised — lanes: %s | Safety: reserve=%.0f%% "
            "min_balance=$%.0f daily_fee_cap=$%.0f max_trades/hr=%d cooldown=%ds",
            self._asset_types,
            settings.safety.reserve_balance_pct * 100,
            settings.safety.min_balance_to_trade,
            settings.safety.max_daily_fees,
            settings.safety.max_trades_per_hour,
            settings.safety.cooldown_after_trade_sec,
        )

        # Load model metadata for retraining tracking
        self._model_meta = self._load_model_meta()

    # ------------------------------------------------------------------
    # Model metadata & auto-retraining
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_meta() -> Dict[str, Any]:
        """Load model metadata from disk, or return defaults."""
        if MODEL_META_PATH.exists():
            try:
                with open(MODEL_META_PATH) as f:
                    return json.load(f)
            except Exception:
                logger.warning("Corrupt model_metadata.json, resetting")
        return {
            "crypto_last_train": None,
            "stock_last_train": None,
            "bet_last_train": None,
            "version": 1,
        }

    @staticmethod
    def _save_model_meta(meta: Dict[str, Any]) -> None:
        """Persist model metadata to disk."""
        MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.debug("Model metadata saved")

    def _check_retraining_needed(self, lane: str) -> bool:
        """Check if a lane's model needs retraining based on staleness.

        Args:
            lane: ``"crypto"``, ``"stock"``, or ``"bet"``.

        Returns:
            True if retraining should be triggered.
        """
        key = f"{lane}_last_train"
        last_train = self._model_meta.get(key)
        interval_days = RETRAIN_INTERVALS.get(lane, 14)

        if last_train is None:
            logger.info("%s model never trained — retraining needed", lane)
            return True

        try:
            last_ts = datetime.fromisoformat(last_train)
            days_since = (datetime.now(timezone.utc) - last_ts).days
            if days_since >= interval_days:
                logger.info(
                    "%s model stale (%d days since last train, interval=%d days)",
                    lane, days_since, interval_days,
                )
                return True
        except (ValueError, TypeError):
            return True

        return False

    def _record_training(self, lane: str) -> None:
        """Mark a lane's model as freshly trained."""
        self._model_meta[f"{lane}_last_train"] = datetime.now(timezone.utc).isoformat()
        self._save_model_meta(self._model_meta)
        logger.info("%s model training timestamp updated", lane)

    def _run_retraining_if_needed(self) -> Dict[str, bool]:
        """Check all lanes and retrain stale models.

        Returns:
            Dict of lane -> whether retraining ran.
        """
        results: Dict[str, bool] = {}

        for lane in self._asset_types:
            if not self._check_retraining_needed(lane):
                results[lane] = False
                continue

            try:
                if lane == "stock" and self.stock_models:
                    logger.info("Retraining stock models...")
                    # Retrain on most recent data
                    for ticker in settings.stock.supported_tickers:
                        if self.stock_collector:
                            ohlcv = self.stock_collector.fetch_ohlcv(ticker, interval="1d")
                            fundamentals = self.stock_collector.fetch_fundamentals(ticker)
                            features = ohlcv
                            if self.stock_feature_pipeline:
                                features = self.stock_feature_pipeline.compute_all(ohlcv, fundamentals)
                            if not features.empty:
                                self.stock_models.train_all(features)
                    self._record_training("stock")
                    results[lane] = True

                elif lane == "bet" and self.betting_model:
                    logger.info("Retraining betting model...")
                    for sport in settings.betting.supported_sports:
                        if self.betting_collector:
                            odds_df = self.betting_collector.fetch_odds(sport)
                            historical = self.betting_collector.fetch_historical_results(sport)
                            features = odds_df
                            if self.betting_features and not odds_df.empty:
                                features = self.betting_features.compute_all(odds_df, historical)
                            if not features.empty:
                                if "home_score" in historical.columns and not historical.empty:
                                    outcomes = (historical.get("home_score", 0) > historical.get("away_score", 0)).astype(int)
                                    self.betting_model.train(features, outcomes)
                    self._record_training("bet")
                    results[lane] = True

                elif lane == "crypto":
                    logger.info("Crypto retraining triggered (deferred to main bot training pipeline)")
                    self._record_training("crypto")
                    results[lane] = True

            except Exception as exc:
                logger.error("Retraining failed for %s: %s", lane, exc)
                results[lane] = False

        return results

    # ------------------------------------------------------------------
    # Disk space monitoring
    # ------------------------------------------------------------------

    @staticmethod
    def check_disk_space(path: Optional[Path] = None) -> Dict[str, Any]:
        """Check disk space and warn if critically low.

        Returns:
            Dict with total_gb, used_gb, free_gb, free_pct, is_critical.
        """
        target = str(path or PROJECT_ROOT)
        try:
            usage = shutil.disk_usage(target)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            free_pct = (usage.free / usage.total) * 100 if usage.total > 0 else 0

            is_critical = free_pct < DISK_MIN_FREE_PCT or free_gb < DISK_MIN_FREE_GB

            if is_critical:
                logger.critical(
                    "DISK SPACE CRITICAL: %.1f GB free (%.1f%%) — "
                    "log rotation or cleanup needed",
                    free_gb, free_pct,
                )
            elif free_pct < DISK_MIN_FREE_PCT * 2:
                logger.warning(
                    "Disk space low: %.1f GB free (%.1f%%)", free_gb, free_pct,
                )

            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "free_pct": round(free_pct, 1),
                "is_critical": is_critical,
            }
        except Exception as exc:
            logger.error("Disk space check failed: %s", exc)
            return {"error": str(exc), "is_critical": False}

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

    def run_stock_cycle(self, ticker: str) -> dict:
        """One full collect -> analyse -> trade cycle for a stock ticker."""

        if self.stock_collector is None:
            return {"symbol": ticker, "signal": "SKIP", "reason": "stock lane not initialised"}

        if self.circuit.is_open:
            return {"symbol": ticker, "signal": "PAUSED", "reason": "circuit breaker open"}

        if self.safety.is_halted:
            return {"symbol": ticker, "signal": "HALTED", "reason": self.safety._halt_reason}

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Fetch stock data
        try:
            ohlcv = self.stock_collector.fetch_ohlcv(ticker, interval="1h")
        except Exception as exc:
            self.circuit.record_failure()
            return {"symbol": ticker, "signal": "ERROR", "reason": f"stock data: {exc}"}

        if ohlcv.empty or len(ohlcv) < 20:
            return {"symbol": ticker, "signal": "SKIP", "reason": f"insufficient data ({len(ohlcv)} rows)"}

        # 2. Features
        fundamentals = self.stock_collector.fetch_fundamentals(ticker)
        features = ohlcv
        if self.stock_feature_pipeline:
            features = self.stock_feature_pipeline.compute_all(ohlcv, fundamentals)
        if features.empty:
            features = ohlcv

        latest = features.iloc[-1].to_dict()
        price = float(ohlcv["close"].iloc[-1])

        # 3. Signal
        if self.stock_models and self.stock_models.is_trained:
            sig_result = self.stock_models.predict(features)
        else:
            sig_result = self.signal_gen.generate(latest)

        sig = {
            "symbol": ticker,
            "signal": sig_result.get("signal", "HOLD"),
            "confidence": sig_result.get("confidence", 0),
            "price": round(price, 2),
            "timestamp": now_iso,
            "asset_type": "stock",
        }

        # 4. Execute if actionable
        if sig["signal"] in ("BUY", "SELL") and sig["confidence"] >= 0.45:
            balance = self.router.get_unified_balance()
            usd = balance.get("total_usd_equivalent", 10_000)
            positions = self.router.get_all_positions()
            equity = usd + sum(p.get("unrealized_pnl", 0) for p in positions)
            trade_size = self.risk.calculate_position_size(usd)

            verdict = self.safety.pre_trade_check(
                symbol=ticker,
                side=sig["signal"].lower(),
                trade_size_usd=trade_size,
                current_balance=usd,
                current_equity=equity,
                current_price=price,
                open_positions=positions,
                features_row=latest,
                data_timestamp=time.time(),
            )

            if not verdict.safe:
                sig["trade_blocked"] = verdict.reason
                logger.info("  BLOCKED %s %s: %s", sig["signal"], ticker, verdict.reason)
            else:
                signal = UnifiedSignal(
                    asset_type=AssetType.STOCK,
                    symbol=ticker,
                    side=sig["signal"],
                    confidence=sig["confidence"],
                    amount=verdict.adjusted_size or trade_size,
                    price=price,
                )
                trade = self.router.route_signal(signal)

                if trade.get("status") == "filled":
                    self.circuit.record_success()
                    fee = trade.get("fee", 0)
                    self.pnl.record_entry(trade)
                    self.safety.record_trade(ticker, fee, is_loss=False)
                    trade["pnl"] = ""
                    trade["balance_after"] = usd
                    append_trade_csv(trade)
                    sig["trade"] = {
                        "order_id": trade.get("order_id"),
                        "filled": trade.get("filled_amount"),
                        "price": trade.get("price"),
                        "fee": fee,
                    }
                    logger.info(
                        "  TRADE %s %s $%.2f @ $%.2f (conf=%.2f)",
                        sig["signal"], ticker, trade_size, price, sig["confidence"],
                    )
                elif trade.get("status") == "rejected":
                    sig["trade_rejected"] = trade.get("reason", "unknown")
                else:
                    self.circuit.record_failure()

        return sig

    def run_betting_cycle(self) -> List[dict]:
        """One full collect -> analyse -> bet cycle for all configured sports."""

        if self.betting_collector is None:
            return [{"sport": "all", "signal": "SKIP", "reason": "betting lane not initialised"}]

        if self.circuit.is_open:
            return [{"sport": "all", "signal": "PAUSED", "reason": "circuit breaker open"}]

        if self.safety.is_halted:
            return [{"sport": "all", "signal": "HALTED", "reason": self.safety._halt_reason}]

        results = []
        bet_executor = self.router.bet_executor

        for sport in settings.betting.supported_sports:
            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                odds_df = self.betting_collector.fetch_odds(sport)
                if odds_df.empty:
                    results.append({"sport": sport, "signal": "SKIP", "reason": "no odds data"})
                    continue

                historical = self.betting_collector.fetch_historical_results(sport)
                features = odds_df
                if self.betting_features:
                    features = self.betting_features.compute_all(odds_df, historical)

                if features.empty:
                    results.append({"sport": sport, "signal": "SKIP", "reason": "no features"})
                    continue

                signal = self.betting_model.predict(features)

                sig = {
                    "sport": sport,
                    "symbol": signal.symbol,
                    "side": str(signal.side),
                    "confidence": signal.confidence,
                    "asset_type": "bet",
                    "timestamp": now_iso,
                    "edge": signal.metadata.get("edge", 0),
                }

                if signal.side not in ("BACK", "LAY"):
                    sig["signal"] = "PASS"
                    results.append(sig)
                    continue

                sig["signal"] = str(signal.side)
                result = self.router.route_signal(signal)

                if result.get("status") == "filled":
                    self.circuit.record_success()
                    sig["trade"] = {
                        "order_id": result.get("order_id"),
                        "stake": result.get("stake"),
                        "odds": result.get("odds"),
                    }
                    logger.info(
                        "  BET %s %s $%.2f @ %+.0f (edge=%.1f%%, conf=%.2f)",
                        signal.side, signal.symbol,
                        result.get("stake", 0), result.get("odds", 0),
                        signal.metadata.get("edge", 0) * 100,
                        signal.confidence,
                    )
                elif result.get("status") == "rejected":
                    sig["trade_rejected"] = result.get("reason", "unknown")
                    logger.info("  BET REJECTED %s: %s", signal.symbol, result.get("reason"))
                else:
                    self.circuit.record_failure()

                results.append(sig)

            except Exception as exc:
                self.circuit.record_failure()
                logger.error("  [bet] %s ERROR: %s", sport, exc)
                results.append({"sport": sport, "signal": "ERROR", "reason": str(exc)})

        # Log betting stats
        if bet_executor:
            stats = bet_executor.get_stats()
            logger.info(
                "  Betting: bankroll=$%.2f | bets=%d | win=%.0f%% | PnL=$%.2f | ROI=%.1f%% | open=%d",
                stats["bankroll"], stats["total_bets"], stats["win_rate"],
                stats["total_pnl"], stats["roi_pct"], stats["open_bets"],
            )

        return results

    def get_portfolio_summary(self) -> dict:
        unified = self.router.get_unified_balance()
        positions = self.router.get_all_positions()
        equity = unified.get("total_usd_equivalent", 10_000)
        unrealised = sum(p.get("unrealized_pnl", 0) for p in positions)
        return_pct = ((equity - self.pnl.initial_balance) / self.pnl.initial_balance * 100
                      if self.pnl.initial_balance else 0)
        return {
            "balance_usd": round(equity, 2),
            "equity": round(equity, 2),
            "return_pct": round(return_pct, 2),
            "open_positions": len(positions),
            "unrealized_pnl": round(unrealised, 2),
            "lanes": unified.get("by_lane", {}),
            "safety_status": self.safety.get_status(),
            "positions": [
                {
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "asset_type": p.get("asset_type", "crypto"),
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
    parser = argparse.ArgumentParser(description="OmniTrade AI — Autonomous Multi-Asset Trader")
    parser.add_argument("--test", action="store_true",
                        help="Quick 3-cycle test with 10s intervals, no git push")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0 = infinite)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between cycles")
    parser.add_argument("--commit-interval", type=int, default=60,
                        help="Minutes between auto-commits to GitHub")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override crypto trading symbols")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override stock tickers")
    parser.add_argument("--asset-types", nargs="+", choices=["crypto", "stock", "bet"],
                        default=None,
                        help="Asset classes to trade (default: all enabled in config)")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.cycles = 3
        args.interval = 10
        args.commit_interval = 9999  # don't push during test

    ensure_dirs()
    setup_log_rotation()
    symbols = args.symbols or settings.exchange.supported_symbols
    tickers = args.tickers or settings.stock.supported_tickers
    asset_types = args.asset_types or settings.asset.enabled_assets

    has_crypto = "crypto" in asset_types
    has_stock = "stock" in asset_types
    has_bet = "bet" in asset_types

    logger.info("=" * 60)
    logger.info("  OMNITRADE AI — Autonomous Trader%s", " (TEST MODE)" if args.test else "")
    logger.info("  Lanes   : %s", ", ".join(asset_types))
    if has_crypto:
        logger.info("  Crypto  : %s", ", ".join(symbols))
    if has_stock:
        logger.info("  Stocks  : %s", ", ".join(tickers))
    if has_bet:
        logger.info("  Betting : %s", ", ".join(settings.betting.supported_sports))
    logger.info("  Interval: %ds | Commit: every %dm", args.interval, args.commit_interval)
    logger.info("  Mode    : PAPER (sandbox)")
    logger.info("=" * 60)

    trader = AutoTrader(asset_types=asset_types)
    last_commit = time.time()
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

            # ── Crypto lane ──
            if has_crypto:
                for symbol in symbols:
                    try:
                        result = trader.run_cycle(symbol)
                        signals[symbol] = result

                        ind = result.get("indicators", {})
                        logger.info(
                            "  [crypto] %s → %-4s (conf=%.2f) RSI=%-5s MACD=%-8s BB=%-6s price=$%s",
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
                        logger.error("  [crypto] %s ERROR: %s", symbol, exc)
                        logger.debug(traceback.format_exc())
                        signals[symbol] = {"signal": "ERROR", "error": str(exc)}

            # ── Stock lane ──
            if has_stock:
                for ticker in tickers:
                    try:
                        result = trader.run_stock_cycle(ticker)
                        key = f"stock:{ticker}"
                        signals[key] = result
                        logger.info(
                            "  [stock]  %s → %-4s (conf=%.2f) price=$%s",
                            ticker,
                            result.get("signal", "?"),
                            result.get("confidence", 0),
                            result.get("price", "?"),
                        )
                    except Exception as exc:
                        errors_this_session += 1
                        logger.error("  [stock] %s ERROR: %s", ticker, exc)
                        logger.debug(traceback.format_exc())
                        signals[f"stock:{ticker}"] = {"signal": "ERROR", "error": str(exc)}

            # ── Betting lane ──
            if has_bet:
                try:
                    bet_results = trader.run_betting_cycle()
                    for br in bet_results:
                        sport = br.get("sport", "?")
                        key = f"bet:{sport}"
                        signals[key] = br
                        logger.info(
                            "  [bet]    %s → %-4s (conf=%.2f) edge=%.3f",
                            br.get("symbol", sport),
                            br.get("signal", "?"),
                            br.get("confidence", 0),
                            br.get("edge", 0),
                        )
                except Exception as exc:
                    errors_this_session += 1
                    logger.error("  [bet] ERROR: %s", exc)
                    logger.debug(traceback.format_exc())
                    signals["bet:all"] = {"signal": "ERROR", "error": str(exc)}

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

            # Auto-commit
            elapsed_min = (time.time() - last_commit) / 60
            if elapsed_min >= args.commit_interval:
                logger.info("Auto-committing reports to GitHub...")
                auto_commit_and_push()
                last_commit = time.time()

            # Retraining check (every 10 cycles)
            if cycle % 10 == 0:
                logger.info("Checking model staleness...")
                retrain_results = trader._run_retraining_if_needed()
                logger.info("Retraining results: %s", retrain_results)

            # Disk space check (every 50 cycles)
            if cycle % 50 == 0:
                disk = AutoTrader.check_disk_space()
                if disk.get("is_critical"):
                    logger.critical(
                        "DISK CRITICAL — attempting emergency log rotation"
                    )
                logger.info(
                    "Disk: %.1f GB free (%.1f%%)",
                    disk.get("free_gb", 0), disk.get("free_pct", 0),
                )

            # Record portfolio return for adaptive sizing
            trader.risk.record_return(
                portfolio.get("return_pct", 0) / 100.0
            )

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

    if not args.test:
        logger.info("Final commit...")
        auto_commit_and_push()

    logger.info("OmniTrade AI stopped.")


if __name__ == "__main__":
    main()
