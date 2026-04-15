#!/usr/bin/env python3
"""
Daily paper-trade runner — test with real data, log every trade.

Fetches real market data, runs the full trading strategy through the backtest
engine with fake money ($10,000), and logs every single trade to a CSV.
Commits whenever there are new trade logs — profit or loss — so there's
always a record of what happened.

The CSV trade log is included in the commit so every change has a clear
audit trail showing each trade's P&L.

Usage:
    python scripts/benchmark_runner.py                # Full run
    python scripts/benchmark_runner.py --dry-run      # Test only, no commit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.config.settings import settings
from crypto_bot.utils.logger import get_logger
from crypto_bot.data.collectors.market_data import MarketDataCollector
from crypto_bot.features.technical import TechnicalFeatures
from crypto_bot.backtesting.engine import BacktestEngine

logger = get_logger("crypto_bot.benchmark_runner")

REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "benchmarks"
TRADE_LOG_DIR = REPORTS_DIR / "trade_logs"
BASELINE_PATH = REPORTS_DIR / "benchmark_baseline.json"

STARTING_BALANCE = 10_000.0

# CSV columns for the trade log
TRADE_LOG_FIELDS = [
    "trade_number",
    "timestamp",
    "symbol",
    "side",
    "price",
    "raw_price",
    "amount",
    "fee",
    "pnl",
    "cumulative_pnl",
    "balance_after",
    "return_pct",
]


# =====================================================================
# Strategy adapter — wraps SignalGenerator for the backtest engine
# =====================================================================

def make_strategy_func(tech: TechnicalFeatures):
    """Return a strategy function compatible with BacktestEngine.run()."""
    from scripts.auto_trader import SignalGenerator

    sig_gen = SignalGenerator()

    def strategy(data, current_index: int) -> Optional[Dict]:
        if current_index < 60:
            return None

        window = data.iloc[max(0, current_index - 499): current_index + 1]
        try:
            features = tech.compute_all(window).dropna()
        except Exception:
            return None

        if features.empty or len(features) < 2:
            return None

        latest = features.iloc[-1].to_dict()
        prev = features.iloc[-2].to_dict()
        sig = sig_gen.generate(latest, prev)

        if sig["signal"] == "BUY" and sig["confidence"] >= 0.45:
            price = float(data["close"].iloc[current_index])
            amount = STARTING_BALANCE * 0.10 / price  # 10% of starting balance
            return {"side": "buy", "amount": amount}
        elif sig["signal"] == "SELL" and sig["confidence"] >= 0.45:
            return {"side": "close"}

        return None

    return strategy


# =====================================================================
# Trade log writer
# =====================================================================

def write_trade_log(
    trades: list,
    symbol: str,
    log_path: Path,
) -> None:
    """Write a per-trade CSV log from the backtest engine's trade history."""
    cumulative_pnl = 0.0
    is_new = not log_path.exists() or log_path.stat().st_size == 0

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        if is_new:
            writer.writeheader()

        for i, trade in enumerate(trades, start=1):
            cumulative_pnl += trade.pnl
            return_pct = (trade.balance - STARTING_BALANCE) / STARTING_BALANCE * 100

            writer.writerow({
                "trade_number": i,
                "timestamp": str(trade.timestamp),
                "symbol": symbol,
                "side": trade.side,
                "price": round(trade.price, 6),
                "raw_price": round(trade.raw_price, 6),
                "amount": round(trade.amount, 8),
                "fee": round(trade.fee, 6),
                "pnl": round(trade.pnl, 4),
                "cumulative_pnl": round(cumulative_pnl, 4),
                "balance_after": round(trade.balance, 2),
                "return_pct": round(return_pct, 2),
            })


def write_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    """Write a one-row summary CSV so it's easy to diff between commits."""
    fields = [
        "date",
        "starting_balance",
        "final_balance",
        "total_pnl",
        "total_return_pct",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "profit_factor",
        "best_trade_pnl",
        "worst_trade_pnl",
        "symbols_traded",
        "verdict",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)


# =====================================================================
# Run paper trades with real data
# =====================================================================

def run_paper_trades() -> Dict[str, Any]:
    """Fetch real market data, run strategy with fake money, log every trade."""
    market = MarketDataCollector(settings)
    tech = TechnicalFeatures(settings)
    engine = BacktestEngine(settings, initial_balance=STARTING_BALANCE)
    strategy = make_strategy_func(tech)

    symbols = settings.exchange.supported_symbols
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Create log directory
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trade_log_path = TRADE_LOG_DIR / f"trades_{today}.csv"
    # Clear previous run for today if exists
    if trade_log_path.exists():
        trade_log_path.unlink()

    all_symbol_results = {}
    total_trades_all: List = []
    total_pnl = 0.0
    total_final_balance = 0.0
    total_starting = 0.0
    best_trade_pnl = 0.0
    worst_trade_pnl = 0.0

    for symbol in symbols:
        logger.info("Paper-trading %s with real data ...", symbol)
        try:
            ohlcv = market.fetch_ohlcv(symbol, timeframe="1h", limit=2000)
        except Exception as exc:
            logger.warning("Failed to fetch data for %s: %s", symbol, exc)
            continue

        if ohlcv.empty or len(ohlcv) < 100:
            logger.warning("Insufficient data for %s (%d rows)", symbol, len(ohlcv))
            continue

        data_start = ohlcv["timestamp"].iloc[0] if "timestamp" in ohlcv.columns else ohlcv.index[0]
        data_end = ohlcv["timestamp"].iloc[-1] if "timestamp" in ohlcv.columns else ohlcv.index[-1]
        logger.info("  Data range: %s to %s (%d candles)", data_start, data_end, len(ohlcv))

        # Run the backtest with real data
        result = engine.run(strategy, ohlcv)

        # Log every trade to CSV
        if result.trade_history:
            write_trade_log(result.trade_history, symbol, trade_log_path)

        # Track per-trade P&L
        trades_with_pnl = [t for t in result.trade_history if t.pnl != 0.0]
        winning = [t for t in trades_with_pnl if t.pnl > 0]
        losing = [t for t in trades_with_pnl if t.pnl <= 0]

        symbol_pnl = sum(t.pnl for t in trades_with_pnl)
        final_balance = result.equity_curve.iloc[-1] if not result.equity_curve.empty else STARTING_BALANCE
        return_pct = result.total_return * 100

        all_symbol_results[symbol] = {
            "total_return": round(result.total_return, 6),
            "return_pct": round(return_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 6),
            "max_drawdown_pct": round(result.max_drawdown * 100, 2),
            "win_rate": round(result.win_rate, 4),
            "profit_factor": round(result.profit_factor, 4),
            "total_trades": result.total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pnl": round(symbol_pnl, 2),
            "final_balance": round(final_balance, 2),
            "best_trade": round(max((t.pnl for t in trades_with_pnl), default=0), 2),
            "worst_trade": round(min((t.pnl for t in trades_with_pnl), default=0), 2),
            "data_candles": len(ohlcv),
        }

        total_pnl += symbol_pnl
        total_final_balance += final_balance
        total_starting += STARTING_BALANCE
        total_trades_all.extend(result.trade_history)

        for t in trades_with_pnl:
            if t.pnl > best_trade_pnl:
                best_trade_pnl = t.pnl
            if t.pnl < worst_trade_pnl:
                worst_trade_pnl = t.pnl

        logger.info(
            "  %s RESULT: %s $%.2f (%.1f%%) | %d trades, %.0f%% win | sharpe=%.2f | maxDD=%.1f%%",
            symbol,
            "PROFIT" if symbol_pnl > 0 else "LOSS",
            symbol_pnl,
            return_pct,
            result.total_trades,
            result.win_rate * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
        )

    if not all_symbol_results:
        logger.error("No results produced — no data available")
        return {}

    # Aggregate metrics
    all_closed = [t for t in total_trades_all if t.pnl != 0.0]
    all_winning = [t for t in all_closed if t.pnl > 0]
    all_losing = [t for t in all_closed if t.pnl <= 0]
    total_return_pct = (total_final_balance - total_starting) / total_starting * 100 if total_starting else 0

    # Average sharpe/drawdown across symbols
    avg_sharpe = sum(r["sharpe_ratio"] for r in all_symbol_results.values()) / len(all_symbol_results)
    avg_drawdown = sum(r["max_drawdown"] for r in all_symbol_results.values()) / len(all_symbol_results)
    win_rate = len(all_winning) / len(all_closed) * 100 if all_closed else 0

    gross_profit = sum(t.pnl for t in all_winning)
    gross_loss = abs(sum(t.pnl for t in all_losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    is_profitable = total_pnl > 0
    verdict = "PROFIT" if is_profitable else "LOSS"

    logger.info("=" * 60)
    logger.info("  OVERALL: %s $%.2f (%.2f%%)", verdict, total_pnl, total_return_pct)
    logger.info("  Trades: %d total | %d wins | %d losses | %.1f%% win rate",
                len(total_trades_all), len(all_winning), len(all_losing), win_rate)
    logger.info("  Sharpe: %.2f | Max DD: %.2f%% | Profit Factor: %.2f",
                avg_sharpe, avg_drawdown * 100, profit_factor)
    logger.info("  Best trade: $%.2f | Worst trade: $%.2f", best_trade_pnl, worst_trade_pnl)
    logger.info("  Trade log: %s", trade_log_path)
    logger.info("=" * 60)

    # Write the summary CSV
    summary_row = {
        "date": today,
        "starting_balance": total_starting,
        "final_balance": round(total_final_balance, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_trades": len(total_trades_all),
        "winning_trades": len(all_winning),
        "losing_trades": len(all_losing),
        "win_rate_pct": round(win_rate, 1),
        "sharpe_ratio": round(avg_sharpe, 4),
        "max_drawdown_pct": round(avg_drawdown * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "best_trade_pnl": round(best_trade_pnl, 2),
        "worst_trade_pnl": round(worst_trade_pnl, 2),
        "symbols_traded": ";".join(all_symbol_results.keys()),
        "verdict": verdict,
    }
    summary_path = RESULTS_DIR / f"summary_{today}.csv"
    write_summary_csv(summary_row, summary_path)
    logger.info("Summary written to %s", summary_path)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "verdict": verdict,
        "is_profitable": is_profitable,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "starting_balance": total_starting,
        "final_balance": round(total_final_balance, 2),
        "total_trades": len(total_trades_all),
        "winning_trades": len(all_winning),
        "losing_trades": len(all_losing),
        "win_rate_pct": round(win_rate, 1),
        "sharpe_ratio": round(avg_sharpe, 4),
        "max_drawdown_pct": round(avg_drawdown * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "per_symbol": all_symbol_results,
        "trade_log_file": str(trade_log_path.relative_to(PROJECT_ROOT)),
        "summary_file": str(summary_path.relative_to(PROJECT_ROOT)),
    }


# =====================================================================
# Comparison against previous baseline
# =====================================================================

def load_baseline() -> Optional[Dict[str, Any]]:
    """Load the last saved benchmark baseline."""
    if not BASELINE_PATH.exists():
        return None
    try:
        with open(BASELINE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_better_than_baseline(new: Dict, old: Dict) -> tuple[bool, list[str]]:
    """Check if new results beat the old baseline.

    Must be profitable AND not worse than previous run.
    """
    reasons = []

    if not new.get("is_profitable"):
        reasons.append(f"LOSS: strategy lost ${abs(new['total_pnl']):.2f} — not committing")
        return False, reasons

    # If no old baseline, any profit counts
    if not old:
        reasons.append(f"PROFIT: ${new['total_pnl']:.2f} — first baseline established")
        return True, reasons

    old_pnl = old.get("total_pnl", 0)
    new_pnl = new.get("total_pnl", 0)
    old_sharpe = old.get("sharpe_ratio", 0)
    new_sharpe = new.get("sharpe_ratio", 0)
    old_dd = old.get("max_drawdown_pct", 0)
    new_dd = new.get("max_drawdown_pct", 0)
    old_wr = old.get("win_rate_pct", 0)
    new_wr = new.get("win_rate_pct", 0)

    # Must be profitable
    if new_pnl <= 0:
        reasons.append(f"LOSS: ${new_pnl:.2f} — not committing")
        return False, reasons

    # Compare to previous
    if new_pnl > old_pnl:
        reasons.append(f"P&L improved: ${old_pnl:.2f} -> ${new_pnl:.2f}")
    elif new_pnl < old_pnl * 0.95:  # more than 5% worse
        reasons.append(f"P&L regressed: ${old_pnl:.2f} -> ${new_pnl:.2f}")
        return False, reasons
    else:
        reasons.append(f"P&L similar: ${old_pnl:.2f} -> ${new_pnl:.2f}")

    if new_sharpe > old_sharpe:
        reasons.append(f"Sharpe improved: {old_sharpe:.2f} -> {new_sharpe:.2f}")
    if new_dd < old_dd:
        reasons.append(f"Drawdown improved: {old_dd:.1f}% -> {new_dd:.1f}%")
    if new_wr > old_wr:
        reasons.append(f"Win rate improved: {old_wr:.1f}% -> {new_wr:.1f}%")

    return True, reasons


# =====================================================================
# Git helpers
# =====================================================================

def git_run(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30,
    )
    return result.stdout.strip()


def commit_and_push(message: str) -> bool:
    """Stage trade logs and results, commit, and push."""
    try:
        git_run("add", "reports/trade_logs/")
        git_run("add", "reports/benchmarks/")
        git_run("add", "reports/benchmark_baseline.json")

        status = git_run("status", "--porcelain")
        if not status:
            logger.info("No changes to commit")
            return False

        git_run("commit", "-m", message)

        for attempt in range(4):
            push = subprocess.run(
                ["git", "push", "origin", "HEAD"],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=60,
            )
            if push.returncode == 0:
                logger.info("Pushed to GitHub")
                return True
            wait = 2 ** (attempt + 1)
            logger.warning("Push attempt %d failed, retrying in %ds: %s",
                           attempt + 1, wait, push.stderr.strip())
            time.sleep(wait)

        logger.error("Push failed after 4 attempts")
        return False
    except Exception as exc:
        logger.error("Git error: %s", exc)
        return False


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paper-trade with real data, log every trade, commit new logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run and log trades, but don't commit")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  BENCHMARK RUNNER — Paper trading with real data")
    logger.info("  Starting balance: $%.2f (fake money)", STARTING_BALANCE)
    logger.info("  Mode: %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("=" * 60)

    # 1. Run paper trades with real market data
    results = run_paper_trades()
    if not results:
        logger.error("No results — aborting")
        return 1

    # 2. Compare against previous baseline (for logging, not gating)
    baseline = load_baseline()
    _, reasons = is_better_than_baseline(results, baseline)

    logger.info("─" * 60)
    for r in reasons:
        logger.info("  %s", r)
    logger.info("─" * 60)

    # 3. Always save the new baseline
    with open(BASELINE_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 4. Commit if there are new trade logs
    has_trades = results.get("total_trades", 0) > 0

    if not has_trades:
        logger.info("No trades executed — nothing to commit")
        return 0

    if args.dry_run:
        logger.info("DRY RUN: would commit (verdict=%s, pnl=$%.2f, trades=%d)",
                    results["verdict"], results["total_pnl"], results["total_trades"])
        return 0

    pnl = results["total_pnl"]
    ret = results["total_return_pct"]
    trades = results["total_trades"]
    wr = results["win_rate_pct"]
    verdict = results["verdict"]

    msg = (f"bot: daily trade log — {verdict} ${pnl:+.2f} ({ret:+.1f}%) "
           f"| {trades} trades, {wr:.0f}% win rate "
           f"| sharpe={results['sharpe_ratio']:.2f}")

    if commit_and_push(msg):
        logger.info("Committed: %s", msg)
        return 0
    else:
        logger.error("Failed to commit/push")
        return 1


if __name__ == "__main__":
    sys.exit(main())
