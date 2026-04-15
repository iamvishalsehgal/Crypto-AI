#!/usr/bin/env python3
"""
Auto-tuner — evolves the trading strategy to beat its previous self.

Loads real market data, tries many variations of signal parameters
(RSI thresholds, indicator weights, confidence levels, etc.),
backtests each variation with fake money, and keeps the one that
performs best.  Only overwrites config/signal_config.json when a
variation beats the current config on real data.

This runs as part of the daily benchmark workflow, before the
paper-trade test.

Strategy:
  1. Load current signal_config.json (the baseline)
  2. Generate N random variations around the current params
  3. Backtest each variation on real market data
  4. Pick the best by a composite score (return * sharpe / drawdown)
  5. If the best variation beats the current config → write new config
  6. Log everything so there's a record of what was tried

Usage:
    python scripts/auto_tuner.py               # Full tuning run
    python scripts/auto_tuner.py --dry-run     # Tune but don't save
    python scripts/auto_tuner.py --variants 50 # Try 50 variations
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
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

logger = get_logger("crypto_bot.auto_tuner")

SIGNAL_CONFIG_PATH = PROJECT_ROOT / "config" / "signal_config.json"
TUNING_LOG_DIR = PROJECT_ROOT / "reports" / "tuning"

STARTING_BALANCE = 10_000.0


# =====================================================================
# Parameter mutation
# =====================================================================

# Bounds for each parameter to keep mutations sensible
PARAM_BOUNDS = {
    "rsi_oversold":       (15, 40),
    "rsi_weak_buy":       (30, 50),
    "rsi_overbought":     (60, 85),
    "rsi_weak_sell":       (50, 70),
    "rsi_strong_weight":  (0.10, 0.40),
    "rsi_weak_weight":    (0.03, 0.20),
    "macd_crossover_weight": (0.10, 0.35),
    "macd_trend_weight":  (0.03, 0.20),
    "bb_buy_strong":      (0.05, 0.25),
    "bb_buy_weak":        (0.15, 0.40),
    "bb_sell_strong":     (0.75, 0.95),
    "bb_sell_weak":       (0.60, 0.85),
    "bb_strong_weight":   (0.10, 0.35),
    "bb_weak_weight":     (0.03, 0.15),
    "stoch_oversold":     (10, 30),
    "stoch_overbought":   (70, 90),
    "stoch_weight":       (0.05, 0.25),
    "ema_weight":         (0.05, 0.25),
    "adx_weak_threshold": (15, 30),
    "adx_dampen_factor":  (0.2, 0.8),
    "min_confidence":     (0.30, 0.60),
    "position_size_pct":  (0.05, 0.20),
}


def mutate_params(base: dict, mutation_rate: float = 0.3) -> dict:
    """Create a mutated copy of signal parameters.

    Each parameter has a `mutation_rate` chance of being perturbed
    by up to +/-20% within its allowed bounds.
    """
    mutated = copy.deepcopy(base)

    for key, (low, high) in PARAM_BOUNDS.items():
        if key not in mutated:
            continue
        if random.random() > mutation_rate:
            continue

        current = mutated[key]
        # Perturb by up to 20%
        delta = current * random.uniform(-0.20, 0.20)
        new_val = current + delta

        # Clamp to bounds
        new_val = max(low, min(high, new_val))

        # Keep integers as integers
        if isinstance(base.get(key), int):
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 4)

        mutated[key] = new_val

    # Enforce RSI ordering: oversold < weak_buy < weak_sell < overbought
    if mutated["rsi_oversold"] >= mutated["rsi_weak_buy"]:
        mutated["rsi_weak_buy"] = mutated["rsi_oversold"] + 5
    if mutated["rsi_weak_sell"] >= mutated["rsi_overbought"]:
        mutated["rsi_weak_sell"] = mutated["rsi_overbought"] - 5

    # Enforce BB ordering: buy_strong < buy_weak < sell_weak < sell_strong
    if mutated["bb_buy_strong"] >= mutated["bb_buy_weak"]:
        mutated["bb_buy_weak"] = mutated["bb_buy_strong"] + 0.05
    if mutated["bb_sell_weak"] >= mutated["bb_sell_strong"]:
        mutated["bb_sell_weak"] = mutated["bb_sell_strong"] - 0.05

    return mutated


# =====================================================================
# Backtest with given params
# =====================================================================

def make_strategy(tech: TechnicalFeatures, params: dict):
    """Build a backtest strategy function using the given signal params."""
    from scripts.auto_trader import SignalGenerator

    sig_gen = SignalGenerator(params=params)
    pos_size_pct = params.get("position_size_pct", 0.10)
    min_conf = params.get("min_confidence", 0.45)

    def strategy(data, idx: int) -> Optional[Dict]:
        if idx < 60:
            return None
        window = data.iloc[max(0, idx - 499): idx + 1]
        try:
            features = tech.compute_all(window).dropna()
        except Exception:
            return None
        if features.empty or len(features) < 2:
            return None

        latest = features.iloc[-1].to_dict()
        prev = features.iloc[-2].to_dict()
        sig = sig_gen.generate(latest, prev)

        if sig["signal"] == "BUY" and sig["confidence"] >= min_conf:
            price = float(data["close"].iloc[idx])
            amount = STARTING_BALANCE * pos_size_pct / price
            return {"side": "buy", "amount": amount}
        elif sig["signal"] == "SELL" and sig["confidence"] >= min_conf:
            return {"side": "close"}
        return None

    return strategy


def score_result(result) -> float:
    """Compute a composite score for a backtest result.

    Balances profit, risk-adjusted return, and drawdown control.
    Higher is better.
    """
    ret = result.total_return
    sharpe = result.sharpe_ratio
    dd = result.max_drawdown

    if result.total_trades < 3:
        return -999.0  # not enough trades to judge

    # Penalise losses hard
    if ret < 0:
        return ret * 10

    # Reward: return * sharpe, penalise drawdown
    dd_penalty = max(dd, 0.01)  # avoid division by zero
    return (ret * max(sharpe, 0.01)) / dd_penalty


# =====================================================================
# Main tuning loop
# =====================================================================

def run_tuning(n_variants: int = 30, dry_run: bool = False) -> Dict[str, Any]:
    """Try parameter variations and keep the best one."""
    market = MarketDataCollector(settings)
    tech = TechnicalFeatures(settings)
    symbols = settings.exchange.supported_symbols
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    TUNING_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load current config as baseline
    if SIGNAL_CONFIG_PATH.exists():
        with open(SIGNAL_CONFIG_PATH) as f:
            current_params = json.load(f)
    else:
        from scripts.auto_trader import DEFAULT_SIGNAL_PARAMS
        current_params = DEFAULT_SIGNAL_PARAMS.copy()

    # Fetch real data for all symbols (once, reused for all variants)
    symbol_data = {}
    for symbol in symbols:
        try:
            ohlcv = market.fetch_ohlcv(symbol, timeframe="1h", limit=2000)
            if not ohlcv.empty and len(ohlcv) >= 100:
                symbol_data[symbol] = ohlcv
                logger.info("Loaded %d candles for %s", len(ohlcv), symbol)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", symbol, exc)

    if not symbol_data:
        logger.error("No market data available for tuning")
        return {}

    # Score the current params (baseline)
    logger.info("=" * 60)
    logger.info("  Scoring current config (baseline)...")
    baseline_score, baseline_details = _evaluate_params(current_params, tech, symbol_data)
    logger.info("  Baseline score: %.4f (return=%.2f%%, sharpe=%.2f)",
                baseline_score,
                baseline_details.get("avg_return", 0) * 100,
                baseline_details.get("avg_sharpe", 0))

    # Generate and test variants
    logger.info("  Testing %d parameter variations...", n_variants)
    logger.info("=" * 60)

    best_score = baseline_score
    best_params = current_params
    best_details = baseline_details
    all_trials = []

    for i in range(n_variants):
        variant = mutate_params(current_params)
        score, details = _evaluate_params(variant, tech, symbol_data)

        trial = {
            "variant": i + 1,
            "score": round(score, 6),
            "return_pct": round(details.get("avg_return", 0) * 100, 2),
            "sharpe": round(details.get("avg_sharpe", 0), 4),
            "max_dd_pct": round(details.get("avg_drawdown", 0) * 100, 2),
            "trades": details.get("total_trades", 0),
            "params_changed": {k: v for k, v in variant.items()
                               if v != current_params.get(k)},
        }
        all_trials.append(trial)

        if score > best_score:
            best_score = score
            best_params = variant
            best_details = details
            logger.info(
                "  [%d/%d] NEW BEST: score=%.4f return=%.2f%% sharpe=%.2f trades=%d",
                i + 1, n_variants, score,
                details.get("avg_return", 0) * 100,
                details.get("avg_sharpe", 0),
                details.get("total_trades", 0),
            )
        elif (i + 1) % 10 == 0:
            logger.info("  [%d/%d] best so far: %.4f", i + 1, n_variants, best_score)

    improved = best_score > baseline_score
    changes_made = {k: v for k, v in best_params.items()
                    if v != current_params.get(k)} if improved else {}

    logger.info("=" * 60)
    if improved:
        logger.info("  IMPROVED: %.4f -> %.4f (%.1f%% better)",
                    baseline_score, best_score,
                    (best_score - baseline_score) / max(abs(baseline_score), 0.001) * 100)
        logger.info("  Parameters changed: %s", json.dumps(changes_made, indent=2))

        if not dry_run:
            SIGNAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SIGNAL_CONFIG_PATH, "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info("  Wrote improved config to %s", SIGNAL_CONFIG_PATH)
        else:
            logger.info("  DRY RUN: would write new config")
    else:
        logger.info("  No improvement found — keeping current config")
    logger.info("=" * 60)

    # Save tuning log
    tuning_report = {
        "date": today,
        "baseline_score": round(baseline_score, 6),
        "best_score": round(best_score, 6),
        "improved": improved,
        "n_variants_tested": n_variants,
        "baseline_params": current_params,
        "best_params": best_params if improved else current_params,
        "changes": changes_made,
        "baseline_details": baseline_details,
        "best_details": best_details,
        "trials": all_trials,
    }

    log_path = TUNING_LOG_DIR / f"tuning_{today}.json"
    with open(log_path, "w") as f:
        json.dump(tuning_report, f, indent=2, default=str)
    logger.info("Tuning log saved to %s", log_path)

    return tuning_report


def _evaluate_params(params: dict, tech: TechnicalFeatures,
                     symbol_data: dict) -> tuple[float, dict]:
    """Backtest a set of params across all symbols, return (score, details)."""
    engine = BacktestEngine(settings, initial_balance=STARTING_BALANCE)
    strategy = make_strategy(tech, params)

    scores = []
    returns = []
    sharpes = []
    drawdowns = []
    total_trades = 0

    for symbol, ohlcv in symbol_data.items():
        result = engine.run(strategy, ohlcv)
        s = score_result(result)
        scores.append(s)
        returns.append(result.total_return)
        sharpes.append(result.sharpe_ratio)
        drawdowns.append(result.max_drawdown)
        total_trades += result.total_trades

    avg_score = sum(scores) / len(scores) if scores else -999
    details = {
        "avg_return": sum(returns) / len(returns) if returns else 0,
        "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0,
        "avg_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        "total_trades": total_trades,
    }
    return avg_score, details


# =====================================================================
# Entry point
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-tune trading strategy parameters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tune but don't save improved config")
    parser.add_argument("--variants", type=int, default=30,
                        help="Number of parameter variations to test (default: 30)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  AUTO-TUNER — evolving the trading strategy")
    logger.info("  Variants to test: %d", args.variants)
    logger.info("  Mode: %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("=" * 60)

    report = run_tuning(n_variants=args.variants, dry_run=args.dry_run)

    if not report:
        logger.error("Tuning failed — no data")
        return 1

    if report.get("improved"):
        logger.info("Strategy improved! New config will be used in next trade run.")
    else:
        logger.info("No improvement found. Current strategy is the best so far.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
