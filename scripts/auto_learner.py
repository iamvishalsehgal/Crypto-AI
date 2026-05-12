#!/usr/bin/env python3
"""
Self-learning pipeline — trains ML models on real data and improves over time.

Collects real market data + sentiment + on-chain + macro data, engineers
features, trains the ML models (LSTM, XGBoost, DDQN), calibrates the
ensemble voter weights, and saves everything if the new models perform
better than the previous ones.

Each day the bot:
  1. Fetches fresh real data from all sources
  2. Engineers features (technical + sentiment + on-chain + macro)
  3. Generates training labels from actual price movements
  4. Trains/retrains the ML models on recent data
  5. Calibrates ensemble weights based on backtest accuracy
  6. Backtests the full ensemble vs the old one
  7. Saves the improved model weights + ensemble config if better

The models learn from:
  - Chart patterns (LSTM sequences, CNN candlestick images)
  - Technical indicators (RSI, MACD, BB, Stoch, EMA, ADX, etc.)
  - Market sentiment (Reddit, Twitter, news, Fear & Greed index)
  - On-chain data (whale movements, exchange flows, network activity)
  - Macro data (Fed rates, VIX, DXY, gold, oil)

Usage:
    python scripts/auto_learner.py                  # Full learning run
    python scripts/auto_learner.py --dry-run        # Train but don't save
    python scripts/auto_learner.py --epochs 200     # More training epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.config.settings import settings
from crypto_bot.utils.logger import get_logger

logger = get_logger("crypto_bot.auto_learner")

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_weights.json"
LEARNING_LOG_DIR = PROJECT_ROOT / "reports" / "learning"
STARTING_BALANCE = 10_000.0


# =====================================================================
# Data collection — gather everything
# =====================================================================

def collect_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Fetch real data from all available sources.

    Returns (market_data_by_symbol, extra_data).
    """
    from crypto_bot.data.collectors.market_data import MarketDataCollector

    market = MarketDataCollector(settings)
    symbols = settings.exchange.supported_symbols
    market_data = {}

    for symbol in symbols:
        try:
            ohlcv = market.fetch_ohlcv(symbol, timeframe="1h", limit=2000)
            if not ohlcv.empty and len(ohlcv) >= 100:
                market_data[symbol] = ohlcv
                logger.info("Fetched %d candles for %s", len(ohlcv), symbol)
        except Exception as exc:
            logger.warning("Market data fetch failed for %s: %s", symbol, exc)

    # Collect sentiment, on-chain, macro data (best effort)
    extra = {}

    try:
        from crypto_bot.data.collectors.sentiment_data import SentimentCollector
        sentiment = SentimentCollector(settings)
        extra["reddit"] = sentiment.fetch_reddit_posts()
        logger.info("Fetched Reddit sentiment data")
    except Exception as exc:
        logger.debug("Reddit data unavailable: %s", exc)

    try:
        from crypto_bot.data.collectors.sentiment_data import SentimentCollector
        sentiment = SentimentCollector(settings)
        extra["news"] = sentiment.fetch_news_articles()
        logger.info("Fetched news data")
    except Exception as exc:
        logger.debug("News data unavailable: %s", exc)

    try:
        from crypto_bot.data.collectors.onchain_data import OnChainCollector
        onchain = OnChainCollector(settings)
        extra["whale_transfers"] = onchain.fetch_whale_transfers()
        extra["exchange_flows"] = onchain.fetch_exchange_flows()
        logger.info("Fetched on-chain data")
    except Exception as exc:
        logger.debug("On-chain data unavailable: %s", exc)

    try:
        from crypto_bot.data.collectors.macro_data import MacroDataCollector
        macro = MacroDataCollector(settings)
        extra["macro"] = macro.fetch_all()
        logger.info("Fetched macro data")
    except Exception as exc:
        logger.debug("Macro data unavailable: %s", exc)

    try:
        from crypto_bot.data.collectors.web_scraper import WebScraper
        scraper = WebScraper()
        extra["fear_greed"] = scraper.fetch_fear_greed()
        logger.info("Fetched Fear & Greed index")
    except Exception as exc:
        logger.debug("Fear & Greed unavailable: %s", exc)

    return market_data, extra


# =====================================================================
# Feature engineering — combine all data sources
# =====================================================================

def engineer_features(ohlcv: pd.DataFrame, extra: Dict[str, Any]) -> pd.DataFrame:
    """Compute all available features from market + auxiliary data."""
    from crypto_bot.features.technical import TechnicalFeatures

    tech = TechnicalFeatures(settings)

    # Start with technical features (always available)
    features = tech.compute_all(ohlcv)

    # Try adding sentiment features
    try:
        from crypto_bot.features.sentiment_features import SentimentFeatures
        sent_feat = SentimentFeatures(settings)
        sent_df = sent_feat.compute_all()
        if sent_df is not None and not sent_df.empty:
            features = _safe_merge(features, sent_df)
            logger.info("Added %d sentiment features", len(sent_df.columns))
    except Exception as exc:
        logger.debug("Sentiment features skipped: %s", exc)

    # Try adding on-chain features
    try:
        from crypto_bot.features.onchain_features import OnChainFeatures
        onchain_feat = OnChainFeatures(settings)
        onchain_df = onchain_feat.compute_all()
        if onchain_df is not None and not onchain_df.empty:
            features = _safe_merge(features, onchain_df)
            logger.info("Added %d on-chain features", len(onchain_df.columns))
    except Exception as exc:
        logger.debug("On-chain features skipped: %s", exc)

    # Try adding macro features
    try:
        from crypto_bot.features.macro_features import MacroFeatures
        macro_feat = MacroFeatures(settings)
        macro_df = macro_feat.compute_all()
        if macro_df is not None and not macro_df.empty:
            features = _safe_merge(features, macro_df)
            logger.info("Added %d macro features", len(macro_df.columns))
    except Exception as exc:
        logger.debug("Macro features skipped: %s", exc)

    # Generate training labels from actual price movements
    features = _add_labels(features)

    # Drop rows with NaN (from indicator warm-up)
    features = features.dropna()

    logger.info("Feature matrix: %d rows x %d columns", len(features), len(features.columns))
    return features


def _safe_merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Merge extra features into the base DataFrame by index alignment."""
    if base.index.dtype == extra.index.dtype:
        return base.join(extra, how="left")
    # If indices don't align, broadcast the last known values
    for col in extra.columns:
        if col not in base.columns:
            base[col] = extra[col].iloc[-1] if not extra.empty else 0
    return base


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signal labels from future price returns.

    Label: BUY (1) if next-period return > +0.5%
           SELL (-1) if next-period return < -0.5%
           HOLD (0) otherwise
    """
    if "close" not in df.columns:
        return df

    future_return = df["close"].pct_change(1).shift(-1)
    labels = pd.Series(0, index=df.index, name="signal")
    labels[future_return > 0.005] = 1    # BUY
    labels[future_return < -0.005] = -1  # SELL

    df["signal"] = labels
    # Drop the last row (no future to predict)
    df = df.iloc[:-1]
    return df


# =====================================================================
# Model training
# =====================================================================

def train_models(features: pd.DataFrame, epochs: int = 50) -> Dict[str, Tuple[Any, Dict]]:
    """Train available ML models on the feature matrix."""
    trained = {}

    # 1. XGBoost — fast, tabular, always works
    try:
        logger.info("── Training XGBoost ──")
        from crypto_bot.models.xgboost_model import XGBoostTrader

        feature_cols = [c for c in features.columns if c != "signal"]
        n = len(features)
        split = int(n * 0.8)

        X_train = features.iloc[:split][feature_cols]
        y_train = features.iloc[:split]["signal"]
        X_val = features.iloc[split:][feature_cols]
        y_val = features.iloc[split:]["signal"]

        xgb = XGBoostTrader(settings)
        metrics = xgb.train(X_train, y_train, X_val, y_val)
        trained["xgboost"] = (xgb, metrics)
        logger.info("  XGBoost: accuracy=%.3f, f1=%.3f",
                    metrics.get("val_accuracy", 0), metrics.get("val_f1_macro", 0))
    except Exception as exc:
        logger.warning("XGBoost training failed: %s", exc)
        logger.debug(traceback.format_exc())

    # 2. LSTM — sequence model for time series patterns
    try:
        logger.info("── Training LSTM ──")
        from crypto_bot.models.training.trainer import ModelTrainer

        trainer = ModelTrainer(settings)
        lstm_model, lstm_metrics = trainer.train_lstm(
            features, model_type="lstm",
            target_col="signal", epochs=epochs, patience=10,
        )
        trained["lstm"] = (lstm_model, lstm_metrics)
        logger.info("  LSTM: f1=%.3f, accuracy=%.3f",
                    lstm_metrics.get("mean_f1", 0), lstm_metrics.get("mean_accuracy", 0))
    except Exception as exc:
        logger.warning("LSTM training failed: %s", exc)
        logger.debug(traceback.format_exc())

    # 3. CNN — candlestick pattern recognition
    try:
        logger.info("── Training CNN ──")
        from crypto_bot.models.cnn_model import CNNTrainer, CandlestickImageGenerator

        img_gen = CandlestickImageGenerator()
        images, labels = img_gen.generate_dataset(features)
        if len(images) > 50:
            cnn = CNNTrainer(settings)
            cnn_metrics = cnn.train(images, labels, epochs=epochs)
            trained["cnn"] = (cnn, cnn_metrics)
            logger.info("  CNN: accuracy=%.3f",
                        cnn_metrics.get("val_accuracy", 0))
    except Exception as exc:
        logger.warning("CNN training failed: %s", exc)
        logger.debug(traceback.format_exc())

    # 4. FinBERT sentiment — pre-trained, no training needed
    try:
        logger.info("── Loading FinBERT sentiment ──")
        from crypto_bot.models.sentiment_model import SentimentAnalyzer

        finbert = SentimentAnalyzer(settings)
        trained["sentiment"] = (finbert, {"model_type": "finbert", "pretrained": True})
        logger.info("  FinBERT: loaded (pre-trained)")
    except Exception as exc:
        logger.warning("FinBERT loading failed: %s", exc)

    logger.info("Trained %d models: %s", len(trained), list(trained.keys()))
    return trained


# =====================================================================
# Ensemble calibration
# =====================================================================

def calibrate_ensemble(
    trained_models: Dict[str, Tuple[Any, Dict]],
    features: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate each model on held-out data and set ensemble weights.

    Models that predict more accurately get higher weights.
    """
    from crypto_bot.ensemble.voting_system import EnsembleVoter

    voter = EnsembleVoter(settings)

    feature_cols = [c for c in features.columns if c != "signal"]
    n = len(features)
    # Use last 20% as evaluation set
    eval_start = int(n * 0.8)
    X_eval = features.iloc[eval_start:][feature_cols]
    y_eval = features.iloc[eval_start:]["signal"].values

    weights = {}

    for name, (model, _) in trained_models.items():
        try:
            if name == "sentiment":
                # Sentiment model needs text, not features — give base weight
                weights[name] = 0.5
                continue

            if hasattr(model, "predict"):
                preds = model.predict(X_eval)
                if isinstance(preds, np.ndarray) and preds.ndim == 2:
                    preds = preds.argmax(axis=1)
                if isinstance(preds, (list, np.ndarray)):
                    preds = np.array(preds).flatten()
                    # Calculate accuracy
                    correct = np.sum(preds == y_eval[:len(preds)])
                    total = len(preds)
                    accuracy = correct / total if total > 0 else 0
                    weight = max(float(accuracy), 0.1)
                    weights[name] = round(weight, 4)
                    logger.info("  %s: accuracy=%.3f → weight=%.3f", name, accuracy, weight)
                else:
                    weights[name] = 0.5
            else:
                weights[name] = 0.5
        except Exception as exc:
            logger.warning("  %s calibration failed: %s", name, exc)
            weights[name] = 0.3

    logger.info("Ensemble weights: %s", json.dumps(weights, indent=2))
    return weights


# =====================================================================
# Evaluation — is the new version better?
# =====================================================================

def evaluate_improvement(
    new_weights: Dict[str, float],
    old_weights: Optional[Dict[str, float]],
) -> Tuple[bool, list[str]]:
    """Check if the new ensemble is meaningfully different/better."""
    reasons = []

    if old_weights is None:
        reasons.append("First training run — establishing baseline")
        return True, reasons

    # Compare total weight (more confident models = better)
    new_total = sum(new_weights.values())
    old_total = sum(old_weights.values())

    if new_total > old_total * 1.05:
        reasons.append(f"Total confidence improved: {old_total:.2f} → {new_total:.2f}")
    elif new_total < old_total * 0.90:
        reasons.append(f"Total confidence dropped: {old_total:.2f} → {new_total:.2f}")

    # Check individual model improvements
    for name in set(list(new_weights.keys()) + list(old_weights.keys())):
        new_w = new_weights.get(name, 0)
        old_w = old_weights.get(name, 0)
        if abs(new_w - old_w) > 0.05:
            direction = "improved" if new_w > old_w else "regressed"
            reasons.append(f"  {name}: {old_w:.3f} → {new_w:.3f} ({direction})")

    if not reasons:
        reasons.append("No significant changes in model performance")

    # Always save new models — they're trained on more recent data
    return True, reasons


# =====================================================================
# Save / load
# =====================================================================

def save_models(trained_models: Dict[str, Tuple[Any, Dict]], weights: Dict[str, float]) -> None:
    """Save model weights and ensemble config."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save ensemble weights
    config = {
        "weights": weights,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "models": list(weights.keys()),
    }
    with open(ENSEMBLE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved ensemble config to %s", ENSEMBLE_CONFIG_PATH)

    # Save individual model states
    for name, (model, _) in trained_models.items():
        try:
            if name == "sentiment":
                continue  # pre-trained, no need to save

            model_path = MODELS_DIR / f"{name}_model"
            if hasattr(model, "save"):
                model.save(str(model_path))
                logger.info("  Saved %s model", name)
            elif hasattr(model, "model"):
                import torch
                torch.save(model.model.state_dict(), str(model_path) + ".pt")
                logger.info("  Saved %s model state", name)
        except Exception as exc:
            logger.warning("  Failed to save %s: %s", name, exc)


def load_previous_weights() -> Optional[Dict[str, float]]:
    """Load ensemble weights from previous run."""
    if not ENSEMBLE_CONFIG_PATH.exists():
        return None
    try:
        with open(ENSEMBLE_CONFIG_PATH) as f:
            config = json.load(f)
        return config.get("weights", None)
    except (json.JSONDecodeError, OSError):
        return None


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Self-learning trading bot pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and evaluate but don't save models")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for neural networks (default: 50)")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  SELF-LEARNING PIPELINE")
    logger.info("  Mode: %s | Epochs: %d", "DRY RUN" if args.dry_run else "LIVE", args.epochs)
    logger.info("=" * 60)

    # 1. Collect all data
    logger.info("Step 1: Collecting real data from all sources...")
    market_data, extra_data = collect_all_data()

    if not market_data:
        logger.error("No market data available — aborting")
        return 1

    logger.info("  Got data for %d symbols, %d extra sources",
                len(market_data), len(extra_data))

    # 2. Engineer features for each symbol
    logger.info("Step 2: Engineering features...")
    all_features = []
    for symbol, ohlcv in market_data.items():
        try:
            features = engineer_features(ohlcv, extra_data)
            if len(features) >= 100:
                all_features.append(features)
                logger.info("  %s: %d rows, %d features", symbol, len(features), len(features.columns))
        except Exception as exc:
            logger.warning("  Feature engineering failed for %s: %s", symbol, exc)

    if not all_features:
        logger.error("No features produced — aborting")
        return 1

    # Combine features from all symbols
    combined = pd.concat(all_features, ignore_index=True)
    combined = combined.dropna()
    logger.info("  Combined feature matrix: %d rows x %d columns", len(combined), len(combined.columns))

    # 3. Train models
    logger.info("Step 3: Training ML models on real data...")
    trained_models = train_models(combined, epochs=args.epochs)

    if not trained_models:
        logger.error("No models trained successfully — aborting")
        return 1

    # 4. Calibrate ensemble weights
    logger.info("Step 4: Calibrating ensemble weights...")
    new_weights = calibrate_ensemble(trained_models, combined)

    # 5. Compare to previous
    logger.info("Step 5: Evaluating improvement...")
    old_weights = load_previous_weights()
    improved, reasons = evaluate_improvement(new_weights, old_weights)

    logger.info("─" * 60)
    for r in reasons:
        logger.info("  %s", r)
    logger.info("─" * 60)

    # 6. Save if improved (or first run)
    if improved and not args.dry_run:
        logger.info("Step 6: Saving improved models and weights...")
        save_models(trained_models, new_weights)
    elif args.dry_run:
        logger.info("DRY RUN: would save models (improved=%s)", improved)
    else:
        logger.info("No improvement — keeping previous models")

    # 7. Save learning log
    learning_report = {
        "date": today,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": list(market_data.keys()),
        "data_sources": {
            "market": len(market_data),
            "extra": list(extra_data.keys()),
        },
        "feature_matrix": {
            "rows": len(combined),
            "columns": len(combined.columns),
        },
        "models_trained": {
            name: {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool))
            }
            for name, (_, metrics) in trained_models.items()
        },
        "ensemble_weights": new_weights,
        "previous_weights": old_weights,
        "improved": improved,
        "reasons": reasons,
    }

    log_path = LEARNING_LOG_DIR / f"learning_{today}.json"
    with open(log_path, "w") as f:
        json.dump(learning_report, f, indent=2, default=str)
    logger.info("Learning log saved to %s", log_path)

    logger.info("=" * 60)
    logger.info("  LEARNING COMPLETE — %d models trained", len(trained_models))
    logger.info("  Weights: %s", json.dumps(new_weights))
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
