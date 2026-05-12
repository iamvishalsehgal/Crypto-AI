#!/usr/bin/env python3
"""
Train all ML models on real market data. One-shot training script.

Usage:
    python scripts/train_all_models.py                # Train everything
    python scripts/train_all_models.py --symbols BTC/USDT ETH/USDT
    python scripts/train_all_models.py --epochs 100
    python scripts/train_all_models.py --dry-run      # Don't save
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omnitrade.config.settings import settings
from omnitrade.utils.logger import get_logger

logger = get_logger("train_all_models")

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_weights.json"
METADATA_PATH = PROJECT_ROOT / "output" / "model_metadata.json"


def collect_crypto_data(symbols: list[str]) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for crypto symbols."""
    from omnitrade.data.collectors.market_data import MarketDataCollector

    collector = MarketDataCollector(settings)
    market_data = {}

    for symbol in symbols:
        try:
            ohlcv = collector.fetch_ohlcv(symbol, timeframe="15m", limit=2000)  # ~3 weeks of 15m candles
            if not ohlcv.empty and len(ohlcv) >= 100:
                market_data[symbol] = ohlcv
                logger.info("Fetched %d candles for %s", len(ohlcv), symbol)
            else:
                logger.warning("Insufficient data for %s: %d rows", symbol, len(ohlcv))
        except Exception as exc:
            logger.warning("Market data fetch failed for %s: %s", symbol, exc)

    return market_data


def engineer_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features and generate labels."""
    from omnitrade.features.technical import TechnicalFeatures

    tech = TechnicalFeatures(settings)
    features = tech.compute_all(ohlcv)

    # Generate labels from future returns
    if "close" in features.columns:
        future_return = features["close"].pct_change(1).shift(-1)
        labels = pd.Series(0, index=features.index, name="signal")
        labels[future_return > 0.005] = 1
        labels[future_return < -0.005] = -1
        features["signal"] = labels
        features = features.iloc[:-1]

    features = features.ffill().bfill().fillna(0)
    return features


def _numeric_feature_cols(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude timestamp, datetime, object)."""
    exclude = {"signal", "timestamp"}
    return [
        c for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]


def train_xgboost(
    features: pd.DataFrame, feature_cols: list[str]
) -> Tuple[Any, Dict]:
    """Train XGBoost model."""
    from omnitrade.models.xgboost_model import XGBoostTrader

    n = len(features)
    split_train = int(n * 0.70)
    split_val = int(n * 0.85)

    X_train = features.iloc[:split_train][feature_cols]
    y_train = features.iloc[:split_train]["signal"]
    X_val = features.iloc[split_train:split_val][feature_cols]
    y_val = features.iloc[split_train:split_val]["signal"]

    model = XGBoostTrader(settings)
    metrics = model.train(X_train, y_train, X_val, y_val)
    return model, metrics


def train_lightgbm(
    features: pd.DataFrame, feature_cols: list[str]
) -> Tuple[Any, Dict]:
    """Train LightGBM model."""
    from omnitrade.models.lightgbm_model import LightGBMTrader

    n = len(features)
    split_train = int(n * 0.70)
    split_val = int(n * 0.85)

    X_train = features.iloc[:split_train][feature_cols]
    y_train = features.iloc[:split_train]["signal"]
    X_val = features.iloc[split_train:split_val][feature_cols]
    y_val = features.iloc[split_train:split_val]["signal"]

    model = LightGBMTrader(settings)
    metrics = model.train(X_train, y_train, X_val, y_val)
    return model, metrics


def train_lstm(
    features: pd.DataFrame, epochs: int = 50
) -> Tuple[Any, Dict]:
    """Train LSTM model via ModelTrainer."""
    from omnitrade.models.training.trainer import ModelTrainer

    trainer = ModelTrainer(settings)
    # Use shorter sequence length for smaller datasets
    seq_len = min(60, max(10, len(features) // 8))
    model, metrics = trainer.train_lstm(
        features, model_type="lstm",
        target_col="signal", epochs=epochs, patience=10,
        sequence_length=seq_len,
    )
    return model, metrics


def train_stock_models() -> Dict[str, Tuple[Any, Dict]]:
    """Train stock models on all configured tickers."""
    from omnitrade.data.collectors.stock_data import StockDataCollector
    from omnitrade.features.stock_features import StockFeaturePipeline
    from omnitrade.models.stock_models import StockModelFactory

    collector = StockDataCollector(settings)
    pipeline = StockFeaturePipeline(settings)
    factory = StockModelFactory(settings)

    results = {}
    all_features = []
    all_labels = []

    for ticker in settings.stock.supported_tickers:
        try:
            ohlcv = collector.fetch_ohlcv(ticker, interval="1h")
            fundamentals = collector.fetch_fundamentals(ticker)

            if ohlcv.empty:
                logger.warning("No OHLCV for %s, skipping", ticker)
                continue

            features = pipeline.compute_all(ohlcv, fundamentals)
            if features.empty or len(features) < 100:
                logger.warning("Insufficient features for %s: %d rows", ticker, len(features))
                continue

            # Generate labels
            if "close" in features.columns:
                future_return = features["close"].pct_change(1).shift(-1)
                features["signal"] = pd.Series(0, index=features.index)
                features.loc[future_return > 0.005, "signal"] = 1
                features.loc[future_return < -0.005, "signal"] = -1
                features = features.iloc[:-1]

            features = features.ffill().bfill().fillna(0)
            all_features.append(features)
            logger.info("%s: %d rows x %d cols", ticker, len(features), len(features.columns))

        except Exception as exc:
            logger.warning("Stock feature pipeline failed for %s: %s", ticker, exc)

    if not all_features:
        logger.warning("No stock features generated")
        return results

    combined = pd.concat(all_features, ignore_index=True)
    feature_cols = _numeric_feature_cols(combined)
    labels = combined["signal"].values

    try:
        factory.create_all()  # Must create models before training
        stock_metrics = factory.train_all(combined[feature_cols], pd.Series(labels))
        results["stock_ensemble"] = (factory, stock_metrics)
        logger.info("Stock models trained: %s", list(stock_metrics.keys()) if stock_metrics else "none")
    except Exception as exc:
        logger.warning("Stock model training failed: %s", exc)
        logger.debug(traceback.format_exc())

    return results


def train_betting_models() -> Dict[str, Tuple[Any, Dict]]:
    """Train betting value-detection model."""
    from omnitrade.data.collectors.betting_data import BettingDataCollector
    from omnitrade.features.betting_features import BettingFeaturePipeline
    from omnitrade.models.betting_models import ValueBettingModel, PoissonModel

    results = {}

    try:
        collector = BettingDataCollector(settings)
        pipeline = BettingFeaturePipeline(settings)

        all_features = []
        for sport in settings.betting.supported_sports:
            try:
                odds_data = collector.fetch_odds(sport_key=sport)
                features = pipeline.compute_all(odds_data)
                if not features.empty:
                    features["sport_key"] = sport
                    all_features.append(features)
                    logger.info("%s: %d rows x %d cols", sport, len(features), len(features.columns))
            except Exception as exc:
                logger.warning("Betting data collection failed for %s: %s", sport, exc)

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            feature_cols = [
                c for c in combined.columns
                if c not in ("sport_key", "outcome", "settled")
            ]

            # Train value betting model on historical data
            if "outcome" in combined.columns:
                vbm = ValueBettingModel(settings)
                vbm_metrics = vbm.train(combined[feature_cols], combined["outcome"])
                results["value_betting"] = (vbm, vbm_metrics)
                logger.info("Value betting model trained: %s", vbm_metrics)

            # Train Poisson model
            poisson = PoissonModel(settings)
            poisson.train(combined)
            results["poisson"] = (poisson, {"model_type": "poisson"})
            logger.info("Poisson model trained")

    except Exception as exc:
        logger.warning("Betting model training failed: %s", exc)
        logger.debug(traceback.format_exc())

    return results


def calibrate_weights(trained: Dict[str, Tuple[Any, Dict]]) -> Dict[str, float]:
    """Assign ensemble weights based on model metrics."""
    weights = {}
    for name, (_, metrics) in trained.items():
        if isinstance(metrics, dict):
            f1 = metrics.get("val_f1_macro", metrics.get("f1", metrics.get("mean_f1", 0)))
            acc = metrics.get("val_accuracy", metrics.get("accuracy", metrics.get("mean_accuracy", 0)))
            # Weight = average of normalized F1 and accuracy
            weight = max((f1 + acc) / 2.0, 0.1)
        else:
            weight = 0.5
        weights[name] = round(float(weight), 4)
    return weights


def save_all(
    trained: Dict[str, Tuple[Any, Dict]],
    weights: Dict[str, float],
    dry_run: bool = False,
) -> None:
    """Save models and ensemble config."""
    if dry_run:
        logger.info("DRY RUN — skipping save")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save ensemble config
    config = {
        "weights": weights,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "models": list(weights.keys()),
    }
    with open(ENSEMBLE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved ensemble config: %s", ENSEMBLE_CONFIG_PATH)

    # Save individual models
    for name, (model, _) in trained.items():
        try:
            model_path = MODELS_DIR / f"{name}_model"
            if hasattr(model, "save"):
                model.save(str(model_path))
                logger.info("Saved %s → %s", name, model_path)
            elif hasattr(model, "save_model"):
                model.save_model(str(model_path))
                logger.info("Saved %s → %s", name, model_path)
            elif hasattr(model, "_models"):
                # StockModelFactory — save individual sub-models
                for sub_name, sub_model in model._models.items():
                    sub_path = MODELS_DIR / f"stock_{sub_name}_model"
                    try:
                        if hasattr(sub_model, "save"):
                            sub_model.save(str(sub_path))
                            logger.info("Saved stock/%s → %s", sub_name, sub_path)
                        elif hasattr(sub_model, "save_model"):
                            sub_model.save_model(str(sub_path))
                            logger.info("Saved stock/%s → %s", sub_name, sub_path)
                    except Exception as sub_exc:
                        logger.warning("Failed to save stock/%s: %s", sub_name, sub_exc)
            else:
                logger.warning("No save method for %s", name)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", name, exc)

    # Save model metadata
    from omnitrade.models.model_metadata import ModelMetadata
    meta = ModelMetadata(path=str(METADATA_PATH))
    for name, (_, metrics) in trained.items():
        asset = "crypto"
        if name.startswith("stock"):
            asset = "stock"
        elif name.startswith(("bet", "value", "poisson")):
            asset = "bet"
        meta.record_training(name, metrics, asset_type=asset)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all OmniTrade AI models")
    parser.add_argument("--symbols", nargs="+",
                        default=settings.exchange.supported_symbols,
                        help="Crypto symbols to train on")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Neural network epochs (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train but don't save")
    parser.add_argument("--skip-stock", action="store_true",
                        help="Skip stock model training")
    parser.add_argument("--skip-betting", action="store_true",
                        help="Skip betting model training")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  OMNITRADE AI — MODEL TRAINING PIPELINE")
    logger.info("  Mode: %s | Epochs: %d", "DRY RUN" if args.dry_run else "SAVE", args.epochs)
    logger.info("  Crypto symbols: %s", args.symbols)
    logger.info("  Stock: %s | Betting: %s",
                "skip" if args.skip_stock else "train",
                "skip" if args.skip_betting else "train")
    logger.info("=" * 60)

    all_trained: Dict[str, Tuple[Any, Dict]] = {}

    # ── 1. Crypto models ──────────────────────────────────────────
    logger.info("── Step 1: Crypto data collection ──")
    market_data = collect_crypto_data(args.symbols)

    if not market_data:
        logger.error("No crypto market data — aborting")
        return 1

    all_features = []
    for symbol, ohlcv in market_data.items():
        try:
            features = engineer_features(ohlcv)
            if len(features) >= 100:
                all_features.append(features)
                logger.info("  %s: %d rows x %d cols", symbol, len(features), len(features.columns))
        except Exception as exc:
            logger.warning("Feature engineering failed for %s: %s", symbol, exc)

    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        feature_cols = _numeric_feature_cols(combined)
        logger.info("Combined features: %d rows x %d cols (%d numeric)", len(combined), len(combined.columns), len(feature_cols))

        # Train XGBoost
        logger.info("── Training XGBoost ──")
        try:
            xgb_model, xgb_metrics = train_xgboost(combined, feature_cols)
            all_trained["xgboost"] = (xgb_model, xgb_metrics)
            logger.info("  XGBoost: acc=%.3f, f1=%.3f",
                        xgb_metrics.get("val_accuracy", 0),
                        xgb_metrics.get("val_f1_macro", 0))
        except Exception as exc:
            logger.warning("XGBoost training failed: %s", exc)

        # Train LightGBM
        logger.info("── Training LightGBM ──")
        try:
            lgb_model, lgb_metrics = train_lightgbm(combined, feature_cols)
            all_trained["lightgbm"] = (lgb_model, lgb_metrics)
            logger.info("  LightGBM: acc=%.3f, f1=%.3f",
                        lgb_metrics.get("val_accuracy", 0),
                        lgb_metrics.get("val_f1_macro", 0))
        except Exception as exc:
            logger.warning("LightGBM training failed: %s", exc)

        # Train LSTM
        logger.info("── Training LSTM ──")
        try:
            lstm_model, lstm_metrics = train_lstm(combined, epochs=args.epochs)
            all_trained["lstm"] = (lstm_model, lstm_metrics)
            logger.info("  LSTM: acc=%.3f, f1=%.3f",
                        lstm_metrics.get("mean_accuracy", 0),
                        lstm_metrics.get("mean_f1", 0))
        except Exception as exc:
            logger.warning("LSTM training failed: %s", exc)
            logger.debug(traceback.format_exc())

    # ── 2. Stock models ───────────────────────────────────────────
    if not args.skip_stock:
        logger.info("── Step 2: Stock model training ──")
        stock_results = train_stock_models()
        all_trained.update(stock_results)

    # ── 3. Betting models ─────────────────────────────────────────
    if not args.skip_betting:
        logger.info("── Step 3: Betting model training ──")
        betting_results = train_betting_models()
        all_trained.update(betting_results)

    if not all_trained:
        logger.error("No models trained — aborting")
        return 1

    # ── 4. Calibrate ensemble ─────────────────────────────────────
    logger.info("── Step 4: Calibrating ensemble weights ──")
    weights = calibrate_weights(all_trained)
    logger.info("Weights: %s", json.dumps(weights, indent=2))

    # ── 5. Save ───────────────────────────────────────────────────
    logger.info("── Step 5: Saving models ──")
    save_all(all_trained, weights, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE — %d models trained", len(all_trained))
    logger.info("  Weights: %s", json.dumps(weights))
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
