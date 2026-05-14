#!/usr/bin/env python3
"""Train models on 5 years of daily OHLCV data with pagination.

Binance limits 1000 candles/request. 5yr daily ≈ 1825 candles → 2 paginated fetches.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omnitrade.config.settings import settings
from omnitrade.utils.logger import get_logger

logger = get_logger("train_5year")

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "config" / "ensemble_weights.json"

# 5 years of daily data
DAYS = 5 * 365
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
BINANCE_MAX_CANDLES = 1000
TIMEFRAME = "1d"


def fetch_all_daily(symbol: str) -> pd.DataFrame:
    """Fetch 5 years of daily OHLCV with pagination."""
    from omnitrade.data.collectors.market_data import MarketDataCollector

    collector = MarketDataCollector(settings)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)

    all_candles: List[List[Any]] = []
    since_ms = start_ms
    page = 0

    while since_ms < end_ms:
        page += 1
        try:
            raw = collector._retry(
                collector._exchange.fetch_ohlcv,
                symbol, TIMEFRAME, since=since_ms, limit=BINANCE_MAX_CANDLES,
            )
        except Exception as exc:
            logger.warning("Page %d fetch failed for %s: %s", page, symbol, exc)
            break

        if not raw or len(raw) <= 1:
            logger.info("Page %d: no more candles (got %d)", page, len(raw))
            break

        # Drop last candle (may be incomplete/current)
        candles = raw[:-1] if len(raw) > 1 else raw
        all_candles.extend(candles)

        last_ts = candles[-1][0]
        logger.info("Page %d: %d candles, %s → %s",
                    page, len(candles),
                    datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"))

        if len(raw) < BINANCE_MAX_CANDLES:
            logger.info("Last page (%d < %d candles)", len(raw), BINANCE_MAX_CANDLES)
            break

        # Next page starts right after last candle to avoid overlap
        since_ms = last_ts + 1
        time.sleep(0.5)  # Respect rate limits

    if not all_candles:
        logger.error("No data fetched for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    logger.info("%s: total %d daily candles, %s → %s",
                symbol, len(df),
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"))
    return df


def engineer_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features and generate labels."""
    from omnitrade.features.technical import TechnicalFeatures

    tech = TechnicalFeatures(settings)
    features = tech.compute_all(ohlcv)

    if "close" in features.columns:
        future_return = features["close"].pct_change(1).shift(-1)
        labels = pd.Series(0, index=features.index, name="signal")
        labels[future_return > 0.0015] = 1
        labels[future_return < -0.0015] = -1
        features["signal"] = labels
        features = features.iloc[:-1]

    features = features.ffill().bfill().fillna(0)
    return features


def _numeric_cols(features: pd.DataFrame) -> list[str]:
    exclude = {"signal", "timestamp"}
    return [
        c for c in features.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(features[c])
    ]


def train_xgboost(features: pd.DataFrame, feature_cols: list[str]) -> Tuple[Any, Dict]:
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


def train_lightgbm(features: pd.DataFrame, feature_cols: list[str]) -> Tuple[Any, Dict]:
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


def save_model(model: Any, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}_model"
    if hasattr(model, "save"):
        model.save(str(model_path))
    elif hasattr(model, "save_model"):
        model.save_model(str(model_path))
    else:
        logger.warning("No save method for %s", name)
        return
    logger.info("Saved %s → %s", name, model_path)


def main() -> int:
    logger.info("=" * 60)
    logger.info("  5-YEAR TRAINING PIPELINE (%s, %d days)", TIMEFRAME, DAYS)
    logger.info("  Symbols: %s", SYMBOLS)
    logger.info("  Sandbox: %s", settings.exchange.sandbox_mode)
    logger.info("=" * 60)

    all_features = []

    for symbol in SYMBOLS:
        logger.info("── Fetching %s ──", symbol)
        ohlcv = fetch_all_daily(symbol)
        if ohlcv.empty or len(ohlcv) < 100:
            logger.warning("SKIP %s: insufficient data (%d candles)", symbol, len(ohlcv))
            continue

        logger.info("── Engineering features for %s ──", symbol)
        features = engineer_features(ohlcv)
        if len(features) < 100:
            logger.warning("SKIP %s: insufficient features (%d rows)", symbol, len(features))
            continue

        logger.info("%s: %d rows x %d cols", symbol, len(features), len(features.columns))
        all_features.append(features)

    if not all_features:
        logger.error("No features generated — aborting")
        return 1

    combined = pd.concat(all_features, ignore_index=True)
    feature_cols = _numeric_cols(combined)
    logger.info("Combined: %d rows x %d numeric features", len(combined), len(feature_cols))

    # Label distribution
    labels = combined["signal"].value_counts().to_dict()
    logger.info("Label distribution: %s", labels)

    all_trained: Dict[str, Tuple[Any, Dict]] = {}

    # ── XGBoost ──
    logger.info("── Training XGBoost ──")
    try:
        xgb_model, xgb_metrics = train_xgboost(combined, feature_cols)
        all_trained["xgboost"] = (xgb_model, xgb_metrics)
        logger.info("XGBoost: acc=%.4f  f1=%.4f",
                    xgb_metrics.get("val_accuracy", 0),
                    xgb_metrics.get("val_f1_macro", 0))
    except Exception as exc:
        logger.warning("XGBoost training failed: %s", exc)

    # ── LightGBM ──
    logger.info("── Training LightGBM ──")
    try:
        lgb_model, lgb_metrics = train_lightgbm(combined, feature_cols)
        all_trained["lightgbm"] = (lgb_model, lgb_metrics)
        logger.info("LightGBM: acc=%.4f  f1=%.4f",
                    lgb_metrics.get("val_accuracy", 0),
                    lgb_metrics.get("val_f1_macro", 0))
    except Exception as exc:
        logger.warning("LightGBM training failed: %s", exc)

    if not all_trained:
        logger.error("No models trained — aborting")
        return 1

    # ── Save ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (model, _) in all_trained.items():
        save_model(model, name)

    # Ensemble config
    ENSEMBLE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, (_, metrics) in all_trained.items():
        f1 = metrics.get("val_f1_macro", metrics.get("f1", 0.5))
        acc = metrics.get("val_accuracy", metrics.get("accuracy", 0.5))
        weights[name] = round(max((f1 + acc) / 2.0, 0.1), 4)

    config = {
        "weights": weights,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "models": list(weights.keys()),
        "training_days": DAYS,
        "timeframe": TIMEFRAME,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Ensemble config saved: %s", ENSEMBLE_CONFIG_PATH)

    # Metadata
    from omnitrade.models.model_metadata import ModelMetadata
    meta = ModelMetadata(path=str(PROJECT_ROOT / "output" / "model_metadata.json"))
    for name, (_, metrics) in all_trained.items():
        meta.record_training(name, metrics, asset_type="crypto")

    logger.info("=" * 60)
    logger.info("  DONE — %d models saved to %s", len(all_trained), MODELS_DIR)
    logger.info("  Weights: %s", json.dumps(weights))
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
