"""Smoke tests for StockModelFactory + EnsembleVoter delegation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from omnitrade.config.settings import Settings
from omnitrade.models.stock_models import StockModelFactory


def test_factory_creates_no_voter_on_init():
    """Factory creates internal voter but no models until asked."""
    settings = Settings()
    factory = StockModelFactory(settings)
    assert not factory.is_trained
    assert factory.voter is not None


def test_predict_returns_hold_when_untrained():
    """Untrained factory returns HOLD with zero confidence."""
    settings = Settings()
    factory = StockModelFactory(settings)
    result = factory.predict(pd.DataFrame({"close": [100, 101, 102]}))
    assert result["signal"] == "HOLD"
    assert result["confidence"] == 0.0


def test_create_all_returns_dict():
    """create_all returns model dict even when backends are unavailable."""
    settings = Settings()
    factory = StockModelFactory(settings)
    models = factory.create_all()
    assert isinstance(models, dict)


if __name__ == "__main__":
    tests = [
        test_factory_creates_no_voter_on_init,
        test_predict_returns_hold_when_untrained,
        test_create_all_returns_dict,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} — {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
