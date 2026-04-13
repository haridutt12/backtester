"""
Shared fixtures for all test modules.
"""

import os

import numpy as np
import pandas as pd
import pytest

# ── make sure dummy credentials are set before any src module is imported ─────
os.environ.setdefault("DHAN_CLIENT_ID", "test-id")
os.environ.setdefault("DHAN_ACCESS_TOKEN", "test-token")


def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 1000 + np.cumsum(rng.normal(0, 5, n))
    high = close + rng.uniform(1, 10, n)
    low = close - rng.uniform(1, 10, n)
    open_ = close - rng.normal(0, 3, n)
    volume = rng.integers(100_000, 1_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """120 business-days of synthetic OHLCV data."""
    return _make_ohlcv(120)


@pytest.fixture
def short_data() -> pd.DataFrame:
    """50 business-days — exercises edge cases (fewer rows than warmup)."""
    return _make_ohlcv(50, seed=7)
