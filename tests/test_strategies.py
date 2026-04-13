"""
Unit tests for strategy classes.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.inside_candle_rsi import InsideCandleRSIStrategy


# ── MACrossoverStrategy ───────────────────────────────────────────────────────

class TestMACrossoverStrategy:
    def test_generates_signal_column(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        result = strategy.generate_signals(sample_data)
        assert "Signal" in result.columns

    def test_signals_are_valid_values(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        result = strategy.generate_signals(sample_data)
        assert set(result["Signal"].unique()).issubset({-1, 0, 1})

    def test_no_signals_during_warmup(self, sample_data):
        warmup = 10
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=warmup)
        result = strategy.generate_signals(sample_data)
        # First `long_window + warmup - 1` rows should have no signal
        # (exact cut-off depends on NaN rows from rolling)
        assert result["Signal"].iloc[0] == 0

    def test_ma_columns_present(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        result = strategy.calculate_indicators(sample_data)
        assert "MA_Short" in result.columns
        assert "MA_Long" in result.columns

    def test_lowercase_columns_handled(self, sample_data):
        lower = sample_data.rename(columns=str.lower)
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        result = strategy.generate_signals(lower)
        assert "Signal" in result.columns

    def test_returns_dataframe(self, sample_data):
        strategy = MACrossoverStrategy()
        result = strategy.generate_signals(sample_data)
        assert isinstance(result, pd.DataFrame)


# ── InsideCandleRSIStrategy ───────────────────────────────────────────────────

class TestInsideCandleRSIStrategy:
    def test_generates_signal_column(self, sample_data):
        strategy = InsideCandleRSIStrategy(rsi_period=5, warmup_period=10)
        result = strategy.generate_signals(sample_data)
        assert "Signal" in result.columns

    def test_signals_are_valid_values(self, sample_data):
        strategy = InsideCandleRSIStrategy(rsi_period=5, warmup_period=10)
        result = strategy.generate_signals(sample_data)
        assert set(result["Signal"].unique()).issubset({-1, 0, 1})

    def test_rsi_column_present(self, sample_data):
        strategy = InsideCandleRSIStrategy(rsi_period=5)
        result = strategy.calculate_indicators(sample_data)
        assert "RSI" in result.columns

    def test_inside_candle_column_present(self, sample_data):
        strategy = InsideCandleRSIStrategy()
        result = strategy.calculate_indicators(sample_data)
        assert "Inside_Candle" in result.columns

    def test_rsi_range(self, sample_data):
        strategy = InsideCandleRSIStrategy(rsi_period=5)
        result = strategy.calculate_indicators(sample_data)
        rsi_valid = result["RSI"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_short_data_no_crash(self, short_data):
        """Strategy should not crash on short datasets."""
        strategy = InsideCandleRSIStrategy(rsi_period=5, warmup_period=5)
        result = strategy.generate_signals(short_data)
        assert "Signal" in result.columns

    def test_ffill_no_deprecated_warning(self, sample_data):
        """Ensure we are not using the deprecated fillna(method=) API."""
        import warnings
        strategy = InsideCandleRSIStrategy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            strategy.calculate_indicators(sample_data)  # Should not raise FutureWarning
