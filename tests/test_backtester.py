"""
Unit tests for the core Backtester engine.
"""

import pytest
import pandas as pd

from src.backtester.backtester import Backtester
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.inside_candle_rsi import InsideCandleRSIStrategy


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(strategy, data, capital=100_000, commission=0.001):
    bt = Backtester(initial_capital=capital, commission=commission)
    return bt.run(strategy, data)


# ── Backtester core ───────────────────────────────────────────────────────────

class TestBacktesterCore:
    def test_returns_required_keys(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, sample_data)
        required = {
            "final_portfolio_value",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "n_trades",
            "trades",
            "portfolio_values",
            "portfolio_df",
        }
        assert required.issubset(results.keys())

    def test_initial_capital_preserved_with_no_trades(self, sample_data):
        """If no signals are generated, portfolio value should stay near initial capital."""
        # Use extreme windows that will never cross during the short test period
        strategy = MACrossoverStrategy(short_window=100, long_window=110, warmup_period=10)
        results = _run(strategy, sample_data, capital=50_000)
        # With no trades the portfolio is unchanged (still 50 000)
        assert results["n_trades"] == 0
        assert abs(results["final_portfolio_value"] - 50_000) < 1

    def test_portfolio_value_positive(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, sample_data)
        assert results["final_portfolio_value"] > 0

    def test_trade_count_matches_list(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, sample_data)
        assert results["n_trades"] == len(results["trades"])

    def test_max_drawdown_non_positive(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, sample_data)
        assert results["max_drawdown"] <= 0

    def test_portfolio_values_length(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, sample_data)
        # portfolio_values should have same length as input data (minus 1 since we start at row 1)
        assert len(results["portfolio_values"]) == len(sample_data)

    def test_accepts_lowercase_columns(self, sample_data):
        lower = sample_data.rename(columns=str.lower)
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results = _run(strategy, lower)
        assert "final_portfolio_value" in results

    def test_commission_reduces_returns(self, sample_data):
        strategy = MACrossoverStrategy(short_window=5, long_window=20, warmup_period=5)
        results_no_commission = _run(strategy, sample_data, commission=0.0)
        results_with_commission = _run(strategy, sample_data, commission=0.01)
        # Higher commission should always produce ≤ portfolio value
        assert (
            results_with_commission["final_portfolio_value"]
            <= results_no_commission["final_portfolio_value"]
        )


# ── Risk management ───────────────────────────────────────────────────────────

class TestRiskManagement:
    def test_stop_loss_limits_downside(self, sample_data):
        """A strategy with a tight stop-loss should produce fewer/smaller losing trades."""
        strategy_sl = InsideCandleRSIStrategy(
            rsi_period=5, warmup_period=10, stop_loss_pct=1.0
        )
        strategy_no_sl = InsideCandleRSIStrategy(
            rsi_period=5, warmup_period=10, stop_loss_pct=None
        )
        results_sl = _run(strategy_sl, sample_data)
        results_no_sl = _run(strategy_no_sl, sample_data)
        # With stop-loss, max drawdown should be ≥ (i.e. less severe) than without
        assert results_sl["max_drawdown"] >= results_no_sl["max_drawdown"]

    def test_max_bars_exits_trade(self, sample_data):
        strategy = InsideCandleRSIStrategy(
            rsi_period=5, warmup_period=10, max_bars=3
        )
        results = _run(strategy, sample_data)
        for trade in results["trades"]:
            if trade["exit_reason"] == "Max Bars":
                entry = pd.Timestamp(trade["entry_date"])
                exit_ = pd.Timestamp(trade["exit_date"])
                # The exit should be within a few bars of entry
                delta = (exit_ - entry).days
                assert delta <= 10  # generous calendar-day buffer
