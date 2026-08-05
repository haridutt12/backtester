import pandas as pd
import numpy as np
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 warmup_period=20, stop_loss_pct=None, take_profit_pct=None,
                 trailing_stop_pct=None, max_bars=None):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.warmup_period = warmup_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_bars = max_bars
        super().__init__()

    def _calc_rsi(self, prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, data):
        return self.generate_signals(data)

    def generate_signals(self, data):
        df = data.copy()
        close_col = 'Close' if 'Close' in df.columns else 'close'
        df['RSI'] = self._calc_rsi(df[close_col])
        df['Signal'] = 0
        valid = df['RSI'].dropna().index
        if len(valid) <= self.warmup_period:
            return df
        trading = valid[self.warmup_period:]
        prev_rsi = df.loc[valid[self.warmup_period - 1], 'RSI']
        for idx in trading:
            rsi = df.loc[idx, 'RSI']
            if prev_rsi <= self.rsi_oversold and rsi > self.rsi_oversold:
                df.loc[idx, 'Signal'] = 1
            elif prev_rsi >= self.rsi_overbought and rsi < self.rsi_overbought:
                df.loc[idx, 'Signal'] = -1
            prev_rsi = rsi
        return df
