import pandas as pd
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MACDCrossoverStrategy(BaseStrategy):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9,
                 warmup_period=35, stop_loss_pct=None, take_profit_pct=None,
                 trailing_stop_pct=None, max_bars=None):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.warmup_period = warmup_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_bars = max_bars
        super().__init__()

    def calculate_indicators(self, data):
        return self.generate_signals(data)

    def generate_signals(self, data):
        df = data.copy()
        close_col = 'Close' if 'Close' in df.columns else 'close'
        ema_fast = df[close_col].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['Signal'] = 0
        if self.warmup_period >= len(df):
            return df
        prev_hist = df['MACD_Hist'].iloc[self.warmup_period - 1]
        for i in range(self.warmup_period, len(df)):
            idx = df.index[i]
            hist = df.loc[idx, 'MACD_Hist']
            if prev_hist <= 0 and hist > 0:
                df.loc[idx, 'Signal'] = 1
            elif prev_hist >= 0 and hist < 0:
                df.loc[idx, 'Signal'] = -1
            prev_hist = hist
        return df
