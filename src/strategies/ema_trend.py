import pandas as pd
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class EMATrendStrategy(BaseStrategy):
    def __init__(self, ema_fast=9, ema_slow=21, warmup_period=25,
                 stop_loss_pct=None, take_profit_pct=None,
                 trailing_stop_pct=None, max_bars=None):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
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
        df['EMA_Fast'] = df[close_col].ewm(span=self.ema_fast, adjust=False).mean()
        df['EMA_Slow'] = df[close_col].ewm(span=self.ema_slow, adjust=False).mean()
        df['Signal'] = 0
        if self.warmup_period >= len(df):
            return df
        prev_fast = df['EMA_Fast'].iloc[self.warmup_period - 1]
        prev_slow = df['EMA_Slow'].iloc[self.warmup_period - 1]
        for i in range(self.warmup_period, len(df)):
            idx = df.index[i]
            fast = df.loc[idx, 'EMA_Fast']
            slow = df.loc[idx, 'EMA_Slow']
            if prev_fast <= prev_slow and fast > slow:
                df.loc[idx, 'Signal'] = 1
            elif prev_fast >= prev_slow and fast < slow:
                df.loc[idx, 'Signal'] = -1
            prev_fast, prev_slow = fast, slow
        return df
