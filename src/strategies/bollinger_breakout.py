import pandas as pd
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class BollingerBreakoutStrategy(BaseStrategy):
    def __init__(self, bb_period=20, bb_std=2.0, warmup_period=25,
                 mode='mean_reversion', stop_loss_pct=None, take_profit_pct=None,
                 trailing_stop_pct=None, max_bars=None):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.warmup_period = warmup_period
        self.mode = mode
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
        df['BB_Mid'] = df[close_col].rolling(self.bb_period).mean()
        df['BB_Std'] = df[close_col].rolling(self.bb_period).std()
        df['BB_Upper'] = df['BB_Mid'] + self.bb_std * df['BB_Std']
        df['BB_Lower'] = df['BB_Mid'] - self.bb_std * df['BB_Std']
        df['Signal'] = 0
        valid = df[['BB_Upper', 'BB_Lower']].dropna().index
        if len(valid) <= self.warmup_period:
            return df
        trading = valid[self.warmup_period:]
        for i in range(1, len(trading)):
            idx = trading[i]
            prev = trading[i - 1]
            close = df.loc[idx, close_col]
            prev_close = df.loc[prev, close_col]
            upper = df.loc[idx, 'BB_Upper']
            lower = df.loc[idx, 'BB_Lower']
            prev_upper = df.loc[prev, 'BB_Upper']
            prev_lower = df.loc[prev, 'BB_Lower']
            if self.mode == 'breakout':
                if prev_close <= prev_upper and close > upper:
                    df.loc[idx, 'Signal'] = 1
                elif prev_close >= prev_lower and close < lower:
                    df.loc[idx, 'Signal'] = -1
            else:
                if prev_close >= prev_lower and close < lower:
                    df.loc[idx, 'Signal'] = 1
                elif prev_close <= prev_upper and close > upper:
                    df.loc[idx, 'Signal'] = -1
        return df
