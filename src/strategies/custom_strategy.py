import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class CustomStrategy(BaseStrategy):
    def __init__(self, entry_indicator='RSI', entry_params=None,
                 filter_indicator=None, filter_params=None,
                 stop_loss_pct=None, take_profit_pct=None,
                 trailing_stop_pct=None, max_bars=None):
        self.entry_indicator = (entry_indicator or 'RSI').upper()
        self.entry_params = entry_params or {}
        self.filter_indicator = ((filter_indicator or '').upper()) or None
        self.filter_params = filter_params or {}
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_bars = max_bars
        super().__init__()

    @staticmethod
    def _calc_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, data):
        return self.generate_signals(data)

    def generate_signals(self, data):
        df = data.copy()
        close_col = 'Close' if 'Close' in df.columns else 'close'
        c = df[close_col]
        p = self.entry_params
        df['_raw'] = 0
        df['Signal'] = 0

        if self.entry_indicator == 'RSI':
            period = int(p.get('period', 14))
            oversold = float(p.get('oversold', 30))
            overbought = float(p.get('overbought', 70))
            warmup = int(p.get('warmup_period', 20))
            df['_rsi'] = self._calc_rsi(c, period)
            valid = df['_rsi'].dropna().index
            if len(valid) > warmup:
                prev = df.loc[valid[warmup - 1], '_rsi']
                for idx in valid[warmup:]:
                    cur = df.loc[idx, '_rsi']
                    if prev <= oversold and cur > oversold:
                        df.loc[idx, '_raw'] = 1
                    elif prev >= overbought and cur < overbought:
                        df.loc[idx, '_raw'] = -1
                    prev = cur

        elif self.entry_indicator == 'MA_CROSSOVER':
            short_w = int(p.get('short_window', 20))
            long_w = int(p.get('long_window', 50))
            warmup = int(p.get('warmup_period', 15))
            df['_ma_s'] = c.rolling(short_w).mean()
            df['_ma_l'] = c.rolling(long_w).mean()
            valid = df[['_ma_s', '_ma_l']].dropna().index
            if len(valid) > warmup:
                prev_s = df.loc[valid[warmup - 1], '_ma_s']
                prev_l = df.loc[valid[warmup - 1], '_ma_l']
                for idx in valid[warmup:]:
                    s, l = df.loc[idx, '_ma_s'], df.loc[idx, '_ma_l']
                    if prev_s <= prev_l and s > l:
                        df.loc[idx, '_raw'] = 1
                    elif prev_s >= prev_l and s < l:
                        df.loc[idx, '_raw'] = -1
                    prev_s, prev_l = s, l

        elif self.entry_indicator == 'EMA_CROSSOVER':
            fast = int(p.get('fast_period', 9))
            slow = int(p.get('slow_period', 21))
            warmup = int(p.get('warmup_period', 25))
            df['_ema_f'] = c.ewm(span=fast, adjust=False).mean()
            df['_ema_s'] = c.ewm(span=slow, adjust=False).mean()
            if warmup < len(df):
                prev_f = df['_ema_f'].iloc[warmup - 1]
                prev_s = df['_ema_s'].iloc[warmup - 1]
                for i in range(warmup, len(df)):
                    idx = df.index[i]
                    f, s = df.loc[idx, '_ema_f'], df.loc[idx, '_ema_s']
                    if prev_f <= prev_s and f > s:
                        df.loc[idx, '_raw'] = 1
                    elif prev_f >= prev_s and f < s:
                        df.loc[idx, '_raw'] = -1
                    prev_f, prev_s = f, s

        elif self.entry_indicator == 'MACD':
            fast = int(p.get('fast_period', 12))
            slow = int(p.get('slow_period', 26))
            sig = int(p.get('signal_period', 9))
            warmup = int(p.get('warmup_period', 35))
            df['_macd'] = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
            df['_macd_h'] = df['_macd'] - df['_macd'].ewm(span=sig, adjust=False).mean()
            if warmup < len(df):
                prev_h = df['_macd_h'].iloc[warmup - 1]
                for i in range(warmup, len(df)):
                    idx = df.index[i]
                    h = df.loc[idx, '_macd_h']
                    if prev_h <= 0 and h > 0:
                        df.loc[idx, '_raw'] = 1
                    elif prev_h >= 0 and h < 0:
                        df.loc[idx, '_raw'] = -1
                    prev_h = h

        elif self.entry_indicator == 'BOLLINGER':
            period = int(p.get('period', 20))
            std_dev = float(p.get('std_dev', 2.0))
            mode = str(p.get('mode', 'mean_reversion'))
            warmup = int(p.get('warmup_period', 25))
            df['_bb_m'] = c.rolling(period).mean()
            df['_bb_u'] = df['_bb_m'] + std_dev * c.rolling(period).std()
            df['_bb_l'] = df['_bb_m'] - std_dev * c.rolling(period).std()
            valid = df[['_bb_u', '_bb_l']].dropna().index
            if len(valid) > warmup:
                trading = valid[warmup:]
                for i in range(1, len(trading)):
                    idx = trading[i]
                    prev = trading[i - 1]
                    cl = df.loc[idx, close_col]
                    pc = df.loc[prev, close_col]
                    pu, pl = df.loc[prev, '_bb_u'], df.loc[prev, '_bb_l']
                    cu, cl_ = df.loc[idx, '_bb_u'], df.loc[idx, '_bb_l']
                    if mode == 'breakout':
                        if pc <= pu and cl > cu:
                            df.loc[idx, '_raw'] = 1
                        elif pc >= pl and cl < cl_:
                            df.loc[idx, '_raw'] = -1
                    else:
                        if pc >= pl and cl < cl_:
                            df.loc[idx, '_raw'] = 1
                        elif pc <= pu and cl > cu:
                            df.loc[idx, '_raw'] = -1

        fp = self.filter_params
        if self.filter_indicator == 'RSI_TREND':
            f_period = int(fp.get('period', 14))
            threshold = float(fp.get('threshold', 50))
            rsi_col = '_rsi' if '_rsi' in df.columns else '_f_rsi'
            if rsi_col not in df.columns:
                df[rsi_col] = self._calc_rsi(c, f_period)
            df.loc[(df['_raw'] == 1) & (df[rsi_col] > threshold), 'Signal'] = 1
            df.loc[(df['_raw'] == -1) & (df[rsi_col] < (100 - threshold)), 'Signal'] = -1
        elif self.filter_indicator == 'MA_TREND':
            f_period = int(fp.get('period', 200))
            df['_f_ma'] = c.rolling(f_period).mean()
            df.loc[(df['_raw'] == 1) & (c > df['_f_ma']), 'Signal'] = 1
            df.loc[(df['_raw'] == -1) & (c < df['_f_ma']), 'Signal'] = -1
        else:
            df['Signal'] = df['_raw']

        return df
