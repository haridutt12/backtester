import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class InsideCandleRSIStrategy(BaseStrategy):
    def __init__(self, rsi_period: int = 14, rsi_overbought: int = 70, 
                 rsi_oversold: int = 30, warmup_period: int = 20,
                 stop_loss_pct: float = None, take_profit_pct: float = None,
                 trailing_stop_pct: float = None, max_bars: int = None):
        """
        Initialize strategy parameters
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            warmup_period: Number of bars to wait after indicators are available
            stop_loss_pct: Stop loss percentage (e.g., 2.0 for 2%) - exits if loss exceeds this percentage
            take_profit_pct: Take profit percentage (e.g., 5.0 for 5%) - exits if profit exceeds this percentage
            trailing_stop_pct: Trailing stop percentage - adjusts stop loss as price moves in favorable direction
            max_bars: Maximum number of bars to hold a position before forced exit
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.warmup_period = warmup_period
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_bars = max_bars
        super().__init__()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices: Series of prices
            period: RSI calculation period
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _identify_inside_candles(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify inside candles in the data
        
        An inside candle is formed when the current candle's high is lower than the previous candle's high
        AND the current candle's low is higher than the previous candle's low
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with boolean values indicating inside candles
        """
        # Get column names based on case
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Shift high and low values to compare with previous candle
        prev_high = df[high_col].shift(1)
        prev_low = df[low_col].shift(1)
        
        # Identify inside candles
        inside_candles = (df[high_col] < prev_high) & (df[low_col] > prev_low)
        
        return inside_candles
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators needed for the strategy
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Determine column names (handle both uppercase and lowercase)
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Calculate RSI
        df['RSI'] = self._calculate_rsi(df[close_col], self.rsi_period)
        
        # Find inside candles
        df['Inside_Candle'] = self._identify_inside_candles(df)
        
        # Calculate inside candle's high and low for breakout reference
        df['Inside_High'] = np.where(df['Inside_Candle'], df[high_col], np.nan)
        df['Inside_Low'] = np.where(df['Inside_Candle'], df[low_col], np.nan)
        
        # Forward fill the inside candle high/low until next inside candle
        df['Inside_High'] = df['Inside_High'].ffill()
        df['Inside_Low'] = df['Inside_Low'].ffill()
        
        # Start high/low from the first inside candle
        first_inside = df['Inside_Candle'].idxmax() if df['Inside_Candle'].any() else None
        if first_inside:
            df.loc[:first_inside, 'Inside_High'] = np.nan
            df.loc[:first_inside, 'Inside_Low'] = np.nan
        
        # RSI trend indication
        df['RSI_Trend'] = np.zeros(len(df))
        df.loc[df['RSI'] > self.rsi_overbought, 'RSI_Trend'] = 1  # Bullish
        df.loc[df['RSI'] < self.rsi_oversold, 'RSI_Trend'] = -1   # Bearish
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on inside candle breakouts with RSI confirmation
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Determine column names
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Initialize signal column
        df['Signal'] = 0
        
        # Apply warmup period to ensure indicator stability
        valid_idx = df.dropna(subset=['RSI', 'Inside_High', 'Inside_Low']).index
        
        if len(valid_idx) > 0 and self.warmup_period > 0:
            # Skip initial data points (warmup period)
            first_valid_idx = valid_idx[0]
            start_trading_idx = df.index.get_loc(first_valid_idx) + self.warmup_period
            if start_trading_idx < len(df):
                valid_idx = valid_idx[valid_idx >= df.index[start_trading_idx]]
            else:
                logger.warning("Not enough data after warmup period")
                return df
        
        # Process each valid bar after warmup
        for i in range(1, len(valid_idx)):
            idx = valid_idx[i]
            prev_idx = valid_idx[i-1]
            
            # Skip if this is an inside candle (wait for breakout)
            if df.loc[idx, 'Inside_Candle']:
                continue
            
            # Check for bullish breakout (price breaks above inside candle high)
            if (df.loc[idx, high_col] > df.loc[prev_idx, 'Inside_High']) and (df.loc[idx, 'RSI_Trend'] >= 0):
                df.loc[idx, 'Signal'] = 1  # Buy signal
            
            # Check for bearish breakout (price breaks below inside candle low)
            elif (df.loc[idx, low_col] < df.loc[prev_idx, 'Inside_Low']) and (df.loc[idx, 'RSI_Trend'] <= 0):
                df.loc[idx, 'Signal'] = -1  # Sell signal
        
        # Log signal statistics
        buy_signals = (df['Signal'] == 1).sum()
        sell_signals = (df['Signal'] == -1).sum()
        logger.debug(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return df 