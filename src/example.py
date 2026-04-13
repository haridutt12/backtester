import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import csv
import numpy as np
from adjustText import adjust_text
import mplfinance as mpf

from src.data.dhan_data import DhanDataFetcher
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.inside_candle_rsi import InsideCandleRSIStrategy
from src.backtester.backtester import Backtester
from src.config.config_manager import ConfigManager

# Configure logging – level controlled by LOG_LEVEL env var (default INFO)
_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
# Strategy selection - only this needs to be specified at runtime
STRATEGY_TYPE = os.environ.get('STRATEGY_TYPE', 'MA_CROSSOVER')  # MA_CROSSOVER or INSIDE_CANDLE_RSI
STRATEGY_NAME = os.environ.get('STRATEGY_NAME', STRATEGY_TYPE)   # The specific strategy name from config

# Other common parameters
BACKTEST_DAYS = int(os.environ.get('BACKTEST_DAYS', 365))
SYMBOL = os.environ.get('SYMBOL', 'RELIANCE')
INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', 100000))
COMMISSION = float(os.environ.get('COMMISSION', 0.001))

# The parameters below are kept for backward compatibility but will be overridden 
# by the configuration file if a strategy is specified
# MA Crossover Strategy parameters
SHORT_MA_PERIOD = int(os.environ.get('SHORT_MA_PERIOD', 20))
LONG_MA_PERIOD = int(os.environ.get('LONG_MA_PERIOD', 50))
MA_WARMUP_PERIOD = int(os.environ.get('MA_WARMUP_PERIOD', 15))

# Inside Candle RSI Strategy parameters
RSI_PERIOD = int(os.environ.get('RSI_PERIOD', 14))
RSI_OVERBOUGHT = int(os.environ.get('RSI_OVERBOUGHT', 70))
RSI_OVERSOLD = int(os.environ.get('RSI_OVERSOLD', 30))
IC_WARMUP_PERIOD = int(os.environ.get('IC_WARMUP_PERIOD', 20))

# Risk management parameters
STOP_LOSS_PCT = float(os.environ.get('STOP_LOSS_PCT', 0)) if os.environ.get('STOP_LOSS_PCT') else None
TAKE_PROFIT_PCT = float(os.environ.get('TAKE_PROFIT_PCT', 0)) if os.environ.get('TAKE_PROFIT_PCT') else None
TRAILING_STOP_PCT = float(os.environ.get('TRAILING_STOP_PCT', 0)) if os.environ.get('TRAILING_STOP_PCT') else None
MAX_BARS = int(os.environ.get('MAX_BARS', 0)) if os.environ.get('MAX_BARS') else None

def save_trade_journal(trades, strategy_params, output_file="trade_journal.csv"):
    """Save trade details to a CSV file"""
    if not trades:
        logger.warning("No trades to save to journal")
        return
    
    # Get first trade to extract all fields
    first_trade = trades[0]
    
    # Add strategy_name to parameters if not present
    if 'strategy_name' not in strategy_params:
        strategy_params['strategy_name'] = STRATEGY_NAME
    
    # Define CSV headers - add all fields from the first trade plus strategy parameters
    headers = list(first_trade.keys()) + list(strategy_params.keys())
    
    # Create CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        # Write each trade row
        for trade in trades:
            # Combine trade data with strategy parameters
            row = {**trade, **strategy_params}
            writer.writerow(row)
    
    logger.info(f"Trade journal saved to {output_file}")

def plot_portfolio_with_annotations(portfolio_df, trades, strategy_params, max_drawdown, data_with_indicators=None, output_file="backtest_results.png"):
    """Create an enhanced plot with trade annotations, strategy parameters, and indicators"""
    # Create figure with subplots - price/indicators on top, portfolio value in middle, trade table at bottom
    fig = plt.figure(figsize=(14, 16))
    
    # Define grid for the three main components
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.3)
    
    # Price and indicators subplot
    ax1 = fig.add_subplot(gs[0])
    
    # Portfolio value subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Trade table subplot
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')  # Hide axes for table
    
    # Get strategy type
    strategy_type = strategy_params.get('strategy_type', 'MA_CROSSOVER')
    
    # Plot price and indicators on top subplot
    if data_with_indicators is not None:
        # Get column names - handle both uppercase and lowercase
        close_col = 'Close' if 'Close' in data_with_indicators.columns else 'close'
        
        # Plot price
        ax1.plot(data_with_indicators.index, data_with_indicators[close_col], label='Price', color='black', alpha=0.7)
        
        # Calculate buy and hold returns
        first_price = data_with_indicators[close_col].iloc[0]
        last_price = data_with_indicators[close_col].iloc[-1]
        buy_hold_return = ((last_price / first_price) - 1) * 100
        
        # Plot strategy-specific indicators
        if strategy_type == 'MA_CROSSOVER':
            # Plot moving averages
            ax1.plot(data_with_indicators.index, data_with_indicators['MA_Short'], 
                    label=f'MA Short ({strategy_params["short_window"]})', color='green', linewidth=1.5)
            ax1.plot(data_with_indicators.index, data_with_indicators['MA_Long'], 
                    label=f'MA Long ({strategy_params["long_window"]})', color='red', linewidth=1.5)
            
            ax1.set_title(f'Price and Moving Averages - {strategy_params["symbol"]}', fontsize=12)
            
        elif strategy_type == 'INSIDE_CANDLE_RSI':
            # Create a secondary y-axis for RSI
            ax1_rsi = ax1.twinx()
            
            # Plot RSI
            ax1_rsi.plot(data_with_indicators.index, data_with_indicators['RSI'], 
                        label=f'RSI ({strategy_params["rsi_period"]})', color='purple', linewidth=1.5)
            
            # Add overbought/oversold lines
            ax1_rsi.axhline(y=strategy_params['rsi_overbought'], color='red', linestyle='--', alpha=0.5)
            ax1_rsi.axhline(y=strategy_params['rsi_oversold'], color='green', linestyle='--', alpha=0.5)
            
            # Highlight inside candles on the price chart
            inside_candles = data_with_indicators.loc[data_with_indicators['Inside_Candle'] == True].index
            for date in inside_candles:
                ax1.axvline(x=date, color='blue', linestyle=':', alpha=0.5)
            
            # Configure RSI axis
            ax1_rsi.set_ylabel('RSI', fontsize=10)
            ax1_rsi.set_ylim(0, 100)
            ax1_rsi.legend(loc='upper right')
            
            ax1.set_title(f'Price with Inside Candles and RSI - {strategy_params["symbol"]}', fontsize=12)
            
        # Add buy/sell markers at the price level for all strategies
        for trade in trades:
            if trade['position'] == 1:  # Buy signal
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
            else:  # Sell signal
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
        
        # Customize price subplot
        ax1.set_ylabel('Price (₹)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Set y-axis to show a bit of padding above/below the price range
        y_min = data_with_indicators[close_col].min() * 0.98
        y_max = data_with_indicators[close_col].max() * 1.02
        ax1.set_ylim(y_min, y_max)
    
    # Plot portfolio value on middle subplot
    ax2.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], label='Portfolio Value', linewidth=2, color='blue')
    
    # Add horizontal line for initial capital
    ax2.axhline(y=portfolio_df['Portfolio_Value'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Prepare for annotations - just simple markers, no text annotations on the chart
    buy_dates, buy_values = [], []
    sell_dates, sell_values = [], []
    
    # Calculate total PNL to match final result
    total_pnl = portfolio_df['Portfolio_Value'].iloc[-1] - portfolio_df['Portfolio_Value'].iloc[0]
    total_pnl_percentage = ((portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0]) - 1) * 100
    
    for trade in trades:
        # Entry points
        entry_date = trade['entry_date']
        entry_value = portfolio_df.loc[entry_date, 'Portfolio_Value']
        
        # Add to collections for scatter plot
        if trade['position'] == 1:
            buy_dates.append(entry_date)
            buy_values.append(entry_value)
        else:
            sell_dates.append(entry_date)
            sell_values.append(entry_value)
        
        # Exit points
        exit_date = trade['exit_date']
        exit_value = portfolio_df.loc[exit_date, 'Portfolio_Value']
        
        # Add to collections for scatter plot
        if trade['position'] == 1:
            sell_dates.append(exit_date)
            sell_values.append(exit_value)
        else:
            buy_dates.append(exit_date)
            buy_values.append(exit_value)
    
    # Add scatter plots for buy/sell points with numbers for reference
    if buy_dates:
        for i, (date, value) in enumerate(zip(buy_dates, buy_values)):
            ax2.scatter(date, value, color='green', s=80, marker='^')
            # Small number label next to marker
            ax2.text(date, value*1.01, str(i+1), fontsize=8, ha='center')
    
    if sell_dates:
        for i, (date, value) in enumerate(zip(sell_dates, sell_values)):
            ax2.scatter(date, value, color='red', s=80, marker='v')
            # Small number label next to marker
            ax2.text(date, value*0.99, str(i+1), fontsize=8, ha='center')
    
    # Customize portfolio subplot
    ax2.set_title('Portfolio Value Over Time', fontsize=12)
    ax2.set_ylabel('Portfolio Value (₹)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Create detailed trade table at the bottom
    # Prepare table data
    table_data = []
    headers = ['#', 'Type', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'PnL (₹)', 'Return (%)', 'Cumulative PnL (₹)']
    cumulative_pnl = 0
    
    for i, trade in enumerate(trades):
        trade_type = "Long" if trade['position'] == 1 else "Short"
        entry_date = trade['entry_date'].strftime('%Y-%m-%d')
        exit_date = trade['exit_date'].strftime('%Y-%m-%d')
        pnl = trade['pnl']
        cumulative_pnl += pnl
        trade_return = trade['return'] * 100  # Convert to percentage
        
        row = [
            i+1,
            trade_type,
            entry_date,
            f"₹{trade['entry_price']:.2f}",
            exit_date,
            f"₹{trade['exit_price']:.2f}",
            f"₹{pnl:.2f}",
            f"{trade_return:.2f}%",
            f"₹{cumulative_pnl:.2f}"
        ]
        table_data.append(row)
    
    # Create and style the table
    table = ax3.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(headers),
        bbox=[0.05, 0.2, 0.9, 0.7]  # [left, bottom, width, height]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Adjust row height
    
    # Style PnL cells based on positive/negative values
    for i, row in enumerate(table_data):
        pnl_value = float(row[6].replace('₹', ''))
        if pnl_value > 0:
            table[(i+1, 6)].set_facecolor('#d4f7d4')  # Light green for profit
        elif pnl_value < 0:
            table[(i+1, 6)].set_facecolor('#f7d4d4')  # Light red for loss
    
    # Add title with strategy details
    if strategy_type == 'MA_CROSSOVER':
        title = f'Backtest Results: MA Crossover Strategy (Short: {strategy_params["short_window"]}, Long: {strategy_params["long_window"]}, Warmup: {strategy_params["warmup_period"]})'
    elif strategy_type == 'INSIDE_CANDLE_RSI':
        title = f'Backtest Results: Inside Candle RSI Strategy (RSI: {strategy_params["rsi_period"]}, OB: {strategy_params["rsi_overbought"]}, OS: {strategy_params["rsi_oversold"]}, Warmup: {strategy_params["warmup_period"]})'
    else:
        title = f'Backtest Results: {strategy_params.get("strategy_type", "Unknown")} Strategy'
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Add statistics text box
    stats_text = (
        f"Initial Capital: ₹{portfolio_df['Portfolio_Value'].iloc[0]:,.2f}\n"
        f"Final Value: ₹{portfolio_df['Portfolio_Value'].iloc[-1]:,.2f}\n"
        f"Total Return: {total_pnl_percentage:.2f}%\n"
        f"Total PnL: ₹{total_pnl:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Sharpe Ratio: {strategy_params.get('sharpe_ratio', 'N/A')}\n"
        f"Buy & Hold Return: {buy_hold_return:.2f}%\n"
        f"Number of Trades: {len(trades)}\n"
        f"Win Rate: {(len([t for t in trades if t['pnl'] > 0]) / max(1, len(trades)) * 100):.1f}%\n"
    )
    
    # Add risk management info if available
    if strategy_params.get('stop_loss_pct') is not None:
        stats_text += f"Stop Loss: {strategy_params['stop_loss_pct']}%\n"
    if strategy_params.get('take_profit_pct') is not None:
        stats_text += f"Take Profit: {strategy_params['take_profit_pct']}%\n"
    if strategy_params.get('trailing_stop_pct') is not None:
        stats_text += f"Trailing Stop: {strategy_params['trailing_stop_pct']}%\n"
    if strategy_params.get('max_bars') is not None:
        stats_text += f"Max Holding Period: {strategy_params['max_bars']} bars\n"
    
    stats_text += f"Date Range: {portfolio_df.index[0].strftime('%Y-%m-%d')} to {portfolio_df.index[-1].strftime('%Y-%m-%d')}"
    
    # Place the text box in the upper right of the portfolio value plot
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Enhanced plot with indicators saved to {output_file}")
    
    # Close figure to free memory
    plt.close(fig)

def plot_inside_candle_visualization(data_with_indicators, trades, strategy_params, output_file="inside_candle_visualization.png"):
    """
    Create a specialized visualization for Inside Candle RSI strategy with candlestick chart
    and clear annotations showing inside candles and entry/exit conditions.
    """
    # Prepare the data
    df = data_with_indicators.copy()
    
    # Limit the data to the most recent 40 days for better visualization
    if len(df) > 40:
        df = df.iloc[-40:]
        # Keep only trades that occurred within this period
        filtered_trades = [trade for trade in trades if trade['entry_date'] in df.index or trade['exit_date'] in df.index]
    else:
        filtered_trades = trades
    
    # Ensure column names are uppercase for mplfinance
    if 'close' in df.columns:
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
    
    # Create figure with subplots (price chart and RSI indicator)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot candlestick chart in the main subplot
    mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False, show_nontrading=False)
    
    # Highlight inside candles with blue rectangle
    inside_candles = df.loc[df['Inside_Candle'] == True]
    for idx in inside_candles.index:
        # Get high and low values for the inside candle
        high = df.loc[idx, 'High']
        low = df.loc[idx, 'Low']
        
        # Calculate date positions for rectangle
        date_loc = df.index.get_loc(idx)
        date_prev = df.index[max(0, date_loc-1)]  # Previous date
        date_next = df.index[min(len(df.index)-1, date_loc+1)]  # Next date
        
        # Calculate width in days
        width_days = (date_next - date_prev).total_seconds() / (24 * 3600) * 0.8
        
        # Create rectangle to highlight inside candle
        rect = plt.Rectangle(
            (mdates.date2num(idx) - width_days/2, low),
            width_days, high - low,
            facecolor='skyblue', alpha=0.5, edgecolor='blue', linewidth=1,
            zorder=0
        )
        ax1.add_patch(rect)
        
        # Add label for inside candle
        ax1.text(idx, high * 1.01, "Inside\nCandle", fontsize=8, 
                 color='blue', ha='center', va='bottom', weight='bold')
    
    # Store annotations to adjust them later
    exit_annotations = []
    
    # Mark entry and exit points for trades
    for trade in filtered_trades:
        # Parse dates
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        
        # Skip if dates are outside our filtered range
        if entry_date not in df.index and exit_date not in df.index:
            continue
        
        # Use only dates within our visible range
        if entry_date in df.index:
            # Add entry marker (triangle)
            if trade['position'] == 1:  # Long entry
                ax1.scatter(entry_date, df.loc[entry_date, 'Low'] * 0.99, 
                            marker='^', color='green', s=120, zorder=5)
                ax1.text(entry_date, df.loc[entry_date, 'Low'] * 0.97, 
                         "BUY", color='green', fontsize=10, ha='center', weight='bold')
            else:  # Short entry
                ax1.scatter(entry_date, df.loc[entry_date, 'High'] * 1.01, 
                            marker='v', color='red', s=120, zorder=5)
                ax1.text(entry_date, df.loc[entry_date, 'High'] * 1.03, 
                         "SELL", color='red', fontsize=10, ha='center', weight='bold')
        
        # Add exit marker if exit date is in our range
        if exit_date in df.index:
            ax1.scatter(exit_date, df.loc[exit_date, 'Close'], 
                        marker='x', color='purple', s=100, zorder=5)
            
            # Add exit reason annotation
            if 'exit_reason' in trade:
                exit_text = f"EXIT: {trade['exit_reason']}"
                if trade['position'] == 1:  # Long exit
                    y_pos = df.loc[exit_date, 'Low'] * 0.97
                else:  # Short exit
                    y_pos = df.loc[exit_date, 'High'] * 1.03
                    
                # Store annotation for later adjustment
                annotation = ax1.text(exit_date, y_pos, exit_text, 
                         color='purple', fontsize=10, ha='center', rotation=45,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                exit_annotations.append(annotation)
        
        # Draw connecting line between entry and exit if both are in range
        if entry_date in df.index and exit_date in df.index:
            ax1.plot([entry_date, exit_date], 
                    [df.loc[entry_date, 'Close'], df.loc[exit_date, 'Close']], 
                    color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Try to prevent annotation overlaps
    if len(exit_annotations) > 0:
        from adjustText import adjust_text
        adjust_text(exit_annotations, arrowprops=dict(arrowstyle='->', color='purple'))
    
    # Plot RSI in the bottom subplot
    ax2.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
    ax2.axhline(y=strategy_params['rsi_overbought'], color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=strategy_params['rsi_oversold'], color='green', linestyle='--', alpha=0.5)
    ax2.text(df.index[-1], strategy_params['rsi_overbought'], f"Overbought ({strategy_params['rsi_overbought']})", 
             va='bottom', ha='right', color='red', fontsize=9)
    ax2.text(df.index[-1], strategy_params['rsi_oversold'], f"Oversold ({strategy_params['rsi_oversold']})", 
             va='top', ha='right', color='green', fontsize=9)
    
    # Add buy/sell signals to RSI chart
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1:  # Buy signal
            ax2.scatter(df.index[i], df['RSI'].iloc[i], marker='^', color='green', s=50)
        elif df['Signal'].iloc[i] == -1:  # Sell signal
            ax2.scatter(df.index[i], df['RSI'].iloc[i], marker='v', color='red', s=50)
    
    # Set RSI subplot properties
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set main chart properties
    ax1.set_title(f"Inside Candle RSI Strategy Visualization - {strategy_params['symbol']} (Last {len(df)} Days)", fontsize=14)
    ax1.set_ylabel('Price (₹)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Add strategy parameters as text box
    params_text = (
        f"RSI Period: {strategy_params['rsi_period']}\n"
        f"RSI Overbought: {strategy_params['rsi_overbought']}\n"
        f"RSI Oversold: {strategy_params['rsi_oversold']}\n"
        f"Warmup Period: {strategy_params['warmup_period']}\n"
    )
    
    # Add risk management info if available
    if strategy_params.get('stop_loss_pct') is not None:
        params_text += f"Stop Loss: {strategy_params['stop_loss_pct']}%\n"
    if strategy_params.get('take_profit_pct') is not None:
        params_text += f"Take Profit: {strategy_params['take_profit_pct']}%\n"
    if strategy_params.get('trailing_stop_pct') is not None:
        params_text += f"Trailing Stop: {strategy_params['trailing_stop_pct']}%\n"
    
    # Add text box with parameters
    ax1.text(0.02, 0.98, params_text, transform=ax1.transAxes, fontsize=10,
             va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend with explanation of indicators
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.5, edgecolor='blue', label='Inside Candle'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Buy Entry'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='Sell Entry'),
        plt.Line2D([0], [0], marker='x', color='purple', markersize=10, label='Exit')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Inside Candle visualization saved to {output_file}")
    plt.close(fig)

def main():
    # Load configuration
    config_manager = ConfigManager()
    available_strategies = config_manager.get_available_strategies()
    logger.info(f"Available strategies: {list(available_strategies.keys())}")
    
    # Get strategy configuration - use STRATEGY_NAME instead of STRATEGY_TYPE to get the specific config
    strategy_config = config_manager.get_strategy_config(STRATEGY_NAME)
    if strategy_config is None:
        logger.error(f"Strategy {STRATEGY_NAME} not found in configuration. Using environment variables.")
        # Strategy will be initialized with environment variables
    else:
        logger.info(f"Loaded configuration for strategy: {STRATEGY_NAME}")
        
        # If this is a custom strategy variant, get the base strategy type
        global STRATEGY_TYPE  # Make sure we're updating the global variable
        base_strategy_type = strategy_config.get('strategy_type', STRATEGY_TYPE)
        if base_strategy_type != STRATEGY_TYPE:
            logger.info(f"Overriding base strategy implementation from {STRATEGY_TYPE} to {base_strategy_type}")
            STRATEGY_TYPE = base_strategy_type
        logger.info(f"Using base strategy implementation: {STRATEGY_TYPE}")
    
    # Initialize data fetcher
    data_fetcher = DhanDataFetcher()

    # Set date range for historical data using configurable duration
    end_date = datetime.now()  # End date is current date
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    
    logger.info(f"Fetching data for date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Using {BACKTEST_DAYS} days of historical data")

    # Fetch historical data for configured symbol
    symbol = SYMBOL
    logger.info(f"Fetching data for symbol: {symbol}")
    
    historical_data = data_fetcher.get_historical_data(symbol, start_date, end_date)
    
    if historical_data is None:
        logger.error("Failed to fetch historical data")
        return
    
    if len(historical_data) == 0:
        logger.error("No historical data found for the specified date range")
        return
        
    logger.info(f"Successfully fetched {len(historical_data)} data points")
    logger.info(f"Data sample:\n{historical_data.head()}")

    # Get current price
    current_price = data_fetcher.get_current_price(symbol)
    logger.info(f"Current price for {symbol}: {current_price}")

    # Initialize selected strategy based on STRATEGY_TYPE
    strategy_params = {
        'symbol': symbol,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'initial_capital': INITIAL_CAPITAL,
        'commission': COMMISSION
    }
    
    if STRATEGY_TYPE == 'MA_CROSSOVER':
        # Use configuration from config file if available, otherwise use environment variables
        if strategy_config:
            short_window = strategy_config.get('short_window', SHORT_MA_PERIOD)
            long_window = strategy_config.get('long_window', LONG_MA_PERIOD)
            warmup_period = strategy_config.get('warmup_period', MA_WARMUP_PERIOD)
        else:
            short_window = SHORT_MA_PERIOD
            long_window = LONG_MA_PERIOD
            warmup_period = MA_WARMUP_PERIOD
            
        logger.info(f"Using Moving Average Crossover strategy with parameters: short_window={short_window}, long_window={long_window}, warmup_period={warmup_period}")
        strategy = MACrossoverStrategy(
            short_window=short_window, 
            long_window=long_window, 
            warmup_period=warmup_period
        )
        # Add strategy-specific parameters for logging
        strategy_params.update({
            'strategy_type': 'MA_CROSSOVER',
            'short_window': short_window,
            'long_window': long_window,
            'warmup_period': warmup_period
        })
    elif STRATEGY_TYPE == 'INSIDE_CANDLE_RSI':
        # Use configuration from config file if available, otherwise use environment variables
        if strategy_config:
            rsi_period = strategy_config.get('rsi_period', RSI_PERIOD)
            rsi_overbought = strategy_config.get('rsi_overbought', RSI_OVERBOUGHT)
            rsi_oversold = strategy_config.get('rsi_oversold', RSI_OVERSOLD)
            warmup_period = strategy_config.get('warmup_period', IC_WARMUP_PERIOD)
            stop_loss_pct = strategy_config.get('stop_loss_pct', STOP_LOSS_PCT)
            take_profit_pct = strategy_config.get('take_profit_pct', TAKE_PROFIT_PCT)
            trailing_stop_pct = strategy_config.get('trailing_stop_pct', TRAILING_STOP_PCT)
            max_bars = strategy_config.get('max_bars', MAX_BARS)
        else:
            rsi_period = RSI_PERIOD
            rsi_overbought = RSI_OVERBOUGHT
            rsi_oversold = RSI_OVERSOLD
            warmup_period = IC_WARMUP_PERIOD
            stop_loss_pct = STOP_LOSS_PCT
            take_profit_pct = TAKE_PROFIT_PCT
            trailing_stop_pct = TRAILING_STOP_PCT
            max_bars = MAX_BARS
            
        logger.info(f"Using Inside Candle Breakout with RSI confirmation strategy with parameters: "
                   f"rsi_period={rsi_period}, rsi_overbought={rsi_overbought}, rsi_oversold={rsi_oversold}, "
                   f"warmup_period={warmup_period}, stop_loss={stop_loss_pct}%, take_profit={take_profit_pct}%, "
                   f"trailing_stop={trailing_stop_pct}%, max_bars={max_bars}")
        strategy = InsideCandleRSIStrategy(
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            warmup_period=warmup_period,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_bars=max_bars
        )
        # Add strategy-specific parameters for logging
        strategy_params.update({
            'strategy_type': 'INSIDE_CANDLE_RSI',
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'warmup_period': warmup_period,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'max_bars': max_bars
        })
    else:
        logger.error(f"Invalid strategy type: {STRATEGY_TYPE}. Using MA_CROSSOVER as default.")
        strategy = MACrossoverStrategy(
            short_window=SHORT_MA_PERIOD, 
            long_window=LONG_MA_PERIOD, 
            warmup_period=MA_WARMUP_PERIOD
        )
        # Add default strategy parameters
        strategy_params.update({
            'strategy_type': 'MA_CROSSOVER',
            'short_window': SHORT_MA_PERIOD,
            'long_window': LONG_MA_PERIOD,
            'warmup_period': MA_WARMUP_PERIOD
        })

    # Add strategy name to parameters
    strategy_params['strategy_name'] = STRATEGY_NAME
    
    # Log strategy information
    logger.info(f"Running backtest with strategy type: {STRATEGY_TYPE}")
    logger.info(f"Strategy configuration: {STRATEGY_NAME}")
    logger.info(f"Backtest period: {BACKTEST_DAYS} days")

    # Initialize and run backtester
    backtester = Backtester(initial_capital=strategy_params['initial_capital'], 
                            commission=strategy_params['commission'])
    results = backtester.run(strategy, historical_data)

    # Print results
    print("\nBacktest Results:")
    print(f"Final Portfolio Value: ₹{results['final_portfolio_value']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {results['n_trades']}")
    
    # Add Sharpe ratio to strategy_params for plotting
    strategy_params['sharpe_ratio'] = results['sharpe_ratio']
    
    # Save trade journal
    save_trade_journal(results['trades'], strategy_params, 'trade_journal.csv')

    # Get data with indicators for plotting
    data_with_indicators = strategy.generate_signals(historical_data.copy())
    
    # Create enhanced portfolio plot with annotations and indicators
    plot_portfolio_with_annotations(
        results['portfolio_df'], 
        results['trades'], 
        strategy_params, 
        results['max_drawdown'],
        data_with_indicators,
        'backtest_results.png'
    )

    # After creating the regular plot, add the specialized visualization for Inside Candle RSI
    if STRATEGY_TYPE == 'INSIDE_CANDLE_RSI':
        plot_inside_candle_visualization(
            data_with_indicators,
            results['trades'],
            strategy_params,
            'inside_candle_visualization.png'
        )
        
    # Update configuration with the latest parameters if they were changed via environment variables
    if strategy_config and (
        (STRATEGY_TYPE == 'MA_CROSSOVER' and (
            SHORT_MA_PERIOD != strategy_config.get('short_window') or 
            LONG_MA_PERIOD != strategy_config.get('long_window') or 
            MA_WARMUP_PERIOD != strategy_config.get('warmup_period')
        )) or 
        (STRATEGY_TYPE == 'INSIDE_CANDLE_RSI' and (
            RSI_PERIOD != strategy_config.get('rsi_period') or 
            RSI_OVERBOUGHT != strategy_config.get('rsi_overbought') or 
            RSI_OVERSOLD != strategy_config.get('rsi_oversold') or 
            IC_WARMUP_PERIOD != strategy_config.get('warmup_period') or
            STOP_LOSS_PCT != strategy_config.get('stop_loss_pct') or
            TAKE_PROFIT_PCT != strategy_config.get('take_profit_pct') or
            TRAILING_STOP_PCT != strategy_config.get('trailing_stop_pct') or
            MAX_BARS != strategy_config.get('max_bars')
        ))
    ):
        # Environment variables override the config, so update the config
        logger.info(f"Updating configuration for strategy {STRATEGY_NAME} with current parameters")
        strategy_config_update = strategy_params.copy()
        # Add description if it exists in the original config
        if strategy_config and 'description' in strategy_config:
            strategy_config_update['description'] = strategy_config['description']
        elif STRATEGY_TYPE == 'MA_CROSSOVER':
            strategy_config_update['description'] = "Moving Average Crossover Strategy"
        elif STRATEGY_TYPE == 'INSIDE_CANDLE_RSI':
            strategy_config_update['description'] = "Inside Candle Breakout with RSI Confirmation Strategy"
            
        # If this is a variation of a base strategy, make sure to preserve the strategy_type
        if 'strategy_type' in strategy_config:
            strategy_config_update['strategy_type'] = strategy_config['strategy_type']
            
        # Remove non-strategy specific parameters
        for param in ['symbol', 'start_date', 'end_date', 'initial_capital', 'commission', 'sharpe_ratio', 'strategy_name']:
            if param in strategy_config_update:
                strategy_config_update.pop(param)
        
        # Save the updated config
        config_manager.save_strategy_config(STRATEGY_NAME, strategy_config_update)

if __name__ == "__main__":
    main()