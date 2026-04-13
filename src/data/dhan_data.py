from dhanhq import dhanhq
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from src.config.dhan_config import CLIENT_ID, ACCESS_TOKEN, EXCHANGE_NSE
import logging
import inspect

logger = logging.getLogger(__name__)

class DhanDataFetcher:
    def __init__(self):
        """Initialize Dhan API client"""
        if not CLIENT_ID or not ACCESS_TOKEN:
            raise ValueError(
                "Dhan API credentials are required. "
                "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables "
                "(copy .env.example → .env and fill in your values)."
            )
        self.dhan = dhanhq(client_id=CLIENT_ID, access_token=ACCESS_TOKEN)
        self._security_map: Dict[str, Any] = {}
        self.security_list = None
        self.exchange = EXCHANGE_NSE  # Use the constant from config, not hardcoded "NSE"
        logger.debug(f"Using exchange format: {self.exchange}")
        
    def _get_security_id(self, symbol: str) -> Optional[str]:
        """Get the security ID for a given symbol."""
        try:
            if self.security_list is None:
                logger.debug(f"Fetching security list to find ID for symbol: {symbol}")
                self.security_list = self.dhan.fetch_security_list("compact")
                logger.debug(f"Received {len(self.security_list)} securities")
                
                # Log column names to help debug
                logger.debug(f"Security list columns: {list(self.security_list.columns)}")
                
                # Log unique exchange values
                if 'SEM_EXM_EXCH_ID' in self.security_list.columns:
                    logger.debug(f"Unique exchange IDs: {self.security_list['SEM_EXM_EXCH_ID'].unique()}")
                
                # Map NSE_EQ to NSE for filtering if needed
                exchange_filter = 'NSE' if self.exchange == 'NSE_EQ' else self.exchange
                logger.debug(f"Using exchange filter: {exchange_filter}")
                
                # Log a sample of equity symbols for the exchange
                segment_filter = self.security_list['SEM_EXM_EXCH_ID'] == exchange_filter
                
                if 'SEM_SEGMENT' in self.security_list.columns:
                    segment_filter &= self.security_list['SEM_SEGMENT'] == 'E'
                
                equities = self.security_list[segment_filter].head(5)
                logger.debug(f"Sample equity symbols for {exchange_filter}:\n{equities['SEM_TRADING_SYMBOL'] if len(equities) > 0 else 'No equities found'}")

            # Map NSE_EQ to NSE for filtering if needed
            exchange_filter = 'NSE' if self.exchange == 'NSE_EQ' else self.exchange
            
            # Filter for the specific symbol and exchange
            filter_condition = (self.security_list['SEM_TRADING_SYMBOL'] == symbol) & (self.security_list['SEM_EXM_EXCH_ID'] == exchange_filter)
            
            # Add segment filter if column exists
            if 'SEM_SEGMENT' in self.security_list.columns:
                filter_condition &= (self.security_list['SEM_SEGMENT'] == 'E')
            
            security = self.security_list[filter_condition]
            
            # Log what we found
            logger.debug(f"Found {len(security)} matching securities for {symbol} on {exchange_filter}")
            
            if len(security) > 0:
                # Log the first match details
                first_match = security.iloc[0]
                logger.debug(f"Matched security details: {first_match[['SEM_TRADING_SYMBOL', 'SEM_EXM_EXCH_ID', 'SEM_SMST_SECURITY_ID']].to_dict()}")
                
                security_id = str(first_match['SEM_SMST_SECURITY_ID'])
                logger.debug(f"Found security ID: {security_id}")
                return security_id
            else:
                # Try to find the symbol without exchange filter for debugging
                symbol_matches = self.security_list[self.security_list['SEM_TRADING_SYMBOL'] == symbol]
                logger.error(f"No security found for symbol {symbol} on exchange {self.exchange}")
                if len(symbol_matches) > 0:
                    logger.debug(f"Symbol {symbol} exists on other exchanges: {symbol_matches['SEM_EXM_EXCH_ID'].unique()}")
                return None

        except Exception as e:
            logger.error(f"Error fetching security ID: {str(e)}")
            logger.debug(f"Exception details:", exc_info=True)
            return None
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol between start_date and end_date."""
        try:
            security_id = self._get_security_id(symbol)
            if security_id is None:
                return None

            logger.debug(f"Found security ID {security_id} for symbol {symbol}")
            
            # Convert datetime objects to string format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Log API request parameters in detail
            logger.debug(f"Fetching historical data with parameters:")
            logger.debug(f"  security_id: {security_id}")
            logger.debug(f"  exchange_segment: {self.exchange}")
            logger.debug(f"  instrument_type: EQUITY")
            logger.debug(f"  expiry_code: 0")
            logger.debug(f"  from_date: {start_str}")
            logger.debug(f"  to_date: {end_str}")

            # Check if dates are in the future
            now = datetime.now()
            if start_date > now or end_date > now:
                logger.warning(f"Attempting to fetch data for future dates: {start_str} to {end_str}")

            response = self.dhan.historical_daily_data(
                security_id=security_id,
                exchange_segment=self.exchange,
                instrument_type='EQUITY',
                expiry_code=0,
                from_date=start_str,
                to_date=end_str
            )

            logger.debug(f"Raw response: {response}")
            
            # Add detailed logging for API response structure
            if isinstance(response, dict):
                logger.debug(f"Response keys: {response.keys()}")
                if 'data' in response and isinstance(response['data'], dict):
                    logger.debug(f"Data keys: {response['data'].keys()}")

            if response.get('status') == 'success':
                data = response.get('data', {})
                if not data:
                    logger.error("No data returned from API")
                    return None

                # Convert response to DataFrame
                df = pd.DataFrame({
                    'datetime': data.get('timestamp', []),
                    'open': data.get('open', []),
                    'high': data.get('high', []),
                    'low': data.get('low', []),
                    'close': data.get('close', []),
                    'volume': data.get('volume', [])
                })

                if len(df) == 0:
                    logger.error("No data points in response")
                    return None

                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)

                logger.debug(f"Processed data:\n{df.head()}")
                return df
            else:
                if isinstance(response, dict) and 'remarks' in response:
                    if isinstance(response['remarks'], dict):
                        logger.error(f"API returned detailed error: {response['remarks']}")
                    else:
                        logger.error(f"API returned error: {response['remarks']}")
                else:
                    logger.error("API returned unknown error format")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price of a security
        
        Args:
            symbol: Symbol of the security
            
        Returns:
            Current price of the security or None if not available
        """
        try:
            security_id = self._get_security_id(symbol)
            if security_id is None:
                logging.error(f"Could not find security ID for {symbol}")
                return None
                
            logging.info(f"Fetching current price for security ID: {security_id}")
            
            # Try to get live quote
            try:
                # Need to specify exchange in NSE_EQ format instead of NSE
                quote_params = {
                    "security_id": security_id,
                    "exchange": EXCHANGE_NSE
                }
                
                # Use quote_data method to get current price
                quote_response = self.dhan.quote_data(quote_params)
                logging.debug(f"Raw quote response: {quote_response}")
                
                # Check if response is valid
                if quote_response:
                    # Handle different response formats
                    if isinstance(quote_response, dict):
                        # Single quote response
                        if "last_price" in quote_response:
                            return float(quote_response["last_price"])
                        elif "ltp" in quote_response:
                            return float(quote_response["ltp"])
                    elif isinstance(quote_response, list) and len(quote_response) > 0:
                        # List of quotes response
                        quote = quote_response[0]
                        if "last_price" in quote:
                            return float(quote["last_price"])
                        elif "ltp" in quote:
                            return float(quote["ltp"])
                    
                    logging.warning(f"Unexpected quote response format: {quote_response}")
            except Exception as e:
                logging.error(f"Error fetching live quote: {e}")
            
            # Fallback to historical data if live quote fails
            logging.info(f"Attempting to fallback to recent historical data for {symbol}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Get last 5 days of data
            
            historical_data = self.get_historical_data(symbol, start_date, end_date)
            
            if historical_data is not None and not historical_data.empty:
                # Return the most recent closing price
                latest_price = historical_data.iloc[-1]["close"]
                logging.info(f"Using fallback price from historical data: {latest_price}")
                return float(latest_price)
                
            logging.error(f"Failed to get current price for {symbol} - both live and historical methods failed")
            return None
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_instrument_info(self, symbol: str, exchange: str = EXCHANGE_NSE) -> Optional[Dict]:
        """Get instrument information"""
        try:
            security_id = self._get_security_id(symbol)
            if not security_id:
                return None
                
            return self.dhan.get_security_info(
                security_id=security_id,
                exchange_segment=exchange
            )
        except Exception as e:
            logger.error("Error getting instrument info for %s: %s", symbol, str(e))
            return None 