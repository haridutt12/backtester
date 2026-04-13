import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables.
# Values are None when not set; DhanDataFetcher will raise a clear error
# at instantiation time so that the API server can still start and serve
# the /health and /strategies endpoints without credentials.
CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')

# Exchange constants
EXCHANGE_NSE = 'NSE_EQ'  # National Stock Exchange Equity
EXCHANGE_BSE = 'BSE_EQ'  # Bombay Stock Exchange Equity 