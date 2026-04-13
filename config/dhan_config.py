"""
Dhan API configuration
Get your client ID and access token from Dhan
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

# NOTE: credentials validated at runtime by DhanDataFetcher, not at import time

# Market segments
EXCHANGE_NSE = "NSE_EQ"
EXCHANGE_BSE = "BSE_EQ" 