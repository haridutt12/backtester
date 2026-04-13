"""
Yahoo Finance data fetcher — no API key required.

NSE symbols: append  .NS   (e.g. RELIANCE.NS, TCS.NS, INFY.NS)
BSE symbols: append  .BO
Global:      plain ticker  (e.g. AAPL, MSFT, TSLA)
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceDataFetcher:
    """Fetch OHLCV data from Yahoo Finance (no credentials required)."""

    def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame with columns: open, high, low, close, volume
        indexed by date, or None on failure.

        Automatically appends '.NS' for plain Indian symbols that have no
        exchange suffix, so users can pass 'RELIANCE' or 'RELIANCE.NS'.
        """
        ticker = self._resolve_ticker(symbol)
        logger.info("Fetching Yahoo Finance data for %s (%s → %s)", ticker,
                    start_date.date(), end_date.date())
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            return None

        if df is None or df.empty:
            # Try with .NS suffix as fallback
            if "." not in symbol:
                fallback = symbol + ".NS"
                logger.warning("%s returned no data; retrying as %s", ticker, fallback)
                return self.get_historical_data(fallback, start_date, end_date)
            logger.error("No data returned from Yahoo Finance for %s", ticker)
            return None

        # Flatten MultiIndex columns produced by yfinance ≥ 0.2
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index.name = "datetime"

        logger.info("Fetched %d rows for %s", len(df), ticker)
        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        ticker = self._resolve_ticker(symbol)
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price:
                return float(price)
        except Exception as exc:
            logger.warning("Could not get current price for %s: %s", ticker, exc)
        return None

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ticker(symbol: str) -> str:
        """Ensure the ticker has an exchange suffix when needed."""
        if "." in symbol or "^" in symbol:
            return symbol
        # Common NSE large-caps — add .NS automatically
        return symbol + ".NS"
