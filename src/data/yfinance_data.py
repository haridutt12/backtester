"""
Yahoo Finance data fetcher — no API key required.

Key fixes vs original:
- 22-second hard timeout using a thread (prevents Cloudflare 'stream idle timeout')
- In-memory 1-hour TTL cache — repeated runs on same symbol are instant
- Auto .NS suffix for Indian tickers with .BO fallback
"""

import concurrent.futures
import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE: dict = {}
_CACHE_TTL = 3600   # 1 hour
_DL_TIMEOUT = 22    # seconds — safely under Cloudflare's 30 s idle limit


class YFinanceDataFetcher:

    def get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        ticker = self._resolve_ticker(symbol)
        key = f"{ticker}|{start_date.date()}|{end_date.date()}"

        if key in _CACHE:
            ts, df = _CACHE[key]
            if time.time() - ts < _CACHE_TTL:
                logger.info("Cache hit for %s", ticker)
                return df.copy()

        logger.info("Fetching Yahoo Finance: %s (%s to %s)",
                    ticker, start_date.date(), end_date.date())

        try:
            df = self._timed_download(ticker,
                                      start_date.strftime("%Y-%m-%d"),
                                      end_date.strftime("%Y-%m-%d"))
        except TimeoutError:
            raise
        except Exception as exc:
            logger.error("yfinance error for %s: %s", ticker, exc)
            return None

        if df is None or df.empty:
            if "." not in symbol:
                fallback = symbol + ".NS"
                logger.warning("%s empty; retrying as %s", ticker, fallback)
                return self.get_historical_data(fallback, start_date, end_date)
            logger.error("No data for %s", ticker)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index.name = "datetime"

        _CACHE[key] = (time.time(), df)
        logger.info("Fetched %d rows for %s", len(df), ticker)
        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        ticker = self._resolve_ticker(symbol)
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            return float(price) if price else None
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", ticker, exc)
            return None

    @staticmethod
    def _resolve_ticker(symbol: str) -> str:
        if "." in symbol or "^" in symbol:
            return symbol
        return symbol + ".NS"

    @staticmethod
    def _timed_download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        def _dl():
            return yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_dl)
            try:
                return future.result(timeout=_DL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error("yfinance timed out after %ds for %s", _DL_TIMEOUT, ticker)
                raise TimeoutError(
                    f"Data fetch for '{ticker}' timed out after {_DL_TIMEOUT}s. "
                    "Yahoo Finance is slow right now — please retry in a moment."
                )
