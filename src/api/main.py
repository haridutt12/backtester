"""
Backtester REST API  v2.0
==========================
GET  /                       - web dashboard
GET  /health                 - liveness probe
GET  /strategies             - list all strategies with metadata
GET  /symbols/search?q=REL   - symbol autocomplete
POST /backtest               - run a preset strategy
POST /backtest/custom        - run a custom indicator combination
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO),
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Backtester API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_DEFAULT_DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance").lower()


@app.get("/", include_in_schema=False)
def root():
    idx = _STATIC_DIR / "index.html"
    return FileResponse(str(idx)) if idx.is_file() else {"message": "Backtester API v2 — /docs"}


# -- shared models -------------------------------------------------------------

class TradeResult(BaseModel):
    entry_date: str; exit_date: str
    entry_price: float; exit_price: float
    position: int; position_size: int
    pnl: float; trade_return: float; exit_reason: str


class BacktestResponse(BaseModel):
    symbol: str; strategy: str; data_source: str
    start_date: str; end_date: str; initial_capital: float
    final_portfolio_value: float; total_return_pct: float
    sharpe_ratio: float; max_drawdown_pct: float
    n_trades: int; win_rate_pct: float; buy_hold_return_pct: float
    trades: List[TradeResult]


# -- preset backtest request ---------------------------------------------------

class BacktestRequest(BaseModel):
    symbol: str = "RELIANCE"
    strategy: str = "MA_CROSSOVER"
    days: int = Field(365, ge=30, le=1825)
    initial_capital: float = Field(100_000, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    data_source: Literal["yfinance", "dhan"] = "yfinance"
    # MA overrides
    short_window: Optional[int] = None
    long_window: Optional[int] = None
    ma_warmup_period: Optional[int] = None
    # RSI / IC overrides
    rsi_period: Optional[int] = None
    rsi_overbought: Optional[int] = None
    rsi_oversold: Optional[int] = None
    ic_warmup_period: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_bars: Optional[int] = None
    # New strategy overrides
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    bb_period: Optional[int] = None
    bb_std: Optional[float] = None
    bb_mode: Optional[str] = None
    ema_fast: Optional[int] = None
    ema_slow: Optional[int] = None


# -- custom strategy request ---------------------------------------------------

class CustomBacktestRequest(BaseModel):
    symbol: str = "RELIANCE"
    days: int = Field(365, ge=30, le=1825)
    initial_capital: float = Field(100_000, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    data_source: Literal["yfinance", "dhan"] = "yfinance"
    entry_indicator: Literal["RSI", "MA_CROSSOVER", "EMA_CROSSOVER", "MACD", "BOLLINGER"] = "RSI"
    entry_params: Dict[str, Any] = Field(default_factory=dict)
    filter_indicator: Optional[Literal["RSI_TREND", "MA_TREND"]] = None
    filter_params: Dict[str, Any] = Field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    max_bars: Optional[int] = None


# -- helpers -------------------------------------------------------------------

def _fetch(symbol, days, data_source):
    if data_source == "dhan":
        from src.data.dhan_data import DhanDataFetcher
        try:
            fetcher = DhanDataFetcher()
        except ValueError as e:
            raise HTTPException(503, str(e))
    else:
        from src.data.yfinance_data import YFinanceDataFetcher
        fetcher = YFinanceDataFetcher()

    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = fetcher.get_historical_data(symbol, start, end)
    except TimeoutError as e:
        raise HTTPException(503, str(e))
    if df is None or len(df) == 0:
        raise HTTPException(404,
            f"No data for '{symbol}'. NSE: use RELIANCE, TCS, INFY (auto-resolved). "
            "US: AAPL, MSFT, TSLA.")
    return df, start, end


def _respond(symbol, strat_name, ds, start, end, capital, results, df):
    cc = "close" if "close" in df.columns else "Close"
    bh = ((float(df[cc].iloc[-1]) / float(df[cc].iloc[0])) - 1) * 100
    trades = results.get("trades", [])
    wr = (sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100) if trades else 0.0
    return BacktestResponse(
        symbol=symbol, strategy=strat_name, data_source=ds,
        start_date=str(start.date()), end_date=str(end.date()),
        initial_capital=capital,
        final_portfolio_value=round(float(results["final_portfolio_value"]), 2),
        total_return_pct=round(float(results["total_return"]), 4),
        sharpe_ratio=round(float(results["sharpe_ratio"]), 4),
        max_drawdown_pct=round(float(results["max_drawdown"]), 4),
        n_trades=results["n_trades"],
        win_rate_pct=round(wr, 2),
        buy_hold_return_pct=round(bh, 4),
        trades=[TradeResult(
            entry_date=str(t["entry_date"])[:10],
            exit_date=str(t["exit_date"])[:10],
            entry_price=round(float(t["entry_price"]), 4),
            exit_price=round(float(t["exit_price"]), 4),
            position=int(t["position"]),
            position_size=int(t["position_size"]),
            pnl=round(float(t["pnl"]), 4),
            trade_return=round(float(t["return"]) * 100, 4),
            exit_reason=t.get("exit_reason", ""),
        ) for t in trades],
    )


# -- routes --------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/strategies", tags=["strategies"])
def list_strategies():
    from src.config.config_manager import ConfigManager
    cm = ConfigManager()
    out = {}
    for name, cfg in cm.config.items():
        out[name] = {k: cfg.get(k, "") for k in
                     ("description", "risk_level", "best_for", "indicators")}
    return {"strategies": out}


@app.get("/symbols/search", tags=["symbols"])
def symbol_search(q: str = Query("", min_length=1)):
    from src.api.symbols_data import search_symbols
    return {"results": search_symbols(q)}


@app.post("/backtest", response_model=BacktestResponse, tags=["backtest"])
def run_backtest(req: BacktestRequest):
    from src.backtester.backtester import Backtester
    from src.config.config_manager import ConfigManager
    from src.strategies.bollinger_breakout import BollingerBreakoutStrategy
    from src.strategies.ema_trend import EMATrendStrategy
    from src.strategies.inside_candle_rsi import InsideCandleRSIStrategy
    from src.strategies.ma_crossover import MACrossoverStrategy
    from src.strategies.macd_crossover import MACDCrossoverStrategy
    from src.strategies.rsi_mean_reversion import RSIMeanReversionStrategy

    cm = ConfigManager()
    if req.strategy not in cm.config:
        raise HTTPException(400, f"Unknown strategy '{req.strategy}'. "
                            f"Available: {list(cm.config.keys())}")
    cfg = cm.get_strategy_config(req.strategy) or {}
    stype = cfg.get("strategy_type", req.strategy)

    ds = req.data_source or _DEFAULT_DATA_SOURCE
    df, start, end = _fetch(req.symbol, req.days, ds)

    def _o(req_val, cfg_key, default):
        return req_val if req_val is not None else cfg.get(cfg_key, default)

    if stype == "MA_CROSSOVER":
        s = MACrossoverStrategy(
            short_window=req.short_window or cfg.get("short_window", 20),
            long_window=req.long_window or cfg.get("long_window", 50),
            warmup_period=req.ma_warmup_period or cfg.get("warmup_period", 15))
    elif stype == "INSIDE_CANDLE_RSI":
        s = InsideCandleRSIStrategy(
            rsi_period=req.rsi_period or cfg.get("rsi_period", 14),
            rsi_overbought=req.rsi_overbought or cfg.get("rsi_overbought", 70),
            rsi_oversold=req.rsi_oversold or cfg.get("rsi_oversold", 30),
            warmup_period=req.ic_warmup_period or cfg.get("warmup_period", 20),
            stop_loss_pct=_o(req.stop_loss_pct, "stop_loss_pct", None),
            take_profit_pct=_o(req.take_profit_pct, "take_profit_pct", None),
            trailing_stop_pct=_o(req.trailing_stop_pct, "trailing_stop_pct", None),
            max_bars=_o(req.max_bars, "max_bars", None))
    elif stype == "RSI_MEAN_REVERSION":
        s = RSIMeanReversionStrategy(
            rsi_period=req.rsi_period or cfg.get("rsi_period", 14),
            rsi_overbought=req.rsi_overbought or cfg.get("rsi_overbought", 70),
            rsi_oversold=req.rsi_oversold or cfg.get("rsi_oversold", 30),
            warmup_period=req.ic_warmup_period or cfg.get("warmup_period", 20),
            stop_loss_pct=_o(req.stop_loss_pct, "stop_loss_pct", None),
            take_profit_pct=_o(req.take_profit_pct, "take_profit_pct", None))
    elif stype == "BOLLINGER_BREAKOUT":
        s = BollingerBreakoutStrategy(
            bb_period=req.bb_period or cfg.get("bb_period", 20),
            bb_std=req.bb_std or cfg.get("bb_std", 2.0),
            warmup_period=cfg.get("warmup_period", 25),
            mode=req.bb_mode or cfg.get("mode", "mean_reversion"),
            stop_loss_pct=_o(req.stop_loss_pct, "stop_loss_pct", None),
            take_profit_pct=_o(req.take_profit_pct, "take_profit_pct", None))
    elif stype == "MACD_CROSSOVER":
        s = MACDCrossoverStrategy(
            fast_period=req.fast_period or cfg.get("fast_period", 12),
            slow_period=req.slow_period or cfg.get("slow_period", 26),
            signal_period=req.signal_period or cfg.get("signal_period", 9),
            warmup_period=cfg.get("warmup_period", 35),
            stop_loss_pct=_o(req.stop_loss_pct, "stop_loss_pct", None),
            take_profit_pct=_o(req.take_profit_pct, "take_profit_pct", None))
    elif stype == "EMA_TREND":
        s = EMATrendStrategy(
            ema_fast=req.ema_fast or cfg.get("ema_fast", 9),
            ema_slow=req.ema_slow or cfg.get("ema_slow", 21),
            warmup_period=cfg.get("warmup_period", 25),
            stop_loss_pct=_o(req.stop_loss_pct, "stop_loss_pct", None),
            take_profit_pct=_o(req.take_profit_pct, "take_profit_pct", None),
            trailing_stop_pct=_o(req.trailing_stop_pct, "trailing_stop_pct", None))
    else:
        raise HTTPException(400, f"Unsupported strategy type: {stype}")

    results = Backtester(req.initial_capital, req.commission).run(s, df)
    return _respond(req.symbol, req.strategy, ds, start, end, req.initial_capital, results, df)


@app.post("/backtest/custom", response_model=BacktestResponse, tags=["backtest"])
def run_custom_backtest(req: CustomBacktestRequest):
    from src.backtester.backtester import Backtester
    from src.strategies.custom_strategy import CustomStrategy

    ds = req.data_source or _DEFAULT_DATA_SOURCE
    df, start, end = _fetch(req.symbol, req.days, ds)

    s = CustomStrategy(
        entry_indicator=req.entry_indicator,
        entry_params=req.entry_params,
        filter_indicator=req.filter_indicator,
        filter_params=req.filter_params,
        stop_loss_pct=req.stop_loss_pct,
        take_profit_pct=req.take_profit_pct,
        trailing_stop_pct=req.trailing_stop_pct,
        max_bars=req.max_bars,
    )
    label = req.entry_indicator + (f" + {req.filter_indicator}" if req.filter_indicator else "")
    results = Backtester(req.initial_capital, req.commission).run(s, df)
    return _respond(req.symbol, label, ds, start, end, req.initial_capital, results, df)
