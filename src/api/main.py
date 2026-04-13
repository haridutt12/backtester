"""
Backtester REST API
===================
Exposes the backtesting engine as a JSON REST API using FastAPI.

Endpoints
---------
GET  /health               – liveness probe
GET  /strategies           – list available strategies
POST /backtest             – run a backtest and return metrics + trades
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ── logging ──────────────────────────────────────────────────────────────────
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Backtester API",
    description="Algorithmic trading backtesting framework REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol: str = Field("RELIANCE", description="NSE equity ticker symbol")
    strategy: str = Field("MA_CROSSOVER", description="Strategy key from config")
    days: int = Field(365, ge=30, le=1825, description="Historical days to backtest")
    initial_capital: float = Field(100_000, gt=0, description="Starting capital in INR")
    commission: float = Field(0.001, ge=0, le=0.05, description="Commission per trade (decimal)")

    # MA Crossover overrides
    short_window: Optional[int] = Field(None, ge=2, description="Short MA window")
    long_window: Optional[int] = Field(None, ge=3, description="Long MA window")
    ma_warmup_period: Optional[int] = Field(None, ge=1, description="MA warmup bars")

    # Inside Candle RSI overrides
    rsi_period: Optional[int] = Field(None, ge=2, description="RSI period")
    rsi_overbought: Optional[int] = Field(None, ge=50, le=100, description="RSI overbought level")
    rsi_oversold: Optional[int] = Field(None, ge=0, le=50, description="RSI oversold level")
    ic_warmup_period: Optional[int] = Field(None, ge=1, description="Inside candle warmup bars")
    stop_loss_pct: Optional[float] = Field(None, ge=0, description="Stop-loss %")
    take_profit_pct: Optional[float] = Field(None, ge=0, description="Take-profit %")
    trailing_stop_pct: Optional[float] = Field(None, ge=0, description="Trailing stop %")
    max_bars: Optional[int] = Field(None, ge=1, description="Max bars in trade before forced exit")


class TradeResult(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position: int
    position_size: int
    pnl: float
    trade_return: float
    exit_reason: str


class BacktestResponse(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_trades: int
    win_rate_pct: float
    buy_hold_return_pct: float
    trades: List[TradeResult]


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_env(key: str, value: Any) -> None:
    """Set an environment variable only when the value is not None."""
    if value is not None:
        os.environ[key] = str(value)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies", tags=["strategies"])
def list_strategies() -> Dict[str, Any]:
    """Return all strategies defined in config.json."""
    from src.config.config_manager import ConfigManager

    cm = ConfigManager()
    return {"strategies": cm.get_available_strategies()}


@app.post("/backtest", response_model=BacktestResponse, tags=["backtest"])
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest and return performance metrics plus individual trades.

    The Dhan API credentials must be available via environment variables
    ``DHAN_CLIENT_ID`` and ``DHAN_ACCESS_TOKEN`` (or a mounted ``.env`` file).
    """
    # Lazy imports so the API starts quickly even when optional deps are slow
    from src.backtester.backtester import Backtester
    from src.config.config_manager import ConfigManager
    from src.data.dhan_data import DhanDataFetcher
    from src.strategies.inside_candle_rsi import InsideCandleRSIStrategy
    from src.strategies.ma_crossover import MACrossoverStrategy

    # ── validate strategy ─────────────────────────────────────────────────
    cm = ConfigManager()
    available = cm.get_available_strategies()
    if req.strategy not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: {list(available.keys())}",
        )

    strategy_config = cm.get_strategy_config(req.strategy) or {}
    base_strategy_type = strategy_config.get("strategy_type", req.strategy)

    # ── fetch market data ─────────────────────────────────────────────────
    try:
        fetcher = DhanDataFetcher()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=req.days)

    logger.info(
        "Fetching %s for %s (%s → %s)",
        req.symbol,
        req.strategy,
        start_date.date(),
        end_date.date(),
    )

    historical_data = fetcher.get_historical_data(req.symbol, start_date, end_date)
    if historical_data is None or len(historical_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for symbol '{req.symbol}'",
        )

    # ── build strategy ────────────────────────────────────────────────────
    if base_strategy_type == "MA_CROSSOVER":
        short_window = req.short_window or strategy_config.get("short_window", 20)
        long_window = req.long_window or strategy_config.get("long_window", 50)
        warmup = req.ma_warmup_period or strategy_config.get("warmup_period", 15)
        strategy = MACrossoverStrategy(
            short_window=short_window,
            long_window=long_window,
            warmup_period=warmup,
        )

    elif base_strategy_type == "INSIDE_CANDLE_RSI":
        strategy = InsideCandleRSIStrategy(
            rsi_period=req.rsi_period or strategy_config.get("rsi_period", 14),
            rsi_overbought=req.rsi_overbought or strategy_config.get("rsi_overbought", 70),
            rsi_oversold=req.rsi_oversold or strategy_config.get("rsi_oversold", 30),
            warmup_period=req.ic_warmup_period or strategy_config.get("warmup_period", 20),
            stop_loss_pct=req.stop_loss_pct if req.stop_loss_pct is not None
                          else strategy_config.get("stop_loss_pct"),
            take_profit_pct=req.take_profit_pct if req.take_profit_pct is not None
                            else strategy_config.get("take_profit_pct"),
            trailing_stop_pct=req.trailing_stop_pct if req.trailing_stop_pct is not None
                              else strategy_config.get("trailing_stop_pct"),
            max_bars=req.max_bars if req.max_bars is not None
                     else strategy_config.get("max_bars"),
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported base strategy type: {base_strategy_type}",
        )

    # ── run backtest ──────────────────────────────────────────────────────
    backtester = Backtester(
        initial_capital=req.initial_capital,
        commission=req.commission,
    )
    results = backtester.run(strategy, historical_data)

    # ── compute buy-and-hold baseline ─────────────────────────────────────
    data_copy = historical_data.copy()
    close_col = "close" if "close" in data_copy.columns else "Close"
    first_price = float(data_copy[close_col].iloc[0])
    last_price = float(data_copy[close_col].iloc[-1])
    buy_hold_return_pct = ((last_price / first_price) - 1) * 100

    # ── win rate ──────────────────────────────────────────────────────────
    trades = results.get("trades", [])
    winning = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = (winning / len(trades) * 100) if trades else 0.0

    # ── format trades ─────────────────────────────────────────────────────
    trade_results = [
        TradeResult(
            entry_date=str(t["entry_date"])[:10],
            exit_date=str(t["exit_date"])[:10],
            entry_price=round(t["entry_price"], 4),
            exit_price=round(t["exit_price"], 4),
            position=t["position"],
            position_size=t["position_size"],
            pnl=round(t["pnl"], 4),
            trade_return=round(t["return"] * 100, 4),
            exit_reason=t.get("exit_reason", ""),
        )
        for t in trades
    ]

    return BacktestResponse(
        symbol=req.symbol,
        strategy=req.strategy,
        start_date=str(start_date.date()),
        end_date=str(end_date.date()),
        initial_capital=req.initial_capital,
        final_portfolio_value=round(results["final_portfolio_value"], 2),
        total_return_pct=round(results["total_return"], 4),
        sharpe_ratio=round(results["sharpe_ratio"], 4),
        max_drawdown_pct=round(results["max_drawdown"], 4),
        n_trades=results["n_trades"],
        win_rate_pct=round(win_rate, 2),
        buy_hold_return_pct=round(buy_hold_return_pct, 4),
        trades=trade_results,
    )
