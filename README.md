# Algorithmic Trading Backtesting Framework

A Python framework for backtesting algorithmic trading strategies against NSE equity data via the [Dhan API](https://dhan.co/).

## Features

- **Two built-in strategies** — Moving Average Crossover & Inside Candle Breakout with RSI confirmation
- **Full risk-management** — stop-loss, take-profit, trailing stop, and maximum bars per trade
- **REST API** — FastAPI server for running backtests programmatically or integrating with UIs
- **Docker-ready** — multi-stage Dockerfile + `docker-compose.yml` for one-command deployment
- **CI/CD** — GitHub Actions pipeline that runs the test suite on every push and validates the Docker build
- **30+ unit tests** covering strategies, the backtesting engine, and config management

---

## Project Structure

```
backtester/
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI application
│   ├── backtester/
│   │   └── backtester.py      # Core backtesting engine
│   ├── config/
│   │   ├── config_manager.py  # Strategy configuration loader/saver
│   │   └── dhan_config.py     # Dhan API credential loader
│   ├── data/
│   │   └── dhan_data.py       # Dhan API data fetcher
│   ├── strategies/
│   │   ├── base_strategy.py   # Abstract base class
│   │   ├── ma_crossover.py    # Moving Average Crossover
│   │   └── inside_candle_rsi.py  # Inside Candle + RSI
│   ├── example.py             # Main execution / visualisation
│   └── run_backtest.py        # CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_backtester.py
│   ├── test_strategies.py
│   └── test_config_manager.py
├── config.json                # Strategy parameter presets
├── .env.example               # Environment variable template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1 · Prerequisites

- Python 3.11+
- A [Dhan](https://dhan.co/) trading account with API access
- (Optional) Docker & Docker Compose for containerised deployment

### 2 · Configure credentials

```bash
cp .env.example .env
# Edit .env and set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN
```

### 3a · Run via CLI

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.run_backtest --strategy MA_CROSSOVER --days 365
```

Available options:

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | interactive | Strategy key from `config.json` |
| `--days` | 180 | Historical calendar days |
| `--config` | `config.json` | Path to strategy config file |
| `--list` | — | List available strategies and exit |

Environment variables that override defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `SYMBOL` | `RELIANCE` | NSE ticker symbol |
| `INITIAL_CAPITAL` | `100000` | Starting capital (INR) |
| `COMMISSION` | `0.001` | Commission per trade (0.1 %) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### 3b · Run the REST API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API documentation is served at **http://localhost:8000/docs**.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/strategies` | List available strategies |
| `POST` | `/backtest` | Run a backtest, returns JSON metrics + trades |

#### Example request

```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "strategy": "MA_CROSSOVER",
    "days": 365,
    "initial_capital": 100000
  }'
```

---

## Docker Deployment

### Build & run with docker-compose (recommended)

```bash
# 1 – Fill in credentials
cp .env.example .env && nano .env

# 2 – Start the API server
docker compose up -d

# 3 – Verify it's running
curl http://localhost:8000/health
```

### Build the image manually

```bash
docker build -t backtester-api .
docker run -p 8000:8000 --env-file .env backtester-api
```

---

## Testing

```bash
pytest tests/ -v --tb=short
# With coverage:
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Strategies

### MA Crossover (`MA_CROSSOVER`)

Generates buy signals when the short moving average crosses above the long MA, and sell signals on the reverse crossover.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 20 | Short MA period (bars) |
| `long_window` | 50 | Long MA period (bars) |
| `warmup_period` | 15 | Bars to skip after indicators stabilise |

### Inside Candle + RSI (`INSIDE_CANDLE_RSI`)

Detects inside-candle patterns and confirms breakouts with RSI. Supports full risk management.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_period` | 14 | RSI calculation period |
| `rsi_overbought` | 70 | RSI overbought threshold |
| `rsi_oversold` | 30 | RSI oversold threshold |
| `warmup_period` | 20 | Warmup bars |
| `stop_loss_pct` | 2.0 | Stop-loss % (null to disable) |
| `take_profit_pct` | 4.0 | Take-profit % (null to disable) |
| `trailing_stop_pct` | 1.5 | Trailing stop % (null to disable) |
| `max_bars` | null | Force exit after N bars |

An **aggressive variant** (`INSIDE_CANDLE_RSI_AGGRESSIVE`) is pre-configured in `config.json` with tighter stops and a higher profit target.

---

## Adding a Custom Strategy

1. Create `src/strategies/my_strategy.py` extending `BaseStrategy`.
2. Implement `calculate_indicators(data)` and `generate_signals(data)`.
3. Add a configuration block to `config.json`.
4. The CLI and API will pick it up automatically.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Lint & Test** — runs `pytest` on Python 3.11 and 3.12
2. **Docker build** — validates the multi-stage Dockerfile

Secrets required (add in GitHub → Settings → Secrets):

| Secret | Purpose |
|--------|---------|
| `DHAN_CLIENT_ID` | Dhan API client ID (for integration tests) |
| `DHAN_ACCESS_TOKEN` | Dhan API access token |
