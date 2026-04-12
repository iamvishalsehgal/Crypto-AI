# AI Crypto Trading Bot

An AI-powered cryptocurrency trading bot that combines deep reinforcement learning, time-series forecasting, sentiment analysis, and ensemble voting to generate trading signals.

## Architecture

```
crypto_bot/
├── config/            # Settings & logging configuration
├── data/
│   ├── collectors/    # Market, on-chain, sentiment, macro data
│   └── storage/       # MongoDB persistence layer
├── features/          # Feature engineering & XGBoost selection
├── models/            # AI models
│   ├── lstm_model     # LSTM / BiLSTM / GRU forecasting
│   ├── ddqn_agent     # Double DQN reinforcement learning
│   ├── xgboost_model  # XGBoost signal classification
│   ├── sentiment_model# FinBERT sentiment analysis
│   ├── cnn_model      # CNN candlestick pattern recognition
│   └── training/      # RL environment & training orchestrator
├── ensemble/          # Weighted voting system
├── backtesting/       # Engine, walk-forward, overfitting detection
├── risk/              # Risk management engine
├── execution/         # Trade execution (live + paper)
└── monitoring/        # Telegram alerts, Grafana metrics, health checks
```

## Features

- **Multi-source data pipeline** — OHLCV prices, on-chain whale tracking, Reddit/Twitter/News sentiment, macroeconomic indicators (VIX, DXY, Fed rate)
- **30+ technical indicators** — RSI, MACD, Bollinger Bands, Ichimoku, ADX, Stochastic, and more
- **XGBoost feature selection** — automatically ranks and selects the most predictive features
- **5 AI models** — LSTM, Double DQN, XGBoost, FinBERT sentiment, CNN on candlestick images
- **Ensemble voting** — weighted majority vote with configurable confidence threshold
- **Backtesting engine** — walk-forward validation, overfitting detection, crash scenario testing
- **Risk management** — position sizing, stop-loss/take-profit, trailing stops, daily drawdown limits
- **Paper trading mode** — test strategies with simulated orders against live market data
- **Real-time monitoring** — Telegram alerts, Prometheus metrics, Grafana dashboards
- **Docker deployment** — containerized with MongoDB, Grafana, and Prometheus

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd Crypto-AI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Minimum required:**
- Binance API key & secret (get from [Binance](https://www.binance.com/en/my/settings/api-management))

**Optional (for full functionality):**
- Reddit API credentials (sentiment)
- Twitter Bearer Token (sentiment)
- NewsAPI key (news sentiment)
- Alchemy API key (on-chain data)
- FRED API key (macro indicators)
- Telegram bot token (alerts)

### 3. Run

```bash
# Paper trading (default, safe)
python -m crypto_bot.main

# Paper trading with specific symbols
python -m crypto_bot.main --symbols BTC/USDT ETH/USDT

# Backtesting mode
python -m crypto_bot.main --backtest

# Live trading (requires real API keys, use with caution)
python -m crypto_bot.main --mode live
```

### 4. Docker Deployment

```bash
cd deployment
docker-compose up -d
```

This starts:
- Trading bot container
- MongoDB (port 27017)
- Grafana dashboard (port 3000)
- Prometheus metrics (port 9090)

## Risk Configuration

Edit via environment variables or `.env`:

| Parameter | Default | Description |
|---|---|---|
| `TRADING_MAX_POSITION_SIZE` | 10% | Max portfolio fraction per trade |
| `TRADING_STOP_LOSS` | 2% | Stop-loss threshold |
| `TRADING_TAKE_PROFIT` | 4% | Take-profit threshold |
| `TRADING_MAX_DAILY_DRAWDOWN` | 5% | Daily drawdown limit (halts bot) |
| `TRADING_MAX_OPEN_TRADES` | 3 | Max simultaneous positions |

## Model Training

The bot supports training individual models or all at once:

```python
from crypto_bot.models.training.trainer import ModelTrainer
from crypto_bot.config.settings import settings

trainer = ModelTrainer(settings)
results = trainer.train_all_models(features_df)
```

## Backtesting

```python
from crypto_bot.backtesting.engine import BacktestEngine
from crypto_bot.backtesting.overfitting_detector import OverfittingDetector

engine = BacktestEngine(settings)
result = engine.run(strategy_func, historical_data)

# Check for overfitting
detector = OverfittingDetector(settings)
report = detector.detect_and_report(in_sample_returns, out_sample_returns)
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest

# Lint
ruff check crypto_bot/
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| RL Environment | Gymnasium |
| NLP | HuggingFace Transformers (FinBERT) |
| ML | XGBoost, scikit-learn |
| Exchange | ccxt (100+ exchanges) |
| Database | MongoDB |
| Monitoring | Prometheus + Grafana |
| Alerts | Telegram Bot API |
| Deployment | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Foundation & Setup | Done |
| Phase 1 | Data Pipeline | Done |
| Phase 2 | Feature Engineering | Done |
| Phase 3 | AI Models | Done |
| Phase 4 | Ensemble Voting | Done |
| Phase 5 | Backtesting | Done |
| Phase 6 | Risk Management | Done |
| Phase 7 | Deployment & Monitoring | Done |
| Ongoing | Model retraining & new signals | In Progress |

## Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

## Author

**Vishal Sehgal** — [@iamvishalsehgal](https://github.com/iamvishalsehgal)

## License

MIT
