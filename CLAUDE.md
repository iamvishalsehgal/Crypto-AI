## Agent skills

### Issue tracker

GitHub Issues on `iamvishalsehgal/Crypto-AI`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `docs/CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Auto-learning pipeline

`AutoRetrainer` (omnitrade/learning/) retrains all lane models every 24h on fresh data. Records every trade outcome → adapts ensemble weights via softmax over cumulative PnL. State persisted to `models/saved/retrain_state.json` across restarts.

### Data sources (no mock/synthetic data)

All training data must be real. Crypto: ccxt OHLCV. Stocks: yfinance + Alpaca. Betting: ESPN public API (free, real scores — no API key) for historical results; The Odds API (free tier, 500 req/month) for live odds. No synthetic/mock data generators remain in the codebase.
