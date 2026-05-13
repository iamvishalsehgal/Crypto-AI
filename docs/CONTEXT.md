# OmniTrade AI — Domain Glossary

Canonical terms for the multi-asset autonomous trading system. Implementation-agnostic.
Definitions resolve ambiguity across the codebase. No implementation details.

---

## Lane
A self-contained trading domain within the system. Three lanes exist:
- **crypto** — digital assets traded via exchange APIs (ccxt/Binance)
- **stock** — equities traded via broker APIs (Alpaca) or data sources (yfinance)
- **bet** — sports wagers placed via odds APIs (The Odds API)

Each lane has its own collector (data ingestion), feature pipeline (indicator computation), model(s) (signal generation), and executor (order placement). Lanes operate independently but share a unified risk/safety layer. The AssetRouter dispatches signals to the correct lane.

---

## Raw Signal
The unconverted output of a signal generator (e.g. `SignalGenerator.generate()`). A dict with keys: `signal` ("BUY"/"SELL"/"HOLD"), `confidence` (0–1), `reasons` (list of strings), and computed `indicators`. Lane-specific. Not yet wrapped for routing.

## UnifiedSignal
A dataclass (`asset_types.UnifiedSignal`) that wraps any lane's raw output into a common shape: `asset_type` (Lane), `symbol`, `side`, `confidence`, `amount`, `price`, plus optional metadata. The canonical form consumed by `AssetRouter.route_signal()`.

---

## Order
A single buy or sell fill. Has `order_id`, `symbol`, `side`, `price`, `amount` (base units filled), `fee`, `status` ("filled"/"rejected"), and `timestamp`. What `PaperWallet.simulate_order()` returns. What `_trade_history` in executors actually stores (despite the misnamed attribute). An Order is **not** a Trade — it's one leg.

## Trade
A matched pair: one entry Order + one exit Order, with computed P&L. Lives only in `PnLTracker`. Has `entry_price`, `exit_price`, `amount`, `pnl` (absolute), `pnl_pct`, `status` ("open"/"closed"). A Trade is always composed of exactly two Orders.

---

## RiskManager
Portfolio-level risk control. Decides **how much** to trade (position sizing, adaptive Sharpe-based multiplier) and **when to exit** (stop-loss, take-profit). Monitors daily drawdown and daily trade count across the whole portfolio. Portfolio-level halt: stops all lanes when daily drawdown exceeds threshold.

## SafetyGuard
Per-trade gate. Runs 12 pre-flight checks before every individual trade: kill switch, equity drawdown, consecutive loss kill, minimum balance, reserve buffer, fee cap, rate limits, symbol cooldown, duplicate position, stale data, volatility spike, daily fee cap. Answers one question: "is this specific trade safe right now?" Can halt all trading (kill switch).

---

## PaperWallet
Shared paper-trading **state**. Holds multi-asset balances, open positions, and order history. Pure state + math — no price fetching, no API calls. Provides `simulate_order()` for market/limit fills with fee and slippage. Used by TradeExecutor and StockExecutor in paper mode. One wallet per executor instance (crypto shares USDT wallet, stocks share USD wallet).

## Executor
The **action** layer. Each lane has one: `TradeExecutor` (crypto), `StockExecutor` (stocks), `BettingExecutor` (betting). An executor takes a UnifiedSignal, places an Order (paper or live), and returns the Order result. Key method: `place_order()` (currently misnamed `execute_trade()` in TradeExecutor — places an Order, not a Trade).

## AssetRouter
Central dispatcher. Holds references to all three lane executors. `route_signal(UnifiedSignal)` inspects `asset_type`, forwards to the correct executor. Also provides `get_unified_balance()` and `get_all_positions()` for cross-lane portfolio views.

---

## Model
Any component that takes features and produces a prediction. Includes classifiers (XGBoost, LightGBM), forecasters (LSTM, CNN), RL agents (DQN, PPO), sentiment analyzers (FinBERT), and factory aggregators (StockModelFactory). Every model exposes a `predict()` method. Models register with the EnsembleVoter to participate in voting.

## EnsembleVoter
Weighted voting system that aggregates predictions from multiple Models into a single signal. Each Model registers with a weight. On `vote()`, every model's prediction is collected, weighted, and the highest-scoring signal class wins. Produces a final signal with aggregated confidence.

## Adapter
A thin wrapper that makes a Model compatible with the EnsembleVoter interface. For example, `_XGBoostStockAdapter` wraps XGBoost's `predict_proba()` output into the BUY/SELL/HOLD string format the voter expects. Adapters live alongside the factory that creates them — not as standalone modules.

---

## Collector
Fetches raw data from an external source. Each lane has one: `MarketDataCollector` (crypto OHLCV via ccxt), `StockDataCollector` (equity OHLCV + fundamentals via yfinance/Alpaca), `BettingDataCollector` (odds + results via The Odds API). Collectors return DataFrames. They do not compute indicators.

## Feature
A computed value derived from raw data. Includes technical indicators (RSI, MACD, Bollinger Bands), transformations (EMA crossovers, volatility), and fundamental ratios. "Indicator" and "feature" are synonyms in this system — both mean a column in the DataFrame fed into a Model. `TechnicalFeatures.compute_all()` and `StockFeaturePipeline.compute_all()` both produce features.

## Pipeline
The three-stage flow: **Collect** (raw data) → **Features** (computed indicators) → **Signal** (model prediction). Each lane runs its own pipeline independently. The pipeline is stateless — each cycle runs fresh.

---

## Cycle
One complete pass through the trading loop for one symbol: collect data, compute features, generate signal, check safety, place order, update P&L. Cycles run sequentially per symbol within a larger loop iteration. Not to be confused with a loop iteration (which covers all symbols across all lanes).

## CircuitBreaker
Trips after N consecutive failures within a time window. When open, all cycles return PAUSED. Separate from SafetyGuard — the breaker guards against systemic failure (exchange down, data corruption), not financial risk. Auto-resets after cooldown.

## Halt
A full trading stop. When halted, all cycles return HALTED immediately. Two paths to halt: (1) SafetyGuard kill switch (financial — drawdown, consecutive losses), (2) RiskManager daily drawdown (delegates to SafetyGuard when wired). A halt persists until explicitly reset (daily reset or manual intervention). Not the same as CircuitBreaker open (self-resetting, failure-guarding).

## SafetyVerdict
The result object from `SafetyGuard.pre_trade_check()`. Three fields: `safe` (bool), `reason` (string, set when unsafe), `adjusted_size` (optional float — cap trade size to respect reserve), `warnings` (list of non-fatal concerns).

## PnLTracker
Tracks every Trade from open to close. Stores two lists: open Trades and closed Trades. On entry, creates an open Trade from an Order. On exit, matches the open Trade by symbol, computes P&L, moves to closed. Provides aggregate stats: win rate, profit factor, best/worst trade, total P&L.
