# Graph Report - .  (2026-05-14)

## Corpus Check
- 113 files · ~106,024 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2001 nodes · 6245 edges · 59 communities detected
- Extraction: 39% EXTRACTED · 61% INFERRED · 0% AMBIGUOUS · INFERRED: 3781 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Execution & Betting Models|Execution & Betting Models]]
- [[_COMMUNITY_Asset Router & Signal Routing|Asset Router & Signal Routing]]
- [[_COMMUNITY_Auto-Learner Pipeline|Auto-Learner Pipeline]]
- [[_COMMUNITY_Asset Router Initialization|Asset Router Initialization]]
- [[_COMMUNITY_CNN & DQN Deep Learning|CNN & DQN Deep Learning]]
- [[_COMMUNITY_Betting Models & Caching|Betting Models & Caching]]
- [[_COMMUNITY_Betting Backtester|Betting Backtester]]
- [[_COMMUNITY_Technical Feature Engineering|Technical Feature Engineering]]
- [[_COMMUNITY_Auto-Retrainer & Tuning|Auto-Retrainer & Tuning]]
- [[_COMMUNITY_Asset Router Status & Betting|Asset Router Status & Betting]]
- [[_COMMUNITY_Architecture Decision Records|Architecture Decision Records]]
- [[_COMMUNITY_Configuration & Settings|Configuration & Settings]]
- [[_COMMUNITY_Backtest Charts & Reports|Backtest Charts & Reports]]
- [[_COMMUNITY_Security Test Suite|Security Test Suite]]
- [[_COMMUNITY_Logging Infrastructure|Logging Infrastructure]]
- [[_COMMUNITY_Safety Guard & Balance|Safety Guard & Balance]]
- [[_COMMUNITY_Utility Helpers|Utility Helpers]]
- [[_COMMUNITY_Betting Feature Engineering|Betting Feature Engineering]]
- [[_COMMUNITY_Backtest Engine Core|Backtest Engine Core]]
- [[_COMMUNITY_Betting Asset Classes & Methods|Betting Asset Classes & Methods]]
- [[_COMMUNITY_Telegram Bot Notifications|Telegram Bot Notifications]]
- [[_COMMUNITY_Backtest Visualization Components|Backtest Visualization Components]]
- [[_COMMUNITY_Asset Type Configuration|Asset Type Configuration]]
- [[_COMMUNITY_Telegram Bot Reporting|Telegram Bot Reporting]]
- [[_COMMUNITY_Triage Labels & Workflow|Triage Labels & Workflow]]
- [[_COMMUNITY_Issue Tracker Integration|Issue Tracker Integration]]
- [[_COMMUNITY_Package Setup|Package Setup]]
- [[_COMMUNITY_Tests Package Init|Tests Package Init]]
- [[_COMMUNITY_Omnitrade Package Init|Omnitrade Package Init]]
- [[_COMMUNITY_Ensemble Package Init|Ensemble Package Init]]
- [[_COMMUNITY_Backtesting Package Init|Backtesting Package Init]]
- [[_COMMUNITY_Asset Type Rationale|Asset Type Rationale]]
- [[_COMMUNITY_Config Package Init|Config Package Init]]
- [[_COMMUNITY_Features Package Init|Features Package Init]]
- [[_COMMUNITY_Utils Package Init|Utils Package Init]]
- [[_COMMUNITY_Models Package Init|Models Package Init]]
- [[_COMMUNITY_Training Package Init|Training Package Init]]
- [[_COMMUNITY_RL Environment Setup|RL Environment Setup]]
- [[_COMMUNITY_RL Environment State|RL Environment State]]
- [[_COMMUNITY_RL Environment Actions|RL Environment Actions]]
- [[_COMMUNITY_Paper Wallet State|Paper Wallet State]]
- [[_COMMUNITY_Execution Package Init|Execution Package Init]]
- [[_COMMUNITY_Monitoring Package Init|Monitoring Package Init]]
- [[_COMMUNITY_Data Package Init|Data Package Init]]
- [[_COMMUNITY_Data Collectors Package Init|Data Collectors Package Init]]
- [[_COMMUNITY_Data Storage Package Init|Data Storage Package Init]]
- [[_COMMUNITY_Risk Package Init|Risk Package Init]]
- [[_COMMUNITY_Project Dependencies|Project Dependencies]]
- [[_COMMUNITY_SPY Backtest Chart|SPY Backtest Chart]]
- [[_COMMUNITY_SPY Equity Curve|SPY Equity Curve]]
- [[_COMMUNITY_SPY Drawdown Analysis|SPY Drawdown Analysis]]
- [[_COMMUNITY_SPY Performance Metrics|SPY Performance Metrics]]
- [[_COMMUNITY_SPY Trade Annotations|SPY Trade Annotations]]
- [[_COMMUNITY_AAPL Backtest Chart|AAPL Backtest Chart]]
- [[_COMMUNITY_AAPL Equity Curve|AAPL Equity Curve]]
- [[_COMMUNITY_AAPL Stock Analysis|AAPL Stock Analysis]]
- [[_COMMUNITY_Backtest Buy Signals|Backtest Buy Signals]]
- [[_COMMUNITY_Backtest Sell Signals|Backtest Sell Signals]]
- [[_COMMUNITY_Matplotlib Backtest Engine|Matplotlib Backtest Engine]]

## God Nodes (most connected - your core abstractions)
1. `Settings` - 743 edges
2. `TechnicalFeatures` - 155 edges
3. `UnifiedSignal` - 131 edges
4. `RiskManager` - 122 edges
5. `AssetType` - 113 edges
6. `AutoTrader` - 93 edges
7. `MarketDataCollector` - 89 edges
8. `AssetRouter` - 83 edges
9. `StockModelFactory` - 77 edges
10. `XGBoostTrader` - 77 edges

## Surprising Connections (you probably didn't know these)
- `test_paper_mode_is_default()` --calls--> `Settings`  [INFERRED]
  tests/test_live_guard.py → omnitrade/config/settings.py
- `test_accepts_live_mode()` --calls--> `Settings`  [INFERRED]
  tests/test_live_guard.py → omnitrade/config/settings.py
- `test_american_positive()` --calls--> `american_to_decimal()`  [INFERRED]
  tests/test_utils.py → omnitrade/utils/odds.py
- `test_american_negative()` --calls--> `american_to_decimal()`  [INFERRED]
  tests/test_utils.py → omnitrade/utils/odds.py
- `test_american_zero()` --calls--> `american_to_decimal()`  [INFERRED]
  tests/test_utils.py → omnitrade/utils/odds.py

## Hyperedges (group relationships)
- **Pipeline: Collect-Features-Signal** — context_collector, context_feature, context_model, context_pipeline [EXTRACTED 1.00]
- **3-Lane Architecture** — readme_crypto_lane, readme_stock_lane, readme_betting_lane [EXTRACTED 1.00]
- **Trading Safety and Halt System** — context_risk_manager, context_safety_guard, context_circuit_breaker, context_halt [EXTRACTED 1.00]

## Communities

### Community 0 - "Execution & Betting Models"
Cohesion: 0.01
Nodes (203): Compute home win / draw / away win probabilities via Poisson., Compute features from odds data for betting signal generation., Train the probability estimator on historical data.          Args:             f, CandlestickCNN, Ensure lowercase column names and required OHLC columns exist., Small convolutional neural network for candlestick pattern recognition.      Arc, Forward pass.          Parameters         ----------         x:             Inpu, Return trading signal predictions for a batch of images.          Parameters (+195 more)

### Community 1 - "Asset Router & Signal Routing"
Cohesion: 0.02
Nodes (189): bet_executor(), crypto_executor(), stock_executor(), _generate_betting_labels(), append_trade_csv(), auto_commit_and_push(), AutoTrader, check_disk_space() (+181 more)

### Community 2 - "Auto-Learner Pipeline"
Cohesion: 0.02
Nodes (187): _add_labels(), calibrate_ensemble(), collect_all_data(), engineer_features(), evaluate_improvement(), load_previous_weights(), main(), Compute all available features from market + auxiliary data. (+179 more)

### Community 3 - "Asset Router Initialization"
Cohesion: 0.05
Nodes (152): AssetRouter, AssetType, Supported asset classes for the trading bot., Normalised trading signal that every pipeline stage understands.      Attributes, UnifiedSignal, AutoRetrainer, Auto-retraining pipeline for self-learning trading agents.  Handles periodic mod, Retrain all enabled lane models on fresh data.          Returns dict of per-lane (+144 more)

### Community 4 - "CNN & DQN Deep Learning"
Cohesion: 0.02
Nodes (76): Train the CNN on candlestick images.          Parameters         ----------, DDQNAgent, QNetwork, Double Deep Q-Network (DDQN) agent for crypto trading.  Provides:     - ``QNetwo, Double DQN agent for discrete-action crypto trading.      The agent maintains tw, Select an action using epsilon-greedy (training) or greedy (eval).          Args, Store a transition in the replay buffer., Sample a mini-batch and perform one DDQN update.          Returns:             T (+68 more)

### Community 5 - "Betting Models & Caching"
Cohesion: 0.03
Nodes (87): Simple TTL cache with per-key time-to-live., Minimalist in-memory cache with per-key time-to-live., TTLCache, CircuitBreaker, is_open(), Circuit breaker -- trips after consecutive failures within cooldown window., Trips after *threshold* consecutive failures within *window* seconds., Fetch the effective Federal Funds Rate.          Returns         ------- (+79 more)

### Community 6 - "Betting Backtester"
Cohesion: 0.04
Nodes (95): BettingBacktestResult, Betting backtester — simulates value betting on historical odds, computes CLV (C, Run a backtest on historical odds for a sport.          Splits data chronologica, Compare full Kelly, fractional Kelly, and flat staking.          Returns:, Simulate betting through the feature DataFrame chronologically.          Args:, Simulate with flat staking (fixed % of initial bankroll per bet)., Aggregate betting backtest metrics., Determine bet outcome from historical data.          For mock data, uses the poi (+87 more)

### Community 7 - "Technical Feature Engineering"
Cohesion: 0.03
Nodes (88): Technical indicator calculations for crypto trading.  Pure pandas/numpy implemen, Relative Strength Index using Wilder's smoothing method.          Parameters, Moving Average Convergence Divergence.          Parameters         ----------, Bollinger Bands with bandwidth and %B.          Parameters         ----------, Exponential Moving Averages for multiple periods.          Parameters         --, Average True Range.          Parameters         ----------         df : pd.DataF, On-Balance Volume.          Parameters         ----------         df : pd.DataFr, Volume-Weighted Average Price (cumulative).          Parameters         -------- (+80 more)

### Community 8 - "Auto-Retrainer & Tuning"
Cohesion: 0.03
Nodes (66): _generate_labels(), _evaluate_params(), main(), make_strategy(), mutate_params(), run_tuning(), score_result(), commit_and_push() (+58 more)

### Community 9 - "Asset Router Status & Betting"
Cohesion: 0.07
Nodes (30): Central asset router — dispatches UnifiedSignals to the correct lane executor., Dispatch a unified signal to the correct executor.          Args:             si, Aggregate balances across all active lanes., Aggregate open positions across all lanes., Top-level dispatcher for multi-asset trading.      Routes every :class:`UnifiedS, Emergency liquidation across all lanes., Return status of each lane., BettingExecutor (+22 more)

### Community 10 - "Architecture Decision Records"
Cohesion: 0.08
Nodes (39): ADR-0001: Extract PaperWallet, ADR-0002: Delegate Voting to EnsembleVoter, ADR-0003: Unified Halt Delegation, Domain Docs Config, Adapter, AssetRouter, CircuitBreaker, Collector (+31 more)

### Community 11 - "Configuration & Settings"
Cohesion: 0.08
Nodes (29): BaseSettings, AssetSettings, BacktestingSettings, DatabaseSettings, DataSettings, ExchangeSettings, FeatureSettings, ModelSettings (+21 more)

### Community 12 - "Backtest Charts & Reports"
Cohesion: 0.08
Nodes (33): BettingBacktester Engine, NFL Betting Backtest Performance Chart, BTC/USDT Backtest Performance Chart, Backtest: 1d timeframe over 365 days, Backtest 365 Days NFL Historical Odds, ETH/USDT Backtest Performance Chart, Backtest Result: +2.42% Total Return, BTC/USDT Trading Pair (+25 more)

### Community 13 - "Security Test Suite"
Cohesion: 0.13
Nodes (16): _file_contains(), Security fix verification tests.  Validates that the 5 security vulnerability fi, token=xxx in stderr should be redacted., Plain stderr with no credentials should pass through unchanged., Return True if *file_path* contains every pattern as a substring., Replicate the exact helper from auto_trader.py / benchmark_runner.py., Token embedded in HTTPS URL should be redacted., password=plaintext in stderr should be redacted. (+8 more)

### Community 14 - "Logging Infrastructure"
Cohesion: 0.11
Nodes (12): get_logger(), Convenience wrapper around the project logging configuration.  Usage::      from, Return a configured logger for the given *name*.      On the first call the root, Logging configuration for the AI Crypto Trading Bot.  Provides a ``setup_logging, Configure and return a logger with console and rotating-file handlers.      Args, setup_logging(), _init_vader(), Social media and news sentiment collector.  Aggregates posts from Reddit, Twitte (+4 more)

### Community 15 - "Safety Guard & Balance"
Cohesion: 0.39
Nodes (13): _default_settings(), _make_bet_executor(), _make_crypto_executor(), _make_stock_executor(), test_get_status_has_correct_booleans(), test_get_status_returns_enabled_lanes(), test_get_unified_balance_aggregates(), test_get_unified_balance_error_handling() (+5 more)

### Community 16 - "Utility Helpers"
Cohesion: 0.12
Nodes (15): calculate_returns(), chunk_list(), datetime_to_timestamp(), normalize_dataframe(), General-purpose utility functions used across the crypto bot., Divide *a* by *b*, returning *default* when *b* is zero or the result     is not, Decorator that retries *func* on failure with exponential backoff.      Can be u, Split *lst* into consecutive chunks of at most *n* elements.      Args: (+7 more)

### Community 17 - "Betting Feature Engineering"
Cohesion: 0.13
Nodes (6): _poisson_match_probs(), Betting-specific feature engineering.  Converts raw odds and historical data int, Detect steam moves — sharp odds shifts across sportsbooks.          When multipl, Iterative ELO calculation from historical match results., Compute full feature matrix from odds and historical data.          Args:, Probability-based features from odds.

### Community 18 - "Backtest Engine Core"
Cohesion: 0.22
Nodes (7): _calculate_max_drawdown(), _calculate_profit_factor(), _calculate_sharpe(), Core backtesting engine for the AI Crypto Trading Bot.  Simulates strategy execu, Execute a backtest over *data*.          Parameters         ----------         s, Compute all performance metrics from the trade list and equity curve., Reset all mutable state for a fresh backtest run.

### Community 19 - "Betting Asset Classes & Methods"
Cohesion: 0.2
Nodes (11): Betting (sports wagering), Kelly criterion staking model for bet sizing, Value betting strategy — bet when modeled probability exceeds implied probability, Bankroll curve over time (value betting simulation), Individual bet profit-loss distribution, Closing line value (CLV) positive percentage, Drawdown percentage over time, Return on investment (ROI) from betting simulation (+3 more)

### Community 20 - "Telegram Bot Notifications"
Cohesion: 0.25
Nodes (4): Send a single message immediately, rate-limit aware., Sleep if we have hit the 30 msg/s ceiling., Background loop that drains the message queue., Start the background queue worker.

### Community 21 - "Backtest Visualization Components"
Cohesion: 0.38
Nodes (7): BacktestEngine, Backtest Stock TSLA Chart, Drawdown Chart (Percentage Decline From Peak), Equity Curve Subplot, Matplotlib 3-Panel Backtest Chart, Trade PnL Bar Chart (Profit/Loss Per Trade), TSLA (Tesla Stock)

### Community 22 - "Asset Type Configuration"
Cohesion: 0.33
Nodes (4): AssetConfig, Unified asset type system for multi-asset trading (crypto, stocks, betting).  De, Per-asset-type configuration container., Enum

### Community 23 - "Telegram Bot Reporting"
Cohesion: 0.4
Nodes (3): _format_daily_report(), Telegram notification bot for the AI Crypto Trading Bot.  Sends trade alerts, da, Send a daily performance summary.          Args:             stats: Dict with ke

### Community 24 - "Triage Labels & Workflow"
Cohesion: 0.4
Nodes (5): Triage Labels Config, Triage Label Mappings, needs-triage label, ready-for-agent label, wontfix label

### Community 25 - "Issue Tracker Integration"
Cohesion: 0.67
Nodes (3): Issue Tracker Config, gh CLI Command Conventions, GitHub Repository (iamvishalsehgal/Crypto-AI)

### Community 26 - "Package Setup"
Cohesion: 1.0
Nodes (1): Package setup for OmniTrade AI — Multi-Asset Autonomous Trading System.

### Community 27 - "Tests Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Omnitrade Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Ensemble Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Backtesting Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Asset Type Rationale"
Cohesion: 1.0
Nodes (1): Return ``True`` if the signal warrants execution.

### Community 32 - "Config Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Features Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Utils Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Models Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Training Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "RL Environment Setup"
Cohesion: 1.0
Nodes (1): Quote-currency amount originally invested.

### Community 38 - "RL Environment State"
Cohesion: 1.0
Nodes (1): Return column name matching *target* case-insensitively, or None.

### Community 39 - "RL Environment Actions"
Cohesion: 1.0
Nodes (1): Return the name of the close-price column (case-insensitive).

### Community 40 - "Paper Wallet State"
Cohesion: 1.0
Nodes (1): Quote-currency balance (excluding unrealized PnL).

### Community 41 - "Execution Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Monitoring Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Data Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Data Collectors Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Data Storage Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Risk Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 47 - "Project Dependencies"
Cohesion: 1.0
Nodes (1): Python Dependencies

### Community 48 - "SPY Backtest Chart"
Cohesion: 1.0
Nodes (1): Backtest Performance Chart - SPY Stock

### Community 49 - "SPY Equity Curve"
Cohesion: 1.0
Nodes (1): SPY Backtest Equity Curve

### Community 50 - "SPY Drawdown Analysis"
Cohesion: 1.0
Nodes (1): SPY Drawdown Reference Line

### Community 51 - "SPY Performance Metrics"
Cohesion: 1.0
Nodes (1): SPY Backtest Performance Metrics Text Block

### Community 52 - "SPY Trade Annotations"
Cohesion: 1.0
Nodes (1): SPY Backtest Trade Annotations

### Community 53 - "AAPL Backtest Chart"
Cohesion: 1.0
Nodes (1): AAPL Backtest Performance Chart

### Community 54 - "AAPL Equity Curve"
Cohesion: 1.0
Nodes (1): Equity Curve (Portfolio Value Over Time)

### Community 55 - "AAPL Stock Analysis"
Cohesion: 1.0
Nodes (1): Apple Inc. (AAPL) Stock

### Community 56 - "Backtest Buy Signals"
Cohesion: 1.0
Nodes (1): Buy Trade Markers (Green Triangles on Equity Curve)

### Community 57 - "Backtest Sell Signals"
Cohesion: 1.0
Nodes (1): Sell Trade Markers (Red Inverted Triangles on Equity Curve)

### Community 58 - "Matplotlib Backtest Engine"
Cohesion: 1.0
Nodes (1): Matplotlib 3.10.9 Backtest Plot Engine

## Knowledge Gaps
- **175 isolated node(s):** `Package setup for OmniTrade AI — Multi-Asset Autonomous Trading System.`, `Security fix verification tests.  Validates that the 5 security vulnerability fi`, `Return True if *file_path* contains every pattern as a substring.`, `Replicate the exact helper from auto_trader.py / benchmark_runner.py.`, `Token embedded in HTTPS URL should be redacted.` (+170 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Package Setup`** (2 nodes): `setup.py`, `Package setup for OmniTrade AI — Multi-Asset Autonomous Trading System.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tests Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Omnitrade Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ensemble Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backtesting Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Asset Type Rationale`** (1 nodes): `Return ``True`` if the signal warrants execution.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Config Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Features Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Utils Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Models Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Training Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RL Environment Setup`** (1 nodes): `Quote-currency amount originally invested.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RL Environment State`** (1 nodes): `Return column name matching *target* case-insensitively, or None.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RL Environment Actions`** (1 nodes): `Return the name of the close-price column (case-insensitive).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Paper Wallet State`** (1 nodes): `Quote-currency balance (excluding unrealized PnL).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Execution Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Monitoring Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Collectors Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Storage Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Risk Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Project Dependencies`** (1 nodes): `Python Dependencies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SPY Backtest Chart`** (1 nodes): `Backtest Performance Chart - SPY Stock`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SPY Equity Curve`** (1 nodes): `SPY Backtest Equity Curve`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SPY Drawdown Analysis`** (1 nodes): `SPY Drawdown Reference Line`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SPY Performance Metrics`** (1 nodes): `SPY Backtest Performance Metrics Text Block`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SPY Trade Annotations`** (1 nodes): `SPY Backtest Trade Annotations`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AAPL Backtest Chart`** (1 nodes): `AAPL Backtest Performance Chart`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AAPL Equity Curve`** (1 nodes): `Equity Curve (Portfolio Value Over Time)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `AAPL Stock Analysis`** (1 nodes): `Apple Inc. (AAPL) Stock`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backtest Buy Signals`** (1 nodes): `Buy Trade Markers (Green Triangles on Equity Curve)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backtest Sell Signals`** (1 nodes): `Sell Trade Markers (Red Inverted Triangles on Equity Curve)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matplotlib Backtest Engine`** (1 nodes): `Matplotlib 3.10.9 Backtest Plot Engine`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Execution & Betting Models` to `Asset Router & Signal Routing`, `Auto-Learner Pipeline`, `Asset Router Initialization`, `CNN & DQN Deep Learning`, `Betting Models & Caching`, `Betting Backtester`, `Technical Feature Engineering`, `Auto-Retrainer & Tuning`, `Asset Router Status & Betting`, `Configuration & Settings`, `Logging Infrastructure`, `Safety Guard & Balance`, `Betting Feature Engineering`, `Backtest Engine Core`, `Telegram Bot Notifications`, `Telegram Bot Reporting`?**
  _High betweenness centrality (0.570) - this node is a cross-community bridge._
- **Why does `BettingRiskManager` connect `Betting Backtester` to `Execution & Betting Models`, `Asset Router & Signal Routing`, `Asset Router Initialization`, `Asset Router Status & Betting`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `AssetRouter` connect `Asset Router Initialization` to `Execution & Betting Models`, `Asset Router & Signal Routing`, `Betting Backtester`, `Asset Router Status & Betting`, `Safety Guard & Balance`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 738 inferred relationships involving `Settings` (e.g. with `Unit tests for SentimentFeatures.` and `Return *n* daily timestamps starting at *base*.`) actually correct?**
  _`Settings` has 738 INFERRED edges - model-reasoned connections that need verification._
- **Are the 140 inferred relationships involving `TechnicalFeatures` (e.g. with `Unit tests for TechnicalFeatures indicators. 12 indicators, every trading signal` and `Synthetic OHLCV with known statistical properties.`) actually correct?**
  _`TechnicalFeatures` has 140 INFERRED edges - model-reasoned connections that need verification._
- **Are the 129 inferred relationships involving `UnifiedSignal` (e.g. with `End-to-end tests: AutoTrader full loop — signal → validate → execute → record.` and `Generate OHLCV with a sustained downtrend (produces oversold RSI + buy signals).`) actually correct?**
  _`UnifiedSignal` has 129 INFERRED edges - model-reasoned connections that need verification._
- **Are the 104 inferred relationships involving `RiskManager` (e.g. with `End-to-end tests: AutoTrader full loop — signal → validate → execute → record.` and `Generate OHLCV with a sustained downtrend (produces oversold RSI + buy signals).`) actually correct?**
  _`RiskManager` has 104 INFERRED edges - model-reasoned connections that need verification._