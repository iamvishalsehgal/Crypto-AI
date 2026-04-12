"""
Central configuration for the AI Crypto Trading Bot.

Uses pydantic BaseSettings for validation and automatic environment variable loading.
All settings can be overridden via environment variables or a .env file.
"""

from typing import List, Optional

from pydantic import Field, MongoDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeSettings(BaseSettings):
    """Configuration for the cryptocurrency exchange connection."""

    model_config = SettingsConfigDict(env_prefix="EXCHANGE_")

    name: str = Field(default="binance", description="Exchange identifier (ccxt-compatible)")
    api_key: str = Field(default="", description="Exchange API key")
    api_secret: str = Field(default="", description="Exchange API secret")
    sandbox_mode: bool = Field(default=True, description="Use exchange sandbox/testnet")
    supported_symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT"],
        description="Trading pairs the bot is allowed to operate on",
    )
    rate_limit: int = Field(default=1200, description="API rate limit in ms between requests")


class DatabaseSettings(BaseSettings):
    """Configuration for MongoDB storage."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI",
    )
    name: str = Field(default="crypto_bot", description="Database name")
    connection_timeout_ms: int = Field(
        default=5000, description="Connection timeout in milliseconds"
    )
    server_selection_timeout_ms: int = Field(
        default=5000, description="Server selection timeout in milliseconds"
    )


class TradingSettings(BaseSettings):
    """Risk and position management parameters."""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    max_position_size: float = Field(
        default=0.10, description="Maximum fraction of portfolio per position"
    )
    stop_loss: float = Field(default=0.02, description="Stop-loss percentage")
    take_profit: float = Field(default=0.04, description="Take-profit percentage")
    max_daily_drawdown: float = Field(
        default=0.05, description="Maximum allowed daily drawdown fraction"
    )
    max_open_trades: int = Field(default=3, description="Maximum concurrent open trades")


class SafetySettings(BaseSettings):
    """Safety guardrails — fees, reserves, cooldowns, kill switches."""

    model_config = SettingsConfigDict(env_prefix="SAFETY_")

    # Fee protection
    max_fee_per_trade_pct: float = Field(
        default=0.002, description="Max acceptable fee as fraction of trade size"
    )
    max_daily_fees: float = Field(
        default=50.0, description="Hard cap on total fees paid per day (USD)"
    )
    min_profit_after_fees: float = Field(
        default=0.005, description="Minimum expected profit after fees to justify trade"
    )

    # Balance reserves
    reserve_balance_pct: float = Field(
        default=0.20, description="Fraction of balance that must NEVER be traded (emergency fund)"
    )
    min_trade_size_usd: float = Field(
        default=10.0, description="Minimum trade size in USD"
    )
    min_balance_to_trade: float = Field(
        default=100.0, description="Stop trading entirely if balance drops below this"
    )

    # Cooldowns
    cooldown_after_trade_sec: int = Field(
        default=30, description="Minimum seconds between trades on the same symbol"
    )
    cooldown_after_loss_sec: int = Field(
        default=120, description="Extra cooldown after a losing trade"
    )
    max_trades_per_hour: int = Field(
        default=10, description="Maximum trades allowed per rolling hour"
    )
    max_trades_per_day: int = Field(
        default=50, description="Maximum trades allowed per day"
    )

    # Data freshness
    max_data_age_sec: int = Field(
        default=300, description="Reject data older than this many seconds"
    )

    # Volatility
    max_atr_multiplier: float = Field(
        default=3.0, description="Skip trading if ATR > this * 20-period average ATR"
    )

    # Kill switches
    kill_on_consecutive_losses: int = Field(
        default=5, description="Halt all trading after N consecutive losses"
    )
    kill_on_equity_drop_pct: float = Field(
        default=0.15, description="Halt all trading if equity drops this much from peak"
    )


class DataSettings(BaseSettings):
    """Data collection and storage parameters."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    ohlcv_timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "1h", "4h", "1d"],
        description="Candlestick timeframes to collect",
    )
    lookback_days: int = Field(
        default=730, description="Number of historical days to fetch"
    )


class ModelSettings(BaseSettings):
    """Hyperparameters for ML/RL models."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # LSTM
    lstm_units: int = Field(default=128, description="Hidden units per LSTM layer")
    lstm_layers: int = Field(default=2, description="Number of stacked LSTM layers")

    # Double DQN
    ddqn_learning_rate: float = Field(default=0.001, description="DDQN optimiser learning rate")
    ddqn_gamma: float = Field(default=0.99, description="Discount factor")
    ddqn_epsilon: float = Field(default=1.0, description="Initial exploration rate")
    ddqn_epsilon_min: float = Field(default=0.01, description="Minimum exploration rate")
    batch_size: int = Field(default=64, description="Training mini-batch size")

    # XGBoost
    xgboost_n_estimators: int = Field(default=500, description="Number of boosting rounds")


class SentimentSettings(BaseSettings):
    """NLP sentiment analysis parameters."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_")

    finbert_model: str = Field(
        default="ProsusAI/finbert", description="HuggingFace model ID for FinBERT"
    )
    reddit_subreddits: List[str] = Field(
        default=["cryptocurrency", "bitcoin", "ethereum"],
        description="Subreddits to scrape for sentiment",
    )
    reddit_client_id: str = Field(default="", description="Reddit API client ID")
    reddit_client_secret: str = Field(default="", description="Reddit API client secret")


class MonitoringSettings(BaseSettings):
    """Alerting and observability settings."""

    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    telegram_bot_token: str = Field(default="", description="Telegram bot token for alerts")
    telegram_chat_id: str = Field(default="", description="Telegram chat/channel ID")
    grafana_url: str = Field(default="http://localhost:3000", description="Grafana dashboard URL")
    health_check_interval: int = Field(
        default=60, description="Seconds between health-check pings"
    )


class FeatureSettings(BaseSettings):
    """Feature engineering parameters."""

    model_config = SettingsConfigDict(env_prefix="FEATURE_")

    top_k_features: int = Field(
        default=20, description="Number of top features to retain after selection"
    )
    feature_importance_threshold: float = Field(
        default=0.01, description="Minimum importance score to keep a feature"
    )


class BacktestingSettings(BaseSettings):
    """Backtesting and walk-forward validation parameters."""

    model_config = SettingsConfigDict(env_prefix="BACKTEST_")

    walk_forward_windows: int = Field(
        default=5, description="Number of walk-forward splits"
    )
    overfitting_threshold: float = Field(
        default=0.30,
        description="Max allowed gap between in-sample and out-of-sample performance",
    )
    min_trades_per_window: int = Field(
        default=30, description="Minimum trades required per validation window"
    )
    transaction_fee: float = Field(
        default=0.001, description="Simulated transaction fee (fraction)"
    )
    slippage: float = Field(
        default=0.0005, description="Simulated slippage (fraction)"
    )


class Settings(BaseSettings):
    """
    Root settings object that aggregates every sub-configuration.

    Environment variables are loaded from a ``.env`` file in the project root
    (if present) and can be overridden by actual environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    feature: FeatureSettings = Field(default_factory=FeatureSettings)
    backtesting: BacktestingSettings = Field(default_factory=BacktestingSettings)


# Module-level singleton -- import this everywhere.
settings = Settings()
