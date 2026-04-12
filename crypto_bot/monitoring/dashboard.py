"""
Prometheus metrics exporter for the AI Crypto Trading Bot.

Exposes key trading, portfolio, and model metrics on an HTTP endpoint
that Prometheus can scrape and Grafana can visualise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from crypto_bot.config.settings import Settings
from crypto_bot.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsExporter:
    """
    Registers and updates Prometheus metrics, and serves them over HTTP.

    Typical usage::

        exporter = MetricsExporter(settings, port=8000)
        exporter.register_metrics()
        exporter.start_server()

        # ...later, from trading loop...
        exporter.update_trade_metric(trade)
        exporter.update_portfolio_metric(portfolio)
    """

    def __init__(self, settings: Settings, port: int = 8000) -> None:
        """
        Initialise the exporter.

        Args:
            settings: Application settings.
            port: TCP port for the Prometheus HTTP endpoint.
        """
        self._settings = settings
        self._port = port
        self._server_started = False

        # Metrics (populated by register_metrics).
        self._trades_total: Optional[Counter] = None
        self._trade_pnl: Optional[Gauge] = None
        self._portfolio_value: Optional[Gauge] = None
        self._model_accuracy: Optional[Gauge] = None
        self._active_positions: Optional[Gauge] = None
        self._daily_drawdown: Optional[Gauge] = None
        self._signal_count: Optional[Counter] = None
        self._latency: Optional[Histogram] = None

    # ------------------------------------------------------------------
    # Metric registration
    # ------------------------------------------------------------------

    def register_metrics(self) -> None:
        """
        Create all Prometheus counters, gauges, and histograms.

        Safe to call multiple times -- subsequent calls are no-ops.
        """
        if self._trades_total is not None:
            logger.debug("Metrics already registered.")
            return

        self._trades_total = Counter(
            "crypto_bot_trades_total",
            "Total number of executed trades",
            ["symbol", "side"],
        )

        self._trade_pnl = Gauge(
            "crypto_bot_trade_pnl",
            "Profit/loss of the most recent trade",
            ["symbol"],
        )

        self._portfolio_value = Gauge(
            "crypto_bot_portfolio_value",
            "Current total portfolio value in USD",
        )

        self._model_accuracy = Gauge(
            "crypto_bot_model_accuracy",
            "Latest accuracy metric per model",
            ["model_name"],
        )

        self._active_positions = Gauge(
            "crypto_bot_active_positions",
            "Number of currently open positions",
        )

        self._daily_drawdown = Gauge(
            "crypto_bot_daily_drawdown",
            "Current daily drawdown as a fraction",
        )

        self._signal_count = Counter(
            "crypto_bot_signal_count",
            "Number of trading signals generated",
            ["model_name", "signal"],
        )

        self._latency = Histogram(
            "crypto_bot_request_latency_seconds",
            "Latency of exchange API requests in seconds",
            ["operation"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        logger.info("Prometheus metrics registered.")

    # ------------------------------------------------------------------
    # Metric updates
    # ------------------------------------------------------------------

    def update_trade_metric(self, trade: Dict[str, Any]) -> None:
        """
        Record a trade in Prometheus counters/gauges.

        Args:
            trade: Dict with keys ``symbol``, ``side``, and optionally ``pnl``.
        """
        if self._trades_total is None:
            logger.warning("Metrics not registered; call register_metrics() first.")
            return

        symbol = trade.get("symbol", "UNKNOWN")
        side = trade.get("side", "UNKNOWN").lower()
        pnl = trade.get("pnl")

        self._trades_total.labels(symbol=symbol, side=side).inc()

        if pnl is not None:
            self._trade_pnl.labels(symbol=symbol).set(float(pnl))

        logger.debug("Trade metric updated: %s %s", side, symbol)

    def update_portfolio_metric(self, portfolio: Dict[str, Any]) -> None:
        """
        Update portfolio-level gauges.

        Args:
            portfolio: Dict with keys ``total_value``, ``active_positions``,
                       and ``daily_drawdown``.
        """
        if self._portfolio_value is None:
            logger.warning("Metrics not registered; call register_metrics() first.")
            return

        total_value = portfolio.get("total_value")
        if total_value is not None:
            self._portfolio_value.set(float(total_value))

        positions = portfolio.get("active_positions")
        if positions is not None:
            self._active_positions.set(int(positions))

        drawdown = portfolio.get("daily_drawdown")
        if drawdown is not None:
            self._daily_drawdown.set(float(drawdown))

        logger.debug("Portfolio metric updated.")

    def update_model_metric(self, model_name: str, accuracy: float) -> None:
        """
        Update the accuracy gauge for a specific model.

        Args:
            model_name: Identifier of the ML model.
            accuracy: Latest accuracy value (0-1 range).
        """
        if self._model_accuracy is None:
            logger.warning("Metrics not registered; call register_metrics() first.")
            return

        self._model_accuracy.labels(model_name=model_name).set(accuracy)
        logger.debug("Model metric updated: %s accuracy=%.4f", model_name, accuracy)

    def update_signal_metric(self, model_name: str, signal: str) -> None:
        """
        Increment the signal counter for a model.

        Args:
            model_name: Model that generated the signal.
            signal: Signal label (e.g. ``"buy"``, ``"sell"``, ``"hold"``).
        """
        if self._signal_count is None:
            logger.warning("Metrics not registered; call register_metrics() first.")
            return

        self._signal_count.labels(model_name=model_name, signal=signal).inc()

    def observe_latency(self, operation: str, seconds: float) -> None:
        """
        Record an API latency observation.

        Args:
            operation: Operation label (e.g. ``"fetch_ohlcv"``).
            seconds: Elapsed time in seconds.
        """
        if self._latency is None:
            logger.warning("Metrics not registered; call register_metrics() first.")
            return

        self._latency.labels(operation=operation).observe(seconds)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """
        Start the Prometheus HTTP metrics server.

        The server runs in a daemon thread managed by
        :func:`prometheus_client.start_http_server` and will be torn down
        automatically when the process exits.
        """
        if self._server_started:
            logger.warning("Prometheus metrics server already running on port %d.", self._port)
            return

        self.register_metrics()

        start_http_server(self._port)
        self._server_started = True
        logger.info("Prometheus metrics server started on port %d.", self._port)

    def stop_server(self) -> None:
        """
        Mark the server as stopped.

        Note: ``prometheus_client.start_http_server`` starts a daemon thread
        that cannot be explicitly shut down.  This method resets internal
        state so that a subsequent :meth:`start_server` call will attempt to
        bind the port again (useful in tests or restarts within the same
        process).
        """
        if not self._server_started:
            logger.debug("Metrics server was not running.")
            return

        self._server_started = False
        logger.info("Prometheus metrics server marked as stopped (port %d).", self._port)
