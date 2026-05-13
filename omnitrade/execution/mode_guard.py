"""
Validate live/paper trading mode and return the effective sandbox flag.

Extracted from the identical safety-lock blocks that previously lived in
both :class:`TradeExecutor` and :class:`StockExecutor`.  See ADR 0001.
"""

from omnitrade.config.settings import Settings
from omnitrade.utils.logger import get_logger

logger = get_logger(__name__)


def validate_trading_mode(settings: Settings) -> bool:
    """Validate ``settings.trading_mode`` and return effective sandbox flag.

    Returns
    -------
    bool
        ``True`` if the system should run in paper/sandbox mode,
        ``False`` for live trading.

    Raises
    ------
    RuntimeError
        If *trading_mode* is ``"live"`` but ``live_trading_confirmed``
        is not ``"true"``, or if the mode string is unrecognised.
    """
    mode = settings.trading_mode

    if mode == "paper":
        return True

    if mode == "live":
        if settings.live_trading_confirmed != "true":
            raise RuntimeError(
                "LIVE TRADING MODE requires LIVE_TRADING_CONFIRMED=true. "
                "Set this environment variable to acknowledge that you "
                "want real orders to be placed."
            )
        logger.critical(
            "!!!!!!!!!! EXECUTOR INITIALISED IN LIVE MODE — "
            "REAL ORDERS WILL BE PLACED !!!!!!!!!!"
        )
        return False

    raise RuntimeError(f"Invalid trading_mode: {mode!r}")
