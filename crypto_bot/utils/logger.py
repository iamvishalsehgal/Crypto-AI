"""
Convenience wrapper around the project logging configuration.

Usage::

    from crypto_bot.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting data collection")
"""

import logging

from crypto_bot.config.logging_config import setup_logging


def get_logger(name: str = "crypto_bot") -> logging.Logger:
    """
    Return a configured logger for the given *name*.

    On the first call the root project logger is initialised via
    :func:`crypto_bot.config.logging_config.setup_logging`.  Subsequent calls
    return child loggers that inherit the same handlers.

    Args:
        name: Dot-delimited logger name, typically ``__name__``.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    # Ensure the root project logger is set up once.
    setup_logging()

    return logging.getLogger(name)
