"""
Logging configuration for the AI Crypto Trading Bot.

Provides a ``setup_logging`` function that wires up console and rotating-file
handlers with consistent formatting across the entire application.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Directory where log files are written.
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")

# Shared format string.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotating-file limits.
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


def setup_logging(
    name: str = "crypto_bot",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str | None = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and rotating-file handlers.

    Args:
        name: Logger name (usually the package or module name).
        console_level: Minimum severity for the console handler.
        file_level: Minimum severity for the file handler.
        log_dir: Directory for log files.  Defaults to ``<project_root>/logs``.

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    log_dir = log_dir or _LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers when ``setup_logging`` is called more than once
    # (e.g. in tests or interactive sessions).
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Let handlers decide what to emit.

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger '%s' initialised (console=%s, file=%s)", name, console_level, file_level)

    return logger
