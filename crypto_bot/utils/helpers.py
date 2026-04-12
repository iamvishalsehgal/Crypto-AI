"""
General-purpose utility functions used across the crypto bot.
"""

from __future__ import annotations

import functools
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

def timestamp_to_datetime(ts: int | float) -> datetime:
    """
    Convert a Unix millisecond timestamp to a timezone-aware UTC datetime.

    Args:
        ts: Unix timestamp in **milliseconds**.

    Returns:
        A :class:`datetime.datetime` in UTC.
    """
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert a datetime to a Unix millisecond timestamp.

    If *dt* is naive (no tzinfo) it is assumed to be UTC.

    Args:
        dt: The datetime to convert.

    Returns:
        Unix timestamp in milliseconds (integer).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.

    Args:
        prices: A :class:`pandas.Series` of prices (must be > 0).

    Returns:
        A :class:`pandas.Series` of log returns with the first entry as NaN.
    """
    return np.log(prices / prices.shift(1))


def normalize_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply min-max normalisation to the specified columns of *df*.

    Columns whose min equals max are set to 0.0 to avoid division by zero.

    Args:
        df: Input dataframe (not modified in place).
        columns: Column names to normalise.  Defaults to all numeric columns.

    Returns:
        A copy of *df* with the selected columns scaled to [0, 1].
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min == col_max:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)

    return df


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Divide *a* by *b*, returning *default* when *b* is zero or the result
    is not finite.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value returned on division-by-zero or non-finite result.

    Returns:
        ``a / b`` or *default*.
    """
    if b == 0:
        return default
    result = a / b
    if not math.isfinite(result):
        return default
    return result


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_on_exception(
    func: Callable | None = None,
    *,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries *func* on failure with exponential backoff.

    Can be used with or without arguments::

        @retry_on_exception
        def fetch_data(): ...

        @retry_on_exception(max_retries=5, delay=2)
        def fetch_data(): ...

    Args:
        func: The function to wrap (supplied automatically when used without
            parentheses).
        max_retries: Total number of attempts before giving up.
        delay: Base delay in seconds (doubled after each failure).
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        The wrapped function.

    Raises:
        The last caught exception if all retries are exhausted.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: BaseException | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= 2  # exponential backoff

            # All retries exhausted -- re-raise.
            raise last_exception  # type: ignore[misc]

        return wrapper

    # Allow bare ``@retry_on_exception`` (no parentheses).
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def chunk_list(lst: Sequence[T], n: int) -> List[List[T]]:
    """
    Split *lst* into consecutive chunks of at most *n* elements.

    Args:
        lst: The sequence to split.
        n: Maximum chunk size (must be >= 1).

    Returns:
        A list of lists.

    Raises:
        ValueError: If *n* < 1.
    """
    if n < 1:
        raise ValueError(f"Chunk size must be >= 1, got {n}")
    return [list(lst[i : i + n]) for i in range(0, len(lst), n)]
