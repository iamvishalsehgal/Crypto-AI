"""Circuit breaker -- trips after consecutive failures within cooldown window."""

from __future__ import annotations

import time
from typing import Optional

from omnitrade.utils.logger import get_logger

logger = get_logger("omnitrade.circuit_breaker")


class CircuitBreaker:
    """Trips after *threshold* consecutive failures within *window* seconds."""

    def __init__(self, threshold: int = 5, cooldown: int = 300) -> None:
        self.threshold = threshold
        self.cooldown = cooldown  # seconds to wait after tripping
        self._failures: int = 0
        self._tripped_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._tripped_at is None:
            return False
        if time.time() - self._tripped_at > self.cooldown:
            self.reset()
            logger.info("Circuit breaker reset after cooldown")
            return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._tripped_at = time.time()
            logger.critical(
                "CIRCUIT BREAKER TRIPPED after %d failures — pausing for %ds",
                self._failures, self.cooldown,
            )

    def record_success(self) -> None:
        self._failures = 0

    def reset(self) -> None:
        self._failures = 0
        self._tripped_at = None
