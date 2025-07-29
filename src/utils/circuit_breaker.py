"""Circuit breaker pattern implementation for external API resilience.

This module provides a circuit breaker implementation to handle failures
and prevent cascading failures when calling external APIs. It implements
the classic circuit breaker pattern with three states: CLOSED, OPEN, and HALF_OPEN.
"""

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime

from src.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failure threshold exceeded, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    # Counters
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0

    # State tracking
    state_changes: int = 0
    last_state_change: Optional[datetime] = None

    # Performance metrics
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")

    # Recent failures for debugging
    recent_failures: List[str] = field(default_factory=list)

    # Time windows
    window_start: datetime = field(default_factory=datetime.utcnow)

    def add_success(self, response_time: float) -> None:
        """Add a successful call."""
        self.success_count += 1
        self._update_response_time(response_time)

    def add_failure(self, error: str) -> None:
        """Add a failed call."""
        self.failure_count += 1
        self.recent_failures.append(f"{datetime.utcnow().isoformat()}: {error}")

        # Keep only recent failures (last 10)
        if len(self.recent_failures) > 10:
            self.recent_failures.pop(0)

    def add_timeout(self) -> None:
        """Add a timeout."""
        self.timeout_count += 1
        self.failure_count += 1

    def state_changed(self, new_state: CircuitBreakerState) -> None:
        """Record a state change."""
        self.state_changes += 1
        self.last_state_change = datetime.utcnow()

    def _update_response_time(self, response_time: float) -> None:
        """Update response time metrics."""
        if self.success_count == 1:
            self.average_response_time = response_time
        else:
            # Calculate running average
            self.average_response_time = (
                self.average_response_time * (self.success_count - 1) + response_time
            ) / self.success_count

        self.max_response_time = max(self.max_response_time, response_time)
        self.min_response_time = min(self.min_response_time, response_time)

    def reset_window(self) -> None:
        """Reset metrics window."""
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.average_response_time = 0.0
        self.max_response_time = 0.0
        self.min_response_time = float("inf")
        self.recent_failures.clear()
        self.window_start = datetime.utcnow()

    @property
    def total_calls(self) -> int:
        """Get total number of calls."""
        return self.success_count + self.failure_count

    @property
    def failure_rate(self) -> float:
        """Get failure rate as a percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failure_count / self.total_calls) * 100.0

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        return 100.0 - self.failure_rate


class CircuitBreaker:
    """Circuit breaker implementation for external API calls.

    The circuit breaker monitors failures and automatically opens when
    a failure threshold is exceeded, preventing further calls until
    the service recovers.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        timeout_seconds: float = None,
        recovery_timeout_seconds: float = None,
        expected_exceptions: tuple = None,
        success_threshold: int = 3,
        window_size_seconds: int = 60,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Timeout for individual calls
            recovery_timeout_seconds: Time to wait before trying to recover
            expected_exceptions: Exceptions that count as failures
            success_threshold: Consecutive successes needed to close circuit
            window_size_seconds: Time window for counting failures
        """
        self.name = name
        self.failure_threshold = (
            failure_threshold or settings.circuit_breaker_failure_threshold
        )
        self.timeout_seconds = (
            timeout_seconds or settings.circuit_breaker_timeout_seconds
        )
        self.recovery_timeout_seconds = (
            recovery_timeout_seconds
            or settings.circuit_breaker_recovery_timeout_seconds
        )
        self.expected_exceptions = expected_exceptions or (Exception,)
        self.success_threshold = success_threshold
        self.window_size_seconds = window_size_seconds

        # State
        self._state = CircuitBreakerState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._consecutive_successes = 0

        # Metrics
        self.metrics = CircuitBreakerMetrics()

        # Thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"Created circuit breaker '{name}' with threshold {self.failure_threshold}"
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitBreakerState.HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerError: If circuit is open
            Any exception raised by the function
        """
        async with self._lock:
            await self._check_and_update_state()

            if self._state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Failure rate: {self.metrics.failure_rate:.1f}%"
                )

        # Execute the function
        start_time = time.time()
        try:
            # Apply timeout if specified
            if self.timeout_seconds:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=self.timeout_seconds
                    )
                else:
                    # For sync functions, run in executor with timeout
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args, **kwargs),
                        timeout=self.timeout_seconds,
                    )
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

            # Record success
            response_time = time.time() - start_time
            await self._record_success(response_time)
            return result

        except asyncio.TimeoutError:
            await self._record_timeout()
            raise
        except self.expected_exceptions as e:
            await self._record_failure(str(e))
            raise

    async def _check_and_update_state(self) -> None:
        """Check and update circuit breaker state."""
        now = time.time()

        # Reset metrics window if needed
        window_age = (datetime.utcnow() - self.metrics.window_start).total_seconds()
        if window_age > self.window_size_seconds:
            logger.debug(f"Resetting metrics window for circuit breaker '{self.name}'")
            self.metrics.reset_window()

        if self._state == CircuitBreakerState.CLOSED:
            # Check if we should open
            if (
                self.metrics.total_calls >= self.failure_threshold
                and self.metrics.failure_rate >= 50
            ):  # 50% failure rate threshold
                await self._open_circuit()

        elif self._state == CircuitBreakerState.OPEN:
            # Check if we should try half-open
            if (
                self._last_failure_time
                and now - self._last_failure_time >= self.recovery_timeout_seconds
            ):
                await self._half_open_circuit()

        elif self._state == CircuitBreakerState.HALF_OPEN:
            # Check if we should close or re-open
            if self._consecutive_successes >= self.success_threshold:
                await self._close_circuit()
            elif self.metrics.failure_count > 0:
                await self._open_circuit()

    async def _record_success(self, response_time: float) -> None:
        """Record a successful call."""
        async with self._lock:
            self.metrics.add_success(response_time)
            self._consecutive_successes += 1

            logger.debug(
                f"Circuit breaker '{self.name}' recorded success. "
                f"Response time: {response_time:.3f}s, "
                f"Consecutive successes: {self._consecutive_successes}"
            )

    async def _record_failure(self, error: str) -> None:
        """Record a failed call."""
        async with self._lock:
            self.metrics.add_failure(error)
            self._consecutive_successes = 0
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {error}. "
                f"Failure rate: {self.metrics.failure_rate:.1f}%"
            )

    async def _record_timeout(self) -> None:
        """Record a timeout."""
        async with self._lock:
            self.metrics.add_timeout()
            self._consecutive_successes = 0
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded timeout. "
                f"Failure rate: {self.metrics.failure_rate:.1f}%"
            )

    async def _open_circuit(self) -> None:
        """Open the circuit."""
        if self._state != CircuitBreakerState.OPEN:
            self._state = CircuitBreakerState.OPEN
            self.metrics.state_changed(self._state)

            logger.error(
                f"Circuit breaker '{self.name}' OPENED. "
                f"Failure rate: {self.metrics.failure_rate:.1f}% "
                f"({self.metrics.failure_count}/{self.metrics.total_calls})"
            )

    async def _half_open_circuit(self) -> None:
        """Half-open the circuit for testing."""
        if self._state != CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.HALF_OPEN
            self._consecutive_successes = 0
            self.metrics.state_changed(self._state)

            logger.info(
                f"Circuit breaker '{self.name}' HALF-OPENED for testing recovery"
            )

    async def _close_circuit(self) -> None:
        """Close the circuit."""
        if self._state != CircuitBreakerState.CLOSED:
            self._state = CircuitBreakerState.CLOSED
            self.metrics.state_changed(self._state)
            self.metrics.reset_window()  # Fresh start

            logger.info(f"Circuit breaker '{self.name}' CLOSED. Service recovered.")

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._consecutive_successes = 0
            self._last_failure_time = None
            self.metrics.reset_window()

            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dict with current statistics
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "consecutive_successes": self._consecutive_successes,
            "last_failure_time": self._last_failure_time,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_count": self.metrics.success_count,
                "failure_count": self.metrics.failure_count,
                "timeout_count": self.metrics.timeout_count,
                "failure_rate": self.metrics.failure_rate,
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "max_response_time": self.metrics.max_response_time,
                "min_response_time": (
                    self.metrics.min_response_time
                    if self.metrics.min_response_time != float("inf")
                    else 0.0
                ),
                "state_changes": self.metrics.state_changes,
                "last_state_change": (
                    self.metrics.last_state_change.isoformat()
                    if self.metrics.last_state_change
                    else None
                ),
                "window_start": self.metrics.window_start.isoformat(),
                "recent_failures": self.metrics.recent_failures[-5:],  # Last 5 failures
            },
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize the registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Name of the circuit breaker
            **kwargs: Additional arguments for circuit breaker creation

        Returns:
            CircuitBreaker instance
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
                logger.info(f"Created new circuit breaker: {name}")

            return self._breakers[name]

    async def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker from the registry.

        Args:
            name: Name of the circuit breaker to remove

        Returns:
            True if breaker was removed, False if not found
        """
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                return True
            return False

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()
            logger.info("Reset all circuit breakers")

    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            Dict mapping breaker names to their statistics
        """
        async with self._lock:
            return {
                name: breaker.get_stats() for name, breaker in self._breakers.items()
            }

    def list_breakers(self) -> List[str]:
        """List all circuit breaker names.

        Returns:
            List of circuit breaker names
        """
        return list(self._breakers.keys())


# Global registry instance
_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get a circuit breaker from the global registry.

    Args:
        name: Name of the circuit breaker
        **kwargs: Additional arguments for circuit breaker creation

    Returns:
        CircuitBreaker instance
    """
    return await _registry.get_breaker(name, **kwargs)


def circuit_breaker(
    name: str = None,
    failure_threshold: int = None,
    timeout_seconds: float = None,
    recovery_timeout_seconds: float = None,
    expected_exceptions: tuple = None,
):
    """Decorator for applying circuit breaker to functions.

    Args:
        name: Name for the circuit breaker (defaults to function name)
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Timeout for individual calls
        recovery_timeout_seconds: Time to wait before trying to recover
        expected_exceptions: Exceptions that count as failures

    Returns:
        Decorated function
    """

    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                breaker = await get_circuit_breaker(
                    breaker_name,
                    failure_threshold=failure_threshold,
                    timeout_seconds=timeout_seconds,
                    recovery_timeout_seconds=recovery_timeout_seconds,
                    expected_exceptions=expected_exceptions,
                )
                return await breaker.call(func, *args, **kwargs)

            return async_wrapper
        else:

            async def sync_wrapper(*args, **kwargs):
                breaker = await get_circuit_breaker(
                    breaker_name,
                    failure_threshold=failure_threshold,
                    timeout_seconds=timeout_seconds,
                    recovery_timeout_seconds=recovery_timeout_seconds,
                    expected_exceptions=expected_exceptions,
                )
                return await breaker.call(func, *args, **kwargs)

            return sync_wrapper

    return decorator
