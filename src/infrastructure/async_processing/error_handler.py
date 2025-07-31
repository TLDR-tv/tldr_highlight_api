"""
Error Handler for async processing pipeline.

This module provides comprehensive error handling functionality including
exponential backoff retry logic, circuit breakers, error categorization,
and recovery strategies for robust async processing.
"""

import asyncio
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

import structlog
from celery.exceptions import WorkerLostError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from src.infrastructure.cache import get_redis_client
from src.infrastructure.config import get_settings


logger = structlog.get_logger(__name__)
settings = get_settings()


class ErrorCategory(str, Enum):
    """Categories of errors for different handling strategies."""

    TRANSIENT = "transient"  # Temporary errors that should be retried
    PERMANENT = "permanent"  # Permanent errors that should not be retried
    RATE_LIMIT = "rate_limit"  # Rate limiting errors with special handling
    NETWORK = "network"  # Network connectivity issues
    DEPENDENCY = "dependency"  # External service dependency failures
    RESOURCE = "resource"  # Resource exhaustion (memory, disk, etc.)
    CONFIGURATION = "configuration"  # Configuration or setup errors
    UNKNOWN = "unknown"  # Unclassified errors


class RetryStrategy(str, Enum):
    """Retry strategies for different error types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


class CircuitBreakerState(str, Enum):
    """States for circuit breaker pattern."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


class ErrorHandler:
    """
    Comprehensive error handler with circuit breakers and retry strategies.

    Provides intelligent error handling with categorization, retry logic,
    circuit breaker pattern, and comprehensive logging and monitoring.
    """

    def __init__(self):
        """Initialize error handler with Redis client and configuration."""
        self.redis_client = get_redis_client()

        # Redis key prefixes
        self.circuit_breaker_prefix = "circuit_breaker:"
        self.error_stats_prefix = "error_stats:"
        self.retry_count_prefix = "retry_count:"

        # Default retry configuration
        self.default_max_retries = 3
        self.default_base_delay = 1.0
        self.default_max_delay = 300.0  # 5 minutes
        self.default_backoff_factor = 2.0
        self.default_jitter = True

        # Circuit breaker configuration
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_recovery_timeout = 60  # seconds
        self.circuit_breaker_success_threshold = 3

        # Error classification rules
        self.error_classifications = {
            # Transient errors that should be retried
            ConnectionError: ErrorCategory.NETWORK,
            TimeoutError: ErrorCategory.NETWORK,
            OSError: ErrorCategory.NETWORK,
            # Permanent errors that should not be retried
            ValueError: ErrorCategory.PERMANENT,
            TypeError: ErrorCategory.PERMANENT,
            AttributeError: ErrorCategory.PERMANENT,
            # Resource errors
            MemoryError: ErrorCategory.RESOURCE,
            # Worker errors
            WorkerLostError: ErrorCategory.TRANSIENT,
        }

    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        Classify an error to determine appropriate handling strategy.

        Args:
            error: Exception to classify

        Returns:
            ErrorCategory: Classification of the error
        """
        error_type = type(error)

        # Check direct type mapping
        if error_type in self.error_classifications:
            return self.error_classifications[error_type]

        # Check error message patterns
        error_message = str(error).lower()

        if any(
            pattern in error_message
            for pattern in ["connection", "timeout", "network", "dns", "socket"]
        ):
            return ErrorCategory.NETWORK

        if any(
            pattern in error_message
            for pattern in ["rate limit", "throttle", "quota", "too many requests"]
        ):
            return ErrorCategory.RATE_LIMIT

        if any(
            pattern in error_message
            for pattern in ["memory", "disk space", "resource", "capacity"]
        ):
            return ErrorCategory.RESOURCE

        if any(
            pattern in error_message
            for pattern in ["config", "setting", "parameter", "missing"]
        ):
            return ErrorCategory.CONFIGURATION

        # Check parent classes
        for error_class, category in self.error_classifications.items():
            if isinstance(error, error_class):
                return category

        return ErrorCategory.UNKNOWN

    def get_retry_strategy(self, error_category: ErrorCategory) -> RetryStrategy:
        """
        Get the appropriate retry strategy for an error category.

        Args:
            error_category: Category of the error

        Returns:
            RetryStrategy: Recommended retry strategy
        """
        strategy_mapping = {
            ErrorCategory.TRANSIENT: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorCategory.NETWORK: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorCategory.DEPENDENCY: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorCategory.RATE_LIMIT: RetryStrategy.LINEAR_BACKOFF,
            ErrorCategory.RESOURCE: RetryStrategy.FIXED_DELAY,
            ErrorCategory.PERMANENT: RetryStrategy.NO_RETRY,
            ErrorCategory.CONFIGURATION: RetryStrategy.NO_RETRY,
            ErrorCategory.UNKNOWN: RetryStrategy.EXPONENTIAL_BACKOFF,
        }

        return strategy_mapping.get(error_category, RetryStrategy.EXPONENTIAL_BACKOFF)

    def should_retry(
        self, error: Exception, attempt_count: int, max_retries: Optional[int] = None
    ) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error: Exception that occurred
            attempt_count: Number of attempts already made
            max_retries: Maximum number of retries allowed

        Returns:
            bool: True if should retry, False otherwise
        """
        max_retries = max_retries or self.default_max_retries

        # Check attempt count
        if attempt_count >= max_retries:
            return False

        # Check error category
        error_category = self.classify_error(error)
        retry_strategy = self.get_retry_strategy(error_category)

        if retry_strategy == RetryStrategy.NO_RETRY:
            return False

        return True

    def calculate_delay(
        self,
        attempt_count: int,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = None,
        max_delay: float = None,
        backoff_factor: float = None,
        jitter: bool = None,
    ) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt_count: Number of previous attempts
            strategy: Retry strategy to use
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Backoff multiplication factor
            jitter: Whether to add random jitter

        Returns:
            float: Delay in seconds
        """
        base_delay = base_delay or self.default_base_delay
        max_delay = max_delay or self.default_max_delay
        backoff_factor = backoff_factor or self.default_backoff_factor
        jitter = jitter if jitter is not None else self.default_jitter

        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (backoff_factor**attempt_count)
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (attempt_count + 1)
        elif strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        elif strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        else:
            delay = base_delay

        # Apply maximum delay limit
        delay = min(delay, max_delay)

        # Add jitter to prevent thundering herd
        if jitter and delay > 0:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount

        return delay

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        retry_func: Optional[Callable] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Handle an error with comprehensive error processing.

        Args:
            error: Exception that occurred
            context: Context information (stream_id, task_name, etc.)
            retry_func: Function to call for retry
            max_retries: Maximum number of retries

        Returns:
            Dict[str, Any]: Error handling result
        """
        logger.error(
            "Handling error",
            error=str(error),
            error_type=type(error).__name__,
            context=context,
        )

        try:
            # Classify error
            error_category = self.classify_error(error)
            retry_strategy = self.get_retry_strategy(error_category)

            # Get current attempt count
            attempt_count = context.get("attempt_count", 0)

            # Check circuit breaker
            service_name = context.get("service_name", "default")
            if self._is_circuit_breaker_open(service_name):
                logger.warning("Circuit breaker is open", service=service_name)
                return {
                    "action": "circuit_breaker_open",
                    "should_retry": False,
                    "error_category": error_category.value,
                    "circuit_breaker_open": True,
                }

            # Record error for circuit breaker
            self._record_error(service_name, error)

            # Log error statistics
            self._log_error_stats(error_category, context)

            # Determine if should retry
            should_retry = self.should_retry(error, attempt_count, max_retries)

            if not should_retry:
                logger.info(
                    "Not retrying error",
                    error_category=error_category.value,
                    attempt_count=attempt_count,
                )
                return {
                    "action": "no_retry",
                    "should_retry": False,
                    "error_category": error_category.value,
                    "reason": "max_retries_exceeded"
                    if attempt_count >= (max_retries or self.default_max_retries)
                    else "permanent_error",
                }

            # Calculate retry delay
            delay = self.calculate_delay(attempt_count, retry_strategy)

            # Prepare retry result
            retry_result = {
                "action": "retry",
                "should_retry": True,
                "error_category": error_category.value,
                "retry_strategy": retry_strategy.value,
                "delay_seconds": delay,
                "attempt_count": attempt_count + 1,
                "max_retries": max_retries or self.default_max_retries,
            }

            # Execute retry if function provided
            if retry_func:
                logger.info(
                    "Executing retry",
                    delay_seconds=delay,
                    attempt_count=attempt_count + 1,
                )

                # Wait for delay
                if delay > 0:
                    await asyncio.sleep(delay)

                try:
                    # Attempt retry
                    retry_result["retry_result"] = await retry_func()
                    retry_result["retry_successful"] = True

                    # Record success for circuit breaker
                    self._record_success(service_name)

                except Exception as retry_error:
                    logger.error("Retry failed", retry_error=str(retry_error))
                    retry_result["retry_successful"] = False
                    retry_result["retry_error"] = str(retry_error)

                    # Record failure for circuit breaker
                    self._record_error(service_name, retry_error)

            return retry_result

        except Exception as handling_error:
            logger.error("Error in error handling", handling_error=str(handling_error))
            return {
                "action": "error_in_handling",
                "should_retry": False,
                "error": str(handling_error),
            }

    def create_retry_decorator(
        self,
        max_retries: int = None,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = None,
        max_delay: float = None,
        error_types: Tuple[Type[Exception], ...] = None,
    ):
        """
        Create a retry decorator with specified parameters.

        Args:
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            error_types: Exception types to retry on

        Returns:
            Decorator function
        """
        max_retries = max_retries or self.default_max_retries
        base_delay = base_delay or self.default_base_delay
        max_delay = max_delay or self.default_max_delay
        error_types = error_types or (Exception,)

        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            wait_strategy = wait_exponential(multiplier=base_delay, max=max_delay)
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            wait_strategy = wait_exponential(
                multiplier=base_delay,
                exp_base=1,  # Linear
                max=max_delay,
            )
        else:
            wait_strategy = wait_exponential(multiplier=base_delay, max=max_delay)

        return retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_strategy,
            retry=retry_if_exception_type(error_types),
            before_sleep=before_sleep_log(logger, "INFO"),
            after=after_log(logger, "INFO"),
            reraise=True,
        )

    def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service."""
        try:
            cb_key = f"{self.circuit_breaker_prefix}{service_name}"
            cb_data = self.redis_client.hgetall(cb_key)

            if not cb_data:
                return False

            state = cb_data.get("state", CircuitBreakerState.CLOSED.value)

            if state == CircuitBreakerState.OPEN.value:
                # Check if recovery timeout has passed
                last_failure = cb_data.get("last_failure_time")
                if last_failure:
                    last_failure_time = datetime.fromisoformat(last_failure)
                    if datetime.utcnow() - last_failure_time > timedelta(
                        seconds=self.circuit_breaker_recovery_timeout
                    ):
                        # Move to half-open state
                        self._set_circuit_breaker_state(
                            service_name, CircuitBreakerState.HALF_OPEN
                        )
                        return False

                return True

            return False

        except Exception as e:
            logger.error(
                "Failed to check circuit breaker", service=service_name, error=str(e)
            )
            return False

    def _record_error(self, service_name: str, error: Exception) -> None:
        """Record an error for circuit breaker tracking."""
        try:
            cb_key = f"{self.circuit_breaker_prefix}{service_name}"

            # Get current state
            cb_data = self.redis_client.hgetall(cb_key)
            current_state = cb_data.get("state", CircuitBreakerState.CLOSED.value)
            failure_count = int(cb_data.get("failure_count", 0))
            success_count = int(cb_data.get("success_count", 0))

            # Increment failure count
            failure_count += 1
            success_count = 0  # Reset success count on failure

            # Update circuit breaker data
            update_data = {
                "failure_count": str(failure_count),
                "success_count": str(success_count),
                "last_failure_time": datetime.utcnow().isoformat(),
                "last_error": str(error)[:200],  # Limit error message length
            }

            # Check if should open circuit breaker
            if (
                current_state == CircuitBreakerState.CLOSED.value
                and failure_count >= self.circuit_breaker_failure_threshold
            ):
                update_data["state"] = CircuitBreakerState.OPEN.value
                logger.warning(
                    "Circuit breaker opened",
                    service=service_name,
                    failure_count=failure_count,
                )
            elif current_state == CircuitBreakerState.HALF_OPEN.value:
                # Failure in half-open state - go back to open
                update_data["state"] = CircuitBreakerState.OPEN.value
                logger.warning("Circuit breaker reopened", service=service_name)
            else:
                update_data["state"] = current_state

            # Update in Redis
            self.redis_client.hset(cb_key, mapping=update_data)
            self.redis_client.expire(cb_key, 3600)  # 1 hour expiration

        except Exception as e:
            logger.error(
                "Failed to record error for circuit breaker",
                service=service_name,
                error=str(e),
            )

    def _record_success(self, service_name: str) -> None:
        """Record a success for circuit breaker tracking."""
        try:
            cb_key = f"{self.circuit_breaker_prefix}{service_name}"

            # Get current state
            cb_data = self.redis_client.hgetall(cb_key)
            current_state = cb_data.get("state", CircuitBreakerState.CLOSED.value)
            success_count = int(cb_data.get("success_count", 0))
            failure_count = int(cb_data.get("failure_count", 0))

            # Increment success count
            success_count += 1

            # Update circuit breaker data
            update_data = {
                "success_count": str(success_count),
                "last_success_time": datetime.utcnow().isoformat(),
            }

            # Check if should close circuit breaker
            if (
                current_state == CircuitBreakerState.HALF_OPEN.value
                and success_count >= self.circuit_breaker_success_threshold
            ):
                update_data["state"] = CircuitBreakerState.CLOSED.value
                update_data["failure_count"] = "0"  # Reset failure count
                logger.info(
                    "Circuit breaker closed",
                    service=service_name,
                    success_count=success_count,
                )
            else:
                update_data["state"] = current_state
                update_data["failure_count"] = str(failure_count)

            # Update in Redis
            self.redis_client.hset(cb_key, mapping=update_data)
            self.redis_client.expire(cb_key, 3600)  # 1 hour expiration

        except Exception as e:
            logger.error(
                "Failed to record success for circuit breaker",
                service=service_name,
                error=str(e),
            )

    def _set_circuit_breaker_state(
        self, service_name: str, state: CircuitBreakerState
    ) -> None:
        """Set circuit breaker state."""
        try:
            cb_key = f"{self.circuit_breaker_prefix}{service_name}"
            self.redis_client.hset(cb_key, "state", state.value)
            self.redis_client.expire(cb_key, 3600)  # 1 hour expiration

        except Exception as e:
            logger.error(
                "Failed to set circuit breaker state",
                service=service_name,
                error=str(e),
            )

    def _log_error_stats(
        self, error_category: ErrorCategory, context: Dict[str, Any]
    ) -> None:
        """Log error statistics for monitoring."""
        try:
            stats_key = f"{self.error_stats_prefix}{error_category.value}"

            # Increment error count
            self.redis_client.hincrby(stats_key, "count", 1)
            self.redis_client.hset(
                stats_key, "last_occurrence", datetime.utcnow().isoformat()
            )

            # Set expiration
            self.redis_client.expire(stats_key, 86400)  # 24 hours

            # Log context-specific stats
            if "stream_id" in context:
                stream_stats_key = (
                    f"{self.error_stats_prefix}stream:{context['stream_id']}"
                )
                self.redis_client.hincrby(stream_stats_key, error_category.value, 1)
                self.redis_client.expire(stream_stats_key, 86400)

        except Exception as e:
            logger.error("Failed to log error stats", error=str(e))

    def get_error_statistics(
        self, service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get error statistics for monitoring and alerting.

        Args:
            service_name: Optional service name to filter by

        Returns:
            Dict[str, Any]: Error statistics
        """
        try:
            stats = {}

            # Get error category statistics
            for category in ErrorCategory:
                stats_key = f"{self.error_stats_prefix}{category.value}"
                category_stats = self.redis_client.hgetall(stats_key)

                if category_stats:
                    stats[category.value] = {
                        "count": int(category_stats.get("count", 0)),
                        "last_occurrence": category_stats.get("last_occurrence"),
                    }

            # Get circuit breaker states
            if service_name:
                cb_key = f"{self.circuit_breaker_prefix}{service_name}"
                cb_data = self.redis_client.hgetall(cb_key)

                if cb_data:
                    stats["circuit_breaker"] = {
                        "service": service_name,
                        "state": cb_data.get("state", CircuitBreakerState.CLOSED.value),
                        "failure_count": int(cb_data.get("failure_count", 0)),
                        "success_count": int(cb_data.get("success_count", 0)),
                        "last_failure_time": cb_data.get("last_failure_time"),
                        "last_success_time": cb_data.get("last_success_time"),
                        "last_error": cb_data.get("last_error"),
                    }

            return stats

        except Exception as e:
            logger.error(
                "Failed to get error statistics", service=service_name, error=str(e)
            )
            return {"error": str(e)}
