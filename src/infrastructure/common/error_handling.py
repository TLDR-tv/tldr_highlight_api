"""Pythonic error handling decorators and utilities.

This module provides decorators and utilities for consistent error handling
across the application, following Python idioms and best practices.
"""

import asyncio
import logging
from functools import wraps
from typing import TypeVar, Callable, Any, Optional, Type, Tuple
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from src.domain.exceptions import (
    DomainError,
    DuplicateEntityError,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ErrorMappingRule:
    """Rule for mapping infrastructure errors to domain errors."""

    source_exception: Type[Exception]
    target_exception: Type[DomainError]
    message_template: Optional[str] = None
    should_log: bool = True
    log_level: str = "error"


# Default error mapping rules
DEFAULT_ERROR_MAPPINGS = [
    ErrorMappingRule(
        source_exception=IntegrityError,
        target_exception=DuplicateEntityError,
        message_template="Duplicate entity: {original_message}",
    ),
    ErrorMappingRule(
        source_exception=SQLAlchemyError,
        target_exception=DomainError,
        message_template="Database error: {original_message}",
    ),
]


def handle_repository_errors(
    error_mappings: Optional[list[ErrorMappingRule]] = None,
) -> Callable[[F], F]:
    """Decorator for handling repository layer errors.

    Maps infrastructure exceptions to domain exceptions following
    clean architecture principles.

    Args:
        error_mappings: Custom error mapping rules

    Returns:
        Decorated function
    """
    mappings = error_mappings or DEFAULT_ERROR_MAPPINGS

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    _handle_error(e, mappings, func.__name__)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _handle_error(e, mappings, func.__name__)

            return sync_wrapper

    return decorator


def _handle_error(
    error: Exception, mappings: list[ErrorMappingRule], func_name: str
) -> None:
    """Handle error using mapping rules."""
    for rule in mappings:
        if isinstance(error, rule.source_exception):
            if rule.should_log:
                log_level = getattr(logger, rule.log_level.lower())
                log_level(f"Error in {func_name}: {error}")

            message = (
                rule.message_template.format(original_message=str(error))
                if rule.message_template
                else str(error)
            )

            raise rule.target_exception(message) from error

    # If no mapping found, re-raise original exception
    logger.error(f"Unhandled error in {func_name}: {error}")
    raise


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator for retrying operations on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Multiplier for delay after each failure
        exceptions: Exception types to retry on

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed "
                                f"for {func.__name__}: {e}. Retrying in {current_delay}s"
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}"
                            )

                raise last_exception

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                import time

                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed "
                                f"for {func.__name__}: {e}. Retrying in {current_delay}s"
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}"
                            )

                raise last_exception

            return sync_wrapper

    return decorator


def log_exceptions(
    level: str = "error",
    reraise: bool = True,
    message_template: str = "Exception in {func_name}: {error}",
) -> Callable[[F], F]:
    """Decorator for logging exceptions.

    Args:
        level: Log level (debug, info, warning, error, critical)
        reraise: Whether to re-raise the exception after logging
        message_template: Template for log message

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        log_func = getattr(logger, level.lower())

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log_func(
                        message_template.format(func_name=func.__name__, error=str(e))
                    )
                    if reraise:
                        raise

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log_func(
                        message_template.format(func_name=func.__name__, error=str(e))
                    )
                    if reraise:
                        raise

            return sync_wrapper

    return decorator


@contextmanager
def handle_errors(
    error_mappings: Optional[list[ErrorMappingRule]] = None,
    context_name: str = "operation",
):
    """Context manager for error handling.

    Args:
        error_mappings: Custom error mapping rules
        context_name: Name of the operation for logging
    """
    mappings = error_mappings or DEFAULT_ERROR_MAPPINGS

    try:
        yield
    except Exception as e:
        _handle_error(e, mappings, context_name)


@asynccontextmanager
async def handle_errors_async(
    error_mappings: Optional[list[ErrorMappingRule]] = None,
    context_name: str = "operation",
):
    """Async context manager for error handling.

    Args:
        error_mappings: Custom error mapping rules
        context_name: Name of the operation for logging
    """
    mappings = error_mappings or DEFAULT_ERROR_MAPPINGS

    try:
        yield
    except Exception as e:
        _handle_error(e, mappings, context_name)


def safe_call(
    func: Callable,
    *args,
    default_return: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Safely call a function with error handling.

    Args:
        func: Function to call
        *args: Positional arguments
        default_return: Value to return on error
        exceptions: Exception types to catch
        log_errors: Whether to log caught errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if log_errors:
            logger.error(f"Error in safe_call to {func.__name__}: {e}")
        return default_return


async def safe_call_async(
    func: Callable,
    *args,
    default_return: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Safely call an async function with error handling.

    Args:
        func: Async function to call
        *args: Positional arguments
        default_return: Value to return on error
        exceptions: Exception types to catch
        log_errors: Whether to log caught errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return await func(*args, **kwargs)
    except exceptions as e:
        if log_errors:
            logger.error(f"Error in safe_call_async to {func.__name__}: {e}")
        return default_return


# Convenience decorators for common patterns
handle_db_errors = handle_repository_errors()
retry_db_operation = retry_on_failure(max_attempts=3, exceptions=(SQLAlchemyError,))
log_and_reraise = log_exceptions()
