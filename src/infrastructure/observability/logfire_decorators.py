"""Decorators for instrumenting functions with Logfire observability.

This module provides decorators for adding tracing, timing, and logging
to functions throughout the application.
"""

import asyncio
import functools
import time
from typing import Callable, Any, Optional, Dict, TypeVar, Union

import logfire
from logfire import LogfireSpan


T = TypeVar("T")


def traced(
    name: Optional[str] = None,
    span_type: str = "span",
    capture_args: bool = True,
    capture_result: bool = True,
    **extra_attributes: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add distributed tracing to a function.

    Args:
        name: Optional span name (defaults to function name)
        span_type: Type of span (span, log, metric)
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        **extra_attributes: Additional attributes to add to the span

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                attributes = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **extra_attributes,
                }

                if capture_args:
                    # Capture args safely (avoid large objects)
                    attributes["args"] = (
                        _safe_repr(args[:5]) if len(args) > 5 else _safe_repr(args)
                    )
                    attributes["kwargs"] = _safe_repr_dict(kwargs)

                with logfire.span(
                    span_name, _span_type=span_type, **attributes
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        if capture_result:
                            span.set_attribute("result", _safe_repr(result))

                        return result
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error_type", type(e).__name__)
                        span.set_attribute("error_message", str(e))
                        raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                attributes = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **extra_attributes,
                }

                if capture_args:
                    attributes["args"] = (
                        _safe_repr(args[:5]) if len(args) > 5 else _safe_repr(args)
                    )
                    attributes["kwargs"] = _safe_repr_dict(kwargs)

                with logfire.span(
                    span_name, _span_type=span_type, **attributes
                ) as span:
                    try:
                        result = func(*args, **kwargs)

                        if capture_result:
                            logfire.set_attribute("result", _safe_repr(result))

                        return result
                    except Exception as e:
                        logfire.set_attribute("error", True)
                        logfire.set_attribute("error_type", type(e).__name__)
                        logfire.set_attribute("error_message", str(e))
                        raise

            return sync_wrapper

    return decorator


def timed(
    name: Optional[str] = None,
    metric_name: Optional[str] = None,
    success_metric: Optional[str] = None,
    error_metric: Optional[str] = None,
    **extra_tags: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time function execution and track metrics.

    Args:
        name: Optional span name
        metric_name: Optional metric name for timing
        success_metric: Optional metric name for success count
        error_metric: Optional metric name for error count
        **extra_tags: Additional tags for metrics

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or f"timer.{func.__module__}.{func.__name__}"
        timer_metric = metric_name or f"{func.__module__}.{func.__name__}.duration"
        success_counter = success_metric or f"{func.__module__}.{func.__name__}.success"
        error_counter = error_metric or f"{func.__module__}.{func.__name__}.error"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                start_time = time.time()

                with logfire.span(span_name) as span:
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Log timing metric
                        logfire.info(
                            f"metric.{timer_metric}",
                            value=duration,
                            unit="seconds",
                            metric_type="timer",
                            **extra_tags,
                        )

                        # Log success metric
                        logfire.info(
                            f"metric.{success_counter}",
                            value=1,
                            metric_type="counter",
                            **extra_tags,
                        )

                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("success", True)

                        return result
                    except Exception as e:
                        duration = time.time() - start_time

                        # Log error metric
                        logfire.info(
                            f"metric.{error_counter}",
                            value=1,
                            metric_type="counter",
                            error_type=type(e).__name__,
                            **extra_tags,
                        )

                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("success", False)
                        span.set_attribute("error_type", type(e).__name__)

                        raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                start_time = time.time()

                with logfire.span(span_name) as span:
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Log timing metric
                        logfire.info(
                            f"metric.{timer_metric}",
                            value=duration,
                            unit="seconds",
                            metric_type="timer",
                            **extra_tags,
                        )

                        # Log success metric
                        logfire.info(
                            f"metric.{success_counter}",
                            value=1,
                            metric_type="counter",
                            **extra_tags,
                        )

                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("success", True)

                        return result
                    except Exception as e:
                        duration = time.time() - start_time

                        # Log error metric
                        logfire.info(
                            f"metric.{error_counter}",
                            value=1,
                            metric_type="counter",
                            error_type=type(e).__name__,
                            **extra_tags,
                        )

                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_attribute("success", False)
                        span.set_attribute("error_type", type(e).__name__)

                        raise

            return sync_wrapper

    return decorator


def with_span(span_name: str, **attributes: Any) -> Union[LogfireSpan, Callable]:
    """Context manager or decorator for creating a Logfire span.

    Can be used as a context manager:
        with with_span("my_operation", user_id=123):
            do_something()

    Or as a decorator:
        @with_span("my_operation", user_id=123)
        def my_function():
            do_something()

    Args:
        span_name: Name of the span
        **attributes: Attributes to add to the span

    Returns:
        Context manager or decorator
    """
    # If called without parentheses as a decorator
    if callable(span_name):
        func = span_name
        span_name = f"{func.__module__}.{func.__name__}"
        return traced(span_name)(func)

    # Return context manager
    return logfire.span(span_name, **attributes)


def log_event(
    event_type: str, **attributes: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log business events.

    Args:
        event_type: Type of event (e.g., 'user_action', 'api_call')
        **attributes: Event attributes

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        event_name = f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                logfire.info(
                    f"event.{event_type}.{event_name}",
                    event_type=event_type,
                    event_name=event_name,
                    **attributes,
                )
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                logfire.info(
                    f"event.{event_type}.{event_name}",
                    event_type=event_type,
                    event_name=event_name,
                    **attributes,
                )
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _safe_repr(obj: Any, max_length: int = 200) -> str:
    """Safely convert object to string representation.

    Args:
        obj: Object to convert
        max_length: Maximum string length

    Returns:
        String representation
    """
    try:
        repr_str = repr(obj)
        if len(repr_str) > max_length:
            return repr_str[:max_length] + "..."
        return repr_str
    except Exception:
        return f"<{type(obj).__name__} object>"


def _safe_repr_dict(d: Dict[str, Any], max_items: int = 10) -> Dict[str, str]:
    """Safely convert dictionary to string representations.

    Args:
        d: Dictionary to convert
        max_items: Maximum number of items to include

    Returns:
        Dictionary with string values
    """
    result = {}
    for i, (key, value) in enumerate(d.items()):
        if i >= max_items:
            result["..."] = f"({len(d) - max_items} more items)"
            break
        result[str(key)] = _safe_repr(value)

    return result


# Convenience decorators with pre-configured settings
def traced_api_endpoint(name: Optional[str] = None) -> Callable:
    """Trace an API endpoint with standard attributes."""
    return traced(name, span_type="span", operation_type="api_endpoint")


def traced_service_method(name: Optional[str] = None) -> Callable:
    """Trace a service method with standard attributes."""
    return traced(name, span_type="span", operation_type="service_method")


def traced_repository_method(name: Optional[str] = None) -> Callable:
    """Trace a repository method with standard attributes."""
    return traced(name, span_type="span", operation_type="repository_method")


def traced_background_task(name: Optional[str] = None) -> Callable:
    """Trace a background task with standard attributes."""
    return traced(name, span_type="span", operation_type="background_task")


def traced_use_case(name: Optional[str] = None) -> Callable:
    """Trace a use case with standard attributes."""
    return traced(name, span_type="span", operation_type="use_case")
