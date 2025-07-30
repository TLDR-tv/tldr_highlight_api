"""Observability infrastructure for TL;DR Highlight API.

This module provides comprehensive observability through Pydantic Logfire,
including distributed tracing, structured logging, and metrics collection.
"""

from .logfire_setup import configure_logfire, get_logfire
from .logfire_decorators import traced, timed, with_span
from .logfire_metrics import metrics, MetricsCollector
from .logfire_middleware import LogfireMiddleware

__all__ = [
    "configure_logfire",
    "get_logfire",
    "traced",
    "timed",
    "with_span",
    "metrics",
    "MetricsCollector",
    "LogfireMiddleware",
]
