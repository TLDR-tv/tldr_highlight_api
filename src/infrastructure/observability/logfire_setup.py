"""Logfire configuration and setup for TL;DR Highlight API.

This module handles the initialization and configuration of Pydantic Logfire
for comprehensive observability across the application.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

import logfire
from logfire import LogfireSpan

from src.infrastructure.config import settings

logger = logging.getLogger(__name__)


def configure_logfire(
    app_instance: Optional[Any] = None,
    additional_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Configure and initialize Logfire with application settings.

    Args:
        app_instance: Optional FastAPI app instance for auto-instrumentation
        additional_config: Additional configuration to merge with defaults
    """
    if not settings.logfire_enabled:
        logger.info("Logfire is disabled in configuration")
        return

    # Base configuration
    config = {
        "service_name": settings.logfire_service_name,
        "service_version": settings.logfire_version,
        "environment": settings.logfire_env,
        "console": settings.logfire_console_enabled,
        "log_level": settings.logfire_log_level,
    }

    # Add API key if provided (for production)
    if settings.logfire_api_key:
        config["token"] = settings.logfire_api_key.get_secret_value()

    # Merge with additional config
    if additional_config:
        config.update(additional_config)

    # Configure Logfire
    try:
        logfire.configure(**config)
        logger.info(
            f"Logfire configured successfully for {settings.logfire_service_name}"
        )

        # Set up integrations
        _setup_integrations(app_instance)

        # Log startup event
        logfire.info(
            "TL;DR Highlight API started",
            environment=settings.environment,
            version=settings.app_version,
            logfire_enabled=True,
        )

    except Exception as e:
        logger.error(f"Failed to configure Logfire: {e}")
        # Don't fail the application if Logfire setup fails
        if settings.is_development:
            logger.warning("Continuing without Logfire in development mode")
        else:
            raise


def _setup_integrations(app_instance: Optional[Any] = None) -> None:
    """Set up Logfire integrations for various libraries.

    Args:
        app_instance: Optional FastAPI app instance
    """
    # FastAPI instrumentation
    if app_instance and settings.logfire_capture_headers:
        try:
            logfire.instrument_fastapi(
                app_instance,
                capture_headers=settings.logfire_capture_headers,
                capture_body=settings.logfire_capture_body,
            )
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")

    # SQLAlchemy instrumentation
    if settings.logfire_sql_enabled:
        try:
            # This will be called when engine is created
            logfire.instrument_sqlalchemy()
            logger.info("SQLAlchemy instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument SQLAlchemy: {e}")

    # Redis instrumentation
    if settings.logfire_redis_enabled:
        try:
            logfire.instrument_redis()
            logger.info("Redis instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument Redis: {e}")

    # Celery instrumentation
    if settings.logfire_celery_enabled:
        try:
            logfire.instrument_celery()
            logger.info("Celery instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument Celery: {e}")

    # HTTPX instrumentation for external API calls
    try:
        logfire.instrument_httpx()
        logger.info("HTTPX instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument HTTPX: {e}")


@lru_cache(maxsize=1)
def get_logfire():
    """Get the configured Logfire instance.

    Returns:
        The Logfire module
    """
    return logfire


def create_span(name: str, span_type: str = "span", **attributes: Any) -> LogfireSpan:
    """Create a new Logfire span with attributes.

    Args:
        name: Span name
        span_type: Type of span (span, log, metric)
        **attributes: Additional attributes for the span

    Returns:
        LogfireSpan: The created span
    """
    return logfire.span(name, _span_type=span_type, **attributes)


def log_metric(
    name: str,
    value: float,
    unit: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a metric to Logfire.

    Args:
        name: Metric name
        value: Metric value
        unit: Optional unit of measurement
        tags: Optional tags for the metric
    """
    attributes = {"metric_name": name, "value": value}

    if unit:
        attributes["unit"] = unit

    if tags:
        attributes.update(tags)

    logfire.info(f"metric.{name}", **attributes)


def log_event(event_name: str, event_type: str, **attributes: Any) -> None:
    """Log a business event to Logfire.

    Args:
        event_name: Name of the event
        event_type: Type of event (e.g., 'api_call', 'webhook', 'processing')
        **attributes: Event attributes
    """
    logfire.info(
        f"event.{event_type}.{event_name}",
        event_type=event_type,
        event_name=event_name,
        **attributes,
    )


def set_correlation_id(correlation_id: str) -> None:
    """Set a correlation ID for request tracing.

    Args:
        correlation_id: The correlation ID to set
    """
    # Use with_tags to add correlation ID to all subsequent spans
    return logfire.with_tags(correlation_id=correlation_id)


def add_user_context(
    span: LogfireSpan,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
) -> None:
    """Add user context to the given span.

    Args:
        span: The Logfire span to add context to
        organization_id: Organization ID
        user_id: User ID
        api_key_id: API key ID
    """
    if organization_id:
        span.set_attribute("user.organization_id", organization_id)

    if user_id:
        span.set_attribute("user.id", user_id)

    if api_key_id:
        span.set_attribute("user.api_key_id", api_key_id)


def add_processing_context(
    span: LogfireSpan,
    stream_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    platform: Optional[str] = None,
    processing_stage: Optional[str] = None,
) -> None:
    """Add processing context to the given span.

    Args:
        span: The Logfire span to add context to
        stream_id: Stream ID being processed
        batch_id: Batch ID being processed
        platform: Platform (twitch, youtube, rtmp)
        processing_stage: Current processing stage
    """
    if stream_id:
        span.set_attribute("processing.stream_id", stream_id)

    if batch_id:
        span.set_attribute("processing.batch_id", batch_id)

    if platform:
        span.set_attribute("processing.platform", platform)

    if processing_stage:
        span.set_attribute("processing.stage", processing_stage)
