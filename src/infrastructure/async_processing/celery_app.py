"""
Celery application configuration for TL;DR Highlight API.

This module configures the Celery application with Redis as message broker,
sets up task routing, serialization, monitoring, and enterprise-grade features
for reliable asynchronous processing.
"""

import os
from typing import Any, Dict

from celery import Celery
from celery.schedules import crontab
from celery.signals import after_setup_logger, worker_process_init, worker_shutdown
from kombu import Queue, Exchange

from src.infrastructure.config import get_settings


# Get settings instance
settings = get_settings()

# Create Celery app instance
celery_app = Celery(
    "tldr_highlight_api",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.infrastructure.async_processing.tasks",
        "src.infrastructure.async_processing.workflow",
    ],
)


class CeleryConfig:
    """Celery configuration class with enterprise-grade settings."""

    # Broker and backend settings
    broker_url = settings.celery_broker_url
    result_backend = settings.celery_result_backend

    # Task execution settings
    task_always_eager = settings.celery_task_always_eager
    task_eager_propagates = True
    task_acks_late = settings.celery_task_acks_late
    task_reject_on_worker_lost = True

    # Worker settings
    worker_prefetch_multiplier = settings.celery_worker_prefetch_multiplier
    worker_max_tasks_per_child = 1000
    worker_disable_rate_limits = False
    worker_log_color = False

    # Task time limits
    task_time_limit = settings.celery_task_time_limit  # Hard limit: 1 hour
    task_soft_time_limit = (
        settings.celery_task_soft_time_limit
    )  # Soft limit: 55 minutes

    # Result settings
    result_expires = settings.celery_result_expires  # 24 hours
    result_compression = "gzip"
    result_serializer = "json"

    # Task serialization
    task_serializer = "json"
    accept_content = ["json"]

    # Timezone settings
    timezone = "UTC"
    enable_utc = True

    # Routing and queues
    task_default_queue = "default"
    task_default_exchange = "default"
    task_default_exchange_type = "direct"
    task_default_routing_key = "default"

    # Priority queues for different customer tiers
    task_routes = {
        # Stream processing tasks
        "src.infrastructure.async_processing.tasks.start_stream_processing": {
            "queue": "stream_processing",
            "routing_key": "stream.start",
        },
        "src.infrastructure.async_processing.tasks.ingest_stream_data": {
            "queue": "stream_ingestion",
            "routing_key": "stream.ingest",
        },
        "src.infrastructure.async_processing.tasks.process_multimodal_content": {
            "queue": "content_processing",
            "routing_key": "content.process",
        },
        "src.infrastructure.async_processing.tasks.detect_highlights": {
            "queue": "ai_processing",
            "routing_key": "ai.detect",
        },
        "src.infrastructure.async_processing.tasks.finalize_highlights": {
            "queue": "finalization",
            "routing_key": "finalize.highlights",
        },
        "src.infrastructure.async_processing.tasks.notify_completion": {
            "queue": "notifications",
            "routing_key": "notify.completion",
        },
        # Maintenance tasks
        "src.infrastructure.async_processing.tasks.cleanup_job_resources": {
            "queue": "maintenance",
            "routing_key": "maintenance.cleanup",
        },
        "src.infrastructure.async_processing.tasks.health_check_task": {
            "queue": "health",
            "routing_key": "health.check",
        },
    }

    # Queue definitions with priorities
    task_queues = [
        # High priority queue for premium customers
        Queue(
            "high_priority",
            Exchange("high_priority", type="direct"),
            routing_key="high_priority",
            queue_arguments={"x-max-priority": 10},
        ),
        # Medium priority queue for standard customers
        Queue(
            "medium_priority",
            Exchange("medium_priority", type="direct"),
            routing_key="medium_priority",
            queue_arguments={"x-max-priority": 5},
        ),
        # Low priority queue for basic customers
        Queue(
            "low_priority",
            Exchange("low_priority", type="direct"),
            routing_key="low_priority",
            queue_arguments={"x-max-priority": 1},
        ),
        # Specialized queues for different task types
        Queue("stream_processing", routing_key="stream.start"),
        Queue("stream_ingestion", routing_key="stream.ingest"),
        Queue("content_processing", routing_key="content.process"),
        Queue("ai_processing", routing_key="ai.detect"),
        Queue("finalization", routing_key="finalize.highlights"),
        Queue("notifications", routing_key="notify.completion"),
        Queue("maintenance", routing_key="maintenance.cleanup"),
        Queue("health", routing_key="health.check"),
    ]

    # Monitoring and logging
    worker_send_task_events = True
    task_send_sent_event = True

    # Error handling
    task_reject_on_worker_lost = True
    task_acks_on_failure_or_timeout = True

    # Beat schedule for periodic tasks
    beat_schedule = {
        "cleanup-expired-jobs": {
            "task": "src.infrastructure.async_processing.tasks.cleanup_job_resources",
            "schedule": crontab(minute=0, hour="*/2"),  # Every 2 hours
            "options": {"queue": "maintenance"},
        },
        "health-check": {
            "task": "src.infrastructure.async_processing.tasks.health_check_task",
            "schedule": crontab(minute="*/5"),  # Every 5 minutes
            "options": {"queue": "health"},
        },
        "process-dead-letter-queue": {
            "task": "src.infrastructure.async_processing.tasks.process_dead_letter_queue",
            "schedule": crontab(minute="*/15"),  # Every 15 minutes
            "options": {"queue": "maintenance"},
        },
    }

    # Performance optimizations
    broker_connection_retry_on_startup = True
    broker_connection_retry = True
    broker_connection_max_retries = 10

    # Memory and resource management
    worker_max_memory_per_child = 200000  # 200MB

    # Custom task annotations for different task types
    task_annotations = {
        "src.infrastructure.async_processing.tasks.detect_highlights": {
            "rate_limit": "10/m",  # Rate limit AI tasks
            "time_limit": 1800,  # 30 minutes for AI processing
            "soft_time_limit": 1680,  # 28 minutes soft limit
        },
        "src.infrastructure.async_processing.tasks.ingest_stream_data": {
            "rate_limit": "50/m",  # Higher rate for ingestion
            "time_limit": 3600,  # 1 hour for stream ingestion
            "soft_time_limit": 3300,  # 55 minutes soft limit
        },
        "src.infrastructure.async_processing.tasks.notify_completion": {
            "rate_limit": "100/m",  # High rate for notifications
            "retry_policy": {
                "max_retries": 5,
                "interval_start": 1,
                "interval_step": 2,
                "interval_max": 60,
            },
        },
    }


# Apply configuration to Celery app
celery_app.config_from_object(CeleryConfig)


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    """Configure Celery loggers to match application logging."""
    import structlog

    # Configure structured logging for Celery
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize worker process resources."""
    import structlog
    from src.infrastructure.database import init_db
    from src.infrastructure.cache import cache as init_cache

    logger = structlog.get_logger(__name__)
    logger.info("Initializing Celery worker process", worker_pid=os.getpid())

    # Initialize database connections
    try:
        init_db()
        logger.info("Database initialized for worker")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

    # Initialize cache connections
    try:
        init_cache()
        logger.info("Cache initialized for worker")
    except Exception as e:
        logger.error("Failed to initialize cache", error=str(e))
        raise


@worker_shutdown.connect
def shutdown_worker(**kwargs):
    """Clean up worker process resources."""
    import structlog

    logger = structlog.get_logger(__name__)
    logger.info("Shutting down Celery worker process", worker_pid=os.getpid())


def queue_for_priority(priority: str) -> str:
    """
    Get the appropriate queue name for a given priority level.

    Args:
        priority: Priority level ("high", "medium", "low")

    Returns:
        str: Queue name for the priority level
    """
    priority_queues = {
        "high": "high_priority",
        "medium": "medium_priority",
        "low": "low_priority",
    }
    return priority_queues.get(priority, "medium_priority")


def get_task_options(priority: str = "medium", **kwargs) -> Dict[str, Any]:
    """
    Get task options for a given priority and additional parameters.

    Args:
        priority: Priority level ("high", "medium", "low")
        **kwargs: Additional task options

    Returns:
        Dict[str, Any]: Task options dictionary
    """
    queue = queue_for_priority(priority)

    options = {
        "queue": queue,
        "routing_key": queue,
        "priority": {"high": 9, "medium": 5, "low": 1}[priority],
        **kwargs,
    }

    return options


# Health check function for monitoring
def health_check() -> Dict[str, Any]:
    """
    Perform health check on Celery application.

    Returns:
        Dict[str, Any]: Health check results
    """
    try:
        # Check broker connection
        celery_app.broker_connection().ensure_connection(max_retries=3)
        broker_status = "healthy"
    except Exception as e:
        broker_status = f"unhealthy: {str(e)}"

    try:
        # Check result backend connection
        with celery_app.backend.client.lock("health_check", timeout=5):
            pass
        backend_status = "healthy"
    except Exception as e:
        backend_status = f"unhealthy: {str(e)}"

    # Get active tasks
    inspect = celery_app.control.inspect()
    active_tasks = inspect.active()

    return {
        "broker_status": broker_status,
        "backend_status": backend_status,
        "active_tasks": len(active_tasks) if active_tasks else 0,
        "queues": list(CeleryConfig.task_queues),
        "timestamp": "2024-01-01T00:00:00Z",  # This would be actual timestamp
    }


# Export the configured Celery app
__all__ = ["celery_app", "queue_for_priority", "get_task_options", "health_check"]
