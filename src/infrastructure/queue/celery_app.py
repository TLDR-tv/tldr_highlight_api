"""Celery application configuration and setup.

This module configures Celery as message queue infrastructure,
keeping it separate from business logic per DDD principles.
"""

import logging
from typing import Optional, TYPE_CHECKING

from celery import Celery, Task
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    task_success,
    worker_ready,
    worker_shutdown,
)
from kombu import Exchange, Queue

from src.core.config import settings

if TYPE_CHECKING:
    from .task_manager import TaskManager

logger = logging.getLogger(__name__)


class LoggingTask(Task):
    """Custom task class with enhanced logging and error handling.

    This provides infrastructure-level logging without coupling
    to business logic.
    """

    def on_success(self, retval, task_id, args, kwargs):
        """Success callback."""
        logger.info(f"Task {self.name}[{task_id}] succeeded")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Retry callback."""
        logger.warning(f"Task {self.name}[{task_id}] retrying: {exc}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure callback."""
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}", exc_info=einfo)


def create_celery_app() -> Celery:
    """Create and configure Celery application.

    This factory function creates a properly configured Celery instance
    following Pythonic patterns.
    """
    app = Celery(
        "tldr_highlights",
        task_cls=LoggingTask,
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )

    # Configure Celery
    app.conf.update(
        # Task execution settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Task behavior
        task_track_started=True,
        task_time_limit=3600,  # 1 hour hard limit
        task_soft_time_limit=3300,  # 55 minutes soft limit
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Result backend settings
        result_expires=86400,  # 24 hours
        result_persistent=True,
        result_compression="gzip",
        # Worker settings
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
        worker_disable_rate_limits=False,
        worker_send_task_events=True,
        # Retry settings
        task_default_retry_delay=60,  # 1 minute
        task_max_retries=3,
        # Message routing
        task_default_queue="default",
        task_create_missing_queues=True,
        # Monitoring
        task_send_sent_event=True,
    )

    # Configure exchanges and queues
    configure_queues(app)

    # Configure task routing
    configure_routing(app)

    # Setup signal handlers
    setup_signal_handlers()

    return app


def configure_queues(app: Celery) -> None:
    """Configure message queues with different priorities."""
    # Define exchanges
    default_exchange = Exchange("default", type="direct")
    priority_exchange = Exchange("priority", type="direct")
    batch_exchange = Exchange("batch", type="direct")

    # Define queues with different priorities
    app.conf.task_queues = (
        Queue(
            "default",
            exchange=default_exchange,
            routing_key="default",
            queue_arguments={"x-max-priority": 5},
        ),
        Queue(
            "high_priority",
            exchange=priority_exchange,
            routing_key="high",
            queue_arguments={"x-max-priority": 10},
        ),
        Queue(
            "low_priority",
            exchange=priority_exchange,
            routing_key="low",
            queue_arguments={"x-max-priority": 1},
        ),
        Queue(
            "batch",
            exchange=batch_exchange,
            routing_key="batch",
            queue_arguments={"x-max-priority": 3},
        ),
    )


def configure_routing(app: Celery) -> None:
    """Configure task routing rules."""
    app.conf.task_routes = {
        # Core stream processing tasks (high priority)
        "ingest_stream_with_ffmpeg": {"queue": "high_priority", "priority": 9},
        "detect_highlights_with_ai": {"queue": "high_priority", "priority": 8},
        # Webhook tasks (highest priority for real-time notifications)
        "src.tasks.webhook.*": {"queue": "high_priority", "priority": 10},
        # Maintenance and cleanup (low priority)
        "cleanup_stream_resources": {"queue": "low_priority", "priority": 1},
        "cleanup_job_resources": {"queue": "low_priority", "priority": 1},
        "health_check_task": {"queue": "default", "priority": 2},
        # Legacy tasks (for backwards compatibility)
        "src.tasks.stream.*": {"queue": "high_priority", "priority": 7},
        "src.tasks.batch.*": {"queue": "batch", "priority": 3},
    }


def setup_signal_handlers() -> None:
    """Setup Celery signal handlers for monitoring."""

    @worker_ready.connect
    def worker_ready_handler(sender=None, **kwargs):
        """Handle worker ready signal."""
        logger.info("Celery worker is ready")

    @worker_shutdown.connect
    def worker_shutdown_handler(sender=None, **kwargs):
        """Handle worker shutdown signal."""
        logger.info("Celery worker is shutting down")

    @task_prerun.connect
    def task_prerun_handler(
        sender=None, task_id=None, task=None, args=None, kwargs=None, **kw
    ):
        """Handle task pre-run signal."""
        logger.info(
            f"Starting task {task.name}[{task_id}] with args={args}, kwargs={kwargs}"
        )

    @task_postrun.connect
    def task_postrun_handler(
        sender=None,
        task_id=None,
        task=None,
        args=None,
        kwargs=None,
        retval=None,
        state=None,
        **kw,
    ):
        """Handle task post-run signal."""
        logger.info(f"Completed task {task.name}[{task_id}] with state={state}")

    @task_success.connect
    def task_success_handler(sender=None, result=None, **kwargs):
        """Handle task success signal."""
        logger.debug(f"Task {sender.name} succeeded with result: {result}")

    @task_retry.connect
    def task_retry_handler(sender=None, reason=None, **kwargs):
        """Handle task retry signal."""
        logger.warning(f"Task {sender.name} retrying, reason: {reason}")

    @task_failure.connect
    def task_failure_handler(
        sender=None, task_id=None, exception=None, traceback=None, **kwargs
    ):
        """Handle task failure signal."""
        logger.error(
            f"Task {sender.name}[{task_id}] failed with exception: {exception}"
        )


# Create the Celery app instance
celery_app = create_celery_app()

# Import task manager to avoid circular imports
_task_manager: Optional["TaskManager"] = None


def get_task_manager() -> "TaskManager":
    """Get or create the global task manager instance.

    This follows a lazy initialization pattern to avoid circular imports.
    """
    global _task_manager
    if _task_manager is None:
        from .task_manager import TaskManager

        _task_manager = TaskManager(celery_app)
    return _task_manager
