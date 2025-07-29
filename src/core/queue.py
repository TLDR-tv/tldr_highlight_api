"""Celery configuration and task queue management for the TL;DR Highlight API."""

import logging
from typing import Any, Callable, Optional

from celery import Celery, Task
from celery.result import AsyncResult
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

logger = logging.getLogger(__name__)


class LoggingTask(Task):
    """Custom task class with enhanced logging and error handling."""

    def on_success(self, retval, task_id, args, kwargs):
        """Success callback."""
        logger.info(f"Task {self.name}[{task_id}] succeeded")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Retry callback."""
        logger.warning(f"Task {self.name}[{task_id}] retrying: {exc}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure callback."""
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}", exc_info=einfo)


# Create Celery app
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

# Task routing
app.conf.task_routes = {
    "src.tasks.stream.*": {"queue": "high_priority", "priority": 8},
    "src.tasks.webhook.*": {"queue": "high_priority", "priority": 9},
    "src.tasks.batch.*": {"queue": "batch", "priority": 3},
    "src.tasks.cleanup.*": {"queue": "low_priority", "priority": 1},
}


# Signal handlers for monitoring and logging
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
    logger.error(f"Task {sender.name}[{task_id}] failed with exception: {exception}")


class TaskManager:
    """Manager for Celery task operations with enhanced functionality."""

    def __init__(self, celery_app: Celery):
        self.app = celery_app

    def send_task(
        self,
        name: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        queue: Optional[str] = None,
        priority: Optional[int] = None,
        countdown: Optional[int] = None,
        eta: Optional[Any] = None,
        expires: Optional[Any] = None,
        retry: bool = True,
        retry_policy: Optional[dict] = None,
    ) -> AsyncResult:
        """
        Send task with enhanced options.

        Args:
            name: Task name
            args: Task positional arguments
            kwargs: Task keyword arguments
            queue: Target queue name
            priority: Task priority (0-10)
            countdown: Delay in seconds
            eta: Exact time of execution
            expires: Task expiration time
            retry: Whether to retry on failure
            retry_policy: Custom retry policy

        Returns:
            AsyncResult instance
        """
        options = {}

        if queue:
            options["queue"] = queue
        if priority is not None:
            options["priority"] = priority
        if countdown:
            options["countdown"] = countdown
        if eta:
            options["eta"] = eta
        if expires:
            options["expires"] = expires
        if retry:
            options["retry"] = retry
        if retry_policy:
            options["retry_policy"] = retry_policy

        try:
            result = self.app.send_task(
                name, args=args or (), kwargs=kwargs or {}, **options
            )
            logger.info(f"Sent task {name}[{result.id}] to queue {queue or 'default'}")
            return result
        except Exception as e:
            logger.error(f"Failed to send task {name}: {e}")
            raise

    def get_task_info(self, task_id: str) -> dict[str, Any]:
        """Get detailed task information."""
        try:
            result = AsyncResult(task_id, app=self.app)
            return {
                "id": task_id,
                "state": result.state,
                "result": result.result,
                "info": result.info,
                "ready": result.ready(),
                "successful": result.successful() if result.ready() else None,
                "failed": result.failed() if result.ready() else None,
                "traceback": result.traceback if result.failed() else None,
            }
        except Exception as e:
            logger.error(f"Failed to get task info for {task_id}: {e}")
            return {"id": task_id, "error": str(e)}

    def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """Revoke a task."""
        try:
            self.app.control.revoke(task_id, terminate=terminate)
            logger.info(f"Revoked task {task_id} (terminate={terminate})")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False

    def get_active_tasks(self) -> dict[str, list]:
        """Get active tasks from all workers."""
        try:
            inspect = self.app.control.inspect()
            active = inspect.active()
            return active or {}
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return {}

    def get_scheduled_tasks(self) -> dict[str, list]:
        """Get scheduled tasks from all workers."""
        try:
            inspect = self.app.control.inspect()
            scheduled = inspect.scheduled()
            return scheduled or {}
        except Exception as e:
            logger.error(f"Failed to get scheduled tasks: {e}")
            return {}

    def get_worker_stats(self) -> dict[str, Any]:
        """Get statistics from all workers."""
        try:
            inspect = self.app.control.inspect()
            return {
                "stats": inspect.stats() or {},
                "active_queues": inspect.active_queues() or {},
                "registered_tasks": inspect.registered() or {},
            }
        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {}

    def purge_queue(self, queue_name: str) -> int:
        """Purge all tasks from a specific queue."""
        try:
            result = self.app.control.purge()
            logger.warning(f"Purged queue {queue_name}: {result} tasks deleted")
            return result
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0


# Global task manager instance
task_manager = TaskManager(app)


# Utility decorators for common task patterns
def rate_limited_task(rate_limit: str):
    """Decorator to apply rate limiting to tasks."""

    def decorator(func: Callable) -> Callable:
        func = app.task(
            bind=True,
            rate_limit=rate_limit,
            base=LoggingTask,
        )(func)
        return func

    return decorator


def priority_task(priority: int = 5, queue: str = "default"):
    """Decorator to create priority tasks."""

    def decorator(func: Callable) -> Callable:
        func = app.task(
            bind=True,
            priority=priority,
            queue=queue,
            base=LoggingTask,
        )(func)
        return func

    return decorator


def batch_task(batch_size: int = 100):
    """Decorator for batch processing tasks."""

    def decorator(func: Callable) -> Callable:
        func = app.task(
            bind=True,
            queue="batch",
            base=LoggingTask,
            batch_size=batch_size,
        )(func)
        return func

    return decorator
