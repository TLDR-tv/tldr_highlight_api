"""Message queue infrastructure for the TL;DR Highlight API.

This module provides task queue management using Celery as infrastructure,
following DDD principles with clean separation of concerns.
"""

from .celery_app import celery_app, get_task_manager
from .task_manager import TaskManager
from .decorators import rate_limited_task, priority_task, batch_task

__all__ = [
    "celery_app",
    "get_task_manager",
    "TaskManager",
    "rate_limited_task",
    "priority_task",
    "batch_task",
]
