"""
Async Processing Pipeline for TL;DR Highlight API.

This module provides the complete asynchronous processing pipeline using Celery
for stream processing workflows, job management, progress tracking, error handling,
and webhook dispatch.

Key Components:
- Celery app configuration and task definitions
- Job manager with priority queues and resource allocation
- Progress tracker with real-time status updates
- Error handler with exponential backoff and circuit breakers
- Webhook dispatcher with HMAC signatures and retry logic
- Workflow orchestration with task chaining and coordination
"""

from .celery_app import celery_app
from .tasks import (
    start_stream_processing,
    ingest_stream_data,
    process_multimodal_content,
    detect_highlights,
    finalize_highlights,
    notify_completion,
    cleanup_job_resources,
    health_check_task,
)
from .job_manager import JobManager, JobPriority, JobStatus
from .progress_tracker import ProgressTracker, ProgressEvent
from .webhook_dispatcher import WebhookDispatcher, WebhookEvent
from .error_handler import ErrorHandler, RetryStrategy
from .workflow import StreamProcessingWorkflow

__all__ = [
    # Core components
    "celery_app",
    "JobManager",
    "ProgressTracker",
    "WebhookDispatcher",
    "ErrorHandler",
    "StreamProcessingWorkflow",
    # Task functions
    "start_stream_processing",
    "ingest_stream_data",
    "process_multimodal_content",
    "detect_highlights",
    "finalize_highlights",
    "notify_completion",
    "cleanup_job_resources",
    "health_check_task",
    # Enums and data classes
    "JobPriority",
    "JobStatus",
    "ProgressEvent",
    "WebhookEvent",
    "RetryStrategy",
]
