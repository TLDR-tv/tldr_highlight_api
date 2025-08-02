"""Celery client for API to queue tasks."""

from celery import Celery
from shared.infrastructure.config.config import get_settings

settings = get_settings()

# Create Celery client (not a worker, just for queueing tasks)
celery_app = Celery(
    "api_client",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Configure Celery client
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)