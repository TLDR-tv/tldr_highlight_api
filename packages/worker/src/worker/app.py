"""Celery application configuration."""

from celery import Celery
from shared.infrastructure.config.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "highlight_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "worker.tasks.stream_processing",
        "worker.tasks.highlight_detection",
        "worker.tasks.webhook_delivery",
        "worker.tasks.email_delivery",
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Worker settings
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
    worker_disable_rate_limits=False,
    
    # Task settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=3900,  # 1 hour 5 min hard limit
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_persistent=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Routing
    task_routes={
        "worker.tasks.stream_processing.*": {"queue": "stream_processing"},
        "worker.tasks.highlight_detection.*": {"queue": "highlight_detection"},
        "worker.tasks.webhook_delivery.*": {"queue": "webhooks"},
        "worker.tasks.email_delivery.*": {"queue": "emails"},
    },
)

# Beat schedule for periodic tasks (if needed)
celery_app.conf.beat_schedule = {
    # Add periodic tasks here
}

if __name__ == "__main__":
    celery_app.start()