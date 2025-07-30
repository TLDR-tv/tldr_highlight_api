"""Task decorators for Celery.

This module provides Pythonic decorators for creating Celery tasks
with common patterns, keeping infrastructure concerns separate.
"""

from typing import Callable, TypeVar, ParamSpec
from functools import wraps

from .celery_app import celery_app, LoggingTask

# Type variables for better type hints
P = ParamSpec('P')
T = TypeVar('T')


def rate_limited_task(rate_limit: str) -> Callable:
    """Decorator to apply rate limiting to tasks.
    
    Args:
        rate_limit: Rate limit string (e.g., "10/m" for 10 per minute)
        
    Returns:
        Decorated task function
        
    Example:
        @rate_limited_task("10/m")
        def process_item(self, item_id: int):
            # Task limited to 10 executions per minute
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return celery_app.task(
            bind=True,
            rate_limit=rate_limit,
            base=LoggingTask,
        )(func)
    
    return decorator


def priority_task(
    priority: int = 5, 
    queue: str = "default"
) -> Callable:
    """Decorator to create priority tasks.
    
    Args:
        priority: Task priority (0-10, higher is more important)
        queue: Target queue name
        
    Returns:
        Decorated task function
        
    Example:
        @priority_task(priority=8, queue="high_priority")
        def urgent_processing(self, data: dict):
            # High priority task
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return celery_app.task(
            bind=True,
            priority=priority,
            queue=queue,
            base=LoggingTask,
        )(func)
    
    return decorator


def batch_task(
    batch_size: int = 100,
    priority: int = 3
) -> Callable:
    """Decorator for batch processing tasks.
    
    Args:
        batch_size: Default batch size
        priority: Task priority
        
    Returns:
        Decorated task function
        
    Example:
        @batch_task(batch_size=50)
        def process_batch(self, items: list):
            # Process items in batches
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Add batch_size as a task attribute
        task = celery_app.task(
            bind=True,
            queue="batch",
            priority=priority,
            base=LoggingTask,
        )(func)
        task.batch_size = batch_size
        return task
    
    return decorator


def retryable_task(
    max_retries: int = 3,
    retry_delay: int = 60,
    retry_backoff: bool = True,
    retry_jitter: bool = True
) -> Callable:
    """Decorator for tasks with custom retry behavior.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries in seconds
        retry_backoff: Whether to use exponential backoff
        retry_jitter: Whether to add jitter to retry delays
        
    Returns:
        Decorated task function
        
    Example:
        @retryable_task(max_retries=5, retry_delay=30)
        def unreliable_api_call(self, url: str):
            # Task with custom retry logic
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                # Calculate retry delay with optional backoff and jitter
                delay = retry_delay
                if retry_backoff and self.request.retries > 0:
                    delay = retry_delay * (2 ** self.request.retries)
                if retry_jitter:
                    import random
                    delay = delay * (0.5 + random.random())
                    
                raise self.retry(
                    exc=exc,
                    countdown=delay,
                    max_retries=max_retries
                )
        
        return celery_app.task(
            bind=True,
            base=LoggingTask,
            autoretry_for=(Exception,),
            retry_kwargs={
                'max_retries': max_retries,
                'countdown': retry_delay
            }
        )(wrapper)
    
    return decorator


def exclusive_task(
    lock_timeout: int = 3600,
    lock_key_prefix: str = "exclusive"
) -> Callable:
    """Decorator for tasks that should only run one at a time.
    
    Uses Redis locking to ensure exclusive execution.
    
    Args:
        lock_timeout: Lock timeout in seconds
        lock_key_prefix: Prefix for lock keys
        
    Returns:
        Decorated task function
        
    Example:
        @exclusive_task(lock_timeout=600)
        def cleanup_database(self):
            # Only one instance runs at a time
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
            from infrastructure.cache import get_redis_cache
            
            # Generate lock key from task name and args
            lock_key = f"{lock_key_prefix}:{self.name}:{hash(args)}"
            
            cache = await get_redis_cache()
            
            # Try to acquire lock
            acquired = await cache.set(
                lock_key, 
                "locked", 
                ttl=lock_timeout, 
                nx=True
            )
            
            if not acquired:
                raise RuntimeError(f"Task {self.name} is already running")
            
            try:
                # Execute the task
                return await func(self, *args, **kwargs)
            finally:
                # Release lock
                await cache.delete(lock_key)
        
        return celery_app.task(
            bind=True,
            base=LoggingTask,
        )(wrapper)
    
    return decorator