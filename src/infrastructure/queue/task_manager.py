"""Task management for Celery operations.

This module provides a Pythonic interface for managing Celery tasks
as infrastructure, keeping task management separate from business logic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from celery import Celery
from celery.result import AsyncResult

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a Celery task."""
    id: str
    state: str
    result: Any = None
    info: Any = None
    ready: bool = False
    successful: Optional[bool] = None
    failed: Optional[bool] = None
    traceback: Optional[str] = None
    error: Optional[str] = None


class TaskManager:
    """Manager for Celery task operations with enhanced functionality.
    
    This class provides a clean interface for task management operations,
    following Pythonic patterns with proper error handling.
    """

    def __init__(self, celery_app: Celery):
        """Initialize task manager with Celery app instance."""
        self.app = celery_app

    async def send_task(
        self,
        name: str,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
        queue: Optional[str] = None,
        priority: Optional[int] = None,
        countdown: Optional[int] = None,
        eta: Optional[datetime] = None,
        expires: Optional[datetime] = None,
        retry: bool = True,
        retry_policy: Optional[Dict] = None,
    ) -> str:
        """
        Send task with enhanced options.
        
        This method provides a Pythonic interface for task submission
        with comprehensive options for routing and scheduling.

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
            Task ID as string
            
        Raises:
            RuntimeError: If task submission fails
        """
        options = self._build_task_options(
            queue=queue,
            priority=priority,
            countdown=countdown,
            eta=eta,
            expires=expires,
            retry=retry,
            retry_policy=retry_policy
        )

        try:
            result = self.app.send_task(
                name, 
                args=args or (), 
                kwargs=kwargs or {}, 
                **options
            )
            logger.info(f"Sent task {name}[{result.id}] to queue {queue or 'default'}")
            return result.id
        except Exception as e:
            logger.error(f"Failed to send task {name}: {e}")
            raise RuntimeError(f"Failed to send task {name}") from e

    def _build_task_options(self, **kwargs) -> Dict[str, Any]:
        """Build task options dictionary from kwargs."""
        return {k: v for k, v in kwargs.items() if v is not None}

    async def get_task_info(self, task_id: str) -> TaskInfo:
        """Get detailed task information.
        
        Args:
            task_id: The task ID to query
            
        Returns:
            TaskInfo object with task details
        """
        try:
            result = AsyncResult(task_id, app=self.app)
            return TaskInfo(
                id=task_id,
                state=result.state,
                result=result.result,
                info=result.info,
                ready=result.ready(),
                successful=result.successful() if result.ready() else None,
                failed=result.failed() if result.ready() else None,
                traceback=result.traceback if result.failed() else None,
            )
        except Exception as e:
            logger.error(f"Failed to get task info for {task_id}: {e}")
            return TaskInfo(
                id=task_id,
                state="UNKNOWN",
                error=str(e)
            )

    async def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """Revoke a task.
        
        Args:
            task_id: Task to revoke
            terminate: Whether to terminate executing tasks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.app.control.revoke(task_id, terminate=terminate)
            logger.info(f"Revoked task {task_id} (terminate={terminate})")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False

    async def get_active_tasks(self) -> Dict[str, List[Dict]]:
        """Get active tasks from all workers.
        
        Returns:
            Dictionary mapping worker names to lists of active tasks
        """
        try:
            inspect = self.app.control.inspect()
            active = inspect.active()
            return active or {}
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return {}

    async def get_scheduled_tasks(self) -> Dict[str, List[Dict]]:
        """Get scheduled tasks from all workers.
        
        Returns:
            Dictionary mapping worker names to lists of scheduled tasks
        """
        try:
            inspect = self.app.control.inspect()
            scheduled = inspect.scheduled()
            return scheduled or {}
        except Exception as e:
            logger.error(f"Failed to get scheduled tasks: {e}")
            return {}

    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics from all workers.
        
        Returns:
            Dictionary with worker statistics
        """
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

    async def purge_queue(self, queue_name: Optional[str] = None) -> int:
        """Purge all tasks from a specific queue.
        
        Args:
            queue_name: Queue to purge (None for all queues)
            
        Returns:
            Number of tasks purged
        """
        try:
            if queue_name:
                # Purge specific queue
                from kombu import Queue as KombuQueue
                with self.app.connection_for_write() as conn:
                    queue = KombuQueue(queue_name, channel=conn.default_channel)
                    count = queue.purge()
            else:
                # Purge all queues
                count = self.app.control.purge()
                
            logger.warning(f"Purged {'queue ' + queue_name if queue_name else 'all queues'}: {count} tasks deleted")
            return count
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskInfo:
        """Wait for a task to complete.
        
        Args:
            task_id: Task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            TaskInfo with final task state
        """
        try:
            result = AsyncResult(task_id, app=self.app)
            result.wait(timeout=timeout)
            return await self.get_task_info(task_id)
        except Exception as e:
            logger.error(f"Error waiting for task {task_id}: {e}")
            return TaskInfo(
                id=task_id,
                state="ERROR",
                error=str(e)
            )