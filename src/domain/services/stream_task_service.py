"""Stream task service protocol for background processing abstraction."""

from typing import Protocol, Optional, runtime_checkable
from dataclasses import dataclass
from uuid import UUID


@dataclass
class TaskResult:
    """Result of a background task submission."""
    
    task_id: str
    status: str
    message: Optional[str] = None


@runtime_checkable
class StreamTaskService(Protocol):
    """Service for managing stream processing background tasks."""

    async def start_stream_ingestion(
        self,
        stream_id: UUID,
        chunk_duration: int,
        agent_config_id: UUID,
    ) -> TaskResult:
        """Start background stream ingestion task.
        
        Args:
            stream_id: ID of the stream to process
            chunk_duration: Duration of chunks in seconds
            agent_config_id: ID of the agent configuration
            
        Returns:
            Task result with ID and status
        """
        ...

    async def cancel_stream_tasks(self, stream_id: UUID) -> bool:
        """Cancel all tasks for a stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            True if tasks were cancelled successfully
        """
        ...

    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a background task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status or None if not found
        """
        ...