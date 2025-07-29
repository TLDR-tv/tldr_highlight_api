"""Batch repository protocol."""

from typing import Protocol, List, Optional
from abc import abstractmethod

from src.domain.repositories.base import Repository
from src.domain.entities.batch import Batch, BatchStatus
from src.domain.value_objects.timestamp import Timestamp


class BatchRepository(Repository[Batch, int], Protocol):
    """Repository protocol for Batch entities.
    
    Extends the base repository with batch-specific operations.
    """
    
    @abstractmethod
    async def get_by_user(self, user_id: int,
                        status: Optional[BatchStatus] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Batch]:
        """Get batches for a user, optionally filtered by status."""
        ...
    
    @abstractmethod
    async def get_active_batches(self) -> List[Batch]:
        """Get all batches currently being processed."""
        ...
    
    @abstractmethod
    async def get_by_date_range(self, start: Timestamp, end: Timestamp,
                              user_id: Optional[int] = None) -> List[Batch]:
        """Get batches within a date range."""
        ...
    
    @abstractmethod
    async def get_with_items(self, batch_id: int) -> Optional[Batch]:
        """Get batch with all its items loaded."""
        ...
    
    @abstractmethod
    async def count_by_status(self, status: BatchStatus,
                            user_id: Optional[int] = None) -> int:
        """Count batches by status."""
        ...
    
    @abstractmethod
    async def get_processing_stats(self, user_id: int) -> dict:
        """Get batch processing statistics for a user."""
        ...
    
    @abstractmethod
    async def get_failed_items_summary(self, user_id: int,
                                     limit: int = 100) -> List[dict]:
        """Get summary of failed items across batches."""
        ...
    
    @abstractmethod
    async def cleanup_old_batches(self, older_than: Timestamp) -> int:
        """Clean up old completed/failed batches."""
        ...
    
    @abstractmethod
    async def get_next_pending_batch(self) -> Optional[Batch]:
        """Get the next batch ready for processing."""
        ...