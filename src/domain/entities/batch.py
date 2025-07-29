"""Batch domain entity."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.url import Url
from src.domain.value_objects.processing_options import ProcessingOptions


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Value object representing an item in a batch."""
    url: Url
    item_id: str  # User-provided ID for tracking
    status: BatchStatus = BatchStatus.PENDING
    stream_id: Optional[int] = None  # ID of created stream
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_processed(self) -> bool:
        """Check if item has been processed."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED
        ]
    
    @property
    def is_successful(self) -> bool:
        """Check if item was processed successfully."""
        return self.status == BatchStatus.COMPLETED


@dataclass
class Batch(Entity[int]):
    """Domain entity representing a batch processing job.
    
    Batches allow users to submit multiple URLs for processing
    in a single request.
    """
    
    name: str
    user_id: int
    status: BatchStatus
    processing_options: ProcessingOptions
    
    # Batch items
    items: List[BatchItem] = field(default_factory=list)
    
    # Processing timestamps
    started_at: Optional[Timestamp] = None
    completed_at: Optional[Timestamp] = None
    
    # Progress tracking
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    
    # Webhook for completion notification
    webhook_url: Optional[Url] = None
    
    def __post_init__(self):
        """Initialize counters after creation."""
        if not self.total_items and self.items:
            object.__setattr__(self, 'total_items', len(self.items))
        super().__post_init__()
    
    @property
    def progress_percentage(self) -> float:
        """Calculate processing progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of processed items."""
        if self.processed_items == 0:
            return 0.0
        return self.successful_items / self.processed_items
    
    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.PARTIALLY_FAILED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED
        ]
    
    def start_processing(self) -> "Batch":
        """Mark batch as processing."""
        if self.status != BatchStatus.PENDING:
            raise ValueError(f"Cannot start batch in {self.status} status")
        
        return Batch(
            id=self.id,
            name=self.name,
            user_id=self.user_id,
            status=BatchStatus.PROCESSING,
            processing_options=self.processing_options,
            items=self.items.copy(),
            started_at=Timestamp.now(),
            completed_at=None,
            total_items=self.total_items,
            processed_items=0,
            successful_items=0,
            failed_items=0,
            webhook_url=self.webhook_url,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def update_item_status(self, item_id: str, status: BatchStatus,
                          stream_id: Optional[int] = None,
                          error_message: Optional[str] = None) -> "Batch":
        """Update status of a batch item."""
        new_items = []
        item_found = False
        
        for item in self.items:
            if item.item_id == item_id:
                item_found = True
                new_item = BatchItem(
                    url=item.url,
                    item_id=item.item_id,
                    status=status,
                    stream_id=stream_id or item.stream_id,
                    error_message=error_message,
                    metadata=item.metadata.copy()
                )
                new_items.append(new_item)
            else:
                new_items.append(item)
        
        if not item_found:
            raise ValueError(f"Item {item_id} not found in batch")
        
        # Update counters
        processed = sum(1 for item in new_items if item.is_processed)
        successful = sum(1 for item in new_items if item.is_successful)
        failed = processed - successful
        
        # Determine batch status
        if processed == self.total_items:
            if failed == 0:
                batch_status = BatchStatus.COMPLETED
            elif successful == 0:
                batch_status = BatchStatus.FAILED
            else:
                batch_status = BatchStatus.PARTIALLY_FAILED
        else:
            batch_status = BatchStatus.PROCESSING
        
        return Batch(
            id=self.id,
            name=self.name,
            user_id=self.user_id,
            status=batch_status,
            processing_options=self.processing_options,
            items=new_items,
            started_at=self.started_at,
            completed_at=Timestamp.now() if batch_status != BatchStatus.PROCESSING else None,
            total_items=self.total_items,
            processed_items=processed,
            successful_items=successful,
            failed_items=failed,
            webhook_url=self.webhook_url,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def cancel(self) -> "Batch":
        """Cancel batch processing."""
        if self.is_complete:
            raise ValueError(f"Cannot cancel batch in {self.status} status")
        
        # Mark unprocessed items as cancelled
        new_items = []
        for item in self.items:
            if not item.is_processed:
                new_item = BatchItem(
                    url=item.url,
                    item_id=item.item_id,
                    status=BatchStatus.CANCELLED,
                    stream_id=item.stream_id,
                    error_message="Batch cancelled",
                    metadata=item.metadata.copy()
                )
                new_items.append(new_item)
            else:
                new_items.append(item)
        
        return Batch(
            id=self.id,
            name=self.name,
            user_id=self.user_id,
            status=BatchStatus.CANCELLED,
            processing_options=self.processing_options,
            items=new_items,
            started_at=self.started_at,
            completed_at=Timestamp.now(),
            total_items=self.total_items,
            processed_items=self.processed_items,
            successful_items=self.successful_items,
            failed_items=self.failed_items,
            webhook_url=self.webhook_url,
            created_at=self.created_at,
            updated_at=Timestamp.now()
        )
    
    def get_item(self, item_id: str) -> Optional[BatchItem]:
        """Get a specific item by ID."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None
    
    def get_failed_items(self) -> List[BatchItem]:
        """Get all failed items."""
        return [item for item in self.items if item.status == BatchStatus.FAILED]
    
    def get_successful_items(self) -> List[BatchItem]:
        """Get all successful items."""
        return [item for item in self.items if item.is_successful]