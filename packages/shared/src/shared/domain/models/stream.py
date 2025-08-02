"""Stream domain model - represents a livestream or video being processed."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class StreamStatus(Enum):
    """Stream processing status."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StreamType(Enum):
    """Type of stream content."""

    LIVESTREAM = "livestream"
    VOD = "vod"  # Video on demand
    FILE = "file"


class StreamSource(Enum):
    """Source type of the stream."""

    RTMP = "rtmp"
    HLS = "hls"
    DIRECT_URL = "direct_url"
    FILE_UPLOAD = "file_upload"


@dataclass
class Stream:
    """Stream being processed for highlights."""

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    url: str = ""  # Stream URL
    name: Optional[str] = None  # Human-readable name
    type: StreamType = StreamType.LIVESTREAM
    status: StreamStatus = StreamStatus.PENDING
    
    # Task tracking
    celery_task_id: Optional[str] = None
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    stream_fingerprint: str = ""  # Unique identifier for the streamer
    source_type: StreamSource = StreamSource.DIRECT_URL

    # Processing info
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    segments_processed: int = 0
    highlights_generated: int = 0
    
    # Stats
    stats: Optional[dict] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0

    @property
    def is_live(self) -> bool:
        """Check if stream is currently being processed."""
        return self.status == StreamStatus.PROCESSING

    @property
    def is_complete(self) -> bool:
        """Check if stream processing is complete."""
        return self.status == StreamStatus.COMPLETED

    @property
    def has_failed(self) -> bool:
        """Check if stream processing failed."""
        return self.status == StreamStatus.FAILED

    @property
    def processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def start_processing(self) -> None:
        """Mark stream as processing."""
        self.status = StreamStatus.PROCESSING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark stream as completed."""
        self.status = StreamStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        """Mark stream as failed with error message."""
        self.status = StreamStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now(timezone.utc)
        self.retry_count += 1

    def increment_segment_count(self) -> None:
        """Increment processed segment counter."""
        self.segments_processed += 1

    def increment_highlight_count(self) -> None:
        """Increment generated highlight counter."""
        self.highlights_generated += 1
