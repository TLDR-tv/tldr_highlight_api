"""Stream domain entity."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from src.domain.entities.base import Entity
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.processing_options import ProcessingOptions


class StreamStatus(Enum):
    """Stream processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamPlatform(Enum):
    """Supported streaming platforms."""

    TWITCH = "twitch"
    YOUTUBE = "youtube"
    RTMP = "rtmp"
    CUSTOM = "custom"


@dataclass
class Stream(Entity[int]):
    """Domain entity representing a stream processing job.

    Streams are individual livestream or video URLs submitted
    for highlight extraction processing.
    """

    url: Url
    platform: StreamPlatform
    status: StreamStatus
    user_id: int

    # Processing configuration
    processing_options: ProcessingOptions

    # Stream metadata
    title: Optional[str] = None
    channel_name: Optional[str] = None
    game_category: Optional[str] = None
    language: Optional[str] = None
    viewer_count: Optional[int] = None
    duration: Optional[Duration] = None

    # Processing results
    started_at: Optional[Timestamp] = None
    completed_at: Optional[Timestamp] = None
    error_message: Optional[str] = None

    # Related entity IDs
    highlight_ids: List[int] = field(default_factory=list)

    # Raw platform data (for debugging/analysis)
    platform_data: Dict[str, Any] = field(default_factory=dict)

    def start_processing(self) -> "Stream":
        """Mark stream as processing."""
        if self.status != StreamStatus.PENDING:
            raise ValueError(f"Cannot start processing stream in {self.status} status")

        return Stream(
            id=self.id,
            url=self.url,
            platform=self.platform,
            status=StreamStatus.PROCESSING,
            user_id=self.user_id,
            processing_options=self.processing_options,
            title=self.title,
            channel_name=self.channel_name,
            game_category=self.game_category,
            language=self.language,
            viewer_count=self.viewer_count,
            duration=self.duration,
            started_at=Timestamp.now(),
            completed_at=self.completed_at,
            error_message=self.error_message,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            highlight_ids=self.highlight_ids.copy(),
            platform_data=self.platform_data.copy(),
        )

    def complete_processing(self, duration: Optional[Duration] = None) -> "Stream":
        """Mark stream as completed."""
        if self.status != StreamStatus.PROCESSING:
            raise ValueError(f"Cannot complete stream in {self.status} status")

        return Stream(
            id=self.id,
            url=self.url,
            platform=self.platform,
            status=StreamStatus.COMPLETED,
            user_id=self.user_id,
            processing_options=self.processing_options,
            title=self.title,
            channel_name=self.channel_name,
            game_category=self.game_category,
            language=self.language,
            viewer_count=self.viewer_count,
            duration=duration or self.duration,
            started_at=self.started_at,
            completed_at=Timestamp.now(),
            error_message=None,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            highlight_ids=self.highlight_ids.copy(),
            platform_data=self.platform_data.copy(),
        )

    def fail_processing(self, error_message: str) -> "Stream":
        """Mark stream as failed."""
        if self.status not in [StreamStatus.PENDING, StreamStatus.PROCESSING]:
            raise ValueError(f"Cannot fail stream in {self.status} status")

        return Stream(
            id=self.id,
            url=self.url,
            platform=self.platform,
            status=StreamStatus.FAILED,
            user_id=self.user_id,
            processing_options=self.processing_options,
            title=self.title,
            channel_name=self.channel_name,
            game_category=self.game_category,
            language=self.language,
            viewer_count=self.viewer_count,
            duration=self.duration,
            started_at=self.started_at,
            completed_at=Timestamp.now(),
            error_message=error_message,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            highlight_ids=self.highlight_ids.copy(),
            platform_data=self.platform_data.copy(),
        )

    def cancel(self) -> "Stream":
        """Cancel stream processing."""
        if self.status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            raise ValueError(f"Cannot cancel stream in {self.status} status")

        return Stream(
            id=self.id,
            url=self.url,
            platform=self.platform,
            status=StreamStatus.CANCELLED,
            user_id=self.user_id,
            processing_options=self.processing_options,
            title=self.title,
            channel_name=self.channel_name,
            game_category=self.game_category,
            language=self.language,
            viewer_count=self.viewer_count,
            duration=self.duration,
            started_at=self.started_at,
            completed_at=Timestamp.now(),
            error_message="Cancelled by user",
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            highlight_ids=self.highlight_ids.copy(),
            platform_data=self.platform_data.copy(),
        )

    def add_highlight(self, highlight_id: int) -> "Stream":
        """Add a highlight to this stream."""
        if highlight_id in self.highlight_ids:
            return self

        new_highlight_ids = self.highlight_ids.copy()
        new_highlight_ids.append(highlight_id)

        return Stream(
            id=self.id,
            url=self.url,
            platform=self.platform,
            status=self.status,
            user_id=self.user_id,
            processing_options=self.processing_options,
            title=self.title,
            channel_name=self.channel_name,
            game_category=self.game_category,
            language=self.language,
            viewer_count=self.viewer_count,
            duration=self.duration,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error_message=self.error_message,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            highlight_ids=new_highlight_ids,
            platform_data=self.platform_data.copy(),
        )

    @property
    def is_live(self) -> bool:
        """Check if this is a live stream (vs VOD)."""
        return self.platform_data.get("is_live", False)

    @property
    def processing_duration(self) -> Optional[Duration]:
        """Calculate how long processing took."""
        if self.started_at and self.completed_at:
            return self.completed_at.duration_since(self.started_at)
        return None

    @property
    def is_terminal_state(self) -> bool:
        """Check if stream is in a terminal state."""
        return self.status in [
            StreamStatus.COMPLETED,
            StreamStatus.FAILED,
            StreamStatus.CANCELLED,
        ]
