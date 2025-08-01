"""Stream domain entity."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from src.domain.entities.base import AggregateRoot
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.exceptions import (
    InvalidStateTransition,
    BusinessRuleViolation,
    StateTransitionInfo,
)
from src.domain.events import (
    StreamProcessingStartedEvent,
    StreamProcessingCompletedEvent,
    StreamProcessingFailedEvent,
    HighlightAddedToStreamEvent,
)


class StreamStatus(Enum):
    """Stream processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamPlatform(Enum):
    """Supported streaming platforms/protocols."""

    RTMP = "rtmp"  # RTMP streams
    RTMPS = "rtmps"  # Secure RTMP
    HLS = "hls"  # HTTP Live Streaming (m3u8)
    DASH = "dash"  # MPEG-DASH
    HTTP = "http"  # Direct HTTP/HTTPS streams
    FILE = "file"  # Local files
    UDP = "udp"  # UDP streams
    RTP = "rtp"  # RTP streams
    RTSP = "rtsp"  # RTSP streams
    SRT = "srt"  # SRT protocol
    CUSTOM = "custom"  # Any other FFmpeg-supported format


@dataclass(kw_only=True)
class Stream(AggregateRoot[int]):
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

    def start_processing(self) -> None:
        """Start processing the stream.

        This follows Pythonic DDD where aggregates enforce state transitions
        and raise appropriate domain events.
        """
        # Business rule: Can only start from PENDING status
        if self.status != StreamStatus.PENDING:
            transition_info = StateTransitionInfo(
                from_state=self.status.value,
                to_state=StreamStatus.PROCESSING.value,
                allowed_states=[StreamStatus.PENDING.value],
                entity_type="Stream",
                entity_id=self.id,
            )
            raise InvalidStateTransition(transition_info)

        # Update state
        self.status = StreamStatus.PROCESSING
        self.started_at = Timestamp.now()
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                StreamProcessingStartedEvent(
                    stream_id=self.id,
                    processing_options=self.processing_options.to_dict(),
                )
            )

    def complete_processing(self, duration: Optional[Duration] = None) -> None:
        """Complete stream processing."""
        # Business rule: Can only complete from PROCESSING status
        if self.status != StreamStatus.PROCESSING:
            transition_info = StateTransitionInfo(
                from_state=self.status.value,
                to_state=StreamStatus.COMPLETED.value,
                allowed_states=[StreamStatus.PROCESSING.value],
                entity_type="Stream",
                entity_id=self.id,
            )
            raise InvalidStateTransition(transition_info)

        # Update state
        self.status = StreamStatus.COMPLETED
        self.completed_at = Timestamp.now()
        self.error_message = None
        if duration:
            self.duration = duration
        self.updated_at = Timestamp.now()

        # Calculate processing duration
        processing_duration = 0.0
        if self.started_at:
            processing_duration = self.completed_at.duration_since(
                self.started_at
            ).total_seconds()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                StreamProcessingCompletedEvent(
                    stream_id=self.id,
                    duration_seconds=processing_duration,
                    highlight_count=len(self.highlight_ids),
                )
            )

    def fail_processing(self, error_message: str) -> None:
        """Mark stream processing as failed."""
        # Business rule: Can fail from PENDING or PROCESSING
        if self.status not in [StreamStatus.PENDING, StreamStatus.PROCESSING]:
            transition_info = StateTransitionInfo(
                from_state=self.status.value,
                to_state=StreamStatus.FAILED.value,
                allowed_states=[
                    StreamStatus.PENDING.value,
                    StreamStatus.PROCESSING.value,
                ],
                entity_type="Stream",
                entity_id=self.id,
            )
            raise InvalidStateTransition(transition_info)

        # Update state
        self.status = StreamStatus.FAILED
        self.completed_at = Timestamp.now()
        self.error_message = error_message
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                StreamProcessingFailedEvent(
                    stream_id=self.id, error_message=error_message
                )
            )

    def cancel(self) -> None:
        """Cancel stream processing."""
        # Business rule: Cannot cancel terminal states
        if self.status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            transition_info = StateTransitionInfo(
                from_state=self.status.value,
                to_state=StreamStatus.CANCELLED.value,
                allowed_states=[
                    StreamStatus.PENDING.value,
                    StreamStatus.PROCESSING.value,
                ],
                entity_type="Stream",
                entity_id=self.id,
            )
            raise InvalidStateTransition(transition_info)

        # Update state
        self.status = StreamStatus.CANCELLED
        self.completed_at = Timestamp.now()
        self.error_message = "Cancelled by user"
        self.updated_at = Timestamp.now()

        # Note: We could add a StreamCancelledEvent if needed

    def add_highlight(self, highlight_id: int, confidence_score: float) -> None:
        """Add a highlight to this stream."""
        # Business rule: Can only add highlights during processing
        if self.status != StreamStatus.PROCESSING:
            raise BusinessRuleViolation(
                "Can only add highlights to streams being processed",
                entity_type="Stream",
                entity_id=self.id,
                context={"status": self.status.value},
            )

        # Business rule: No duplicate highlights
        if highlight_id in self.highlight_ids:
            return  # Idempotent operation

        # Update state
        self.highlight_ids.append(highlight_id)
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                HighlightAddedToStreamEvent(
                    stream_id=self.id,
                    highlight_id=highlight_id,
                    confidence_score=confidence_score,
                )
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

    def __str__(self) -> str:
        """Human-readable string representation."""
        title = self.title or "Untitled"
        return f"Stream({title} - {self.status.value})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"Stream(id={self.id}, platform={self.platform.value}, "
            f"status={self.status.value}, highlights={len(self.highlight_ids)})"
        )

    @classmethod
    def create(
        cls,
        url: Url,
        platform: StreamPlatform,
        user_id: int,
        processing_options: ProcessingOptions,
        title: Optional[str] = None,
        channel_name: Optional[str] = None,
        **kwargs,
    ) -> "Stream":
        """Factory method to create a new stream.

        This is the Pythonic way to handle entity creation with
        proper initialization.
        """
        stream = cls(
            id=None,  # Will be assigned by repository
            url=url,
            platform=platform,
            status=StreamStatus.PENDING,
            user_id=user_id,
            processing_options=processing_options,
            title=title,
            channel_name=channel_name,
            game_category=kwargs.get("game_category"),
            language=kwargs.get("language"),
            viewer_count=kwargs.get("viewer_count"),
            duration=None,
            started_at=None,
            completed_at=None,
            error_message=None,
            highlight_ids=[],
            platform_data=kwargs.get("platform_data", {}),
        )

        return stream

    @classmethod
    def from_model(cls, model) -> "Stream":
        """Create domain entity from persistence model.

        This method provides Pythonic mapping without separate mapper classes.
        """
        import json

        # Parse processing options from JSON
        processing_opts_data = (
            json.loads(model.processing_options) if model.processing_options else {}
        )
        processing_options = (
            ProcessingOptions(**processing_opts_data)
            if processing_opts_data
            else ProcessingOptions()
        )

        # Parse platform data
        platform_data = json.loads(model.platform_data) if model.platform_data else {}

        return cls(
            id=model.id,
            url=Url(model.url),
            platform=StreamPlatform(model.platform),
            status=StreamStatus(model.status),
            user_id=model.user_id,
            processing_options=processing_options,
            title=model.title,
            channel_name=model.channel_name,
            game_category=model.game_category,
            language=model.language,
            viewer_count=model.viewer_count,
            duration=Duration(model.duration_seconds)
            if model.duration_seconds
            else None,
            started_at=Timestamp(model.started_at) if model.started_at else None,
            completed_at=Timestamp(model.completed_at) if model.completed_at else None,
            error_message=model.error_message,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
            highlight_ids=[h.id for h in model.highlights] if model.highlights else [],
            platform_data=platform_data,
        )

    def to_model(self):
        """Convert domain entity to persistence model.

        Returns a new persistence model instance with data from this entity.
        """
        import json
        from src.infrastructure.persistence.models.stream import Stream as StreamModel

        model = StreamModel()

        # Set basic attributes
        if self.id is not None:
            model.id = self.id

        model.url = self.url.value
        model.platform = self.platform.value
        model.status = self.status.value
        model.user_id = self.user_id

        # Serialize processing options
        model.processing_options = json.dumps(self.processing_options.to_dict())

        # Set optional attributes
        model.title = self.title
        model.channel_name = self.channel_name
        model.game_category = self.game_category
        model.language = self.language
        model.viewer_count = self.viewer_count
        model.duration_seconds = float(self.duration) if self.duration else None

        # Set timestamps
        model.started_at = self.started_at.value if self.started_at else None
        model.completed_at = self.completed_at.value if self.completed_at else None
        model.error_message = self.error_message

        # Serialize platform data
        model.platform_data = (
            json.dumps(self.platform_data) if self.platform_data else None
        )

        # Set audit timestamps for existing entities
        if self.id is not None:
            model.created_at = self.created_at.value
            model.updated_at = self.updated_at.value

        return model
