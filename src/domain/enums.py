"""Domain enums for improved validation.

This module provides Pythonic enums to replace string-based validation
throughout the application, improving type safety and reducing errors.
"""

from enum import Enum


class ProcessingPriority(str, Enum):
    """Processing priority levels for stream processing."""

    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"

    def __str__(self) -> str:
        return self.value


class OutputFormat(str, Enum):
    """Supported output video formats."""

    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"

    def __str__(self) -> str:
        return self.value


class OutputQuality(str, Enum):
    """Supported output video quality levels."""

    LOW = "480p"
    STANDARD = "720p"
    HIGH = "1080p"
    ULTRA = "1440p"
    FOUR_K = "2160p"

    def __str__(self) -> str:
        return self.value

    @property
    def height(self) -> int:
        """Get height in pixels."""
        return int(self.value.rstrip("p"))


class WebhookEvent(str, Enum):
    """Supported webhook event types."""

    STREAM_STARTED = "stream.started"
    STREAM_COMPLETED = "stream.completed"
    STREAM_FAILED = "stream.failed"
    HIGHLIGHT_CREATED = "highlight.created"
    HIGHLIGHT_UPDATED = "highlight.updated"
    PROCESSING_PROGRESS = "processing.progress"

    def __str__(self) -> str:
        return self.value


class AnalysisMethod(str, Enum):
    """Analysis method types for highlight detection."""

    AI_ONLY = "ai_only"

    def __str__(self) -> str:
        return self.value


class FusionStrategy(str, Enum):
    """Strategy for fusing multi-modal analysis results."""

    WEIGHTED = "weighted"
    CONSENSUS = "consensus"
    CASCADE = "cascade"
    MAX_CONFIDENCE = "max_confidence"

    def __str__(self) -> str:
        return self.value


class ContentType(str, Enum):
    """Content type presets for processing optimization."""

    GAMING = "gaming"
    EDUCATION = "education"
    SPORTS = "sports"
    CORPORATE = "corporate"
    ENTERTAINMENT = "entertainment"
    MUSIC = "music"

    def __str__(self) -> str:
        return self.value


class SourceType(str, Enum):
    """Source type for highlights."""

    STREAM = "stream"
    BATCH = "batch"

    def __str__(self) -> str:
        return self.value


class SortOrder(str, Enum):
    """Sort order options."""

    ASC = "asc"
    DESC = "desc"

    def __str__(self) -> str:
        return self.value


class HighlightSortField(str, Enum):
    """Available sort fields for highlights."""

    CREATED_AT = "created_at"
    CONFIDENCE_SCORE = "confidence_score"
    DURATION = "duration"
    TIMESTAMP = "timestamp"

    def __str__(self) -> str:
        return self.value
