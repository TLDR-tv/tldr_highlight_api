"""Stream processing request/response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.infrastructure.persistence.models.stream import StreamPlatform, StreamStatus


class StreamOptions(BaseModel):
    """Stream processing configuration options."""

    # AI Model Configuration
    highlight_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for highlight extraction (0-1)",
    )
    max_highlights: int = Field(
        default=10, ge=1, le=100, description="Maximum number of highlights to extract"
    )
    min_duration: int = Field(
        default=5, ge=1, le=300, description="Minimum highlight duration in seconds"
    )
    max_duration: int = Field(
        default=60, ge=5, le=600, description="Maximum highlight duration in seconds"
    )

    # Processing Options
    enable_audio_analysis: bool = Field(
        default=True, description="Enable audio/speech analysis for highlights"
    )
    enable_chat_analysis: bool = Field(
        default=True, description="Enable chat sentiment analysis (if available)"
    )
    enable_scene_detection: bool = Field(
        default=True, description="Enable scene change detection"
    )

    # Output Configuration
    output_format: str = Field(default="mp4", description="Output video format")
    output_quality: str = Field(default="720p", description="Output video quality")
    generate_thumbnails: bool = Field(
        default=True, description="Generate thumbnail images for highlights"
    )

    # Advanced Options
    custom_tags: List[str] = Field(
        default_factory=list, description="Custom tags to apply to all highlights"
    )
    webhook_events: List[str] = Field(
        default_factory=list, description="Webhook events to trigger for this stream"
    )

    @field_validator("min_duration", "max_duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        """Validate duration values."""
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v

    @field_validator("max_duration")
    @classmethod
    def validate_max_duration(cls, v: int, info) -> int:
        """Validate max duration is greater than min duration."""
        if "min_duration" in info.data and v <= info.data["min_duration"]:
            raise ValueError("max_duration must be greater than min_duration")
        return v


class StreamCreate(BaseModel):
    """Request schema for creating a new stream."""

    source_url: str = Field(description="URL of the stream to process (any FFmpeg-supported format)")
    platform: Optional[StreamPlatform] = Field(default=None, description="Streaming platform type (auto-detected if not provided)")
    options: StreamOptions = Field(
        default_factory=StreamOptions, description="Processing configuration options"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "RTMP Stream",
                    "value": {
                        "source_url": "rtmp://live.example.com/stream/live_stream_key",
                        "options": {
                            "highlight_threshold": 0.85,
                            "max_highlights": 15,
                            "min_duration": 10,
                            "max_duration": 45,
                            "output_quality": "1080p",
                        },
                    }
                },
                {
                    "summary": "HLS Stream",
                    "value": {
                        "source_url": "https://example.com/live/stream.m3u8",
                        "options": {
                            "highlight_threshold": 0.8,
                            "max_highlights": 20,
                        },
                    }
                },
                {
                    "summary": "YouTube Live",
                    "value": {
                        "source_url": "https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
                        "options": {
                            "highlight_threshold": 0.75,
                        },
                    }
                },
                {
                    "summary": "Local Video File",
                    "value": {
                        "source_url": "/path/to/video.mp4",
                        "options": {
                            "highlight_threshold": 0.9,
                            "max_highlights": 10,
                        },
                    }
                },
            ]
        }


class StreamUpdate(BaseModel):
    """Request schema for updating stream configuration."""

    options: Optional[StreamOptions] = Field(
        default=None, description="Updated processing configuration options"
    )

    class Config:
        json_schema_extra = {
            "example": {"options": {"highlight_threshold": 0.9, "max_highlights": 20}}
        }


class StreamResponse(BaseModel):
    """Response schema for stream details."""

    id: int = Field(description="Unique stream identifier")
    source_url: str = Field(description="Source stream URL")
    platform: StreamPlatform = Field(description="Streaming platform")
    status: StreamStatus = Field(description="Current processing status")
    options: Dict[str, Any] = Field(description="Processing configuration")
    user_id: int = Field(description="Owner user ID")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    completed_at: Optional[datetime] = Field(
        default=None, description="Completion timestamp"
    )

    # Computed fields
    is_active: bool = Field(description="Whether stream is actively processing")
    processing_duration: Optional[float] = Field(
        default=None, description="Processing duration in seconds (if completed)"
    )
    highlight_count: int = Field(
        default=0, description="Number of highlights extracted"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 123,
                "source_url": "rtmp://live.example.com/stream/live_stream_key",
                "platform": "rtmp",
                "status": "processing",
                "options": {"highlight_threshold": 0.85, "max_highlights": 15},
                "user_id": 456,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "completed_at": None,
                "is_active": True,
                "processing_duration": None,
                "highlight_count": 0,
            }
        }


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.per_page


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    page: int = Field(description="Current page number")
    per_page: int = Field(description="Items per page")
    total: int = Field(description="Total number of items")
    pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")


class StreamListResponse(PaginatedResponse):
    """Response schema for paginated stream list."""

    items: List[StreamResponse] = Field(description="Stream items")

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "per_page": 20,
                "total": 45,
                "pages": 3,
                "has_next": True,
                "has_prev": False,
                "items": [
                    {
                        "id": 123,
                        "source_url": "rtmp://live.example.com/stream/live_stream_key",
                        "platform": "rtmp",
                        "status": "completed",
                        "user_id": 456,
                        "is_active": False,
                        "highlight_count": 12,
                    }
                ],
            }
        }
