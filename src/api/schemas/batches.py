"""Batch processing request/response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.infrastructure.persistence.models.batch import BatchStatus
from .streams import PaginatedResponse


class VideoInput(BaseModel):
    """Input video specification for batch processing."""

    url: HttpUrl = Field(description="URL of the video to process")
    title: Optional[str] = Field(
        default=None, max_length=255, description="Optional title for the video"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the video"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/video1.mp4",
                "title": "Game Highlights - Episode 1",
                "metadata": {
                    "game": "CS:GO",
                    "tournament": "Major Championship",
                    "date": "2024-01-15",
                },
            }
        }


class BatchOptions(BaseModel):
    """Batch processing configuration options."""

    # AI Model Configuration
    highlight_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for highlight extraction (0-1)",
    )
    max_highlights_per_video: int = Field(
        default=10, ge=1, le=50, description="Maximum highlights to extract per video"
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
    enable_scene_detection: bool = Field(
        default=True, description="Enable scene change detection"
    )
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing of videos"
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
        default_factory=list, description="Webhook events to trigger for this batch"
    )
    priority: str = Field(
        default="normal", description="Processing priority (low, normal, high)"
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority value."""
        allowed = {"low", "normal", "high"}
        if v not in allowed:
            raise ValueError(f"Priority must be one of: {allowed}")
        return v

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


class BatchCreate(BaseModel):
    """Request schema for creating a new batch job."""

    videos: List[VideoInput] = Field(
        min_length=1,
        max_length=100,
        description="List of videos to process (1-100 videos)",
    )
    options: BatchOptions = Field(
        default_factory=BatchOptions, description="Processing configuration options"
    )

    @field_validator("videos")
    @classmethod
    def validate_videos(cls, v: List[VideoInput]) -> List[VideoInput]:
        """Validate video inputs."""
        if not v:
            raise ValueError("At least one video is required")
        if len(v) > 100:
            raise ValueError("Maximum 100 videos per batch")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "videos": [
                    {"url": "https://example.com/video1.mp4", "title": "Match 1"},
                    {"url": "https://example.com/video2.mp4", "title": "Match 2"},
                ],
                "options": {
                    "highlight_threshold": 0.85,
                    "max_highlights_per_video": 8,
                    "priority": "high",
                },
            }
        }


class BatchUpdate(BaseModel):
    """Request schema for updating batch configuration."""

    options: Optional[BatchOptions] = Field(
        default=None, description="Updated processing configuration options"
    )

    class Config:
        json_schema_extra = {
            "example": {"options": {"highlight_threshold": 0.9, "priority": "high"}}
        }


class BatchResponse(BaseModel):
    """Response schema for batch job details."""

    id: int = Field(description="Unique batch identifier")
    status: BatchStatus = Field(description="Current processing status")
    options: Dict[str, Any] = Field(description="Processing configuration")
    user_id: int = Field(description="Owner user ID")
    video_count: int = Field(description="Total number of videos in batch")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    # Computed fields
    is_active: bool = Field(description="Whether batch is actively processing")
    processed_count: int = Field(default=0, description="Number of videos processed")
    progress_percentage: float = Field(
        default=0.0, description="Processing progress as percentage (0-100)"
    )
    highlight_count: int = Field(
        default=0, description="Total number of highlights extracted"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None, description="Estimated completion time"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 789,
                "status": "processing",
                "options": {"highlight_threshold": 0.85, "max_highlights_per_video": 8},
                "user_id": 456,
                "video_count": 25,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:15:00Z",
                "is_active": True,
                "processed_count": 12,
                "progress_percentage": 48.0,
                "highlight_count": 86,
                "estimated_completion": "2024-01-15T13:45:00Z",
            }
        }


class BatchListResponse(PaginatedResponse):
    """Response schema for paginated batch list."""

    items: List[BatchResponse] = Field(description="Batch items")

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "per_page": 20,
                "total": 15,
                "pages": 1,
                "has_next": False,
                "has_prev": False,
                "items": [
                    {
                        "id": 789,
                        "status": "completed",
                        "user_id": 456,
                        "video_count": 25,
                        "is_active": False,
                        "progress_percentage": 100.0,
                        "highlight_count": 156,
                    }
                ],
            }
        }
