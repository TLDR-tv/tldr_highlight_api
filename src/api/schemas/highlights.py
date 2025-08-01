"""Highlight management request/response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .streams import PaginatedResponse
from src.domain.enums import SourceType, SortOrder, HighlightSortField


class HighlightFilters(BaseModel):
    """Filters for highlight search and listing."""

    # Source filters
    stream_id: Optional[int] = Field(
        default=None, description="Filter by specific stream ID"
    )
    source_type: Optional[SourceType] = Field(
        default=None, description="Filter by source type (stream)"
    )

    # Content filters
    min_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence score"
    )
    max_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Maximum confidence score"
    )
    min_duration: Optional[int] = Field(
        default=None, ge=1, description="Minimum duration in seconds"
    )
    max_duration: Optional[int] = Field(
        default=None, ge=1, description="Maximum duration in seconds"
    )

    # Tag and text filters
    tags: Optional[List[str]] = Field(
        default=None, description="Filter by tags (any match)"
    )
    title_contains: Optional[str] = Field(
        default=None, description="Filter by title containing text"
    )
    description_contains: Optional[str] = Field(
        default=None, description="Filter by description containing text"
    )

    # Date filters
    created_after: Optional[datetime] = Field(
        default=None, description="Filter highlights created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Filter highlights created before this date"
    )

    # Sorting
    sort_by: HighlightSortField = Field(
        default=HighlightSortField.CREATED_AT,
        description="Sort field (created_at, confidence_score, duration, timestamp)",
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC, description="Sort order (asc, desc)"
    )

    @field_validator("max_confidence")
    @classmethod
    def validate_max_confidence(cls, v: Optional[float], info) -> Optional[float]:
        """Validate max confidence is greater than min confidence."""
        if v is not None and "min_confidence" in info.data:
            min_conf = info.data["min_confidence"]
            if min_conf is not None and v <= min_conf:
                raise ValueError("max_confidence must be greater than min_confidence")
        return v

    @field_validator("max_duration")
    @classmethod
    def validate_max_duration(cls, v: Optional[int], info) -> Optional[int]:
        """Validate max duration is greater than min duration."""
        if v is not None and "min_duration" in info.data:
            min_dur = info.data["min_duration"]
            if min_dur is not None and v <= min_dur:
                raise ValueError("max_duration must be greater than min_duration")
        return v


class HighlightSearch(BaseModel):
    """Advanced search request for highlights."""

    query: Optional[str] = Field(
        default=None, description="Full-text search query across title and description"
    )
    filters: HighlightFilters = Field(
        default_factory=HighlightFilters, description="Search filters"
    )
    include_low_confidence: bool = Field(
        default=False, description="Include highlights with confidence < 0.7"
    )
    group_by_source: bool = Field(
        default=False, description="Group results by source stream"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "epic moment",
                "filters": {
                    "min_confidence": 0.8,
                    "tags": ["gaming", "highlights"],
                    "min_duration": 10,
                    "created_after": "2024-01-01T00:00:00Z",
                },
                "include_low_confidence": False,
                "group_by_source": True,
            }
        }


class HighlightUpdate(BaseModel):
    """Request schema for updating highlight metadata."""

    title: Optional[str] = Field(
        default=None, max_length=255, description="Updated title"
    )
    description: Optional[str] = Field(default=None, description="Updated description")
    tags: Optional[List[str]] = Field(default=None, description="Updated tags list")
    extra_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated extra metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Amazing Triple Kill",
                "description": "Incredible three-player elimination sequence",
                "tags": ["gaming", "fps", "clutch"],
                "extra_metadata": {
                    "game_mode": "competitive",
                    "map": "dust2",
                    "weapon": "ak47",
                },
            }
        }


class HighlightResponse(BaseModel):
    """Response schema for highlight details."""

    id: int = Field(description="Unique highlight identifier")
    stream_id: Optional[int] = Field(description="Source stream ID")
    title: str = Field(description="Highlight title")
    description: Optional[str] = Field(description="Highlight description")
    video_url: str = Field(description="URL to highlight video")
    thumbnail_url: Optional[str] = Field(description="URL to thumbnail image")
    duration: float = Field(description="Duration in seconds")
    timestamp: int = Field(description="Original timestamp in source (seconds)")
    confidence_score: float = Field(description="AI confidence score (0-1)")
    tags: List[str] = Field(description="Associated tags")
    extra_metadata: Dict[str, Any] = Field(description="Additional metadata")
    created_at: datetime = Field(description="Creation timestamp")

    # Computed fields
    source_type: str = Field(description="Source type (stream)")
    is_high_confidence: bool = Field(description="Whether confidence > 0.8")
    download_url: Optional[str] = Field(
        default=None, description="Presigned download URL (if requested)"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 12345,
                "stream_id": 123,
                "title": "Epic Gaming Moment",
                "description": "Player makes incredible comeback",
                "video_url": "https://storage.example.com/highlights/12345.mp4",
                "thumbnail_url": "https://storage.example.com/thumbnails/12345.jpg",
                "duration": 28.5,
                "timestamp": 1847,
                "confidence_score": 0.92,
                "tags": ["gaming", "comeback", "clutch"],
                "extra_metadata": {"game": "CS:GO", "round": 15, "score": "14-15"},
                "created_at": "2024-01-15T12:34:56Z",
                "source_type": "stream",
                "is_high_confidence": True,
                "download_url": "https://storage.example.com/downloads/12345?expires=1642345200",
            }
        }


class HighlightListResponse(PaginatedResponse):
    """Response schema for paginated highlight list."""

    items: List[HighlightResponse] = Field(description="Highlight items")

    # Aggregation data
    total_duration: float = Field(
        default=0.0, description="Total duration of all highlights in seconds"
    )
    avg_confidence: float = Field(default=0.0, description="Average confidence score")
    tag_counts: Dict[str, int] = Field(
        default_factory=dict, description="Tag frequency counts"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "page": 1,
                "per_page": 20,
                "total": 156,
                "pages": 8,
                "has_next": True,
                "has_prev": False,
                "items": [
                    {
                        "id": 12345,
                        "title": "Epic Gaming Moment",
                        "confidence_score": 0.92,
                        "duration": 28.5,
                        "source_type": "stream",
                    }
                ],
                "total_duration": 2847.3,
                "avg_confidence": 0.86,
                "tag_counts": {"gaming": 45, "highlights": 156, "clutch": 23},
            }
        }
