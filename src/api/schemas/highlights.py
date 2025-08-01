"""Pydantic schemas for highlight endpoints."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DimensionScoreResponse(BaseModel):
    """Dimension score response model."""

    name: str = Field(..., description="Dimension name")
    score: float = Field(..., ge=0.0, le=1.0, description="Score value (0-1)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in score (0-1)"
    )


class HighlightResponse(BaseModel):
    """Highlight response model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    stream_id: UUID
    organization_id: UUID

    # Timing
    start_time: float = Field(
        ..., description="Start time in seconds from stream start"
    )
    end_time: float = Field(..., description="End time in seconds from stream start")
    duration: float = Field(..., description="Duration in seconds")

    # Content
    title: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

    # Scoring
    overall_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: list[DimensionScoreResponse] = Field(default_factory=list)

    # Media
    clip_url: Optional[str] = Field(None, description="URL to highlight clip")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail image")

    # Metadata
    transcript: Optional[str] = None
    wake_word_triggered: bool = False
    wake_word_detected: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime


class HighlightListResponse(BaseModel):
    """Response for highlight listing."""

    highlights: list[HighlightResponse]
    total: int = Field(..., description="Total number of highlights matching filters")
    limit: int = Field(..., description="Maximum results per page")
    offset: int = Field(..., description="Current offset for pagination")
    has_more: bool = Field(..., description="Whether more results are available")


class StreamHighlightResponse(BaseModel):
    """Simplified highlight response for stream endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    start_time: float
    end_time: float
    duration: float
    title: str
    overall_score: float
    wake_word_triggered: bool
    created_at: datetime


class StreamHighlightListResponse(BaseModel):
    """Response for stream highlights listing."""

    stream_id: UUID
    highlights: list[StreamHighlightResponse]
    total: int


# Query parameter models
class HighlightListParams(BaseModel):
    """Query parameters for highlight listing."""

    stream_id: Optional[UUID] = Field(None, description="Filter by stream ID")
    wake_word_triggered: Optional[bool] = Field(
        None, description="Filter by wake word trigger status"
    )
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum overall score"
    )
    order_by: str = Field(
        "created_at", pattern="^(created_at|score)$", description="Sort order"
    )
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Pagination offset")
