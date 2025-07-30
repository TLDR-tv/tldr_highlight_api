"""Pydantic schemas for Gemini structured outputs.

This module defines the structured response schemas used by Gemini's video understanding API
to ensure consistent, type-safe highlight detection results.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# Removed HighlightType enum - types are now dynamic from HighlightTypeRegistry


class AudioQuality(str, Enum):
    """Audio quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NO_AUDIO = "no_audio"


class VideoQuality(str, Enum):
    """Video quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CORRUPTED = "corrupted"


class DimensionScores(BaseModel):
    """Dynamic scoring dimensions for highlight evaluation.

    This model is dynamically constructed based on the DimensionSet
    being used for analysis. The actual fields are determined at runtime.
    """

    # Dynamic fields - actual dimension scores will be set as dict items
    __root__: Dict[str, float] = Field(
        description="Dictionary of dimension ID to score mappings"
    )

    @validator("__root__")
    def validate_scores(cls, v):
        """Ensure all scores are within valid range."""
        for dim_id, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Score for dimension '{dim_id}' must be between 0.0 and 1.0, got {score}"
                )
        return v

    def get_score(self, dimension_id: str, default: float = 0.0) -> float:
        """Get score for a specific dimension."""
        return self.__root__.get(dimension_id, default)

    def to_dict(self) -> Dict[str, float]:
        """Convert to simple dictionary."""
        return self.__root__


class SceneDescription(BaseModel):
    """Description of a specific scene or moment in the video."""

    timestamp: str = Field(description="Timestamp in MM:SS format")
    description: str = Field(description="What's visible or happening in the scene")
    objects_detected: Optional[List[str]] = Field(
        default=None, description="Objects or entities detected in the scene"
    )
    actions_detected: Optional[List[str]] = Field(
        default=None, description="Actions or movements detected"
    )


class GeminiHighlight(BaseModel):
    """Individual highlight detected by Gemini."""

    start_time: str = Field(description="Start timestamp in MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS format")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for this highlight"
    )
    type: str = Field(
        default="general",
        description="Type of highlight detected (from organization's type registry)",
    )
    description: str = Field(description="Detailed description of what happens")
    ranking_score: float = Field(
        ge=0.0, le=1.0, description="Overall ranking score based on all dimensions"
    )
    dimension_scores: Dict[str, float] = Field(
        description="Individual scores for each dimension"
    )
    key_moments: Optional[List[str]] = Field(
        default=None,
        description="Specific timestamps of key moments within the highlight",
    )
    transcript_excerpt: Optional[str] = Field(
        default=None, description="Relevant transcript from this highlight"
    )
    viewer_impact: Optional[str] = Field(
        default=None, description="Expected viewer reaction or impact"
    )


class SegmentQuality(BaseModel):
    """Overall quality assessment of a video segment."""

    overall_score: float = Field(
        ge=0.0, le=1.0, description="Overall quality score of the segment"
    )
    has_highlights: bool = Field(
        description="Whether the segment contains any highlights"
    )
    audio_quality: AudioQuality = Field(description="Audio quality assessment")
    video_quality: VideoQuality = Field(description="Video quality assessment")
    content_richness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How content-rich this segment is"
    )
    engagement_potential: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Potential for viewer engagement"
    )


class GeminiVideoAnalysis(BaseModel):
    """Complete video analysis response from Gemini."""

    highlights: List[GeminiHighlight] = Field(description="List of detected highlights")
    transcript: Optional[str] = Field(
        default=None, description="Full transcript of speech and important audio"
    )
    scene_descriptions: Optional[List[SceneDescription]] = Field(
        default=None, description="Descriptions of key visual moments"
    )
    segment_quality: SegmentQuality = Field(
        description="Overall segment quality assessment"
    )
    content_summary: Optional[str] = Field(
        default=None, description="Brief summary of the segment's content"
    )
    recommended_clips: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Suggested clip boundaries for optimal highlights"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the analysis"
    )


class HighlightRefinement(BaseModel):
    """Refinement suggestions for a highlight."""

    highlight_id: str = Field(description="ID of the highlight being refined")
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality assessment of the highlight"
    )
    adjusted_start_time: Optional[str] = Field(
        default=None, description="Suggested adjusted start time"
    )
    adjusted_end_time: Optional[str] = Field(
        default=None, description="Suggested adjusted end time"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improving the highlight"
    )
    merge_with: Optional[str] = Field(
        default=None, description="ID of another highlight to merge with"
    )
    should_keep: bool = Field(
        default=True, description="Whether this highlight should be kept"
    )
    refined_description: Optional[str] = Field(
        default=None, description="Improved description if needed"
    )
    refined_type: Optional[str] = Field(
        default=None,
        description="More accurate type classification (from organization's type registry)",
    )


class HighlightRefinementBatch(BaseModel):
    """Batch refinement response for multiple highlights."""

    refinements: List[HighlightRefinement] = Field(
        description="List of refinement suggestions"
    )
    overall_quality: float = Field(
        ge=0.0, le=1.0, description="Overall quality of the highlight batch"
    )
    optimization_notes: Optional[str] = Field(
        default=None, description="General notes about optimization opportunities"
    )
