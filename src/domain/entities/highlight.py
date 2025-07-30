"""Highlight domain entity."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.domain.entities.base import Entity
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.url import Url


@dataclass
class Highlight(Entity[int]):
    """Domain entity representing a detected highlight.

    Highlights are specific moments in streams that have been
    identified as noteworthy by the AI analysis.
    """

    stream_id: int
    start_time: Duration  # Offset from stream start
    end_time: Duration  # Offset from stream start
    confidence_score: ConfidenceScore

    # Content details
    title: str
    description: str
    highlight_types: List[str] = field(default_factory=list)  # Flexible type IDs
    thumbnail_url: Optional[Url] = None
    clip_url: Optional[Url] = None

    # Analysis metadata
    tags: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    viewer_engagement: Optional[float] = None  # 0.0 to 1.0

    # AI analysis results
    video_analysis: Dict[str, Any] = field(default_factory=dict)
    audio_analysis: Dict[str, Any] = field(default_factory=dict)
    chat_analysis: Dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    processed_by: Optional[str] = None  # AI model/version

    @property
    def duration(self) -> Duration:
        """Calculate highlight duration."""
        return self.end_time - self.start_time

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence highlight."""
        return self.confidence_score.is_high_confidence()

    def update_urls(
        self, thumbnail_url: Optional[Url] = None, clip_url: Optional[Url] = None
    ) -> "Highlight":
        """Update highlight URLs after clip generation."""
        return Highlight(
            id=self.id,
            stream_id=self.stream_id,
            start_time=self.start_time,
            end_time=self.end_time,
            confidence_score=self.confidence_score,
            highlight_types=self.highlight_types.copy(),
            title=self.title,
            description=self.description,
            thumbnail_url=thumbnail_url or self.thumbnail_url,
            clip_url=clip_url or self.clip_url,
            tags=self.tags.copy(),
            sentiment_score=self.sentiment_score,
            viewer_engagement=self.viewer_engagement,
            video_analysis=self.video_analysis.copy(),
            audio_analysis=self.audio_analysis.copy(),
            chat_analysis=self.chat_analysis.copy(),
            processed_by=self.processed_by,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def add_tag(self, tag: str) -> "Highlight":
        """Add a tag to the highlight."""
        if tag in self.tags:
            return self

        new_tags = self.tags.copy()
        new_tags.append(tag)

        return Highlight(
            id=self.id,
            stream_id=self.stream_id,
            start_time=self.start_time,
            end_time=self.end_time,
            confidence_score=self.confidence_score,
            highlight_types=self.highlight_types.copy(),
            title=self.title,
            description=self.description,
            thumbnail_url=self.thumbnail_url,
            clip_url=self.clip_url,
            tags=new_tags,
            sentiment_score=self.sentiment_score,
            viewer_engagement=self.viewer_engagement,
            video_analysis=self.video_analysis.copy(),
            audio_analysis=self.audio_analysis.copy(),
            chat_analysis=self.chat_analysis.copy(),
            processed_by=self.processed_by,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
        )

    def matches_filter(
        self,
        min_confidence: Optional[float] = None,
        types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Check if highlight matches given filters."""
        if min_confidence and self.confidence_score.value < min_confidence:
            return False

        if types and not any(t in self.highlight_types for t in types):
            return False

        if tags and not any(tag in self.tags for tag in tags):
            return False

        return True

    @property
    def sentiment_label(self) -> str:
        """Get sentiment as a label."""
        if self.sentiment_score is None:
            return "neutral"
        elif self.sentiment_score > 0.5:
            return "positive"
        elif self.sentiment_score < -0.5:
            return "negative"
        else:
            return "neutral"

    @property
    def engagement_level(self) -> str:
        """Get engagement level as a label."""
        if self.viewer_engagement is None:
            return "unknown"
        elif self.viewer_engagement > 0.8:
            return "very_high"
        elif self.viewer_engagement > 0.6:
            return "high"
        elif self.viewer_engagement > 0.4:
            return "medium"
        elif self.viewer_engagement > 0.2:
            return "low"
        else:
            return "very_low"
