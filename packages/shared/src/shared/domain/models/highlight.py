"""Highlight domain model - represents a detected highlight/clip."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class DimensionScore:
    """A dimension score result for storage.
    
    Represents a single dimensional score with confidence value.
    Both score and confidence must be in the range [0.0, 1.0].
    """

    name: str
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    def __post_init__(self) -> None:
        """Validate that scores are within valid range.
        
        Raises:
            ValueError: If score or confidence is not between 0 and 1.

        """
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )


@dataclass
class Highlight:
    """A detected highlight from a stream.
    
    Represents a significant moment in a stream that was identified
    by the highlight detection system. Contains timing information,
    dimensional scoring results, and associated media files.
    """

    id: UUID = field(default_factory=uuid4)
    stream_id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)

    # Timing
    start_time: float = 0.0  # Seconds from stream start
    end_time: float = 0.0  # Seconds from stream start
    duration: float = 0.0  # Seconds

    # Content
    title: str = ""
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Multi-dimensional scoring results
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    overall_score: float = 0.0  # Aggregated score

    # Media files (S3 paths)
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    # Metadata
    transcript: Optional[str] = None
    wake_word_triggered: bool = False
    wake_word_detected: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Calculate duration if not provided.
        
        Automatically computes the duration from start_time and end_time
        if duration is not explicitly set.
        """
        if not self.duration and self.end_time > self.start_time:
            self.duration = self.end_time - self.start_time

    @property
    def is_high_confidence(self) -> bool:
        """Check if highlight has high confidence scores.
        
        Returns:
            True if average confidence across all dimensions is >= 0.7,
            False otherwise or if no dimension scores exist.

        """
        if not self.dimension_scores:
            return False
        avg_confidence = sum(d.confidence for d in self.dimension_scores) / len(
            self.dimension_scores
        )
        return avg_confidence >= 0.7

    @property
    def top_dimensions(self) -> list[DimensionScore]:
        """Get dimensions sorted by score in descending order.
        
        Returns:
            List of DimensionScore objects sorted from highest to lowest score.

        """
        return sorted(self.dimension_scores, key=lambda d: d.score, reverse=True)

    def add_dimension_score(self, name: str, score: float, confidence: float) -> None:
        """Add a dimension score result.
        
        Args:
            name: Name of the dimension being scored.
            score: Score value between 0.0 and 1.0.
            confidence: Confidence value between 0.0 and 1.0.

        """
        self.dimension_scores.append(DimensionScore(name, score, confidence))
        self._recalculate_overall_score()

    def _recalculate_overall_score(self) -> None:
        """Recalculate overall score from dimension scores.
        
        Computes a weighted average using confidence values as weights.
        If no confidence weights are available, uses simple average.
        """
        if not self.dimension_scores:
            self.overall_score = 0.0
            return

        # Weighted average by confidence
        total_weight = sum(d.confidence for d in self.dimension_scores)
        if total_weight > 0:
            self.overall_score = (
                sum(d.score * d.confidence for d in self.dimension_scores)
                / total_weight
            )
        else:
            self.overall_score = sum(d.score for d in self.dimension_scores) / len(
                self.dimension_scores
            )
