"""Processing options value object for highlight detection.

Simple configuration object holding processing parameters.
Complex logic should be handled by domain services.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from src.domain.exceptions import InvalidValueError
from src.domain.enums import ProcessingPriority


@dataclass(frozen=True)
class ProcessingOptions:
    """Processing configuration for highlight detection.

    This immutable value object holds configuration parameters.
    All processing logic is handled by domain services.
    """

    # Core configuration
    dimension_set_id: Optional[int] = None

    # Timing constraints (seconds)
    min_highlight_duration: float = 10.0
    max_highlight_duration: float = 300.0
    typical_highlight_duration: float = 60.0

    # Quality thresholds (0.0 to 1.0)
    min_confidence_threshold: float = 0.5
    target_confidence_threshold: float = 0.7
    exceptional_threshold: float = 0.85

    # Analysis windows (seconds)
    analysis_window_seconds: float = 30.0
    context_window_seconds: float = 120.0
    lookahead_seconds: float = 15.0

    # Post-processing
    merge_nearby_highlights: bool = True
    merge_threshold_seconds: float = 10.0
    remove_duplicates: bool = True
    similarity_threshold: float = 0.8

    # Performance
    processing_priority: ProcessingPriority = ProcessingPriority.BALANCED

    # Custom configuration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate processing options."""
        # Validate thresholds
        thresholds = [
            self.min_confidence_threshold,
            self.target_confidence_threshold,
            self.exceptional_threshold,
        ]
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise InvalidValueError(
                    f"Confidence thresholds must be between 0.0 and 1.0, got {threshold}"
                )

        if not (
            self.min_confidence_threshold
            <= self.target_confidence_threshold
            <= self.exceptional_threshold
        ):
            raise InvalidValueError(
                "Thresholds must be in ascending order: min <= target <= exceptional"
            )

        # Validate durations
        if self.min_highlight_duration < 0:
            raise InvalidValueError("Minimum highlight duration cannot be negative")

        if self.max_highlight_duration < self.min_highlight_duration:
            raise InvalidValueError("Maximum duration must be >= minimum duration")

        if not (
            self.min_highlight_duration
            <= self.typical_highlight_duration
            <= self.max_highlight_duration
        ):
            raise InvalidValueError("Typical duration must be between min and max")

        # Priority validation is handled by the enum

    @classmethod
    def for_gaming(cls) -> "ProcessingOptions":
        """Create processing options optimized for gaming content."""
        return cls(
            min_highlight_duration=15.0,
            max_highlight_duration=90.0,
            typical_highlight_duration=45.0,
            target_confidence_threshold=0.75,
            processing_priority=ProcessingPriority.BALANCED,
            metadata={"preset": "gaming"},
        )

    @classmethod
    def for_education(cls) -> "ProcessingOptions":
        """Create processing options optimized for educational content."""
        return cls(
            min_highlight_duration=30.0,
            max_highlight_duration=300.0,
            typical_highlight_duration=120.0,
            min_confidence_threshold=0.6,
            target_confidence_threshold=0.8,
            processing_priority=ProcessingPriority.QUALITY,
            metadata={"preset": "education"},
        )

    @classmethod
    def for_sports(cls) -> "ProcessingOptions":
        """Create processing options optimized for sports content."""
        return cls(
            min_highlight_duration=10.0,
            max_highlight_duration=60.0,
            typical_highlight_duration=30.0,
            target_confidence_threshold=0.8,
            processing_priority=ProcessingPriority.SPEED,
            merge_threshold_seconds=5.0,
            metadata={"preset": "sports"},
        )

    @classmethod
    def for_corporate(cls) -> "ProcessingOptions":
        """Create processing options optimized for corporate/meeting content."""
        return cls(
            min_highlight_duration=20.0,
            max_highlight_duration=180.0,
            typical_highlight_duration=60.0,
            min_confidence_threshold=0.7,
            target_confidence_threshold=0.85,
            processing_priority=ProcessingPriority.QUALITY,
            metadata={"preset": "corporate"},
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dimension_set_id": self.dimension_set_id,
            "min_highlight_duration": self.min_highlight_duration,
            "max_highlight_duration": self.max_highlight_duration,
            "typical_highlight_duration": self.typical_highlight_duration,
            "min_confidence_threshold": self.min_confidence_threshold,
            "target_confidence_threshold": self.target_confidence_threshold,
            "exceptional_threshold": self.exceptional_threshold,
            "analysis_window_seconds": self.analysis_window_seconds,
            "context_window_seconds": self.context_window_seconds,
            "lookahead_seconds": self.lookahead_seconds,
            "merge_nearby_highlights": self.merge_nearby_highlights,
            "merge_threshold_seconds": self.merge_threshold_seconds,
            "remove_duplicates": self.remove_duplicates,
            "similarity_threshold": self.similarity_threshold,
            "processing_priority": self.processing_priority,
            "metadata": self.metadata,
        }
