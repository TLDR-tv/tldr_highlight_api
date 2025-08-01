"""Processing options value object for highlight detection.

Simple configuration object holding processing parameters.
Complex logic should be handled by domain services.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.domain.exceptions import InvalidValueError


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
    processing_priority: str = "balanced"  # speed, quality, balanced
    
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
            raise InvalidValueError(
                "Maximum duration must be >= minimum duration"
            )

        if not (
            self.min_highlight_duration
            <= self.typical_highlight_duration
            <= self.max_highlight_duration
        ):
            raise InvalidValueError("Typical duration must be between min and max")

        # Validate priority
        valid_priorities = {"speed", "quality", "balanced"}
        if self.processing_priority not in valid_priorities:
            raise InvalidValueError(
                f"Priority must be one of {valid_priorities}, got {self.processing_priority}"
            )

    @classmethod
    def for_gaming(cls) -> "ProcessingOptions":
        """Create processing options optimized for gaming content."""
        return cls(
            min_highlight_duration=15.0,
            max_highlight_duration=90.0,
            typical_highlight_duration=45.0,
            target_confidence_threshold=0.75,
            processing_priority="balanced",
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
            processing_priority="quality",
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
            processing_priority="speed",
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
            processing_priority="quality",
            metadata={"preset": "corporate"},
        )