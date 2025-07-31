"""Dimension Score value object.

This immutable value object represents a score for a specific dimension,
encapsulating scoring logic and validation.
"""

from dataclasses import dataclass
from typing import Optional
from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class DimensionScore:
    """Represents a score for a specific dimension.

    This value object ensures score validity and provides
    score-related operations and comparisons.
    """

    dimension_id: str
    value: float
    confidence: Optional[str] = None  # high, medium, low, uncertain
    evidence: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate score value."""
        if not 0.0 <= self.value <= 1.0:
            raise InvalidValueError(
                f"Score must be between 0.0 and 1.0, got {self.value}"
            )

        if not self.dimension_id:
            raise InvalidValueError("Dimension ID cannot be empty")

        if self.confidence and self.confidence not in [
            "high",
            "medium",
            "low",
            "uncertain",
        ]:
            raise InvalidValueError(f"Invalid confidence level: {self.confidence}")

    def meets_threshold(self, threshold: float) -> bool:
        """Check if score meets a given threshold."""
        return self.value >= threshold

    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence score."""
        return self.confidence == "high"

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Check if score is significant."""
        return self.value >= threshold

    def combine_with(
        self, other: "DimensionScore", method: str = "average"
    ) -> "DimensionScore":
        """Combine with another score using specified method."""
        if self.dimension_id != other.dimension_id:
            raise InvalidValueError("Cannot combine scores from different dimensions")

        if method == "average":
            new_value = (self.value + other.value) / 2
        elif method == "max":
            new_value = max(self.value, other.value)
        elif method == "min":
            new_value = min(self.value, other.value)
        else:
            raise InvalidValueError(f"Unknown combination method: {method}")

        # Determine combined confidence
        confidence_levels = ["uncertain", "low", "medium", "high"]
        new_confidence: Optional[str] = None
        if self.confidence and other.confidence:
            self_idx = confidence_levels.index(self.confidence)
            other_idx = confidence_levels.index(other.confidence)
            # Take the lower confidence
            new_confidence = confidence_levels[min(self_idx, other_idx)]
        elif self.confidence:
            new_confidence = self.confidence
        elif other.confidence:
            new_confidence = other.confidence

        return DimensionScore(
            dimension_id=self.dimension_id,
            value=new_value,
            confidence=new_confidence,
            evidence=f"Combined: {self.evidence or 'N/A'} + {other.evidence or 'N/A'}",
        )

    def with_confidence(self, confidence: str) -> "DimensionScore":
        """Create a new score with updated confidence."""
        return DimensionScore(
            dimension_id=self.dimension_id,
            value=self.value,
            confidence=confidence,
            evidence=self.evidence,
        )

    def __str__(self) -> str:
        conf = f" ({self.confidence})" if self.confidence else ""
        return f"{self.dimension_id}: {self.value:.3f}{conf}"
