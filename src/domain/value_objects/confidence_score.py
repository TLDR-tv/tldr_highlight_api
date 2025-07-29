"""Confidence score value object."""

from dataclasses import dataclass
from typing import ClassVar

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ConfidenceScore:
    """Value object representing an AI confidence score.
    
    This is an immutable value object that ensures confidence scores
    are within the valid range of 0.0 to 1.0.
    """
    
    value: float
    
    # Thresholds for different confidence levels
    HIGH_THRESHOLD: ClassVar[float] = 0.8
    MEDIUM_THRESHOLD: ClassVar[float] = 0.5
    LOW_THRESHOLD: ClassVar[float] = 0.3
    
    def __post_init__(self):
        """Validate confidence score range after initialization."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(
                f"Confidence score must be a number, got {type(self.value).__name__}"
            )
        
        if not 0.0 <= self.value <= 1.0:
            raise InvalidValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.value}"
            )
    
    @property
    def percentage(self) -> float:
        """Get confidence as a percentage (0-100)."""
        return self.value * 100
    
    @property
    def level(self) -> str:
        """Get confidence level as a string."""
        if self.value >= self.HIGH_THRESHOLD:
            return "high"
        elif self.value >= self.MEDIUM_THRESHOLD:
            return "medium"
        elif self.value >= self.LOW_THRESHOLD:
            return "low"
        else:
            return "very_low"
    
    def is_high_confidence(self, threshold: float = None) -> bool:
        """Check if this is a high confidence score."""
        threshold = threshold or self.HIGH_THRESHOLD
        return self.value >= threshold
    
    def is_above_threshold(self, threshold: float) -> bool:
        """Check if confidence exceeds a given threshold."""
        return self.value >= threshold
    
    def __str__(self) -> str:
        """String representation shows percentage."""
        return f"{self.percentage:.1f}%"
    
    def __float__(self) -> float:
        """Allow conversion to float."""
        return self.value