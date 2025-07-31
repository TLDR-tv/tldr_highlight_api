"""Dimension Weight value object.

This immutable value object represents the weight of a dimension
in a dimension set, ensuring valid weight values.
"""

from dataclasses import dataclass
from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class DimensionWeight:
    """Represents the weight of a dimension in scoring calculations.

    This is a value object that ensures weight validity and provides
    weight-related operations.
    """

    dimension_id: str
    value: float

    def __post_init__(self) -> None:
        """Validate weight value."""
        if not 0.0 <= self.value <= 1.0:
            raise InvalidValueError(
                f"Weight must be between 0.0 and 1.0, got {self.value}"
            )

        if not self.dimension_id:
            raise InvalidValueError("Dimension ID cannot be empty")

    def is_zero(self) -> bool:
        """Check if this is a zero weight."""
        return self.value == 0.0

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if weight is significant (above threshold)."""
        return self.value >= threshold

    def normalize_against(self, total: float) -> "DimensionWeight":
        """Create a new normalized weight."""
        if total <= 0:
            raise InvalidValueError("Cannot normalize against zero or negative total")

        return DimensionWeight(
            dimension_id=self.dimension_id, value=min(1.0, self.value / total)
        )

    def __mul__(self, other: float) -> float:
        """Multiply weight by a value (for scoring calculations)."""
        return self.value * other

    def __str__(self) -> str:
        return f"{self.dimension_id}: {self.value:.3f}"
