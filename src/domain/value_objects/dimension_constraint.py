"""Dimension constraint value object.

This value object represents constraints and dependencies between dimensions.
The evaluation logic is handled by domain services, not the value object itself.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from src.domain.exceptions import InvalidValueError


class ConstraintType(str, Enum):
    """Types of constraints between dimensions."""

    REQUIRES = "requires"  # This dimension requires others
    EXCLUDES = "excludes"  # This dimension excludes others
    CONDITIONAL = "conditional"  # Conditional requirement
    MINIMUM_SCORE = "minimum_score"  # Requires minimum score in other dimension
    CORRELATES = "correlates"  # Dimensions should correlate
    INVERSE = "inverse"  # Dimensions should be inversely related


class ConstraintOperator(str, Enum):
    """Operators for constraint conditions."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"


@dataclass(frozen=True)
class DimensionConstraint:
    """Immutable constraint defining relationships between dimensions.

    This value object holds the constraint data. Evaluation logic
    is handled by the constraint_evaluator service.
    """

    # Core constraint definition
    constraint_type: ConstraintType
    source_dimension_id: str
    target_dimension_ids: List[str]

    # Optional conditional logic
    condition_dimension_id: Optional[str] = None
    condition_operator: Optional[ConstraintOperator] = None
    condition_value: Optional[float] = None

    # Constraint metadata
    description: str = ""
    severity: str = "error"  # error, warning, info
    custom_message: Optional[str] = None

    # Additional configuration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate constraint configuration."""
        # Validate source dimension
        if not self.source_dimension_id:
            raise InvalidValueError("Source dimension ID cannot be empty")

        # Validate target dimensions
        if not self.target_dimension_ids:
            raise InvalidValueError("At least one target dimension required")

        # Validate no self-reference
        if self.source_dimension_id in self.target_dimension_ids:
            raise InvalidValueError("Dimension cannot constrain itself")

        # Validate conditional constraint
        if self.constraint_type == ConstraintType.CONDITIONAL:
            if not all(
                [
                    self.condition_dimension_id,
                    self.condition_operator,
                    self.condition_value is not None,
                ]
            ):
                raise InvalidValueError(
                    "Conditional constraints require dimension, operator, and value"
                )

        # Validate severity
        if self.severity not in ["error", "warning", "info"]:
            raise InvalidValueError(f"Invalid severity: {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_type": self.constraint_type.value,
            "source_dimension_id": self.source_dimension_id,
            "target_dimension_ids": self.target_dimension_ids,
            "condition_dimension_id": self.condition_dimension_id,
            "condition_operator": self.condition_operator.value
            if self.condition_operator
            else None,
            "condition_value": self.condition_value,
            "description": self.description,
            "severity": self.severity,
            "custom_message": self.custom_message,
            "metadata": self.metadata,
        }