"""Dimension constraint value object.

This value object represents constraints and dependencies between dimensions,
enabling complex relationships and validation rules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
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

    This value object enables expressing complex business rules about
    how dimensions relate to each other.
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

    def evaluate(
        self,
        dimension_scores: Dict[str, float],
        dimension_presence: Optional[Dict[str, bool]] = None,
    ) -> "ConstraintEvaluationResult":
        """Evaluate if this constraint is satisfied.

        Args:
            dimension_scores: Current dimension scores
            dimension_presence: Which dimensions are present/evaluated

        Returns:
            Evaluation result with satisfaction status and messages
        """
        # Check if constraint applies (conditional logic)
        if not self._constraint_applies(dimension_scores):
            return ConstraintEvaluationResult(
                satisfied=True,
                applicable=False,
                message="Constraint not applicable in current context",
            )

        # Evaluate based on constraint type
        if self.constraint_type == ConstraintType.REQUIRES:
            return self._evaluate_requires(dimension_scores, dimension_presence)
        elif self.constraint_type == ConstraintType.EXCLUDES:
            return self._evaluate_excludes(dimension_scores, dimension_presence)
        elif self.constraint_type == ConstraintType.MINIMUM_SCORE:
            return self._evaluate_minimum_score(dimension_scores)
        elif self.constraint_type == ConstraintType.CORRELATES:
            return self._evaluate_correlation(dimension_scores)
        elif self.constraint_type == ConstraintType.INVERSE:
            return self._evaluate_inverse(dimension_scores)
        else:
            return self._evaluate_conditional(dimension_scores)

    def _constraint_applies(self, dimension_scores: Dict[str, float]) -> bool:
        """Check if conditional constraint applies."""
        if self.constraint_type != ConstraintType.CONDITIONAL:
            return True

        if not self.condition_dimension_id:
            return True

        if self.condition_dimension_id not in dimension_scores:
            return False

        condition_score = dimension_scores[self.condition_dimension_id]

        # Ensure condition_value is not None
        if self.condition_value is None:
            return True

        # Evaluate condition
        if self.condition_operator == ConstraintOperator.GREATER_THAN:
            return condition_score > self.condition_value
        elif self.condition_operator == ConstraintOperator.LESS_THAN:
            return condition_score < self.condition_value
        elif self.condition_operator == ConstraintOperator.EQUALS:
            return abs(condition_score - self.condition_value) < 0.001
        elif self.condition_operator == ConstraintOperator.NOT_EQUALS:
            return abs(condition_score - self.condition_value) >= 0.001
        elif self.condition_operator == ConstraintOperator.GREATER_EQUAL:
            return condition_score >= self.condition_value
        elif self.condition_operator == ConstraintOperator.LESS_EQUAL:
            return condition_score <= self.condition_value

        return True

    def _evaluate_requires(
        self,
        dimension_scores: Dict[str, float],
        dimension_presence: Optional[Dict[str, bool]],
    ) -> "ConstraintEvaluationResult":
        """Evaluate REQUIRES constraint."""
        if dimension_presence is None:
            dimension_presence = {
                dim_id: dim_id in dimension_scores
                for dim_id in self.target_dimension_ids
            }

        # Check if source dimension is present
        if self.source_dimension_id not in dimension_scores:
            return ConstraintEvaluationResult(
                satisfied=True,
                applicable=False,
                message=f"Source dimension '{self.source_dimension_id}' not evaluated",
            )

        # Check if source has significant score
        if dimension_scores[self.source_dimension_id] < 0.1:
            return ConstraintEvaluationResult(
                satisfied=True,
                applicable=False,
                message="Source dimension score too low to require dependencies",
            )

        # Check required dimensions
        missing = [
            dim_id
            for dim_id in self.target_dimension_ids
            if not dimension_presence.get(dim_id, False)
        ]

        if missing:
            message = self.custom_message or (
                f"Dimension '{self.source_dimension_id}' requires: {', '.join(missing)}"
            )
            return ConstraintEvaluationResult(
                satisfied=False,
                applicable=True,
                message=message,
                severity=self.severity,
                missing_dimensions=missing,
            )

        return ConstraintEvaluationResult(satisfied=True, applicable=True)

    def _evaluate_excludes(
        self,
        dimension_scores: Dict[str, float],
        dimension_presence: Optional[Dict[str, bool]],
    ) -> "ConstraintEvaluationResult":
        """Evaluate EXCLUDES constraint."""
        if dimension_presence is None:
            dimension_presence = {
                dim_id: dim_id in dimension_scores
                for dim_id in self.target_dimension_ids
            }

        # Check if source dimension is present with significant score
        if (
            self.source_dimension_id not in dimension_scores
            or dimension_scores[self.source_dimension_id] < 0.1
        ):
            return ConstraintEvaluationResult(satisfied=True, applicable=False)

        # Check for excluded dimensions
        present_excluded = [
            dim_id
            for dim_id in self.target_dimension_ids
            if dimension_presence.get(dim_id, False)
            and dimension_scores.get(dim_id, 0) >= 0.1
        ]

        if present_excluded:
            message = self.custom_message or (
                f"Dimension '{self.source_dimension_id}' excludes: {', '.join(present_excluded)}"
            )
            return ConstraintEvaluationResult(
                satisfied=False,
                applicable=True,
                message=message,
                severity=self.severity,
                conflicting_dimensions=present_excluded,
            )

        return ConstraintEvaluationResult(satisfied=True, applicable=True)

    def _evaluate_minimum_score(
        self, dimension_scores: Dict[str, float]
    ) -> "ConstraintEvaluationResult":
        """Evaluate MINIMUM_SCORE constraint."""
        # Source must have minimum score
        source_score = dimension_scores.get(self.source_dimension_id, 0)

        if source_score < 0.1:
            return ConstraintEvaluationResult(satisfied=True, applicable=False)

        # Check target dimensions meet minimum
        below_minimum = []
        for target_id in self.target_dimension_ids:
            target_score = dimension_scores.get(target_id, 0)
            if target_score < (self.condition_value or 0.5):
                below_minimum.append(f"{target_id}={target_score:.2f}")

        if below_minimum:
            message = self.custom_message or (
                f"Dimensions below minimum score {self.condition_value}: "
                f"{', '.join(below_minimum)}"
            )
            return ConstraintEvaluationResult(
                satisfied=False,
                applicable=True,
                message=message,
                severity=self.severity,
            )

        return ConstraintEvaluationResult(satisfied=True, applicable=True)

    def _evaluate_correlation(
        self, dimension_scores: Dict[str, float]
    ) -> "ConstraintEvaluationResult":
        """Evaluate CORRELATES constraint."""
        source_score = dimension_scores.get(self.source_dimension_id, 0)

        # Check correlation with targets
        uncorrelated = []
        for target_id in self.target_dimension_ids:
            target_score = dimension_scores.get(target_id, 0)

            # Simple correlation check - scores should be similar
            if abs(source_score - target_score) > 0.3:
                uncorrelated.append(
                    f"{target_id} ({source_score:.2f} vs {target_score:.2f})"
                )

        if uncorrelated:
            message = self.custom_message or (
                f"Expected correlation not found: {', '.join(uncorrelated)}"
            )
            return ConstraintEvaluationResult(
                satisfied=False,
                applicable=True,
                message=message,
                severity="warning",  # Correlation is usually a warning
            )

        return ConstraintEvaluationResult(satisfied=True, applicable=True)

    def _evaluate_inverse(
        self, dimension_scores: Dict[str, float]
    ) -> "ConstraintEvaluationResult":
        """Evaluate INVERSE constraint."""
        source_score = dimension_scores.get(self.source_dimension_id, 0)

        # Check inverse relationship with targets
        not_inverse = []
        for target_id in self.target_dimension_ids:
            target_score = dimension_scores.get(target_id, 0)

            # Check if both are high or both are low
            if (source_score > 0.7 and target_score > 0.7) or (
                source_score < 0.3 and target_score < 0.3
            ):
                not_inverse.append(
                    f"{target_id} ({source_score:.2f} vs {target_score:.2f})"
                )

        if not_inverse:
            message = self.custom_message or (
                f"Expected inverse relationship not found: {', '.join(not_inverse)}"
            )
            return ConstraintEvaluationResult(
                satisfied=False,
                applicable=True,
                message=message,
                severity=self.severity,
            )

        return ConstraintEvaluationResult(satisfied=True, applicable=True)

    def _evaluate_conditional(
        self, dimension_scores: Dict[str, float]
    ) -> "ConstraintEvaluationResult":
        """Evaluate CONDITIONAL constraint."""
        # This is already handled by _constraint_applies
        # If we get here, the condition was met, so evaluate as REQUIRES
        return self._evaluate_requires(dimension_scores, None)

    def get_affected_dimensions(self) -> Set[str]:
        """Get all dimensions affected by this constraint."""
        affected = {self.source_dimension_id}
        affected.update(self.target_dimension_ids)
        if self.condition_dimension_id:
            affected.add(self.condition_dimension_id)
        return affected

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


@dataclass
class ConstraintEvaluationResult:
    """Result of constraint evaluation."""

    satisfied: bool
    applicable: bool
    message: Optional[str] = None
    severity: str = "error"
    missing_dimensions: List[str] = field(default_factory=list)
    conflicting_dimensions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
