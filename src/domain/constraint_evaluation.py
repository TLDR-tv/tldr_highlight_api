"""Domain logic for evaluating dimension constraints.

Pure functions for constraint evaluation - no infrastructure dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from src.domain.value_objects.dimension_constraint import (
    DimensionConstraint,
    ConstraintType,
    ConstraintOperator,
)


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


def evaluate_constraint(
    constraint: DimensionConstraint,
    dimension_scores: Dict[str, float],
    dimension_presence: Optional[Dict[str, bool]] = None,
) -> ConstraintEvaluationResult:
    """Evaluate if a constraint is satisfied.

    Args:
        constraint: The constraint to evaluate
        dimension_scores: Current dimension scores
        dimension_presence: Which dimensions are present/evaluated

    Returns:
        Evaluation result with satisfaction status and messages
    """
    # Check if constraint applies (conditional logic)
    if not _constraint_applies(constraint, dimension_scores):
        return ConstraintEvaluationResult(
            satisfied=True,
            applicable=False,
            message="Constraint not applicable in current context",
        )

    # Evaluate based on constraint type
    if constraint.constraint_type == ConstraintType.REQUIRES:
        return _evaluate_requires(constraint, dimension_scores, dimension_presence)
    elif constraint.constraint_type == ConstraintType.EXCLUDES:
        return _evaluate_excludes(constraint, dimension_scores, dimension_presence)
    elif constraint.constraint_type == ConstraintType.MINIMUM_SCORE:
        return _evaluate_minimum_score(constraint, dimension_scores)
    elif constraint.constraint_type == ConstraintType.CORRELATES:
        return _evaluate_correlation(constraint, dimension_scores)
    elif constraint.constraint_type == ConstraintType.INVERSE:
        return _evaluate_inverse(constraint, dimension_scores)
    else:
        return _evaluate_conditional(constraint, dimension_scores, dimension_presence)


def _constraint_applies(
    constraint: DimensionConstraint, dimension_scores: Dict[str, float]
) -> bool:
    """Check if conditional constraint applies."""
    if constraint.constraint_type != ConstraintType.CONDITIONAL:
        return True

    if not constraint.condition_dimension_id:
        return True

    if constraint.condition_dimension_id not in dimension_scores:
        return False

    condition_score = dimension_scores[constraint.condition_dimension_id]

    # Ensure condition_value is not None
    if constraint.condition_value is None:
        return True

    # Evaluate condition
    if constraint.condition_operator == ConstraintOperator.GREATER_THAN:
        return condition_score > constraint.condition_value
    elif constraint.condition_operator == ConstraintOperator.LESS_THAN:
        return condition_score < constraint.condition_value
    elif constraint.condition_operator == ConstraintOperator.EQUALS:
        return abs(condition_score - constraint.condition_value) < 0.001
    elif constraint.condition_operator == ConstraintOperator.NOT_EQUALS:
        return abs(condition_score - constraint.condition_value) >= 0.001
    elif constraint.condition_operator == ConstraintOperator.GREATER_EQUAL:
        return condition_score >= constraint.condition_value
    elif constraint.condition_operator == ConstraintOperator.LESS_EQUAL:
        return condition_score <= constraint.condition_value

    return True


def _evaluate_requires(
    constraint: DimensionConstraint,
    dimension_scores: Dict[str, float],
    dimension_presence: Optional[Dict[str, bool]],
) -> ConstraintEvaluationResult:
    """Evaluate REQUIRES constraint."""
    if dimension_presence is None:
        dimension_presence = {
            dim_id: dim_id in dimension_scores
            for dim_id in constraint.target_dimension_ids
        }

    # Check if source dimension is present
    if constraint.source_dimension_id not in dimension_scores:
        return ConstraintEvaluationResult(
            satisfied=True,
            applicable=False,
            message=f"Source dimension '{constraint.source_dimension_id}' not evaluated",
        )

    # Check if source has significant score
    if dimension_scores[constraint.source_dimension_id] < 0.1:
        return ConstraintEvaluationResult(
            satisfied=True,
            applicable=False,
            message="Source dimension score too low to require dependencies",
        )

    # Check required dimensions
    missing = [
        dim_id
        for dim_id in constraint.target_dimension_ids
        if not dimension_presence.get(dim_id, False)
    ]

    if missing:
        message = constraint.custom_message or (
            f"Dimension '{constraint.source_dimension_id}' requires: {', '.join(missing)}"
        )
        return ConstraintEvaluationResult(
            satisfied=False,
            applicable=True,
            message=message,
            severity=constraint.severity,
            missing_dimensions=missing,
        )

    return ConstraintEvaluationResult(satisfied=True, applicable=True)


def _evaluate_excludes(
    constraint: DimensionConstraint,
    dimension_scores: Dict[str, float],
    dimension_presence: Optional[Dict[str, bool]],
) -> ConstraintEvaluationResult:
    """Evaluate EXCLUDES constraint."""
    if dimension_presence is None:
        dimension_presence = {
            dim_id: dim_id in dimension_scores
            for dim_id in constraint.target_dimension_ids
        }

    # Check if source dimension is present with significant score
    if (
        constraint.source_dimension_id not in dimension_scores
        or dimension_scores[constraint.source_dimension_id] < 0.1
    ):
        return ConstraintEvaluationResult(satisfied=True, applicable=False)

    # Check for excluded dimensions
    present_excluded = [
        dim_id
        for dim_id in constraint.target_dimension_ids
        if dimension_presence.get(dim_id, False)
        and dimension_scores.get(dim_id, 0) >= 0.1
    ]

    if present_excluded:
        message = constraint.custom_message or (
            f"Dimension '{constraint.source_dimension_id}' excludes: {', '.join(present_excluded)}"
        )
        return ConstraintEvaluationResult(
            satisfied=False,
            applicable=True,
            message=message,
            severity=constraint.severity,
            conflicting_dimensions=present_excluded,
        )

    return ConstraintEvaluationResult(satisfied=True, applicable=True)


def _evaluate_minimum_score(
    constraint: DimensionConstraint, dimension_scores: Dict[str, float]
) -> ConstraintEvaluationResult:
    """Evaluate MINIMUM_SCORE constraint."""
    # Source must have minimum score
    source_score = dimension_scores.get(constraint.source_dimension_id, 0)

    if source_score < 0.1:
        return ConstraintEvaluationResult(satisfied=True, applicable=False)

    # Check target dimensions meet minimum
    below_minimum = []
    for target_id in constraint.target_dimension_ids:
        target_score = dimension_scores.get(target_id, 0)
        if target_score < (constraint.condition_value or 0.5):
            below_minimum.append(f"{target_id}={target_score:.2f}")

    if below_minimum:
        message = constraint.custom_message or (
            f"Dimensions below minimum score {constraint.condition_value}: "
            f"{', '.join(below_minimum)}"
        )
        return ConstraintEvaluationResult(
            satisfied=False,
            applicable=True,
            message=message,
            severity=constraint.severity,
        )

    return ConstraintEvaluationResult(satisfied=True, applicable=True)


def _evaluate_correlation(
    constraint: DimensionConstraint, dimension_scores: Dict[str, float]
) -> ConstraintEvaluationResult:
    """Evaluate CORRELATES constraint."""
    source_score = dimension_scores.get(constraint.source_dimension_id, 0)

    # Check correlation with targets
    uncorrelated = []
    for target_id in constraint.target_dimension_ids:
        target_score = dimension_scores.get(target_id, 0)

        # Simple correlation check - scores should be similar
        if abs(source_score - target_score) > 0.3:
            uncorrelated.append(
                f"{target_id} ({source_score:.2f} vs {target_score:.2f})"
            )

    if uncorrelated:
        message = constraint.custom_message or (
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
    constraint: DimensionConstraint, dimension_scores: Dict[str, float]
) -> ConstraintEvaluationResult:
    """Evaluate INVERSE constraint."""
    source_score = dimension_scores.get(constraint.source_dimension_id, 0)

    # Check inverse relationship with targets
    not_inverse = []
    for target_id in constraint.target_dimension_ids:
        target_score = dimension_scores.get(target_id, 0)

        # Check if both are high or both are low
        if (source_score > 0.7 and target_score > 0.7) or (
            source_score < 0.3 and target_score < 0.3
        ):
            not_inverse.append(
                f"{target_id} ({source_score:.2f} vs {target_score:.2f})"
            )

    if not_inverse:
        message = constraint.custom_message or (
            f"Expected inverse relationship not found: {', '.join(not_inverse)}"
        )
        return ConstraintEvaluationResult(
            satisfied=False,
            applicable=True,
            message=message,
            severity=constraint.severity,
        )

    return ConstraintEvaluationResult(satisfied=True, applicable=True)


def _evaluate_conditional(
    constraint: DimensionConstraint,
    dimension_scores: Dict[str, float],
    dimension_presence: Optional[Dict[str, bool]],
) -> ConstraintEvaluationResult:
    """Evaluate CONDITIONAL constraint."""
    # This is already handled by _constraint_applies
    # If we get here, the condition was met, so evaluate as REQUIRES
    return _evaluate_requires(constraint, dimension_scores, dimension_presence)


def get_affected_dimensions(constraint: DimensionConstraint) -> Set[str]:
    """Get all dimensions affected by a constraint."""
    affected = {constraint.source_dimension_id}
    affected.update(constraint.target_dimension_ids)
    if constraint.condition_dimension_id:
        affected.add(constraint.condition_dimension_id)
    return affected


def evaluate_all_constraints(
    constraints: List[DimensionConstraint],
    dimension_scores: Dict[str, float],
    dimension_presence: Optional[Dict[str, bool]] = None,
) -> List[ConstraintEvaluationResult]:
    """Evaluate multiple constraints and return all results."""
    results = []
    for constraint in constraints:
        result = evaluate_constraint(constraint, dimension_scores, dimension_presence)
        if result.applicable:  # Only include applicable results
            results.append(result)
    return results