"""Dimension set policy functions.

This module provides validation functions for dimension sets,
following a more Pythonic functional approach instead of classes.
"""

from typing import Dict, List, Callable
from functools import partial

from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.domain.value_objects.dimension_weight import DimensionWeight
from src.domain.exceptions import BusinessRuleViolation


# Type aliases for clarity
DimensionSet = "DimensionSetAggregate"  # Avoid circular import
PolicyFunction = Callable[..., bool]
ValidationFunction = Callable[[Dict[str, DimensionWeight]], List[str]]


def check_dimension_not_exists(
    dimension_set: DimensionSet, dimension: DimensionDefinition
) -> None:
    """Ensure dimension doesn't already exist."""
    if dimension.id in dimension_set.dimensions:
        raise BusinessRuleViolation(
            f"Dimension '{dimension.id}' already exists in this set"
        )


def check_max_dimensions(dimension_set: DimensionSet, max_dimensions: int = 20) -> None:
    """Ensure we don't exceed max dimensions."""
    if len(dimension_set.dimensions) >= max_dimensions:
        raise BusinessRuleViolation(
            f"Cannot add dimension. Maximum of {max_dimensions} dimensions allowed"
        )


def check_dimension_exists(dimension_set: DimensionSet, dimension_id: str) -> None:
    """Ensure dimension exists in the set."""
    if dimension_id not in dimension_set.dimensions:
        raise BusinessRuleViolation(f"Dimension '{dimension_id}' not found in this set")


def check_min_dimensions(dimension_set: DimensionSet, min_dimensions: int = 3) -> None:
    """Ensure we maintain minimum dimensions."""
    if len(dimension_set.dimensions) - 1 < min_dimensions:
        raise BusinessRuleViolation(
            f"Cannot remove dimension. Set requires at least {min_dimensions} dimensions"
        )


def validate_weights(
    weights: Dict[str, DimensionWeight],
    require_normalized: bool = True,
    tolerance: float = 0.01,
) -> List[str]:
    """Validate weight configuration."""
    errors = []

    if not weights:
        errors.append("Dimension set must have at least one weighted dimension")
        return errors

    # Check for normalized weights if required
    if require_normalized:
        total = sum(w.value for w in weights.values())
        if abs(total - 1.0) > tolerance:
            errors.append(
                f"Weights must sum to 1.0 when normalization is required (sum: {total:.3f})"
            )

    # Check for zero weights
    zero_weights = [dim_id for dim_id, w in weights.items() if w.value == 0.0]
    if zero_weights:
        errors.append(f"Dimensions with zero weight: {zero_weights}")

    return errors


def create_dimension_set_validators(
    min_dimensions: int = 3,
    max_dimensions: int = 20,
    require_normalized_weights: bool = True,
) -> Dict[str, Callable]:
    """Create a set of validation functions with specific configuration.

    This factory function returns configured validators that can be used
    by DimensionSetAggregate without needing a policy class.
    """
    return {
        "can_add": partial(check_max_dimensions, max_dimensions=max_dimensions),
        "can_remove": partial(check_min_dimensions, min_dimensions=min_dimensions),
        "validate_weights": partial(
            validate_weights, require_normalized=require_normalized_weights
        ),
    }


# Default validators for convenience
DEFAULT_VALIDATORS = create_dimension_set_validators()


def can_add_dimension(
    dimension_set: DimensionSet,
    dimension: DimensionDefinition,
    max_dimensions: int = 20,
) -> bool:
    """Check if a dimension can be added to the set."""
    check_dimension_not_exists(dimension_set, dimension)
    check_max_dimensions(dimension_set, max_dimensions)
    return True


def can_remove_dimension(
    dimension_set: DimensionSet, dimension_id: str, min_dimensions: int = 3
) -> bool:
    """Check if a dimension can be removed from the set."""
    check_dimension_exists(dimension_set, dimension_id)
    check_min_dimensions(dimension_set, min_dimensions)
    return True
