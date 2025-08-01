"""Domain validation with focused validator classes.

Pure domain logic for dimension and scoring validation.
No infrastructure dependencies - just business rule validation.
"""

from typing import Dict, List, Set, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.value_objects.dimension_definition import (
    DimensionDefinition,
    DimensionType,
)
from src.domain.value_objects.dimension_weight import DimensionWeight


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Validator(Protocol):
    """Protocol for validators."""

    def validate(self, target: Any) -> ValidationResult:
        """Validate the target object."""
        ...


@dataclass
class DimensionCountValidator:
    """Validates dimension count constraints."""

    min_dimensions: int = 3
    max_dimensions: int = 20

    def validate(self, dimension_set: DimensionSetAggregate) -> ValidationResult:
        """Validate dimension count is within bounds."""
        errors = []
        warnings = []
        metadata = {}

        dimension_count = len(dimension_set.dimensions)
        metadata["dimension_count"] = dimension_count

        if dimension_count < self.min_dimensions:
            errors.append(
                f"Too few dimensions: {dimension_count} < {self.min_dimensions}"
            )
        elif dimension_count > self.max_dimensions:
            warnings.append(
                f"Many dimensions may impact performance: {dimension_count} > {self.max_dimensions}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


@dataclass
class WeightValidator:
    """Validates dimension weights."""

    require_normalized: bool = True
    normalization_tolerance: float = 0.001
    dominance_threshold: float = 0.5

    def validate(self, weights: Dict[str, DimensionWeight]) -> ValidationResult:
        """Validate dimension weights configuration."""
        errors = []
        warnings = []
        metadata = {}

        if not weights:
            errors.append("No weights defined")
            return ValidationResult(False, errors, warnings, metadata)

        # Check individual weights
        for dim_id, weight in weights.items():
            if weight.value < 0.0:
                errors.append(f"Negative weight for '{dim_id}': {weight.value}")
            elif weight.value > 1.0:
                warnings.append(f"Weight > 1.0 for '{dim_id}': {weight.value}")

        # Calculate statistics
        weight_values = [w.value for w in weights.values()]
        total_weight = sum(weight_values)
        metadata["total_weight"] = total_weight
        metadata["min_weight"] = min(weight_values) if weight_values else 0
        metadata["max_weight"] = max(weight_values) if weight_values else 0
        metadata["zero_weights"] = sum(1 for w in weight_values if w == 0.0)

        # Check normalization
        if (
            self.require_normalized
            and abs(total_weight - 1.0) > self.normalization_tolerance
        ):
            errors.append(f"Weights not normalized: sum={total_weight:.3f} != 1.0")

        # Check for all zero weights
        if all(w == 0.0 for w in weight_values):
            errors.append("All weights are zero")

        # Check weight distribution
        if metadata["max_weight"] > self.dominance_threshold:
            warnings.append(
                f"One dimension dominates with weight {metadata['max_weight']:.2f}"
            )

        if metadata["zero_weights"] > len(weights) * 0.5:
            warnings.append("More than half of dimensions have zero weight")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


@dataclass
class DimensionDefinitionValidator:
    """Validates individual dimension definitions."""

    valid_aggregations: List[str] = field(
        default_factory=lambda: ["mean", "max", "min", "sum", "mode", "weighted_mean"]
    )

    def validate(self, dimension: DimensionDefinition) -> ValidationResult:
        """Validate a single dimension definition."""
        errors = []
        warnings = []
        metadata = {"dimension_id": dimension.id}

        # Check required fields
        if not dimension.name:
            errors.append("Dimension name is required")

        if not dimension.description:
            warnings.append("Dimension should have a description")

        # Validate dimension type specific rules
        if dimension.dimension_type == DimensionType.NUMERIC:
            if dimension.threshold < 0.0 or dimension.threshold > 1.0:
                errors.append(
                    f"Numeric threshold must be in [0, 1]: {dimension.threshold}"
                )

        elif dimension.dimension_type == DimensionType.BINARY:
            if dimension.threshold not in [0.0, 1.0]:
                warnings.append(
                    f"Binary dimension threshold should be 0 or 1: {dimension.threshold}"
                )

        elif dimension.dimension_type == DimensionType.CATEGORICAL:
            if not hasattr(dimension, "categories") or not dimension.categories:
                errors.append("Categorical dimension must define categories")

        # Check aggregation method
        if dimension.aggregation_method not in self.valid_aggregations:
            errors.append(f"Invalid aggregation method: {dimension.aggregation_method}")

        # Validate examples if provided
        if hasattr(dimension, "examples") and dimension.examples:
            for i, example in enumerate(dimension.examples):
                if "input" not in example or "expected_score" not in example:
                    warnings.append(f"Example {i} missing required fields")
                elif not (0.0 <= example["expected_score"] <= 1.0):
                    errors.append(
                        f"Example {i} score out of range: {example['expected_score']}"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


@dataclass
class DependencyValidator:
    """Validates dimension dependency graphs."""

    max_depth_warning: int = 3

    def validate(self, dimension_set: DimensionSetAggregate) -> ValidationResult:
        """Validate dimension dependency graph."""
        errors = []
        warnings = []
        metadata = {}

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(dimension_set)

        # Check for missing dependencies
        all_dimension_ids = set(dimension_set.dimensions.keys())
        for dim_id, deps in dependency_graph.items():
            missing_deps = set(deps) - all_dimension_ids
            if missing_deps:
                errors.append(
                    f"Dimension '{dim_id}' has missing dependencies: {missing_deps}"
                )

        # Check for cycles
        cycles = self._detect_cycles(dependency_graph)
        if cycles:
            for cycle in cycles:
                errors.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        # Check dependency depth
        max_depth = self._calculate_max_depth(dependency_graph)
        metadata["max_dependency_depth"] = max_depth

        if max_depth > self.max_depth_warning:
            warnings.append(f"Deep dependency chain detected: depth={max_depth}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _build_dependency_graph(
        self, dimension_set: DimensionSetAggregate
    ) -> Dict[str, List[str]]:
        """Build dependency graph from dimension set."""
        dependency_graph = {}
        for dim_id, dimension in dimension_set.dimensions.items():
            if hasattr(dimension, "depends_on") and dimension.depends_on:
                dependency_graph[dim_id] = dimension.depends_on
            else:
                dependency_graph[dim_id] = []
        return dependency_graph

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _calculate_max_depth(self, graph: Dict[str, List[str]]) -> int:
        """Calculate maximum dependency depth in the graph."""

        def get_depth(node: str, visited: Set[str]) -> int:
            if node in visited:
                return 0  # Cycle detected

            visited.add(node)
            deps = graph.get(node, [])

            if not deps:
                return 0

            max_child_depth = 0
            for dep in deps:
                child_depth = get_depth(dep, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)

            return 1 + max_child_depth

        max_depth = 0
        for node in graph:
            depth = get_depth(node, set())
            max_depth = max(max_depth, depth)

        return max_depth


@dataclass
class ScoringResultValidator:
    """Validates scoring results against dimension sets."""

    allow_missing: bool = False

    def validate(
        self, scoring_result: Dict[str, float], dimension_set: DimensionSetAggregate
    ) -> ValidationResult:
        """Validate a scoring result against dimension set."""
        errors = []
        warnings = []
        metadata = {}

        # Check all scores are in valid range
        for dim_id, score in scoring_result.items():
            if not (0.0 <= score <= 1.0):
                errors.append(f"Score for '{dim_id}' out of range: {score}")

        # Check for missing dimensions
        expected_dims = set(
            dim_id
            for dim_id, weight in dimension_set.weights.items()
            if weight.is_significant()
        )
        scored_dims = set(scoring_result.keys())

        missing_dims = expected_dims - scored_dims
        if missing_dims and not self.allow_missing:
            errors.append(f"Missing scores for dimensions: {missing_dims}")
        elif missing_dims:
            warnings.append(f"Missing scores for dimensions: {missing_dims}")

        # Check for extra dimensions
        extra_dims = scored_dims - expected_dims
        if extra_dims:
            warnings.append(f"Scores for unknown dimensions: {extra_dims}")

        # Validate composite dimension calculations if any
        for dim_id, dimension in dimension_set.dimensions.items():
            if hasattr(dimension, "depends_on") and dimension.depends_on:
                # This is a composite dimension
                required_deps = set(dimension.depends_on)
                available_deps = required_deps & scored_dims

                if len(available_deps) < len(required_deps):
                    missing = required_deps - available_deps
                    warnings.append(
                        f"Composite dimension '{dim_id}' missing dependencies: {missing}"
                    )

        metadata["scored_dimensions"] = len(scored_dims)
        metadata["expected_dimensions"] = len(expected_dims)
        metadata["coverage"] = (
            len(scored_dims & expected_dims) / len(expected_dims)
            if expected_dims
            else 0
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


@dataclass
class DimensionSetValidator:
    """Composite validator for complete dimension set validation."""

    dimension_count_validator: DimensionCountValidator = field(
        default_factory=DimensionCountValidator
    )
    weight_validator: WeightValidator = field(default_factory=WeightValidator)
    dimension_validator: DimensionDefinitionValidator = field(
        default_factory=DimensionDefinitionValidator
    )
    dependency_validator: DependencyValidator = field(
        default_factory=DependencyValidator
    )

    def validate(self, dimension_set: DimensionSetAggregate) -> ValidationResult:
        """Validate a dimension set for completeness and consistency."""
        all_errors = []
        all_warnings = []
        all_metadata = {}

        # Run all validators
        validators = [
            ("dimension_count", self.dimension_count_validator.validate(dimension_set)),
            ("weights", self.weight_validator.validate(dimension_set.weights)),
            ("dependencies", self.dependency_validator.validate(dimension_set)),
        ]

        # Validate individual dimensions
        for dim_id, dimension in dimension_set.dimensions.items():
            result = self.dimension_validator.validate(dimension)
            if not result.is_valid or result.warnings:
                prefixed_errors = [f"{dim_id}: {err}" for err in result.errors]
                prefixed_warnings = [f"{dim_id}: {warn}" for warn in result.warnings]
                all_errors.extend(prefixed_errors)
                all_warnings.extend(prefixed_warnings)

        # Aggregate results
        for validator_name, result in validators:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_metadata[validator_name] = result.metadata

        # Check weight-dimension consistency
        weight_dims = set(dimension_set.weights.keys())
        dimension_ids = set(dimension_set.dimensions.keys())
        orphaned_weights = weight_dims - dimension_ids
        missing_weights = dimension_ids - weight_dims

        if orphaned_weights:
            all_errors.append(f"Weights without dimensions: {orphaned_weights}")
        if missing_weights:
            all_errors.append(f"Dimensions without weights: {missing_weights}")

        # Check for inactive dimensions with significant weights
        for dim_id in dimension_set.dimensions:
            if (
                dim_id in dimension_set.weights
                and dimension_set.weights[dim_id].value > 0.1
                and hasattr(dimension_set.dimensions[dim_id], "is_active")
                and not dimension_set.dimensions[dim_id].is_active
            ):
                all_warnings.append(
                    f"Inactive dimension '{dim_id}' has significant weight"
                )

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            metadata=all_metadata,
        )


# Convenience function for backward compatibility
def validate_dimension_set(
    dimension_set: DimensionSetAggregate,
    min_dimensions: int = 3,
    max_dimensions: int = 20,
    require_normalized_weights: bool = True,
) -> ValidationResult:
    """Validate a dimension set for completeness and consistency.

    This is a convenience function that creates a validator with
    the specified configuration and runs validation.
    """
    validator = DimensionSetValidator(
        dimension_count_validator=DimensionCountValidator(
            min_dimensions=min_dimensions, max_dimensions=max_dimensions
        ),
        weight_validator=WeightValidator(require_normalized=require_normalized_weights),
    )
    return validator.validate(dimension_set)
