"""Validation service for dimension framework.

This module provides comprehensive validation for dimensions, dimension sets,
and their relationships to ensure consistency and correctness.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re
from collections import defaultdict
import numpy as np

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.entities.highlight_type_registry import HighlightTypeRegistry
from src.domain.value_objects.dimension_definition import DimensionDefinition, DimensionType
from src.domain.value_objects.dimension_score import DimensionScore
import logfire


@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""

    severity: str  # "error", "warning", "info"
    dimension_id: Optional[str]
    issue_type: str
    message: str
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dimension configuration."""

    is_valid: bool
    issues: List[ValidationIssue]
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def get_issues_by_dimension(self, dimension_id: str) -> List[ValidationIssue]:
        """Get all issues for a specific dimension."""
        return [issue for issue in self.issues if issue.dimension_id == dimension_id]


class DimensionValidationService:
    """Service for validating dimension configurations and relationships.

    Provides comprehensive validation including:
    - Semantic consistency
    - Cross-dimension relationships
    - Score distribution analysis
    - Configuration completeness
    - Performance optimization suggestions
    """

    def __init__(self):
        self.logger = logfire.get_logger(__name__)

    def validate_dimension_set(
        self,
        dimension_set: DimensionSetAggregate,
        type_registry: Optional[HighlightTypeRegistry] = None,
        sample_scores: Optional[Dict[str, List[float]]] = None,
    ) -> ValidationReport:
        """Perform comprehensive validation of a dimension set.

        Args:
            dimension_set: The dimension set to validate
            type_registry: Optional highlight type registry for cross-validation
            sample_scores: Optional sample scores for distribution analysis

        Returns:
            ValidationReport with all findings
        """
        issues = []
        statistics = {}
        recommendations = []

        # Basic structure validation
        issues.extend(self._validate_basic_structure(dimension_set))

        # Semantic validation
        issues.extend(self._validate_semantic_consistency(dimension_set))

        # Weight validation
        issues.extend(self._validate_weights(dimension_set))
        statistics["weight_analysis"] = self._analyze_weights(dimension_set)

        # Dimension relationships
        issues.extend(self._validate_relationships(dimension_set))

        # Cross-validation with type registry
        if type_registry:
            issues.extend(
                self._validate_with_type_registry(dimension_set, type_registry)
            )

        # Score distribution analysis
        if sample_scores:
            issues.extend(
                self._validate_score_distributions(dimension_set, sample_scores)
            )
            statistics["score_analysis"] = self._analyze_score_distributions(
                sample_scores
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            dimension_set, issues, statistics
        )

        # Determine overall validity
        is_valid = all(issue.severity != "error" for issue in issues)

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations,
        )

    def _validate_basic_structure(
        self, dimension_set: DimensionSetAggregate
    ) -> List[ValidationIssue]:
        """Validate basic structural requirements."""
        issues = []

        # Check minimum dimensions
        if len(dimension_set.dimensions) < dimension_set.config.minimum_dimensions_required:
            issues.append(
                ValidationIssue(
                    severity="error",
                    dimension_id=None,
                    issue_type="insufficient_dimensions",
                    message=f"Set has {len(dimension_set.dimensions)} dimensions but requires {dimension_set.config.minimum_dimensions_required}",
                    suggestion="Add more dimensions or reduce minimum requirement",
                )
            )

        # Check dimension IDs
        for dim_id, dimension in dimension_set.dimensions.items():
            if not re.match(r"^[a-z][a-z0-9_]*$", dim_id):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        dimension_id=dim_id,
                        issue_type="invalid_id_format",
                        message=f"Dimension ID '{dim_id}' doesn't follow naming convention",
                        suggestion="Use lowercase alphanumeric with underscores (e.g., 'skill_level')",
                    )
                )

            # Check required fields
            if not dimension.name or not dimension.description:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        dimension_id=dim_id,
                        issue_type="missing_required_field",
                        message=f"Dimension '{dim_id}' missing name or description",
                        suggestion="Provide clear name and description for all dimensions",
                    )
                )

        return issues

    def _validate_semantic_consistency(
        self, dimension_set: DimensionSetAggregate
    ) -> List[ValidationIssue]:
        """Validate semantic consistency across dimensions."""
        issues = []

        # Check for duplicate or overlapping dimensions
        # Find similar names
        for i, (id1, dim1) in enumerate(dimension_set.dimensions.items()):
            for j, (id2, dim2) in enumerate(
                list(dimension_set.dimensions.items())[i + 1 :], i + 1
            ):
                # Check name similarity
                if self._calculate_similarity(dim1.name, dim2.name) > 0.8:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            dimension_id=id1,
                            issue_type="similar_dimensions",
                            message=f"Dimensions '{id1}' and '{id2}' have very similar names",
                            suggestion="Consider consolidating or differentiating these dimensions",
                            metadata={"related_dimension": id2},
                        )
                    )

                # Check description overlap
                if self._calculate_overlap(dim1.description, dim2.description) > 0.7:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            dimension_id=id1,
                            issue_type="overlapping_descriptions",
                            message=f"Dimensions '{id1}' and '{id2}' have overlapping descriptions",
                            suggestion="Clarify the distinction between these dimensions",
                        )
                    )

        # Check for conflicting dimension types
        for dim_id, dimension in dimension_set.dimensions.items():
            if (
                dimension.dimension_type == "binary"
                and dimension.aggregation_method == "average"
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        dimension_id=dim_id,
                        issue_type="inconsistent_configuration",
                        message=f"Binary dimension '{dim_id}' uses 'average' aggregation",
                        suggestion="Consider using 'max' or 'consensus' for binary dimensions",
                    )
                )

        return issues

    def _validate_weights(self, dimension_set: DimensionSetAggregate) -> List[ValidationIssue]:
        """Validate weight configuration."""
        issues = []

        # Check for zero weights
        zero_weight_dims = [
            dim_id
            for dim_id, weight in dimension_set.weights.items()
            if weight.value == 0.0
        ]

        if zero_weight_dims:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    dimension_id=None,
                    issue_type="zero_weights",
                    message=f"Dimensions with zero weight: {zero_weight_dims}",
                    suggestion="Remove unused dimensions or assign appropriate weights",
                    metadata={"dimensions": zero_weight_dims},
                )
            )

        # Check weight distribution
        weights = [w.value for w in dimension_set.weights.values()]
        if weights:
            max_weight = max(weights)
            min_weight = min(w for w in weights if w > 0)

            if max_weight / min_weight > 10:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        dimension_id=None,
                        issue_type="imbalanced_weights",
                        message=f"Weight ratio is {max_weight / min_weight:.1f}:1 (very imbalanced)",
                        suggestion="Consider more balanced weights to avoid single-dimension dominance",
                    )
                )

        # Check normalization
        if dimension_set.config.normalize_weights:
            total_weight = sum(weights)
            if abs(total_weight - 1.0) > 0.01:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        dimension_id=None,
                        issue_type="weight_normalization_error",
                        message=f"Weights sum to {total_weight:.3f} instead of 1.0",
                        suggestion="Enable automatic normalization or fix weights manually",
                    )
                )

        return issues

    def _validate_relationships(
        self, dimension_set: DimensionSetAggregate
    ) -> List[ValidationIssue]:
        """Validate relationships between dimensions."""
        issues = []

        # Check for orphaned modalities
        all_modalities = set()
        for dimension in dimension_set.dimensions.values():
            all_modalities.update(dimension.applicable_modalities)

        # Check each dimension uses available modalities
        for dim_id, dimension in dimension_set.dimensions.items():
            if not dimension.applicable_modalities:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        dimension_id=dim_id,
                        issue_type="no_modalities",
                        message=f"Dimension '{dim_id}' has no applicable modalities",
                        suggestion="Specify which modalities this dimension should use",
                    )
                )

        # Check for category consistency
        categories = defaultdict(list)
        for dim_id, dimension in dimension_set.dimensions.items():
            if dimension.category:
                categories[dimension.category].append(dim_id)

        # Warn about single-dimension categories
        for category, dims in categories.items():
            if len(dims) == 1:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        dimension_id=dims[0],
                        issue_type="single_dimension_category",
                        message=f"Category '{category}' has only one dimension",
                        suggestion="Consider adding related dimensions or removing the category",
                    )
                )

        return issues

    def _validate_with_type_registry(
        self, dimension_set: DimensionSetAggregate, type_registry: HighlightTypeRegistry
    ) -> List[ValidationIssue]:
        """Cross-validate with highlight type registry."""
        issues = []

        # Check all required dimensions exist
        for type_id, type_def in type_registry.types.items():
            if not type_def.is_active:
                continue

            # Check required dimensions
            missing_dims = set(type_def.required_dimensions) - set(
                dimension_set.dimensions.keys()
            )
            if missing_dims:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        dimension_id=None,
                        issue_type="missing_required_dimensions",
                        message=f"Type '{type_id}' requires missing dimensions: {missing_dims}",
                        suggestion="Add required dimensions or update type definition",
                        metadata={"type_id": type_id, "missing": list(missing_dims)},
                    )
                )

            # Check dimension score requirements
            for dim_id, min_score in type_def.required_dimension_scores.items():
                if dim_id in dimension_set.dimensions:
                    dimension = dimension_set.dimensions[dim_id]
                    if dimension.threshold > min_score:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                dimension_id=dim_id,
                                issue_type="threshold_mismatch",
                                message=f"Dimension threshold ({dimension.threshold}) > type requirement ({min_score})",
                                suggestion="Align dimension threshold with type requirements",
                            )
                        )

        return issues

    def _validate_score_distributions(
        self, dimension_set: DimensionSetAggregate, sample_scores: Dict[str, List[float]]
    ) -> List[ValidationIssue]:
        """Validate score distributions for anomalies."""
        issues = []

        for dim_id, scores in sample_scores.items():
            if dim_id not in dimension_set.dimensions:
                continue

            dimension = dimension_set.dimensions[dim_id]
            scores_array = np.array(scores)

            # Check for always-zero dimensions
            if np.all(scores_array == 0):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        dimension_id=dim_id,
                        issue_type="always_zero",
                        message=f"Dimension '{dim_id}' always scores 0",
                        suggestion="Review scoring logic or remove dimension",
                    )
                )

            # Check for always-max dimensions
            elif np.all(scores_array == 1.0):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        dimension_id=dim_id,
                        issue_type="always_max",
                        message=f"Dimension '{dim_id}' always scores 1.0",
                        suggestion="Adjust scoring criteria to be more selective",
                    )
                )

            # Check for extreme skew
            elif len(scores) > 10:
                skewness = self._calculate_skewness(scores_array)
                if abs(skewness) > 2:
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            dimension_id=dim_id,
                            issue_type="extreme_skew",
                            message=f"Dimension '{dim_id}' has extreme skew ({skewness:.2f})",
                            suggestion="Consider if this distribution is intentional",
                        )
                    )

            # Check threshold effectiveness
            below_threshold = np.sum(scores_array < dimension.threshold) / len(scores)
            if below_threshold > 0.95:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        dimension_id=dim_id,
                        issue_type="ineffective_threshold",
                        message=f"95% of scores below threshold for '{dim_id}'",
                        suggestion="Lower threshold or adjust scoring",
                    )
                )

        return issues

    def _analyze_weights(self, dimension_set: DimensionSetAggregate) -> Dict[str, Any]:
        """Analyze weight distribution statistics."""
        weights = [w.value for w in dimension_set.weights.values()]
        if not weights:
            return {}

        return {
            "total": sum(weights),
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": min(weights),
            "max": max(weights),
            "coefficient_of_variation": np.std(weights) / np.mean(weights)
            if np.mean(weights) > 0
            else 0,
            "gini_coefficient": self._calculate_gini_coefficient(weights),
        }

    def _analyze_score_distributions(
        self, sample_scores: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Analyze score distribution patterns."""
        analysis = {}

        for dim_id, scores in sample_scores.items():
            if not scores:
                continue

            scores_array = np.array(scores)
            analysis[dim_id] = {
                "mean": np.mean(scores_array),
                "median": np.median(scores_array),
                "std": np.std(scores_array),
                "min": np.min(scores_array),
                "max": np.max(scores_array),
                "percentiles": {
                    "25": np.percentile(scores_array, 25),
                    "50": np.percentile(scores_array, 50),
                    "75": np.percentile(scores_array, 75),
                    "90": np.percentile(scores_array, 90),
                },
                "skewness": self._calculate_skewness(scores_array),
                "zero_ratio": np.sum(scores_array == 0) / len(scores_array),
            }

        return analysis

    # Consolidated validation methods from DimensionValidationPolicy
    
    def validate_dimension_definition(
        self, dimension: DimensionDefinition
    ) -> ValidationReport:
        """Validate a single dimension definition (from DimensionValidationPolicy)."""
        issues = []
        
        # Validate ID
        if not dimension.id or not dimension.id.strip():
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="invalid_id",
                message="Dimension ID cannot be empty"
            ))
        elif not dimension.id.replace("_", "").isalnum():
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="invalid_id_format",
                message=f"Dimension ID '{dimension.id}' must be alphanumeric with underscores only"
            ))

        # Validate name
        if not dimension.name or not dimension.name.strip():
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="missing_name",
                message="Dimension name cannot be empty"
            ))

        # Validate scoring prompt
        if not dimension.scoring_prompt:
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="missing_prompt",
                message="Scoring prompt is required"
            ))
        elif len(dimension.scoring_prompt) < 20:
            issues.append(ValidationIssue(
                severity="warning",
                dimension_id=dimension.id,
                issue_type="short_prompt",
                message="Scoring prompt is very short (may reduce accuracy)"
            ))

        # Validate modalities
        if not dimension.applicable_modalities:
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="no_modalities",
                message="At least one modality must be specified"
            ))

        # Validate weights and thresholds
        if dimension.default_weight < 0 or dimension.default_weight > 1:
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="invalid_weight",
                message=f"Default weight must be between 0 and 1 (got {dimension.default_weight})"
            ))

        return ValidationReport(
            is_valid=all(issue.severity != "error" for issue in issues),
            issues=issues
        )

    def validate_dimension_score(
        self, score: DimensionScore, dimension: DimensionDefinition
    ) -> ValidationReport:
        """Validate a dimension score against its definition."""
        issues = []

        # Validate score value
        if score.value < 0 or score.value > 1:
            issues.append(ValidationIssue(
                severity="error",
                dimension_id=dimension.id,
                issue_type="invalid_score_value",
                message=f"Score value must be between 0 and 1 (got {score.value})"
            ))

        # Validate against dimension type
        if dimension.dimension_type == DimensionType.BINARY:
            if score.value not in [0.0, 1.0]:
                issues.append(ValidationIssue(
                    severity="warning",
                    dimension_id=dimension.id,
                    issue_type="non_binary_score",
                    message=f"Binary dimension has non-binary score: {score.value}"
                ))

        # Validate confidence
        if score.confidence:
            valid_confidence = {"high", "medium", "low", "uncertain"}
            if score.confidence not in valid_confidence:
                issues.append(ValidationIssue(
                    severity="warning",
                    dimension_id=dimension.id,
                    issue_type="invalid_confidence",
                    message=f"Non-standard confidence level: {score.confidence}"
                ))

        return ValidationReport(
            is_valid=all(issue.severity != "error" for issue in issues),
            issues=issues
        )

    def _generate_recommendations(
        self,
        dimension_set: DimensionSetAggregate,
        issues: List[ValidationIssue],
        statistics: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Weight balance recommendation
        if statistics.get("weight_analysis", {}).get("gini_coefficient", 0) > 0.4:
            recommendations.append(
                "Consider rebalancing dimension weights for more even influence. "
                "Current distribution is highly concentrated."
            )

        # Dimension count recommendation
        if len(dimension_set.dimensions) < 5:
            recommendations.append(
                "Consider adding more dimensions for richer evaluation. "
                "Most effective sets have 5-10 well-defined dimensions."
            )
        elif len(dimension_set.dimensions) > 15:
            recommendations.append(
                "Consider consolidating dimensions. Too many dimensions can "
                "reduce clarity and increase processing time."
            )

        # Error-based recommendations
        error_types = set(
            issue.issue_type for issue in issues if issue.severity == "error"
        )
        if "missing_required_dimensions" in error_types:
            recommendations.append(
                "Add missing dimensions required by highlight types or update "
                "type definitions to match available dimensions."
            )

        # Category recommendations
        categories = set(
            d.category for d in dimension_set.dimensions.values() if hasattr(d, 'category') and d.category
        )
        if not categories:
            recommendations.append(
                "Consider categorizing dimensions (e.g., 'technical', 'emotional', "
                "'contextual') for better organization."
            )

        return recommendations

    # Utility methods

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (0-1)."""
        # Simple character-based similarity
        str1, str2 = str1.lower(), str2.lower()
        if str1 == str2:
            return 1.0

        # Levenshtein-like similarity
        common_chars = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
        return common_chars / max(len(str1), len(str2))

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap ratio."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        common_words = words1 & words2
        return len(common_words) / min(len(words1), len(words2))

    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0

        return np.mean(((values - mean) / std) ** 3)

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        index = range(1, n + 1)

        return (2 * sum(index[i] * sorted_values[i] for i in range(n))) / (
            n * sum(sorted_values)
        ) - (n + 1) / n
