"""Composite dimension entity for calculated dimensions.

This entity allows dimensions to be calculated from other dimensions
using formulas and aggregation methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import re

from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.domain.value_objects.dimension_score import DimensionScore
from src.domain.entities.base import Entity
from src.domain.exceptions import BusinessRuleViolation, InvalidValueError


class CompositeAggregationMethod(str, Enum):
    """Aggregation methods for composite dimensions."""

    WEIGHTED_SUM = "weighted_sum"  # w1*d1 + w2*d2 + ...
    AVERAGE = "average"  # Mean of all dependencies
    WEIGHTED_AVERAGE = "weighted_avg"  # Weighted mean
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    PRODUCT = "product"  # d1 * d2 * ...
    HARMONIC_MEAN = "harmonic_mean"  # Harmonic mean
    GEOMETRIC_MEAN = "geometric_mean"  # Geometric mean
    CUSTOM = "custom"  # Custom formula


@dataclass
class CompositeDimension(Entity):
    """A dimension calculated from other dimensions.

    This entity enables complex scoring by combining multiple
    dimensions using various aggregation methods and formulas.
    """

    # Identity
    dimension_set_id: int
    dimension_id: str

    # Base dimension definition
    base_definition: DimensionDefinition

    # Composite configuration
    formula: str  # e.g., "0.4 * action_intensity + 0.6 * emotional_impact"
    dependencies: List[str] = field(default_factory=list)  # dimension IDs
    weights: Dict[str, float] = field(default_factory=dict)  # dependency weights
    aggregation_method: CompositeAggregationMethod = (
        CompositeAggregationMethod.WEIGHTED_SUM
    )

    # Optional configuration
    require_all_dependencies: bool = (
        True  # If false, calculate with available dimensions
    )
    min_dependencies_required: int = 1  # Minimum dependencies needed for calculation
    fallback_value: float = 0.0  # Value if calculation fails

    # Metadata
    calculation_explanation: Optional[str] = None
    version: int = 1

    def __post_init__(self) -> None:
        """Validate composite dimension configuration."""
        # Validate dependencies
        if not self.dependencies:
            raise InvalidValueError(
                "Composite dimension must have at least one dependency"
            )

        # Extract dependencies from formula if not provided
        if not self.dependencies and self.formula:
            self.dependencies = self._extract_dependencies_from_formula(self.formula)

        # Validate weights for weighted methods
        if self.aggregation_method in [
            CompositeAggregationMethod.WEIGHTED_SUM,
            CompositeAggregationMethod.WEIGHTED_AVERAGE,
        ]:
            if not self.weights:
                raise InvalidValueError(
                    f"Weights required for {self.aggregation_method}"
                )

            # Check all dependencies have weights
            missing_weights = set(self.dependencies) - set(self.weights.keys())
            if missing_weights:
                raise InvalidValueError(
                    f"Missing weights for dependencies: {missing_weights}"
                )

        # Validate formula syntax for custom method
        if self.aggregation_method == CompositeAggregationMethod.CUSTOM:
            if not self.formula:
                raise InvalidValueError("Formula required for custom aggregation")
            self._validate_formula_syntax(self.formula)

    def calculate_score(
        self, dimension_scores: Dict[str, DimensionScore], strict: bool = True
    ) -> DimensionScore:
        """Calculate composite score from dependency scores.

        Args:
            dimension_scores: Scores for all available dimensions
            strict: If True, raise error on missing dependencies

        Returns:
            Calculated composite dimension score
        """
        # Get available dependency scores
        available_scores = {}
        missing_dependencies = []

        for dep_id in self.dependencies:
            if dep_id in dimension_scores:
                available_scores[dep_id] = dimension_scores[dep_id]
            else:
                missing_dependencies.append(dep_id)

        # Check if we have enough dependencies
        if self.require_all_dependencies and missing_dependencies:
            if strict:
                raise BusinessRuleViolation(
                    f"Missing required dependencies: {missing_dependencies}"
                )
            else:
                return DimensionScore(
                    dimension_id=self.dimension_id,
                    value=self.fallback_value,
                    confidence="uncertain",
                    evidence=f"Missing dependencies: {missing_dependencies}",
                )

        if len(available_scores) < self.min_dependencies_required:
            return DimensionScore(
                dimension_id=self.dimension_id,
                value=self.fallback_value,
                confidence="uncertain",
                evidence=f"Insufficient dependencies: {len(available_scores)} < {self.min_dependencies_required}",
            )

        # Calculate score based on aggregation method
        try:
            value = self._calculate_aggregated_value(available_scores)
            confidence = self._determine_confidence(available_scores)
            evidence = self._generate_evidence(available_scores, value)

            return DimensionScore(
                dimension_id=self.dimension_id,
                value=min(1.0, max(0.0, value)),  # Clamp to [0, 1]
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as e:
            if strict:
                raise
            return DimensionScore(
                dimension_id=self.dimension_id,
                value=self.fallback_value,
                confidence="uncertain",
                evidence=f"Calculation error: {str(e)}",
            )

    def _calculate_aggregated_value(self, scores: Dict[str, DimensionScore]) -> float:
        """Calculate value based on aggregation method."""
        values = [score.value for score in scores.values()]

        if self.aggregation_method == CompositeAggregationMethod.WEIGHTED_SUM:
            return sum(
                scores[dep_id].value * self.weights[dep_id] for dep_id in scores.keys()
            )

        elif self.aggregation_method == CompositeAggregationMethod.AVERAGE:
            return sum(values) / len(values) if values else 0.0

        elif self.aggregation_method == CompositeAggregationMethod.WEIGHTED_AVERAGE:
            total_weight = sum(self.weights[dep_id] for dep_id in scores.keys())
            if total_weight == 0:
                return 0.0
            return (
                sum(
                    scores[dep_id].value * self.weights[dep_id]
                    for dep_id in scores.keys()
                )
                / total_weight
            )

        elif self.aggregation_method == CompositeAggregationMethod.MAX:
            return max(values) if values else 0.0

        elif self.aggregation_method == CompositeAggregationMethod.MIN:
            return min(values) if values else 0.0

        elif self.aggregation_method == CompositeAggregationMethod.PRODUCT:
            result = 1.0
            for value in values:
                result *= value
            return result

        elif self.aggregation_method == CompositeAggregationMethod.HARMONIC_MEAN:
            if not values or any(v == 0 for v in values):
                return 0.0
            return len(values) / sum(1 / v for v in values)

        elif self.aggregation_method == CompositeAggregationMethod.GEOMETRIC_MEAN:
            if not values:
                return 0.0
            product = 1.0
            for value in values:
                product *= value
            return product ** (1.0 / len(values))

        elif self.aggregation_method == CompositeAggregationMethod.CUSTOM:
            return self._evaluate_custom_formula(scores)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _determine_confidence(self, scores: Dict[str, DimensionScore]) -> str:
        """Determine confidence level based on dependency confidences."""
        confidences = [s.confidence for s in scores.values() if s.confidence]

        if not confidences:
            return "medium"

        # Map confidence levels to numeric values
        confidence_values = {"high": 3, "medium": 2, "low": 1, "uncertain": 0}

        # Calculate average confidence
        avg_confidence = sum(confidence_values.get(c, 2) for c in confidences) / len(
            confidences
        )

        if avg_confidence >= 2.5:
            return "high"
        elif avg_confidence >= 1.5:
            return "medium"
        elif avg_confidence >= 0.5:
            return "low"
        else:
            return "uncertain"

    def _generate_evidence(
        self, scores: Dict[str, DimensionScore], calculated_value: float
    ) -> str:
        """Generate evidence string for the calculation."""
        parts = []

        # Add calculation method
        parts.append(f"Calculated using {self.aggregation_method.value}")

        # Add dependency values
        dep_values = [
            f"{dep_id}={scores[dep_id].value:.2f}" for dep_id in sorted(scores.keys())
        ]
        parts.append(f"Dependencies: {', '.join(dep_values)}")

        # Add result
        parts.append(f"Result: {calculated_value:.3f}")

        # Add custom explanation if available
        if self.calculation_explanation:
            parts.append(self.calculation_explanation)

        return "; ".join(parts)

    def _extract_dependencies_from_formula(self, formula: str) -> List[str]:
        """Extract dimension IDs from formula string."""
        # Match valid dimension IDs (alphanumeric with underscores)
        pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        tokens = re.findall(pattern, formula)

        # Filter out common functions and keywords
        keywords = {
            "max",
            "min",
            "avg",
            "sum",
            "sqrt",
            "pow",
            "abs",
            "log",
            "if",
            "then",
            "else",
            "and",
            "or",
            "not",
        }

        dependencies = [t for t in tokens if t not in keywords]
        return list(dict.fromkeys(dependencies))  # Remove duplicates, preserve order

    def _validate_formula_syntax(self, formula: str) -> None:
        """Validate custom formula syntax."""
        # Check balanced parentheses
        open_count = formula.count("(")
        close_count = formula.count(")")
        if open_count != close_count:
            raise InvalidValueError("Formula has unbalanced parentheses")
        
        # Basic syntax validation - simpleeval will handle security
        if not formula.strip():
            raise InvalidValueError("Formula cannot be empty")
        
        # Check for common syntax errors
        if formula.count('"') % 2 != 0 or formula.count("'") % 2 != 0:
            raise InvalidValueError("Formula has unbalanced quotes")

    def _evaluate_custom_formula(self, scores: Dict[str, DimensionScore]) -> float:
        """Evaluate custom formula with dimension scores using safe evaluation.

        Uses simpleeval library for safe mathematical expression evaluation.
        """
        from simpleeval import simple_eval, SimpleEval
        import math

        # Create evaluation context with dimension values
        context = {dep_id: scores[dep_id].value for dep_id in scores.keys()}

        # Safe math functions
        safe_functions = {
            "max": max,
            "min": min,
            "abs": abs,
            "sqrt": math.sqrt,
            "pow": pow,
            "log": lambda x: math.log(x) if x > 0 else 0.0,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        try:
            # Create evaluator with safe functions
            evaluator = SimpleEval()
            evaluator.functions = safe_functions
            evaluator.names = context
            
            # Evaluate the formula safely
            result = evaluator.eval(self.formula)
            return float(result)

        except Exception as e:
            raise BusinessRuleViolation(
                f"Formula evaluation failed: {str(e)}. Check formula syntax and variable names."
            )

    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get dependency graph for cycle detection."""
        return {self.dimension_id: set(self.dependencies)}

    def update_formula(
        self,
        new_formula: str,
        new_dependencies: Optional[List[str]] = None,
        new_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update the composite dimension formula."""
        # Validate new formula
        self._validate_formula_syntax(new_formula)

        # Extract dependencies if not provided
        if new_dependencies is None:
            new_dependencies = self._extract_dependencies_from_formula(new_formula)

        # Update fields
        self.formula = new_formula
        self.dependencies = new_dependencies
        if new_weights is not None:
            self.weights = new_weights

        # Increment version
        self.version += 1

        # Re-validate
        self.__post_init__()

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"CompositeDimension({self.dimension_id})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"CompositeDimension(id={self.dimension_id}, "
            f"method={self.aggregation_method.value}, "
            f"dependencies={len(self.dependencies)})"
        )
