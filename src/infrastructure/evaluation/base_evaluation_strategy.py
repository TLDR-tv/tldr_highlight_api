"""Base implementation for dimension evaluation strategies.

This provides common functionality for concrete strategy implementations
while maintaining protocol compliance.
"""

from typing import Dict, Optional, Any

from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.domain.value_objects.dimension_score import DimensionScore
from src.domain.services.dimension_evaluation_strategy import (
    DimensionEvaluationResult,
    EvaluationContext,
)
import logfire


class BaseEvaluationStrategy:
    """Base class providing common functionality for evaluation strategies."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logfire.get_logger(f"{__name__}.{name}")

    async def evaluate_all_dimensions(
        self, context: EvaluationContext
    ) -> Dict[str, DimensionScore]:
        """Evaluate all dimensions in the dimension set.

        Args:
            context: Evaluation context

        Returns:
            Dictionary of dimension IDs to DimensionScore objects
        """
        results = {}
        tasks = []

        # Create tasks for parallel evaluation
        for dim_id, dimension in context.dimension_set.dimensions.items():
            task = self.evaluate_dimension(dimension, context)
            tasks.append((dim_id, task))

        # Execute evaluations in parallel
        for dim_id, task in tasks:
            try:
                result = await task
                # Create proper DimensionScore object
                results[dim_id] = DimensionScore(
                    dimension_id=dim_id,
                    value=result.score,
                    confidence=result.confidence.value if result.confidence else "medium",
                    evidence=result.reasoning or f"Evaluated by {self.name} strategy"
                )

                # Store detailed result in context for reference
                context.previous_evaluations.append(result)

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate dimension {dim_id}",
                    error=str(e),
                    dimension_id=dim_id,
                )
                # Use default threshold as fallback
                fallback_score = context.dimension_set.dimensions[dim_id].threshold
                results[dim_id] = DimensionScore(
                    dimension_id=dim_id,
                    value=fallback_score,
                    confidence="low",
                    evidence=f"Fallback score due to evaluation error: {str(e)}"
                )

        return results

    def validate_evaluation_result(
        self, result: DimensionEvaluationResult, dimension: DimensionDefinition
    ) -> bool:
        """Validate an evaluation result against dimension constraints.

        Args:
            result: The evaluation result to validate
            dimension: The dimension definition

        Returns:
            True if valid, False otherwise
        """
        # Check score is in valid range
        if not 0.0 <= result.score <= 1.0:
            return False

        # Check dimension type constraints
        if dimension.dimension_type == "binary":
            if result.score not in [0.0, 1.0]:
                return False

        return True

    async def calibrate_scores(
        self,
        raw_scores: Dict[str, float],
        calibration_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Calibrate raw scores based on historical data or reference points.

        Args:
            raw_scores: Raw dimension scores
            calibration_data: Optional calibration information

        Returns:
            Calibrated scores
        """
        # Default implementation: no calibration
        return raw_scores
