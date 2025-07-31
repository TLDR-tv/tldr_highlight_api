"""Domain service for dimension scoring operations.

This service encapsulates complex scoring logic that doesn't naturally
fit within a single aggregate or entity.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from collections import defaultdict
import hashlib
import json

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.value_objects.dimension_score import DimensionScore
from src.domain.services.dimension_evaluation_strategy import (
    DimensionEvaluationStrategy,
    EvaluationContext,
    DimensionEvaluationResult,
    EvaluationConfidence,
)
from src.domain.services.dimension_calibration_service import (
    DimensionCalibrationService,
    CalibrationProfile,
)
from src.domain.protocols.dimension_score_cache import (
    DimensionScoreCache,
    CacheKeyGenerator,
)
from src.domain.value_objects.dimension_definition import DimensionDefinition
import logfire


@dataclass
class ScoringResult:
    """Result of dimension scoring operation."""

    dimension_scores: Dict[str, DimensionScore]
    weighted_score: float
    quality_level: str
    confidence_level: str
    meets_criteria: bool
    evaluation_metadata: Dict[str, Any]


class DimensionScoringService:
    """Domain service for complex dimension scoring operations.

    This service coordinates between multiple domain objects to provide
    comprehensive scoring functionality while maintaining domain integrity.
    """

    def __init__(
        self,
        evaluation_strategy: DimensionEvaluationStrategy,
        calibration_service: Optional[DimensionCalibrationService] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        cache: Optional[DimensionScoreCache] = None,
        cache_key_generator: Optional[CacheKeyGenerator] = None,
        enable_batch_optimization: bool = True,
        max_batch_size: int = 50,
    ):
        self.evaluation_strategy = evaluation_strategy
        self.calibration_service = calibration_service
        self.quality_thresholds = quality_thresholds or {
            "legendary": 0.95,
            "exceptional": 0.85,
            "good": 0.70,
            "viable": 0.50,
            "below_threshold": 0.0,
        }
        self.cache = cache
        self.cache_key_generator = cache_key_generator
        self.enable_batch_optimization = enable_batch_optimization
        self.max_batch_size = max_batch_size
        self.logger = logfire.get_logger(__name__)

    async def score_content(
        self,
        dimension_set: DimensionSetAggregate,
        segment_data: Dict[str, Any],
        modalities: Dict[str, Any],
        calibration_profile: Optional[CalibrationProfile] = None,
        min_dimensions_required: int = 3,
    ) -> ScoringResult:
        """Score content using a dimension set.

        This method orchestrates the complete scoring process including:
        - Dimension evaluation using the configured strategy
        - Score calibration if available
        - Quality assessment
        - Confidence calculation

        Args:
            dimension_set: The dimension set aggregate to use
            segment_data: Video segment information
            modalities: Available modality data
            calibration_profile: Optional calibration profile
            min_dimensions_required: Minimum dimensions that must be scored

        Returns:
            Complete scoring result with metadata
        """
        # Record usage
        dimension_set.record_usage("content_scoring")

        # Create evaluation context
        context = EvaluationContext(
            segment_data=segment_data,
            dimension_set=dimension_set,
            modalities=modalities,
        )

        # Evaluate all dimensions
        evaluation_results = await self._evaluate_dimensions(dimension_set, context)

        # Convert to dimension scores
        raw_scores = self._create_dimension_scores(evaluation_results)

        # Apply calibration if available
        if self.calibration_service and calibration_profile:
            calibrated_values = await self.calibration_service.calibrate_scores(
                {dim_id: score.value for dim_id, score in raw_scores.items()},
                dimension_set,
                calibration_profile,
            )

            # Update scores with calibrated values
            dimension_scores = {
                dim_id: DimensionScore(
                    dimension_id=dim_id,
                    value=calibrated_values[dim_id],
                    confidence=raw_scores[dim_id].confidence,
                    evidence=raw_scores[dim_id].evidence,
                )
                for dim_id in raw_scores
            }
        else:
            dimension_scores = raw_scores

        # Calculate weighted score
        weighted_score = dimension_set.calculate_highlight_score(dimension_scores)

        # Determine quality level
        quality_level = self._determine_quality_level(weighted_score)

        # Calculate overall confidence
        confidence_level = self._calculate_confidence(dimension_scores)

        # Check if meets criteria
        meets_criteria = dimension_set.meets_evaluation_criteria(
            dimension_scores, min_dimensions_required
        )

        # Compile metadata
        metadata = {
            "evaluated_dimensions": len(dimension_scores),
            "total_dimensions": len(dimension_set.dimensions),
            "calibration_applied": calibration_profile is not None,
            "evaluation_strategy": self.evaluation_strategy.name,
            "modalities_used": list(modalities.keys()),
        }

        return ScoringResult(
            dimension_scores=dimension_scores,
            weighted_score=weighted_score,
            quality_level=quality_level,
            confidence_level=confidence_level,
            meets_criteria=meets_criteria,
            evaluation_metadata=metadata,
        )

    async def _evaluate_dimensions(
        self, dimension_set: DimensionSetAggregate, context: EvaluationContext
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate all dimensions in parallel."""
        tasks = []

        for dim_id, dimension in dimension_set.dimensions.items():
            # Only evaluate dimensions with non-zero weights
            if dimension_set.weights[dim_id].is_significant():
                task = self.evaluation_strategy.evaluate_dimension(dimension, context)
                tasks.append((dim_id, task))

        # Execute evaluations in parallel
        results = {}
        for dim_id, task in tasks:
            try:
                result = await task
                results[dim_id] = result
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate dimension {dim_id}",
                    error=str(e),
                    dimension_id=dim_id,
                )
                # Create fallback result
                results[dim_id] = DimensionEvaluationResult(
                    dimension_id=dim_id,
                    score=dimension_set.dimensions[dim_id].threshold,
                    confidence=EvaluationConfidence.UNCERTAIN,
                    reasoning=f"Evaluation failed: {str(e)}",
                )

        return results

    def _create_dimension_scores(
        self, evaluation_results: Dict[str, DimensionEvaluationResult]
    ) -> Dict[str, DimensionScore]:
        """Convert evaluation results to dimension scores."""
        return {
            dim_id: DimensionScore(
                dimension_id=dim_id,
                value=result.score,
                confidence=result.confidence.value,
                evidence=result.reasoning,
            )
            for dim_id, result in evaluation_results.items()
        }

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        for level in ["legendary", "exceptional", "good", "viable"]:
            if score >= self.quality_thresholds[level]:
                return level
        return "below_threshold"

    def _calculate_confidence(self, dimension_scores: Dict[str, DimensionScore]) -> str:
        """Calculate overall confidence from individual dimension confidences."""
        if not dimension_scores:
            return "uncertain"

        confidence_values = {"high": 3, "medium": 2, "low": 1, "uncertain": 0}

        # Calculate weighted average confidence
        total_confidence = 0
        count = 0

        for score in dimension_scores.values():
            if score.confidence:
                total_confidence += confidence_values.get(score.confidence, 0)
                count += 1

        if count == 0:
            return "uncertain"

        avg_confidence = total_confidence / count

        if avg_confidence >= 2.5:
            return "high"
        elif avg_confidence >= 1.5:
            return "medium"
        elif avg_confidence >= 0.5:
            return "low"
        else:
            return "uncertain"

    async def compare_dimension_sets(
        self,
        content_data: Dict[str, Any],
        dimension_sets: List[DimensionSetAggregate],
        modalities: Dict[str, Any],
    ) -> List[Tuple[DimensionSetAggregate, ScoringResult]]:
        """Compare multiple dimension sets on the same content.

        This is useful for A/B testing or finding the best dimension set
        for specific content.
        """
        results = []

        for dimension_set in dimension_sets:
            try:
                result = await self.score_content(
                    dimension_set, content_data, modalities
                )
                results.append((dimension_set, result))
            except Exception as e:
                self.logger.error(
                    f"Failed to score with dimension set {dimension_set.id}",
                    error=str(e),
                )

        # Sort by weighted score
        results.sort(key=lambda x: x[1].weighted_score, reverse=True)

        return results

    def validate_scoring_consistency(
        self,
        dimension_set: DimensionSetAggregate,
        sample_scores: List[Dict[str, float]],
        expected_variance: float = 0.2,
    ) -> Dict[str, Any]:
        """Validate scoring consistency across multiple samples.

        This helps identify dimensions that may be too volatile or
        always return the same score.
        """
        import numpy as np

        dimension_stats = {}

        for dim_id in dimension_set.dimensions:
            scores = [s.get(dim_id, 0.0) for s in sample_scores]

            if scores:
                dimension_stats[dim_id] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "variance": np.var(scores),
                }

                # Flag potential issues
                stats = dimension_stats[dim_id]
                issues = []

                if stats["std"] < 0.01:
                    issues.append("near_constant")
                elif stats["std"] > expected_variance:
                    issues.append("high_variance")

                if stats["mean"] < 0.1:
                    issues.append("typically_low")
                elif stats["mean"] > 0.9:
                    issues.append("typically_high")

                dimension_stats[dim_id]["issues"] = issues

        return {
            "dimension_statistics": dimension_stats,
            "overall_consistency": self._calculate_overall_consistency(dimension_stats),
            "recommendations": self._generate_consistency_recommendations(
                dimension_stats
            ),
        }

    def _calculate_overall_consistency(
        self, dimension_stats: Dict[str, Dict[str, Any]]
    ) -> str:
        """Calculate overall consistency rating."""
        issue_count = sum(
            len(stats.get("issues", [])) for stats in dimension_stats.values()
        )

        if issue_count == 0:
            return "excellent"
        elif issue_count <= 2:
            return "good"
        elif issue_count <= 5:
            return "fair"
        else:
            return "poor"

    def _generate_consistency_recommendations(
        self, dimension_stats: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on consistency analysis."""
        recommendations = []

        for dim_id, stats in dimension_stats.items():
            issues = stats.get("issues", [])

            if "near_constant" in issues:
                recommendations.append(
                    f"Consider removing or reconfiguring '{dim_id}' - it shows no variation"
                )
            elif "high_variance" in issues:
                recommendations.append(
                    f"Review scoring criteria for '{dim_id}' - high variance detected"
                )

            if "typically_low" in issues:
                recommendations.append(
                    f"'{dim_id}' rarely scores high - consider adjusting threshold or criteria"
                )
            elif "typically_high" in issues:
                recommendations.append(
                    f"'{dim_id}' scores too easily - consider tightening criteria"
                )

        return recommendations

    async def score_content_batch(
        self,
        dimension_set: DimensionSetAggregate,
        content_batch: List[Dict[str, Any]],
        modalities_batch: List[Dict[str, Any]],
        calibration_profile: Optional[CalibrationProfile] = None,
        min_dimensions_required: int = 3,
        parallel_limit: int = 10,
    ) -> List[ScoringResult]:
        """Score multiple contents in batch for efficiency.

        This method optimizes batch processing by:
        - Checking cache for existing scores
        - Grouping by evaluation cost
        - Processing in parallel with controlled concurrency
        - Caching results for future use

        Args:
            dimension_set: The dimension set to use
            content_batch: List of segment data dictionaries
            modalities_batch: List of modality data dictionaries
            calibration_profile: Optional calibration profile
            min_dimensions_required: Minimum dimensions required
            parallel_limit: Maximum parallel evaluations

        Returns:
            List of scoring results in the same order as input
        """
        if len(content_batch) != len(modalities_batch):
            raise ValueError("Content and modalities batch sizes must match")

        # Record batch usage
        dimension_set.record_usage(f"batch_scoring_{len(content_batch)}")

        # Process in chunks if batch is too large
        if self.enable_batch_optimization and len(content_batch) > self.max_batch_size:
            results = []
            for i in range(0, len(content_batch), self.max_batch_size):
                chunk_results = await self.score_content_batch(
                    dimension_set,
                    content_batch[i : i + self.max_batch_size],
                    modalities_batch[i : i + self.max_batch_size],
                    calibration_profile,
                    min_dimensions_required,
                    parallel_limit,
                )
                results.extend(chunk_results)
            return results

        # Check cache if available
        if self.cache and self.cache_key_generator:
            cached_results = await self._check_batch_cache(
                dimension_set, content_batch, modalities_batch
            )
        else:
            cached_results = [None] * len(content_batch)

        # Identify items that need scoring
        to_score_indices = [
            i for i, cached in enumerate(cached_results) if cached is None
        ]

        if not to_score_indices:
            # All results were cached
            return cached_results

        # Group by evaluation cost if optimization is enabled
        if self.enable_batch_optimization:
            grouped_indices = self._group_by_evaluation_cost(
                dimension_set,
                [content_batch[i] for i in to_score_indices],
                [modalities_batch[i] for i in to_score_indices],
            )
        else:
            grouped_indices = {"default": to_score_indices}

        # Process each group
        scoring_tasks = []
        for group_name, indices in grouped_indices.items():
            # Create semaphore for this group
            semaphore = asyncio.Semaphore(parallel_limit)

            for idx in indices:
                original_idx = (
                    to_score_indices[idx] if self.enable_batch_optimization else idx
                )
                task = self._score_with_semaphore(
                    semaphore,
                    dimension_set,
                    content_batch[original_idx],
                    modalities_batch[original_idx],
                    calibration_profile,
                    min_dimensions_required,
                )
                scoring_tasks.append((original_idx, task))

        # Execute all scoring tasks
        scored_results = {}
        for idx, task in scoring_tasks:
            try:
                result = await task
                scored_results[idx] = result

                # Cache the result if caching is enabled
                if self.cache and self.cache_key_generator:
                    await self._cache_result(
                        dimension_set, content_batch[idx], modalities_batch[idx], result
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to score content at index {idx}", error=str(e)
                )
                # Create error result
                scored_results[idx] = self._create_error_result(str(e))

        # Combine cached and newly scored results
        final_results = []
        for i in range(len(content_batch)):
            if cached_results[i] is not None:
                final_results.append(cached_results[i])
            else:
                final_results.append(
                    scored_results.get(i, self._create_error_result("Unknown error"))
                )

        return final_results

    async def _score_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        dimension_set: DimensionSetAggregate,
        segment_data: Dict[str, Any],
        modalities: Dict[str, Any],
        calibration_profile: Optional[CalibrationProfile],
        min_dimensions_required: int,
    ) -> ScoringResult:
        """Score content with concurrency control."""
        async with semaphore:
            return await self.score_content(
                dimension_set,
                segment_data,
                modalities,
                calibration_profile,
                min_dimensions_required,
            )

    def _group_by_evaluation_cost(
        self,
        dimension_set: DimensionSetAggregate,
        content_batch: List[Dict[str, Any]],
        modalities_batch: List[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """Group content by estimated evaluation cost.

        This allows processing expensive evaluations separately from cheap ones.
        """
        groups = defaultdict(list)

        for idx, (content, modalities) in enumerate(
            zip(content_batch, modalities_batch)
        ):
            # Estimate cost based on:
            # 1. Number of modalities
            # 2. Content complexity (e.g., duration)
            # 3. Number of dimensions to evaluate

            modality_count = len(modalities)
            duration = content.get("duration", 0)

            # Simple cost estimation
            if modality_count >= 3 and duration > 60:
                cost_group = "high"
            elif modality_count >= 2 or duration > 30:
                cost_group = "medium"
            else:
                cost_group = "low"

            groups[cost_group].append(idx)

        return dict(groups)

    async def _check_batch_cache(
        self,
        dimension_set: DimensionSetAggregate,
        content_batch: List[Dict[str, Any]],
        modalities_batch: List[Dict[str, Any]],
    ) -> List[Optional[ScoringResult]]:
        """Check cache for batch of contents."""
        cache_keys = []

        for content, modalities in zip(content_batch, modalities_batch):
            content_hash = self._generate_content_hash(content, modalities)
            cache_key = self.cache_key_generator.generate_batch_key(
                dimension_set.id, content_hash, dimension_set.version.version_string
            )
            cache_keys.append(cache_key)

        # Batch get from cache
        cached_scores = await self.cache.get_batch(cache_keys)

        # Convert cached scores to results
        results = []
        for cache_key, cached in cached_scores.items():
            if cached is not None:
                # Reconstruct ScoringResult from cached data
                result = self._reconstruct_result_from_cache(cached)
                results.append(result)
            else:
                results.append(None)

        return results

    async def _cache_result(
        self,
        dimension_set: DimensionSetAggregate,
        segment_data: Dict[str, Any],
        modalities: Dict[str, Any],
        result: ScoringResult,
    ) -> None:
        """Cache a scoring result."""
        content_hash = self._generate_content_hash(segment_data, modalities)

        # Cache individual dimension scores
        cache_tasks = []
        for dim_id, score in result.dimension_scores.items():
            cache_key = self.cache_key_generator.generate_score_key(
                dimension_set.id,
                dim_id,
                content_hash,
                dimension_set.version.version_string,
            )
            cache_tasks.append(self.cache.set(cache_key, score))

        # Also cache the complete result
        batch_key = self.cache_key_generator.generate_batch_key(
            dimension_set.id, content_hash, dimension_set.version.version_string
        )
        cache_data = self._serialize_result_for_cache(result)
        cache_tasks.append(self.cache.set(batch_key, cache_data))

        await asyncio.gather(*cache_tasks, return_exceptions=True)

    def _generate_content_hash(
        self, content: Dict[str, Any], modalities: Dict[str, Any]
    ) -> str:
        """Generate a stable hash for content and modalities."""
        # Create a stable representation
        content_repr = {
            "segment": content.get("segment_id", ""),
            "start": content.get("start_time", 0),
            "end": content.get("end_time", 0),
            "modalities": sorted(modalities.keys()),
        }

        # Generate hash
        content_str = json.dumps(content_repr, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _serialize_result_for_cache(self, result: ScoringResult) -> Dict[str, Any]:
        """Serialize ScoringResult for caching."""
        return {
            "dimension_scores": {
                dim_id: {
                    "value": score.value,
                    "confidence": score.confidence,
                    "evidence": score.evidence,
                }
                for dim_id, score in result.dimension_scores.items()
            },
            "weighted_score": result.weighted_score,
            "quality_level": result.quality_level,
            "confidence_level": result.confidence_level,
            "meets_criteria": result.meets_criteria,
            "evaluation_metadata": result.evaluation_metadata,
        }

    def _reconstruct_result_from_cache(
        self, cached_data: Dict[str, Any]
    ) -> ScoringResult:
        """Reconstruct ScoringResult from cached data."""
        dimension_scores = {
            dim_id: DimensionScore(
                dimension_id=dim_id,
                value=data["value"],
                confidence=data.get("confidence"),
                evidence=data.get("evidence"),
            )
            for dim_id, data in cached_data["dimension_scores"].items()
        }

        return ScoringResult(
            dimension_scores=dimension_scores,
            weighted_score=cached_data["weighted_score"],
            quality_level=cached_data["quality_level"],
            confidence_level=cached_data["confidence_level"],
            meets_criteria=cached_data["meets_criteria"],
            evaluation_metadata=cached_data["evaluation_metadata"],
        )

    def _create_error_result(self, error_message: str) -> ScoringResult:
        """Create an error result for failed scoring."""
        return ScoringResult(
            dimension_scores={},
            weighted_score=0.0,
            quality_level="below_threshold",
            confidence_level="uncertain",
            meets_criteria=False,
            evaluation_metadata={
                "error": error_message,
                "evaluation_strategy": self.evaluation_strategy.name,
            },
        )

    async def evaluate_dimensions_with_optimization(
        self,
        dimension_set: DimensionSetAggregate,
        context: EvaluationContext,
        evaluation_hints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate dimensions with performance optimization.

        This method groups dimensions by:
        - Evaluation cost (from dimension definition)
        - Modality requirements
        - Whether they can be batched together
        """
        # Group dimensions by characteristics
        dimension_groups = self._group_dimensions_for_evaluation(
            dimension_set, evaluation_hints or {}
        )

        results = {}

        # Process each group optimally
        for group_name, dimensions in dimension_groups.items():
            if group_name == "batchable":
                # Process these together if the strategy supports it
                batch_results = await self._evaluate_dimension_batch(
                    dimensions, context
                )
                results.update(batch_results)
            else:
                # Process individually but in parallel
                tasks = [
                    (dim.id, self.evaluation_strategy.evaluate_dimension(dim, context))
                    for dim in dimensions
                ]

                for dim_id, task in tasks:
                    try:
                        result = await task
                        results[dim_id] = result
                    except Exception as e:
                        self.logger.error(
                            f"Failed to evaluate dimension {dim_id}", error=str(e)
                        )
                        results[dim_id] = self._create_fallback_result(dim_id, str(e))

        return results

    def _group_dimensions_for_evaluation(
        self, dimension_set: DimensionSetAggregate, hints: Dict[str, Any]
    ) -> Dict[str, List[DimensionDefinition]]:
        """Group dimensions for optimal evaluation."""
        groups = {"high_cost": [], "medium_cost": [], "low_cost": [], "batchable": []}

        for dim_id, dimension in dimension_set.dimensions.items():
            # Skip zero-weight dimensions
            if not dimension_set.weights[dim_id].is_significant():
                continue

            # Check if dimension supports batching
            if hasattr(dimension, "can_batch") and dimension.can_batch:
                groups["batchable"].append(dimension)
            else:
                # Group by cost
                cost = getattr(dimension, "evaluation_cost", "medium")
                groups[f"{cost}_cost"].append(dimension)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    async def _evaluate_dimension_batch(
        self, dimensions: List[DimensionDefinition], context: EvaluationContext
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate multiple dimensions as a batch."""
        # This is a placeholder - actual implementation would depend on
        # the evaluation strategy supporting batch operations
        results = {}

        # For now, fall back to individual evaluation
        tasks = [
            (dim.id, self.evaluation_strategy.evaluate_dimension(dim, context))
            for dim in dimensions
        ]

        for dim_id, task in tasks:
            try:
                result = await task
                results[dim_id] = result
            except Exception as e:
                results[dim_id] = self._create_fallback_result(dim_id, str(e))

        return results

    def _create_fallback_result(
        self, dimension_id: str, error_message: str
    ) -> DimensionEvaluationResult:
        """Create a fallback result for failed dimension evaluation."""
        return DimensionEvaluationResult(
            dimension_id=dimension_id,
            score=0.0,
            confidence=EvaluationConfidence.UNCERTAIN,
            reasoning=f"Evaluation failed: {error_message}",
        )
