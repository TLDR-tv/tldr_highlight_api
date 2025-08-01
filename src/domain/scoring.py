"""Domain scoring functions.

Pure domain logic for dimension scoring operations.
Following Pythonic DDD - these are pure functions with no infrastructure dependencies.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import json

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.value_objects.dimension_score import DimensionScore


@dataclass
class ScoringResult:
    """Result of dimension scoring operation."""

    dimension_scores: Dict[str, DimensionScore]
    weighted_score: float
    quality_level: str
    confidence_level: str
    meets_criteria: bool
    evaluation_metadata: Dict[str, Any]


def calculate_dimension_scores(
    dimension_set: DimensionSetAggregate,
    segment_data: Dict[str, Any],
    dimension_evaluations: Dict[str, float],
    min_dimensions_required: int = 3,
    quality_thresholds: Optional[Dict[str, float]] = None,
) -> ScoringResult:
    """Calculate dimension scores for content.

    This is pure domain logic - it takes evaluated dimension values
    and produces a scoring result based on business rules.

    Args:
        dimension_set: The dimension set aggregate
        segment_data: Video segment information
        dimension_evaluations: Raw dimension evaluation results (from infrastructure)
        min_dimensions_required: Minimum dimensions that must be scored
        quality_thresholds: Quality level thresholds

    Returns:
        Complete scoring result
    """
    # Record usage
    dimension_set.record_usage("content_scoring")

    # Default quality thresholds
    if quality_thresholds is None:
        quality_thresholds = {
            "legendary": 0.95,
            "exceptional": 0.85,
            "good": 0.70,
            "viable": 0.50,
            "below_threshold": 0.0,
        }

    # Create dimension scores from evaluations
    dimension_scores = {}
    for dim_id, value in dimension_evaluations.items():
        if (
            dim_id in dimension_set.dimensions
            and dimension_set.weights[dim_id].is_significant()
        ):
            # Simple confidence calculation based on value
            if value >= 0.9:
                confidence = "high"
            elif value >= 0.7:
                confidence = "medium"
            elif value >= 0.5:
                confidence = "low"
            else:
                confidence = "uncertain"

            dimension_scores[dim_id] = DimensionScore(
                dimension_id=dim_id,
                value=value,
                confidence=confidence,
                evidence=f"Evaluated from {segment_data.get('segment_id', 'unknown')}",
            )

    # Calculate weighted score using aggregate method
    weighted_score = dimension_set.calculate_highlight_score(dimension_scores)

    # Determine quality level
    quality_level = determine_quality_level(weighted_score, quality_thresholds)

    # Calculate overall confidence
    confidence_level = calculate_overall_confidence(dimension_scores)

    # Check if meets criteria
    meets_criteria = dimension_set.meets_evaluation_criteria(
        dimension_scores, min_dimensions_required
    )

    # Compile metadata
    metadata = {
        "evaluated_dimensions": len(dimension_scores),
        "total_dimensions": len(dimension_set.dimensions),
        "segment_id": segment_data.get("segment_id"),
        "timestamp": segment_data.get("timestamp"),
    }

    return ScoringResult(
        dimension_scores=dimension_scores,
        weighted_score=weighted_score,
        quality_level=quality_level,
        confidence_level=confidence_level,
        meets_criteria=meets_criteria,
        evaluation_metadata=metadata,
    )


def determine_quality_level(score: float, thresholds: Dict[str, float]) -> str:
    """Determine quality level based on score."""
    for level in ["legendary", "exceptional", "good", "viable"]:
        if score >= thresholds[level]:
            return level
    return "below_threshold"


def calculate_overall_confidence(dimension_scores: Dict[str, DimensionScore]) -> str:
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


def compare_dimension_sets(
    content_data: Dict[str, Any],
    dimension_sets: List[DimensionSetAggregate],
    dimension_evaluations: Dict[str, float],
) -> List[Tuple[DimensionSetAggregate, ScoringResult]]:
    """Compare multiple dimension sets on the same content.

    Useful for A/B testing or finding the best dimension set.
    """
    results = []

    for dimension_set in dimension_sets:
        result = calculate_dimension_scores(
            dimension_set, content_data, dimension_evaluations
        )
        results.append((dimension_set, result))

    # Sort by weighted score
    results.sort(key=lambda x: x[1].weighted_score, reverse=True)

    return results


def validate_scoring_consistency(
    dimension_set: DimensionSetAggregate,
    sample_scores: List[Dict[str, float]],
    expected_variance: float = 0.2,
) -> Dict[str, Any]:
    """Validate scoring consistency across multiple samples.

    Identifies dimensions that may be too volatile or constant.
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
        "overall_consistency": _calculate_overall_consistency(dimension_stats),
        "recommendations": _generate_consistency_recommendations(dimension_stats),
    }


def _calculate_overall_consistency(dimension_stats: Dict[str, Dict[str, Any]]) -> str:
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
    dimension_stats: Dict[str, Dict[str, Any]],
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
                f"'{dim_id}' rarely scores high - consider adjusting threshold"
            )
        elif "typically_high" in issues:
            recommendations.append(
                f"'{dim_id}' scores too easily - consider tightening criteria"
            )

    return recommendations


def generate_content_hash(content: Dict[str, Any], modalities: List[str]) -> str:
    """Generate a stable hash for content identification."""
    content_repr = {
        "segment": content.get("segment_id", ""),
        "start": content.get("start_time", 0),
        "end": content.get("end_time", 0),
        "modalities": sorted(modalities),
    }

    content_str = json.dumps(content_repr, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]
