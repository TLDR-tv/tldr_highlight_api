"""Domain calibration functions.

Pure domain logic for dimension calibration.
No infrastructure dependencies - just business rules for score adjustment.
"""

from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate


@dataclass
class CalibrationProfile:
    """Profile for dimension calibration."""
    dimension_set_id: int
    target_distribution: str  # "normal", "uniform", "exponential"
    scale_factors: Dict[str, float]
    offsets: Dict[str, float]
    sample_count: int


def calibrate_dimension_scores(
    raw_scores: Dict[str, float],
    dimension_set: DimensionSetAggregate,
    calibration_profile: Optional[CalibrationProfile] = None
) -> Dict[str, float]:
    """Apply calibration to raw dimension scores.
    
    Pure domain logic for score calibration based on business rules.
    
    Args:
        raw_scores: Raw dimension scores
        dimension_set: The dimension set aggregate
        calibration_profile: Optional calibration profile
        
    Returns:
        Calibrated scores
    """
    if not calibration_profile:
        return raw_scores
    
    calibrated = {}
    
    for dim_id, raw_score in raw_scores.items():
        if dim_id not in dimension_set.dimensions:
            continue
            
        # Apply calibration formula: calibrated = (raw * scale) + offset
        scale = calibration_profile.scale_factors.get(dim_id, 1.0)
        offset = calibration_profile.offsets.get(dim_id, 0.0)
        
        calibrated_score = (raw_score * scale) + offset
        
        # Ensure score stays within [0, 1] bounds
        calibrated[dim_id] = max(0.0, min(1.0, calibrated_score))
    
    return calibrated


def calculate_calibration_parameters(
    historical_scores: List[Dict[str, float]],
    dimension_set: DimensionSetAggregate,
    target_distribution: str = "normal"
) -> CalibrationProfile:
    """Calculate calibration parameters from historical data.
    
    Args:
        historical_scores: List of historical score dictionaries
        dimension_set: The dimension set aggregate
        target_distribution: Target distribution type
        
    Returns:
        Calibration profile with calculated parameters
    """
    scale_factors = {}
    offsets = {}
    
    for dim_id in dimension_set.dimensions:
        # Extract scores for this dimension
        scores = [s.get(dim_id, 0.0) for s in historical_scores if dim_id in s]
        
        if not scores:
            scale_factors[dim_id] = 1.0
            offsets[dim_id] = 0.0
            continue
        
        # Calculate current statistics
        current_mean = np.mean(scores)
        current_std = np.std(scores)
        
        # Calculate target parameters based on distribution
        if target_distribution == "normal":
            # Target: mean=0.5, std=0.15
            target_mean = 0.5
            target_std = 0.15
        elif target_distribution == "uniform":
            # Target: full range [0, 1]
            target_mean = 0.5
            target_std = 0.29  # std of uniform distribution
        else:  # exponential
            # Target: skewed toward lower values
            target_mean = 0.3
            target_std = 0.2
        
        # Calculate scale and offset
        if current_std > 0:
            scale = target_std / current_std
            offset = target_mean - (current_mean * scale)
        else:
            scale = 1.0
            offset = target_mean - current_mean
        
        scale_factors[dim_id] = scale
        offsets[dim_id] = offset
    
    return CalibrationProfile(
        dimension_set_id=dimension_set.id if dimension_set.id else 0,
        target_distribution=target_distribution,
        scale_factors=scale_factors,
        offsets=offsets,
        sample_count=len(historical_scores)
    )


def validate_calibration_effectiveness(
    pre_calibration_scores: List[Dict[str, float]],
    post_calibration_scores: List[Dict[str, float]],
    target_distribution: str
) -> Dict[str, Any]:
    """Validate how well calibration achieved target distribution.
    
    Returns metrics on calibration effectiveness.
    """
    if len(pre_calibration_scores) != len(post_calibration_scores):
        raise ValueError("Pre and post calibration score lists must have same length")
    
    effectiveness_metrics = {}
    
    # Analyze each dimension
    all_dimensions = set()
    for scores in pre_calibration_scores + post_calibration_scores:
        all_dimensions.update(scores.keys())
    
    for dim_id in all_dimensions:
        pre_scores = [s.get(dim_id, 0.0) for s in pre_calibration_scores]
        post_scores = [s.get(dim_id, 0.0) for s in post_calibration_scores]
        
        if not pre_scores or not post_scores:
            continue
        
        # Calculate distribution metrics
        pre_mean, pre_std = np.mean(pre_scores), np.std(pre_scores)
        post_mean, post_std = np.mean(post_scores), np.std(post_scores)
        
        # Calculate target achievement
        if target_distribution == "normal":
            target_mean, target_std = 0.5, 0.15
        elif target_distribution == "uniform":
            target_mean, target_std = 0.5, 0.29
        else:  # exponential
            target_mean, target_std = 0.3, 0.2
        
        mean_error = abs(post_mean - target_mean)
        std_error = abs(post_std - target_std)
        
        effectiveness_metrics[dim_id] = {
            "pre_calibration": {"mean": pre_mean, "std": pre_std},
            "post_calibration": {"mean": post_mean, "std": post_std},
            "target": {"mean": target_mean, "std": target_std},
            "mean_error": mean_error,
            "std_error": std_error,
            "effectiveness": 1.0 - (mean_error + std_error) / 2.0  # Simple effectiveness score
        }
    
    # Overall effectiveness
    if effectiveness_metrics:
        overall_effectiveness = np.mean([
            m["effectiveness"] for m in effectiveness_metrics.values()
        ])
    else:
        overall_effectiveness = 0.0
    
    return {
        "dimension_metrics": effectiveness_metrics,
        "overall_effectiveness": overall_effectiveness,
        "target_distribution": target_distribution
    }


def merge_calibration_profiles(
    profiles: List[CalibrationProfile],
    weights: Optional[List[float]] = None
) -> CalibrationProfile:
    """Merge multiple calibration profiles with optional weighting.
    
    Useful for combining calibrations from different time periods or sources.
    """
    if not profiles:
        raise ValueError("At least one calibration profile required")
    
    if weights is None:
        weights = [1.0] * len(profiles)
    elif len(weights) != len(profiles):
        raise ValueError("Weights must match number of profiles")
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Merge parameters
    merged_scale_factors = {}
    merged_offsets = {}
    all_dimensions = set()
    
    for profile in profiles:
        all_dimensions.update(profile.scale_factors.keys())
    
    for dim_id in all_dimensions:
        weighted_scale = 0.0
        weighted_offset = 0.0
        
        for profile, weight in zip(profiles, normalized_weights):
            scale = profile.scale_factors.get(dim_id, 1.0)
            offset = profile.offsets.get(dim_id, 0.0)
            weighted_scale += scale * weight
            weighted_offset += offset * weight
        
        merged_scale_factors[dim_id] = weighted_scale
        merged_offsets[dim_id] = weighted_offset
    
    # Use most common target distribution
    distributions = [p.target_distribution for p in profiles]
    target_distribution = max(set(distributions), key=distributions.count)
    
    # Sum sample counts
    total_samples = sum(p.sample_count for p in profiles)
    
    return CalibrationProfile(
        dimension_set_id=profiles[0].dimension_set_id,
        target_distribution=target_distribution,
        scale_factors=merged_scale_factors,
        offsets=merged_offsets,
        sample_count=total_samples
    )


def detect_calibration_drift(
    current_profile: CalibrationProfile,
    new_sample_scores: List[Dict[str, float]],
    drift_threshold: float = 0.1
) -> Dict[str, Any]:
    """Detect if calibration parameters have drifted significantly.
    
    Returns drift analysis and recommendations.
    """
    drift_analysis = {}
    dimensions_with_drift = []
    
    for dim_id in current_profile.scale_factors:
        scores = [s.get(dim_id, 0.0) for s in new_sample_scores if dim_id in s]
        
        if not scores:
            continue
        
        # Apply current calibration
        calibrated_scores = [
            (score * current_profile.scale_factors[dim_id]) + current_profile.offsets[dim_id]
            for score in scores
        ]
        
        # Check if calibrated scores match expected distribution
        actual_mean = np.mean(calibrated_scores)
        actual_std = np.std(calibrated_scores)
        
        # Expected values based on target distribution
        if current_profile.target_distribution == "normal":
            expected_mean, expected_std = 0.5, 0.15
        elif current_profile.target_distribution == "uniform":
            expected_mean, expected_std = 0.5, 0.29
        else:
            expected_mean, expected_std = 0.3, 0.2
        
        mean_drift = abs(actual_mean - expected_mean)
        std_drift = abs(actual_std - expected_std)
        total_drift = (mean_drift + std_drift) / 2.0
        
        drift_analysis[dim_id] = {
            "mean_drift": mean_drift,
            "std_drift": std_drift,
            "total_drift": total_drift,
            "requires_recalibration": total_drift > drift_threshold
        }
        
        if total_drift > drift_threshold:
            dimensions_with_drift.append(dim_id)
    
    return {
        "drift_analysis": drift_analysis,
        "dimensions_requiring_recalibration": dimensions_with_drift,
        "overall_drift": np.mean([d["total_drift"] for d in drift_analysis.values()]),
        "recommendation": "Recalibrate" if dimensions_with_drift else "Calibration stable"
    }