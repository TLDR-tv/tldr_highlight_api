"""Calibration service for dimension scoring.

This module provides calibration capabilities to ensure consistent and
accurate dimension scoring across different content types and evaluators.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from collections import defaultdict
import json

from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.services.dimension_evaluation_strategy import DimensionEvaluationResult
import logfire


@dataclass
class CalibrationData:
    """Calibration data for a single dimension."""

    dimension_id: str
    raw_scores: List[float]
    calibrated_scores: List[float]
    confidence_levels: List[str]

    # Statistical parameters
    mean_raw: float = 0.0
    std_raw: float = 0.0
    mean_calibrated: float = 0.0
    std_calibrated: float = 0.0

    # Calibration parameters
    scale_factor: float = 1.0
    offset: float = 0.0

    # Mapping for percentile calibration
    percentile_map: Dict[float, float] = field(default_factory=dict)

    # Metadata
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Calculate statistics on initialization."""
        if self.raw_scores:
            self.mean_raw = np.mean(self.raw_scores)
            self.std_raw = np.std(self.raw_scores)
            self.sample_count = len(self.raw_scores)

        if self.calibrated_scores:
            self.mean_calibrated = np.mean(self.calibrated_scores)
            self.std_calibrated = np.std(self.calibrated_scores)


@dataclass
class CalibrationProfile:
    """Complete calibration profile for a dimension set."""

    dimension_set_id: str
    organization_id: int
    calibration_data: Dict[str, CalibrationData] = field(default_factory=dict)

    # Global calibration settings
    target_distribution: str = "normal"  # "normal", "uniform", "beta"
    target_mean: float = 0.5
    target_std: float = 0.2

    # Confidence calibration
    confidence_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "uncertain": 0.2,
        }
    )

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def is_calibrated(self, dimension_id: str) -> bool:
        """Check if a dimension has calibration data."""
        return (
            dimension_id in self.calibration_data
            and self.calibration_data[dimension_id].sample_count >= 10
        )


class DimensionCalibrationService:
    """Service for calibrating dimension scores.

    Provides methods to:
    - Collect calibration data from evaluations
    - Calculate calibration parameters
    - Apply calibration to raw scores
    - Monitor calibration effectiveness
    """

    def __init__(self):
        self.logger = logfire.get_logger(__name__)
        self.profiles: Dict[str, CalibrationProfile] = {}

    async def calibrate_scores(
        self,
        raw_scores: Dict[str, float],
        dimension_set: DimensionSetAggregate,
        profile: Optional[CalibrationProfile] = None,
    ) -> Dict[str, float]:
        """Apply calibration to raw dimension scores.

        Args:
            raw_scores: Raw scores from evaluation
            dimension_set: The dimension set used
            profile: Optional calibration profile (will load if not provided)

        Returns:
            Calibrated scores
        """
        # Get or load calibration profile
        if profile is None:
            profile = self._get_or_create_profile(
                str(dimension_set.id), dimension_set.organization_id
            )

        calibrated = {}

        for dim_id, raw_score in raw_scores.items():
            if profile.is_calibrated(dim_id):
                # Apply calibration
                calibrated[dim_id] = self._apply_calibration(
                    raw_score, profile.calibration_data[dim_id]
                )
            else:
                # No calibration available, use raw score
                calibrated[dim_id] = raw_score
                self.logger.info(
                    f"No calibration data for dimension {dim_id}, using raw score",
                    dimension_id=dim_id,
                    raw_score=raw_score,
                )

        return calibrated

    def collect_calibration_sample(
        self,
        evaluation_results: List[DimensionEvaluationResult],
        human_scores: Optional[Dict[str, float]] = None,
        dimension_set_id: str = None,
        organization_id: int = None,
    ) -> None:
        """Collect samples for calibration from evaluations.

        Args:
            evaluation_results: Results from dimension evaluation
            human_scores: Optional human-provided scores for training
            dimension_set_id: ID of the dimension set
            organization_id: Organization ID
        """
        if not dimension_set_id or not organization_id:
            self.logger.warning(
                "Missing dimension_set_id or organization_id for calibration"
            )
            return

        profile = self._get_or_create_profile(dimension_set_id, organization_id)

        for result in evaluation_results:
            dim_id = result.dimension_id

            # Initialize calibration data if needed
            if dim_id not in profile.calibration_data:
                profile.calibration_data[dim_id] = CalibrationData(
                    dimension_id=dim_id,
                    raw_scores=[],
                    calibrated_scores=[],
                    confidence_levels=[],
                )

            cal_data = profile.calibration_data[dim_id]

            # Add raw score
            cal_data.raw_scores.append(result.score)
            cal_data.confidence_levels.append(result.confidence.value)

            # Add calibrated score (human score if available, otherwise use raw)
            if human_scores and dim_id in human_scores:
                cal_data.calibrated_scores.append(human_scores[dim_id])
            else:
                cal_data.calibrated_scores.append(result.score)

            # Update statistics
            cal_data.sample_count += 1
            cal_data.last_updated = datetime.utcnow()

        # Recalculate calibration parameters if enough samples
        for cal_data in profile.calibration_data.values():
            if cal_data.sample_count >= 10:
                self._calculate_calibration_parameters(cal_data)

        profile.updated_at = datetime.utcnow()

    def _apply_calibration(
        self, raw_score: float, calibration_data: CalibrationData
    ) -> float:
        """Apply calibration transformation to a raw score."""
        # Method 1: Linear calibration (scale and offset)
        linear_calibrated = (
            raw_score * calibration_data.scale_factor
        ) + calibration_data.offset

        # Method 2: Percentile mapping (if available)
        if calibration_data.percentile_map:
            percentile_calibrated = self._apply_percentile_mapping(
                raw_score, calibration_data.percentile_map
            )
            # Average the two methods
            calibrated = (linear_calibrated + percentile_calibrated) / 2
        else:
            calibrated = linear_calibrated

        # Ensure bounds
        return max(0.0, min(1.0, calibrated))

    def _calculate_calibration_parameters(self, cal_data: CalibrationData) -> None:
        """Calculate calibration parameters from collected samples."""
        if len(cal_data.raw_scores) < 10:
            return

        raw_array = np.array(cal_data.raw_scores)
        calibrated_array = np.array(cal_data.calibrated_scores)

        # Update statistics
        cal_data.mean_raw = np.mean(raw_array)
        cal_data.std_raw = np.std(raw_array)
        cal_data.mean_calibrated = np.mean(calibrated_array)
        cal_data.std_calibrated = np.std(calibrated_array)

        # Method 1: Linear regression calibration
        if cal_data.std_raw > 0:
            # Calculate scale and offset for linear transformation
            cal_data.scale_factor = cal_data.std_calibrated / cal_data.std_raw
            cal_data.offset = cal_data.mean_calibrated - (
                cal_data.scale_factor * cal_data.mean_raw
            )
        else:
            # No variance in raw scores
            cal_data.scale_factor = 1.0
            cal_data.offset = cal_data.mean_calibrated - cal_data.mean_raw

        # Method 2: Percentile mapping
        percentiles = [10, 25, 50, 75, 90]
        cal_data.percentile_map = {}

        for p in percentiles:
            raw_percentile = np.percentile(raw_array, p)
            calibrated_percentile = np.percentile(calibrated_array, p)
            cal_data.percentile_map[raw_percentile] = calibrated_percentile

    def _apply_percentile_mapping(
        self, raw_score: float, percentile_map: Dict[float, float]
    ) -> float:
        """Apply percentile-based calibration."""
        if not percentile_map:
            return raw_score

        # Get sorted percentile keys
        raw_percentiles = sorted(percentile_map.keys())

        # Find surrounding percentiles
        if raw_score <= raw_percentiles[0]:
            return percentile_map[raw_percentiles[0]]
        elif raw_score >= raw_percentiles[-1]:
            return percentile_map[raw_percentiles[-1]]

        # Linear interpolation between percentiles
        for i in range(len(raw_percentiles) - 1):
            if raw_percentiles[i] <= raw_score <= raw_percentiles[i + 1]:
                # Interpolate
                raw_low, raw_high = raw_percentiles[i], raw_percentiles[i + 1]
                cal_low, cal_high = percentile_map[raw_low], percentile_map[raw_high]

                ratio = (raw_score - raw_low) / (raw_high - raw_low)
                return cal_low + ratio * (cal_high - cal_low)

        return raw_score

    def calculate_confidence_calibration(
        self,
        evaluation_results: List[DimensionEvaluationResult],
        actual_accuracy: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculate confidence calibration metrics.

        Args:
            evaluation_results: Evaluation results with confidence
            actual_accuracy: Optional actual accuracy for each result

        Returns:
            Calibration metrics including ECE (Expected Calibration Error)
        """
        # Group by confidence level
        confidence_groups = defaultdict(list)

        for result in evaluation_results:
            confidence_groups[result.confidence.value].append(result)

        calibration_metrics = {}

        # Calculate metrics for each confidence level
        for confidence_level, results in confidence_groups.items():
            if not results:
                continue

            # Expected confidence based on level
            expected_confidence = self._get_expected_confidence(confidence_level)

            # Calculate actual performance if accuracy data provided
            if actual_accuracy:
                accuracies = [actual_accuracy.get(r.dimension_id, 0.5) for r in results]
                actual_confidence = np.mean(accuracies)
            else:
                # Use score distribution as proxy
                scores = [r.score for r in results]
                actual_confidence = np.mean(scores)

            calibration_metrics[confidence_level] = {
                "expected": expected_confidence,
                "actual": actual_confidence,
                "error": abs(expected_confidence - actual_confidence),
                "sample_size": len(results),
            }

        # Calculate overall ECE (Expected Calibration Error)
        total_samples = sum(m["sample_size"] for m in calibration_metrics.values())
        if total_samples > 0:
            ece = sum(
                m["error"] * m["sample_size"] / total_samples
                for m in calibration_metrics.values()
            )
            calibration_metrics["expected_calibration_error"] = ece

        return calibration_metrics

    def _get_expected_confidence(self, confidence_level: str) -> float:
        """Get expected accuracy for a confidence level."""
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5, "uncertain": 0.3}
        return confidence_map.get(confidence_level, 0.5)

    def get_calibration_summary(
        self, dimension_set_id: str, organization_id: int
    ) -> Dict[str, Any]:
        """Get summary of calibration status for a dimension set."""
        profile = self._get_or_create_profile(dimension_set_id, organization_id)

        summary = {
            "dimension_set_id": dimension_set_id,
            "organization_id": organization_id,
            "last_updated": profile.updated_at.isoformat(),
            "version": profile.version,
            "dimensions": {},
        }

        for dim_id, cal_data in profile.calibration_data.items():
            summary["dimensions"][dim_id] = {
                "is_calibrated": cal_data.sample_count >= 10,
                "sample_count": cal_data.sample_count,
                "last_updated": cal_data.last_updated.isoformat(),
                "statistics": {
                    "mean_raw": cal_data.mean_raw,
                    "std_raw": cal_data.std_raw,
                    "mean_calibrated": cal_data.mean_calibrated,
                    "std_calibrated": cal_data.std_calibrated,
                },
                "calibration_params": {
                    "scale_factor": cal_data.scale_factor,
                    "offset": cal_data.offset,
                },
            }

        return summary

    def export_calibration_profile(
        self, dimension_set_id: str, organization_id: int
    ) -> str:
        """Export calibration profile as JSON."""
        profile = self._get_or_create_profile(dimension_set_id, organization_id)

        export_data = {
            "dimension_set_id": profile.dimension_set_id,
            "organization_id": profile.organization_id,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
            "version": profile.version,
            "target_distribution": profile.target_distribution,
            "target_mean": profile.target_mean,
            "target_std": profile.target_std,
            "calibration_data": {},
        }

        for dim_id, cal_data in profile.calibration_data.items():
            export_data["calibration_data"][dim_id] = {
                "sample_count": cal_data.sample_count,
                "mean_raw": cal_data.mean_raw,
                "std_raw": cal_data.std_raw,
                "mean_calibrated": cal_data.mean_calibrated,
                "std_calibrated": cal_data.std_calibrated,
                "scale_factor": cal_data.scale_factor,
                "offset": cal_data.offset,
                "percentile_map": cal_data.percentile_map,
            }

        return json.dumps(export_data, indent=2)

    def import_calibration_profile(self, json_data: str) -> CalibrationProfile:
        """Import calibration profile from JSON."""
        data = json.loads(json_data)

        profile = CalibrationProfile(
            dimension_set_id=data["dimension_set_id"],
            organization_id=data["organization_id"],
            target_distribution=data.get("target_distribution", "normal"),
            target_mean=data.get("target_mean", 0.5),
            target_std=data.get("target_std", 0.2),
            version=data.get("version", 1),
        )

        # Import calibration data
        for dim_id, cal_dict in data.get("calibration_data", {}).items():
            cal_data = CalibrationData(
                dimension_id=dim_id,
                raw_scores=[],  # Not importing raw data
                calibrated_scores=[],
                confidence_levels=[],
                mean_raw=cal_dict["mean_raw"],
                std_raw=cal_dict["std_raw"],
                mean_calibrated=cal_dict["mean_calibrated"],
                std_calibrated=cal_dict["std_calibrated"],
                scale_factor=cal_dict["scale_factor"],
                offset=cal_dict["offset"],
                percentile_map=cal_dict.get("percentile_map", {}),
                sample_count=cal_dict["sample_count"],
            )
            profile.calibration_data[dim_id] = cal_data

        # Store in cache
        key = f"{profile.dimension_set_id}:{profile.organization_id}"
        self.profiles[key] = profile

        return profile

    def _get_or_create_profile(
        self, dimension_set_id: str, organization_id: int
    ) -> CalibrationProfile:
        """Get existing or create new calibration profile."""
        key = f"{dimension_set_id}:{organization_id}"

        if key not in self.profiles:
            self.profiles[key] = CalibrationProfile(
                dimension_set_id=dimension_set_id, organization_id=organization_id
            )

        return self.profiles[key]
