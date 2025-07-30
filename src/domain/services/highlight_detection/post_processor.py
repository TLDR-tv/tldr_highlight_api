"""
Post-processing and refinement for the TL;DR Highlight API.

This module implements sophisticated post-processing algorithms for refining
highlight candidates, including temporal smoothing, quality enhancement,
and boundary optimization.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import savgol_filter

from .base_detector import HighlightCandidate
from ...utils.scoring_utils import (
    apply_temporal_smoothing,
)

logger = logging.getLogger(__name__)


class SmoothingMethod(str, Enum):
    """Methods for temporal smoothing."""

    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    SAVGOL = "savgol"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"


class BoundaryOptimization(str, Enum):
    """Methods for optimizing highlight boundaries."""

    NONE = "none"
    SCENE_AWARE = "scene_aware"
    AUDIO_BASED = "audio_based"
    CONTENT_ADAPTIVE = "content_adaptive"
    ML_GUIDED = "ml_guided"


class QualityEnhancement(str, Enum):
    """Methods for enhancing highlight quality."""

    NONE = "none"
    CONFIDENCE_BOOSTING = "confidence_boosting"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    CONTEXTUAL_REFINEMENT = "contextual_refinement"
    ADAPTIVE_SCORING = "adaptive_scoring"


@dataclass
class ProcessingMetrics:
    """
    Metrics for post-processing performance evaluation.

    Contains statistics about processing improvements and
    quality enhancements applied.
    """

    candidates_processed: int
    candidates_improved: int
    avg_score_improvement: float
    avg_confidence_improvement: float
    boundary_adjustments: int
    temporal_smoothing_applied: bool
    quality_enhancements: int
    processing_time_ms: float = 0.0

    @property
    def improvement_ratio(self) -> float:
        """Get ratio of improved candidates."""
        return self.candidates_improved / max(1, self.candidates_processed)

    @property
    def processing_efficiency(self) -> float:
        """Get processing efficiency metric."""
        if self.processing_time_ms == 0:
            return float("inf")
        return (
            self.candidates_processed / self.processing_time_ms * 1000
        )  # Candidates per second


class PostProcessorConfig(BaseModel):
    """
    Configuration for highlight post-processing.

    Defines parameters for temporal smoothing, quality enhancement,
    and boundary optimization algorithms.
    """

    # General processing settings
    enabled: bool = Field(default=True, description="Enable post-processing")
    max_processing_time_seconds: float = Field(
        default=30.0, gt=0.0, description="Maximum processing time limit"
    )

    # Temporal smoothing configuration
    temporal_smoothing_enabled: bool = Field(
        default=True, description="Enable temporal smoothing"
    )
    smoothing_method: SmoothingMethod = Field(
        default=SmoothingMethod.SAVGOL, description="Method for temporal smoothing"
    )
    smoothing_window_size: float = Field(
        default=10.0, gt=0.0, description="Smoothing window size in seconds"
    )
    smoothing_strength: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Strength of smoothing effect"
    )

    # Boundary optimization configuration
    boundary_optimization: BoundaryOptimization = Field(
        default=BoundaryOptimization.CONTENT_ADAPTIVE,
        description="Method for boundary optimization",
    )
    boundary_tolerance: float = Field(
        default=5.0, ge=0.0, description="Maximum boundary adjustment in seconds"
    )
    min_highlight_duration: float = Field(
        default=10.0,
        gt=0.0,
        description="Minimum highlight duration after optimization",
    )
    max_highlight_duration: float = Field(
        default=120.0,
        gt=0.0,
        description="Maximum highlight duration after optimization",
    )

    # Quality enhancement configuration
    quality_enhancement: QualityEnhancement = Field(
        default=QualityEnhancement.ADAPTIVE_SCORING,
        description="Method for quality enhancement",
    )
    confidence_boost_factor: float = Field(
        default=1.1, ge=1.0, le=2.0, description="Factor for confidence boosting"
    )
    feature_enhancement_enabled: bool = Field(
        default=True, description="Enable feature-based quality enhancement"
    )

    # Filtering and refinement
    min_score_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold after processing",
    )
    max_score_boost: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Maximum allowed score boost"
    )
    outlier_detection_enabled: bool = Field(
        default=True, description="Enable outlier detection and removal"
    )
    outlier_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description="Standard deviation threshold for outlier detection",
    )

    # Context-aware processing
    context_window_size: float = Field(
        default=60.0, gt=0.0, description="Context window size for refinement (seconds)"
    )
    neighbor_influence_factor: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Influence factor of neighboring highlights",
    )

    # Performance optimization
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing of candidates"
    )
    max_concurrent_workers: int = Field(
        default=4, ge=1, description="Maximum number of concurrent workers"
    )


class TemporalSmoother:
    """
    Handles temporal smoothing of highlight scores and boundaries.

    Implements various smoothing algorithms to reduce noise
    and improve temporal consistency of highlights.
    """

    def __init__(self, config: PostProcessorConfig):
        """
        Initialize temporal smoother.

        Args:
            config: Post-processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TemporalSmoother")

    async def smooth_highlights(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """
        Apply temporal smoothing to highlight candidates.

        Args:
            candidates: List of highlight candidates

        Returns:
            List of smoothed highlight candidates
        """
        if not self.config.temporal_smoothing_enabled or len(candidates) < 3:
            return candidates

        self.logger.debug(
            f"Applying temporal smoothing to {len(candidates)} candidates"
        )

        # Sort candidates by timestamp
        sorted_candidates = sorted(
            candidates, key=lambda c: (c.start_time + c.end_time) / 2
        )

        # Extract timestamps and scores
        timestamps = [(c.start_time + c.end_time) / 2 for c in sorted_candidates]
        scores = [c.score for c in sorted_candidates]
        confidences = [c.confidence for c in sorted_candidates]

        # Apply smoothing to scores
        smoothed_scores = await self._apply_smoothing(timestamps, scores)
        smoothed_confidences = await self._apply_smoothing(timestamps, confidences)

        # Create smoothed candidates
        smoothed_candidates = []
        for i, candidate in enumerate(sorted_candidates):
            # Create new candidate with smoothed scores
            smoothed_candidate = HighlightCandidate(
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                score=min(1.0, max(0.0, smoothed_scores[i])),
                confidence=min(1.0, max(0.0, smoothed_confidences[i])),
                modality_results=candidate.modality_results,
                features=candidate.features.copy() if candidate.features else {},
                tags=candidate.tags.copy(),
                candidate_id=candidate.candidate_id,
                created_at=candidate.created_at,
            )

            # Add smoothing metadata
            if not smoothed_candidate.features:
                smoothed_candidate.features = {}

            smoothed_candidate.features.update(
                {
                    "original_score": candidate.score,
                    "original_confidence": candidate.confidence,
                    "smoothing_applied": True,
                    "smoothing_method": self.config.smoothing_method,
                }
            )

            smoothed_candidates.append(smoothed_candidate)

        return smoothed_candidates

    async def _apply_smoothing(
        self, timestamps: List[float], values: List[float]
    ) -> List[float]:
        """Apply smoothing algorithm to values."""
        if len(values) < 3:
            return values

        try:
            if self.config.smoothing_method == SmoothingMethod.MOVING_AVERAGE:
                return self._moving_average_smoothing(timestamps, values)
            elif self.config.smoothing_method == SmoothingMethod.EXPONENTIAL:
                return self._exponential_smoothing(values)
            elif self.config.smoothing_method == SmoothingMethod.SAVGOL:
                return self._savgol_smoothing(values)
            elif self.config.smoothing_method == SmoothingMethod.GAUSSIAN:
                return self._gaussian_smoothing(timestamps, values)
            elif self.config.smoothing_method == SmoothingMethod.MEDIAN:
                return self._median_smoothing(values)
            else:
                return self._savgol_smoothing(values)
        except Exception as e:
            self.logger.warning(f"Smoothing failed: {e}, returning original values")
            return values

    def _moving_average_smoothing(
        self, timestamps: List[float], values: List[float]
    ) -> List[float]:
        """Apply moving average smoothing."""
        smoothed = apply_temporal_smoothing(
            timestamps,
            values,
            method="moving_average",
            window_size=self.config.smoothing_window_size,
        )

        # Blend with original values based on smoothing strength
        blended = []
        for i, (original, smooth) in enumerate(zip(values, smoothed)):
            blended_value = (
                original * (1.0 - self.config.smoothing_strength)
                + smooth * self.config.smoothing_strength
            )
            blended.append(blended_value)

        return blended

    def _exponential_smoothing(self, values: List[float]) -> List[float]:
        """Apply exponential smoothing."""
        if not values:
            return values

        alpha = self.config.smoothing_strength
        smoothed = [values[0]]

        for value in values[1:]:
            smoothed_value = alpha * value + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)

        return smoothed

    def _savgol_smoothing(self, values: List[float]) -> List[float]:
        """Apply Savitzky-Golay smoothing."""
        if len(values) < 5:
            return values

        window_length = min(len(values) // 2 * 2 + 1, 11)  # Ensure odd, max 11
        try:
            smoothed = savgol_filter(values, window_length, 2)

            # Blend with original values
            blended = []
            for original, smooth in zip(values, smoothed):
                blended_value = (
                    original * (1.0 - self.config.smoothing_strength)
                    + smooth * self.config.smoothing_strength
                )
                blended.append(blended_value)

            return blended
        except Exception:
            return values

    def _gaussian_smoothing(
        self, timestamps: List[float], values: List[float]
    ) -> List[float]:
        """Apply Gaussian smoothing."""
        smoothed = []
        sigma = self.config.smoothing_window_size / 3.0  # 3-sigma rule

        for i, target_time in enumerate(timestamps):
            weighted_sum = 0.0
            weight_sum = 0.0

            for j, (time, value) in enumerate(zip(timestamps, values)):
                distance = abs(time - target_time)
                weight = np.exp(-(distance**2) / (2 * sigma**2))
                weighted_sum += weight * value
                weight_sum += weight

            if weight_sum > 0:
                smoothed_value = weighted_sum / weight_sum
            else:
                smoothed_value = values[i]

            # Blend with original
            blended_value = (
                values[i] * (1.0 - self.config.smoothing_strength)
                + smoothed_value * self.config.smoothing_strength
            )
            smoothed.append(blended_value)

        return smoothed

    def _median_smoothing(self, values: List[float]) -> List[float]:
        """Apply median smoothing."""
        window_size = min(5, len(values))
        smoothed = []

        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            window_values = values[start:end]

            median_value = np.median(window_values)

            # Blend with original
            blended_value = (
                values[i] * (1.0 - self.config.smoothing_strength)
                + median_value * self.config.smoothing_strength
            )
            smoothed.append(blended_value)

        return smoothed


class BoundaryOptimizer:
    """
    Optimizes highlight boundaries for better quality.

    Implements algorithms to adjust start and end times
    based on content analysis and quality metrics.
    """

    def __init__(self, config: PostProcessorConfig):
        """
        Initialize boundary optimizer.

        Args:
            config: Post-processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BoundaryOptimizer")

    async def optimize_boundaries(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """
        Optimize boundaries of highlight candidates.

        Args:
            candidates: List of highlight candidates

        Returns:
            List of candidates with optimized boundaries
        """
        if self.config.boundary_optimization == BoundaryOptimization.NONE:
            return candidates

        self.logger.debug(f"Optimizing boundaries for {len(candidates)} candidates")

        optimized_candidates = []
        adjustments_made = 0

        for candidate in candidates:
            try:
                optimized_candidate = await self._optimize_single_boundary(candidate)
                optimized_candidates.append(optimized_candidate)

                # Check if boundaries were adjusted
                if (
                    abs(optimized_candidate.start_time - candidate.start_time) > 0.1
                    or abs(optimized_candidate.end_time - candidate.end_time) > 0.1
                ):
                    adjustments_made += 1

            except Exception as e:
                self.logger.warning(
                    f"Boundary optimization failed for candidate {candidate.candidate_id}: {e}"
                )
                optimized_candidates.append(candidate)

        self.logger.debug(f"Made boundary adjustments to {adjustments_made} candidates")
        return optimized_candidates

    async def _optimize_single_boundary(
        self, candidate: HighlightCandidate
    ) -> HighlightCandidate:
        """Optimize boundaries for a single candidate."""
        original_start = candidate.start_time
        original_end = candidate.end_time
        _original_duration = candidate.duration

        # Determine optimization method
        if self.config.boundary_optimization == BoundaryOptimization.SCENE_AWARE:
            new_start, new_end = await self._scene_aware_optimization(candidate)
        elif self.config.boundary_optimization == BoundaryOptimization.AUDIO_BASED:
            new_start, new_end = await self._audio_based_optimization(candidate)
        elif self.config.boundary_optimization == BoundaryOptimization.CONTENT_ADAPTIVE:
            new_start, new_end = await self._content_adaptive_optimization(candidate)
        elif self.config.boundary_optimization == BoundaryOptimization.ML_GUIDED:
            new_start, new_end = await self._ml_guided_optimization(candidate)
        else:
            new_start, new_end = original_start, original_end

        # Apply constraints
        new_start = max(0, new_start)
        new_end = max(new_start + self.config.min_highlight_duration, new_end)

        # Limit boundary adjustments
        max_adjustment = self.config.boundary_tolerance
        new_start = max(
            original_start - max_adjustment,
            min(original_start + max_adjustment, new_start),
        )
        new_end = max(
            original_end - max_adjustment, min(original_end + max_adjustment, new_end)
        )

        # Ensure duration constraints
        new_duration = new_end - new_start
        if new_duration < self.config.min_highlight_duration:
            extension = (self.config.min_highlight_duration - new_duration) / 2
            new_start -= extension
            new_end += extension
        elif new_duration > self.config.max_highlight_duration:
            reduction = (new_duration - self.config.max_highlight_duration) / 2
            new_start += reduction
            new_end -= reduction

        # Create optimized candidate
        optimized_candidate = HighlightCandidate(
            start_time=new_start,
            end_time=new_end,
            score=candidate.score,
            confidence=candidate.confidence,
            modality_results=candidate.modality_results,
            features=candidate.features.copy() if candidate.features else {},
            tags=candidate.tags.copy(),
            candidate_id=candidate.candidate_id,
            created_at=candidate.created_at,
        )

        # Add optimization metadata
        if not optimized_candidate.features:
            optimized_candidate.features = {}

        optimized_candidate.features.update(
            {
                "original_start_time": original_start,
                "original_end_time": original_end,
                "boundary_optimization": self.config.boundary_optimization,
                "boundary_adjusted": abs(new_start - original_start) > 0.1
                or abs(new_end - original_end) > 0.1,
            }
        )

        return optimized_candidate

    async def _scene_aware_optimization(
        self, candidate: HighlightCandidate
    ) -> Tuple[float, float]:
        """Optimize boundaries based on scene changes."""
        # Placeholder implementation - would use actual scene change detection
        # For now, apply minor adjustments based on modality results

        adjustments = []

        for result in candidate.modality_results:
            if result.modality.value == "video":
                # Look for scene change indicators in metadata
                if "scene_analysis" in result.metadata:
                    scene_data = result.metadata["scene_analysis"]
                    if scene_data.get("scene_stability", 1.0) < 0.5:
                        # Suggest expanding boundaries for unstable scenes
                        adjustments.append(("expand", 2.0))
                    else:
                        # Suggest tightening boundaries for stable scenes
                        adjustments.append(("contract", 1.0))

        # Apply adjustments
        start_adjustment = 0.0
        end_adjustment = 0.0

        for action, magnitude in adjustments:
            if action == "expand":
                start_adjustment -= magnitude
                end_adjustment += magnitude
            elif action == "contract":
                start_adjustment += magnitude * 0.5
                end_adjustment -= magnitude * 0.5

        new_start = candidate.start_time + start_adjustment
        new_end = candidate.end_time + end_adjustment

        return new_start, new_end

    async def _audio_based_optimization(
        self, candidate: HighlightCandidate
    ) -> Tuple[float, float]:
        """Optimize boundaries based on audio cues."""
        # Placeholder implementation - would use actual audio analysis

        adjustments = []

        for result in candidate.modality_results:
            if result.modality.value == "audio":
                # Look for audio patterns that suggest boundary adjustments
                if "volume_analysis" in result.metadata:
                    volume_data = result.metadata["volume_analysis"]
                    spike_count = volume_data.get("spike_count", 0)

                    if spike_count > 3:
                        # High activity - might want to expand
                        adjustments.append(("expand", 1.5))
                    elif spike_count == 0:
                        # Low activity - might want to contract
                        adjustments.append(("contract", 1.0))

        # Apply adjustments (similar to scene-aware)
        start_adjustment = 0.0
        end_adjustment = 0.0

        for action, magnitude in adjustments:
            if action == "expand":
                start_adjustment -= magnitude
                end_adjustment += magnitude
            elif action == "contract":
                start_adjustment += magnitude * 0.5
                end_adjustment -= magnitude * 0.5

        new_start = candidate.start_time + start_adjustment
        new_end = candidate.end_time + end_adjustment

        return new_start, new_end

    async def _content_adaptive_optimization(
        self, candidate: HighlightCandidate
    ) -> Tuple[float, float]:
        """Optimize boundaries adaptively based on all content."""
        # Combine multiple optimization strategies
        scene_start, scene_end = await self._scene_aware_optimization(candidate)
        audio_start, audio_end = await self._audio_based_optimization(candidate)

        # Average the suggestions
        new_start = (scene_start + audio_start) / 2
        new_end = (scene_end + audio_end) / 2

        return new_start, new_end

    async def _ml_guided_optimization(
        self, candidate: HighlightCandidate
    ) -> Tuple[float, float]:
        """Optimize boundaries using ML models (placeholder)."""
        # This would use trained ML models to predict optimal boundaries
        # For now, fall back to content adaptive
        return await self._content_adaptive_optimization(candidate)


class QualityEnhancer:
    """
    Enhances highlight quality through various algorithms.

    Implements methods to boost confidence, enhance features,
    and improve overall highlight quality scores.
    """

    def __init__(self, config: PostProcessorConfig):
        """
        Initialize quality enhancer.

        Args:
            config: Post-processor configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QualityEnhancer")

    async def enhance_quality(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """
        Enhance quality of highlight candidates.

        Args:
            candidates: List of highlight candidates

        Returns:
            List of quality-enhanced candidates
        """
        if self.config.quality_enhancement == QualityEnhancement.NONE:
            return candidates

        self.logger.debug(f"Enhancing quality for {len(candidates)} candidates")

        enhanced_candidates = []
        enhancements_applied = 0

        for candidate in candidates:
            try:
                enhanced_candidate = await self._enhance_single_candidate(
                    candidate, candidates
                )
                enhanced_candidates.append(enhanced_candidate)

                # Check if quality was enhanced
                if (
                    enhanced_candidate.score > candidate.score
                    or enhanced_candidate.confidence > candidate.confidence
                ):
                    enhancements_applied += 1

            except Exception as e:
                self.logger.warning(
                    f"Quality enhancement failed for candidate {candidate.candidate_id}: {e}"
                )
                enhanced_candidates.append(candidate)

        self.logger.debug(
            f"Applied quality enhancements to {enhancements_applied} candidates"
        )
        return enhanced_candidates

    async def _enhance_single_candidate(
        self, candidate: HighlightCandidate, all_candidates: List[HighlightCandidate]
    ) -> HighlightCandidate:
        """Enhance quality for a single candidate."""
        enhanced_score = candidate.score
        enhanced_confidence = candidate.confidence
        enhanced_features = candidate.features.copy() if candidate.features else {}

        # Apply enhancement method
        if self.config.quality_enhancement == QualityEnhancement.CONFIDENCE_BOOSTING:
            enhanced_confidence = await self._apply_confidence_boosting(candidate)
        elif self.config.quality_enhancement == QualityEnhancement.FEATURE_ENHANCEMENT:
            enhanced_features = await self._apply_feature_enhancement(
                candidate, enhanced_features
            )
        elif (
            self.config.quality_enhancement == QualityEnhancement.CONTEXTUAL_REFINEMENT
        ):
            (
                enhanced_score,
                enhanced_confidence,
            ) = await self._apply_contextual_refinement(candidate, all_candidates)
        elif self.config.quality_enhancement == QualityEnhancement.ADAPTIVE_SCORING:
            enhanced_score, enhanced_confidence = await self._apply_adaptive_scoring(
                candidate, all_candidates
            )

        # Apply limits
        enhanced_score = min(1.0, max(0.0, enhanced_score))
        enhanced_confidence = min(1.0, max(0.0, enhanced_confidence))

        # Create enhanced candidate
        enhanced_candidate = HighlightCandidate(
            start_time=candidate.start_time,
            end_time=candidate.end_time,
            score=enhanced_score,
            confidence=enhanced_confidence,
            modality_results=candidate.modality_results,
            features=enhanced_features,
            tags=candidate.tags.copy(),
            candidate_id=candidate.candidate_id,
            created_at=candidate.created_at,
        )

        # Add enhancement metadata
        enhanced_candidate.features.update(
            {
                "original_score": candidate.score,
                "original_confidence": candidate.confidence,
                "quality_enhancement": self.config.quality_enhancement,
                "enhanced": enhanced_score > candidate.score
                or enhanced_confidence > candidate.confidence,
            }
        )

        return enhanced_candidate

    async def _apply_confidence_boosting(self, candidate: HighlightCandidate) -> float:
        """Apply confidence boosting algorithm."""
        # Boost confidence based on modality agreement
        modality_scores = []
        for result in candidate.modality_results:
            modality_scores.append(result.score)

        if len(modality_scores) > 1:
            # Calculate agreement between modalities
            score_variance = np.var(modality_scores)
            agreement = 1.0 - min(1.0, score_variance)

            # Boost confidence based on agreement
            boost_factor = 1.0 + (
                agreement * (self.config.confidence_boost_factor - 1.0)
            )
            enhanced_confidence = candidate.confidence * boost_factor
        else:
            # Single modality - minor boost
            enhanced_confidence = candidate.confidence * 1.05

        return min(1.0, enhanced_confidence)

    async def _apply_feature_enhancement(
        self, candidate: HighlightCandidate, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply feature enhancement algorithms."""
        if not self.config.feature_enhancement_enabled:
            return features

        enhanced_features = features.copy()

        # Add derived features
        enhanced_features["duration_score"] = self._calculate_duration_score(candidate)
        enhanced_features["modality_diversity"] = len(candidate.modality_results)
        enhanced_features["weighted_score"] = candidate.weighted_score

        # Add temporal features if available
        if candidate.features:
            # Calculate feature statistics
            numeric_features = [
                v for v in candidate.features.values() if isinstance(v, (int, float))
            ]

            if numeric_features:
                enhanced_features["feature_mean"] = np.mean(numeric_features)
                enhanced_features["feature_std"] = np.std(numeric_features)
                enhanced_features["feature_max"] = np.max(numeric_features)

        return enhanced_features

    def _calculate_duration_score(self, candidate: HighlightCandidate) -> float:
        """Calculate score based on duration preference."""
        duration = candidate.duration
        optimal_min = (
            self.config.min_highlight_duration * 2
        )  # Prefer longer than minimum
        optimal_max = self.config.max_highlight_duration * 0.8  # But not too long

        if optimal_min <= duration <= optimal_max:
            return 1.0
        elif duration < optimal_min:
            return duration / optimal_min
        else:
            return max(0.1, optimal_max / duration)

    async def _apply_contextual_refinement(
        self, candidate: HighlightCandidate, all_candidates: List[HighlightCandidate]
    ) -> Tuple[float, float]:
        """Apply contextual refinement based on neighboring highlights."""
        enhanced_score = candidate.score
        enhanced_confidence = candidate.confidence

        # Find neighboring candidates
        candidate_time = (candidate.start_time + candidate.end_time) / 2
        neighbors = []

        for other in all_candidates:
            if other.candidate_id == candidate.candidate_id:
                continue

            other_time = (other.start_time + other.end_time) / 2
            distance = abs(other_time - candidate_time)

            if distance <= self.config.context_window_size:
                neighbors.append((other, distance))

        if neighbors:
            # Calculate neighbor influence
            neighbor_scores = []
            neighbor_weights = []

            for neighbor, distance in neighbors:
                weight = 1.0 / (
                    1.0 + distance / 10.0
                )  # Closer neighbors have more weight
                neighbor_scores.append(neighbor.score)
                neighbor_weights.append(weight)

            # Weighted average of neighbor scores
            if neighbor_weights:
                weighted_neighbor_score = np.average(
                    neighbor_scores, weights=neighbor_weights
                )

                # Adjust candidate score based on neighbor influence
                influence = self.config.neighbor_influence_factor
                enhanced_score = (
                    candidate.score * (1.0 - influence)
                    + weighted_neighbor_score * influence
                )

                # Boost confidence if neighbors agree
                neighbor_agreement = (
                    1.0 - np.var(neighbor_scores) if len(neighbor_scores) > 1 else 1.0
                )
                enhanced_confidence = candidate.confidence * (
                    1.0 + neighbor_agreement * 0.1
                )

        return enhanced_score, enhanced_confidence

    async def _apply_adaptive_scoring(
        self, candidate: HighlightCandidate, all_candidates: List[HighlightCandidate]
    ) -> Tuple[float, float]:
        """Apply adaptive scoring based on global context."""
        # Combine multiple enhancement strategies
        enhanced_confidence = await self._apply_confidence_boosting(candidate)
        (
            contextual_score,
            contextual_confidence,
        ) = await self._apply_contextual_refinement(candidate, all_candidates)

        # Adaptive weighting based on candidate characteristics
        if candidate.score > 0.7:
            # High-scoring candidates get more confidence boosting
            final_confidence = enhanced_confidence * 0.7 + contextual_confidence * 0.3
            final_score = candidate.score * 0.8 + contextual_score * 0.2
        else:
            # Lower-scoring candidates benefit more from contextual refinement
            final_confidence = enhanced_confidence * 0.3 + contextual_confidence * 0.7
            final_score = candidate.score * 0.5 + contextual_score * 0.5

        # Apply maximum boost constraint
        max_score_boost = candidate.score + self.config.max_score_boost
        final_score = min(max_score_boost, final_score)

        return final_score, final_confidence


class HighlightPostProcessor:
    """
    Main post-processing coordinator for highlight refinement.

    Orchestrates temporal smoothing, boundary optimization,
    and quality enhancement for highlight candidates.
    """

    def __init__(self, config: Optional[PostProcessorConfig] = None):
        """
        Initialize highlight post-processor.

        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessorConfig()
        self.temporal_smoother = TemporalSmoother(self.config)
        self.boundary_optimizer = BoundaryOptimizer(self.config)
        self.quality_enhancer = QualityEnhancer(self.config)
        self.logger = logging.getLogger(f"{__name__}.HighlightPostProcessor")

    async def process_highlights(
        self, candidates: List[HighlightCandidate]
    ) -> Tuple[List[HighlightCandidate], ProcessingMetrics]:
        """
        Process highlight candidates through full post-processing pipeline.

        Args:
            candidates: List of highlight candidates

        Returns:
            Tuple of (processed highlights, processing metrics)
        """
        if not self.config.enabled or not candidates:
            return candidates, ProcessingMetrics(
                candidates_processed=len(candidates),
                candidates_improved=0,
                avg_score_improvement=0.0,
                avg_confidence_improvement=0.0,
                boundary_adjustments=0,
                temporal_smoothing_applied=False,
                quality_enhancements=0,
            )

        start_time = asyncio.get_event_loop().time()
        self.logger.info(f"Starting post-processing for {len(candidates)} candidates")

        original_scores = [c.score for c in candidates]
        original_confidences = [c.confidence for c in candidates]

        processed_candidates = candidates.copy()

        try:
            # Step 1: Temporal smoothing
            if self.config.temporal_smoothing_enabled:
                processed_candidates = await self.temporal_smoother.smooth_highlights(
                    processed_candidates
                )
                self.logger.debug("Applied temporal smoothing")

            # Step 2: Boundary optimization
            if self.config.boundary_optimization != BoundaryOptimization.NONE:
                processed_candidates = (
                    await self.boundary_optimizer.optimize_boundaries(
                        processed_candidates
                    )
                )
                self.logger.debug("Applied boundary optimization")

            # Step 3: Quality enhancement
            if self.config.quality_enhancement != QualityEnhancement.NONE:
                processed_candidates = await self.quality_enhancer.enhance_quality(
                    processed_candidates
                )
                self.logger.debug("Applied quality enhancement")

            # Step 4: Final filtering
            processed_candidates = self._apply_final_filtering(processed_candidates)

            # Calculate metrics
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            metrics = self._calculate_processing_metrics(
                candidates,
                processed_candidates,
                original_scores,
                original_confidences,
                processing_time,
            )

            self.logger.info(
                f"Post-processing complete: {metrics.candidates_improved}/{metrics.candidates_processed} "
                f"improved, {processing_time:.1f}ms"
            )

            return processed_candidates, metrics

        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            # Return original candidates if processing fails
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            metrics = ProcessingMetrics(
                candidates_processed=len(candidates),
                candidates_improved=0,
                avg_score_improvement=0.0,
                avg_confidence_improvement=0.0,
                boundary_adjustments=0,
                temporal_smoothing_applied=False,
                quality_enhancements=0,
                processing_time_ms=processing_time,
            )
            return candidates, metrics

    def _apply_final_filtering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Apply final filtering and outlier removal."""
        if not self.config.outlier_detection_enabled:
            return candidates

        # Filter by minimum score threshold
        filtered = [c for c in candidates if c.score >= self.config.min_score_threshold]

        # Outlier detection based on score distribution
        if len(filtered) > 3:
            scores = [c.score for c in filtered]
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Remove outliers beyond threshold
            outlier_filtered = []
            for candidate in filtered:
                z_score = abs(candidate.score - mean_score) / max(0.1, std_score)
                if z_score <= self.config.outlier_threshold:
                    outlier_filtered.append(candidate)

            if outlier_filtered:  # Keep at least some candidates
                filtered = outlier_filtered

        return filtered

    def _calculate_processing_metrics(
        self,
        original_candidates: List[HighlightCandidate],
        processed_candidates: List[HighlightCandidate],
        original_scores: List[float],
        original_confidences: List[float],
        processing_time_ms: float,
    ) -> ProcessingMetrics:
        """Calculate processing performance metrics."""
        candidates_improved = 0
        score_improvements = []
        confidence_improvements = []
        boundary_adjustments = 0
        quality_enhancements = 0

        # Match processed candidates with originals
        for i, (orig, proc) in enumerate(
            zip(original_candidates, processed_candidates[: len(original_candidates)])
        ):
            # Score improvement
            if proc.score > orig.score:
                candidates_improved += 1
                score_improvements.append(proc.score - orig.score)
            else:
                score_improvements.append(0.0)

            # Confidence improvement
            if proc.confidence > orig.confidence:
                confidence_improvements.append(proc.confidence - orig.confidence)
            else:
                confidence_improvements.append(0.0)

            # Check for boundary adjustments
            if (
                abs(proc.start_time - orig.start_time) > 0.1
                or abs(proc.end_time - orig.end_time) > 0.1
            ):
                boundary_adjustments += 1

            # Check for quality enhancements
            if proc.features and proc.features.get("enhanced", False):
                quality_enhancements += 1

        return ProcessingMetrics(
            candidates_processed=len(original_candidates),
            candidates_improved=candidates_improved,
            avg_score_improvement=np.mean(score_improvements)
            if score_improvements
            else 0.0,
            avg_confidence_improvement=np.mean(confidence_improvements)
            if confidence_improvements
            else 0.0,
            boundary_adjustments=boundary_adjustments,
            temporal_smoothing_applied=self.config.temporal_smoothing_enabled,
            quality_enhancements=quality_enhancements,
            processing_time_ms=processing_time_ms,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get current post-processing configuration."""
        return self.config.dict()

    def update_config(self, **kwargs) -> None:
        """Update post-processing configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
