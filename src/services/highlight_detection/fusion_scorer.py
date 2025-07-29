"""
Multi-modal fusion scoring for the TL;DR Highlight API.

This module implements sophisticated algorithms for combining detection results
from multiple modalities (video, audio, chat) into unified highlight scores
with advanced temporal alignment and confidence calculation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy.stats import pearsonr

from .base_detector import (
    DetectionResult,
    HighlightCandidate,
    ModalityType,
)
from ...utils.scoring_utils import (
    calculate_confidence,
    calculate_temporal_correlation,
)

logger = logging.getLogger(__name__)


class FusionMethod(str, Enum):
    """Methods for fusing multi-modal detection results."""

    WEIGHTED_AVERAGE = "weighted_average"
    LEARNED_WEIGHTS = "learned_weights"
    TEMPORAL_CORRELATION = "temporal_correlation"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE_FUSION = "adaptive_fusion"


class TemporalAlignment(str, Enum):
    """Methods for temporal alignment of modalities."""

    STRICT = "strict"  # Exact timestamp matching
    WINDOW = "window"  # Within time window
    INTERPOLATION = "interpolation"  # Interpolate missing values
    CORRELATION = "correlation"  # Maximize temporal correlation


@dataclass
class ModalityScore:
    """
    Represents a score from a specific modality at a timestamp.

    Contains the detection result, temporal information, and
    fusion-specific metadata.
    """

    timestamp: float
    modality: ModalityType
    score: float
    confidence: float
    detection_result: DetectionResult
    weight: float = 1.0
    aligned: bool = False
    interpolated: bool = False

    @property
    def weighted_score(self) -> float:
        """Get confidence-weighted score."""
        return self.score * self.confidence * self.weight

    @property
    def quality_score(self) -> float:
        """Get overall quality score."""
        quality_factors = [
            self.score,
            self.confidence,
            1.0 if self.aligned else 0.5,
            1.0 if not self.interpolated else 0.8,
        ]
        return np.mean(quality_factors)


class FusionConfig(BaseModel):
    """
    Configuration for multi-modal fusion scoring.

    Defines how different modalities are combined and weighted
    in the fusion process.
    """

    # Fusion method configuration
    fusion_method: FusionMethod = Field(
        default=FusionMethod.ADAPTIVE_FUSION,
        description="Method for fusing multi-modal scores",
    )
    temporal_alignment: TemporalAlignment = Field(
        default=TemporalAlignment.WINDOW, description="Method for temporal alignment"
    )

    # Modality weights
    video_weight: float = Field(
        default=0.5, ge=0.0, description="Weight for video modality in fusion"
    )
    audio_weight: float = Field(
        default=0.5, ge=0.0, description="Weight for audio modality in fusion"
    )
    chat_weight: float = Field(
        default=0.0, ge=0.0, description="Weight for chat modality in fusion (bonus only)"
    )

    # Temporal alignment parameters
    alignment_window_seconds: float = Field(
        default=5.0, gt=0.0, description="Time window for temporal alignment (seconds)"
    )
    max_interpolation_gap: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum gap for score interpolation (seconds)",
    )

    # Confidence and quality parameters
    min_modality_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required per modality",
    )
    confidence_penalty_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Penalty factor for low confidence scores",
    )
    missing_modality_penalty: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Penalty for missing modalities"
    )

    # Adaptive fusion parameters
    adaptive_learning_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate for adaptive weight adjustment",
    )
    correlation_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum correlation for modality agreement",
    )

    # Quality filtering
    min_fusion_score: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Minimum fusion score threshold"
    )
    min_fusion_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum fusion confidence threshold"
    )
    require_multiple_modalities: bool = Field(
        default=False, description="Require agreement from multiple modalities (chat optional)"
    )

    @field_validator("video_weight", "audio_weight", "chat_weight")
    def validate_weights(cls, v):
        """Ensure weights are non-negative."""
        if v < 0:
            raise ValueError("Weights must be non-negative")
        return v

    @property
    def modality_weights(self) -> Dict[ModalityType, float]:
        """Get modality weights as dictionary."""
        return {
            ModalityType.VIDEO: self.video_weight,
            ModalityType.AUDIO: self.audio_weight,
            ModalityType.CHAT: self.chat_weight,
        }

    @property
    def normalized_weights(self) -> Dict[ModalityType, float]:
        """Get normalized modality weights."""
        weights = self.modality_weights.copy()
        
        # Separate chat weight for bonus scoring
        chat_weight = weights.pop(ModalityType.CHAT, 0.0)
        
        # Normalize video and audio weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if all are zero
            normalized = {k: 1.0 / len(weights) for k in weights.keys()}
        
        # Add chat back as bonus weight (not normalized)
        normalized[ModalityType.CHAT] = chat_weight
        
        return normalized


class TemporalAligner:
    """
    Handles temporal alignment of multi-modal detection results.

    Provides various strategies for aligning results from different
    modalities that may not have exact timestamp matches.
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize the temporal aligner.

        Args:
            config: Fusion configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TemporalAligner")

    async def align_results(
        self, results_by_modality: Dict[ModalityType, List[DetectionResult]]
    ) -> List[Tuple[float, Dict[ModalityType, ModalityScore]]]:
        """
        Align detection results across modalities.

        Args:
            results_by_modality: Detection results grouped by modality

        Returns:
            List of (timestamp, aligned_scores) tuples
        """
        if not results_by_modality:
            return []

        # Get all unique timestamps
        all_timestamps = set()
        for results in results_by_modality.values():
            for result in results:
                # Use segment midpoint as representative timestamp
                timestamp = (
                    result.metadata.get("start_time", 0)
                    + result.metadata.get("end_time", 0)
                ) / 2
                all_timestamps.add(timestamp)

        if not all_timestamps:
            return []

        sorted_timestamps = sorted(all_timestamps)

        # Align based on method
        if self.config.temporal_alignment == TemporalAlignment.STRICT:
            return await self._strict_alignment(results_by_modality, sorted_timestamps)
        elif self.config.temporal_alignment == TemporalAlignment.WINDOW:
            return await self._window_alignment(results_by_modality, sorted_timestamps)
        elif self.config.temporal_alignment == TemporalAlignment.INTERPOLATION:
            return await self._interpolation_alignment(
                results_by_modality, sorted_timestamps
            )
        elif self.config.temporal_alignment == TemporalAlignment.CORRELATION:
            return await self._correlation_alignment(
                results_by_modality, sorted_timestamps
            )
        else:
            return await self._window_alignment(results_by_modality, sorted_timestamps)

    async def _strict_alignment(
        self,
        results_by_modality: Dict[ModalityType, List[DetectionResult]],
        timestamps: List[float],
    ) -> List[Tuple[float, Dict[ModalityType, ModalityScore]]]:
        """Strict timestamp matching alignment."""
        aligned_results = []

        for timestamp in timestamps:
            aligned_scores = {}

            for modality, results in results_by_modality.items():
                for result in results:
                    result_timestamp = (
                        result.metadata.get("start_time", 0)
                        + result.metadata.get("end_time", 0)
                    ) / 2

                    if abs(result_timestamp - timestamp) < 0.1:  # 100ms tolerance
                        score = ModalityScore(
                            timestamp=timestamp,
                            modality=modality,
                            score=result.score,
                            confidence=result.confidence,
                            detection_result=result,
                            weight=self.config.normalized_weights[modality],
                            aligned=True,
                        )
                        aligned_scores[modality] = score
                        break

            if aligned_scores:
                aligned_results.append((timestamp, aligned_scores))

        return aligned_results

    async def _window_alignment(
        self,
        results_by_modality: Dict[ModalityType, List[DetectionResult]],
        timestamps: List[float],
    ) -> List[Tuple[float, Dict[ModalityType, ModalityScore]]]:
        """Window-based alignment with time tolerance."""
        aligned_results = []
        window_size = self.config.alignment_window_seconds

        for timestamp in timestamps:
            aligned_scores = {}

            for modality, results in results_by_modality.items():
                best_result = None
                best_distance = float("inf")

                for result in results:
                    result_timestamp = (
                        result.metadata.get("start_time", 0)
                        + result.metadata.get("end_time", 0)
                    ) / 2
                    distance = abs(result_timestamp - timestamp)

                    if distance <= window_size and distance < best_distance:
                        best_result = result
                        best_distance = distance

                if best_result is not None:
                    score = ModalityScore(
                        timestamp=timestamp,
                        modality=modality,
                        score=best_result.score,
                        confidence=best_result.confidence,
                        detection_result=best_result,
                        weight=self.config.normalized_weights[modality],
                        aligned=best_distance < 1.0,  # Within 1 second
                    )
                    aligned_scores[modality] = score

            if aligned_scores:
                aligned_results.append((timestamp, aligned_scores))

        return aligned_results

    async def _interpolation_alignment(
        self,
        results_by_modality: Dict[ModalityType, List[DetectionResult]],
        timestamps: List[float],
    ) -> List[Tuple[float, Dict[ModalityType, ModalityScore]]]:
        """Interpolation-based alignment for missing values."""
        # First do window alignment
        window_aligned = await self._window_alignment(results_by_modality, timestamps)

        # Then interpolate missing values
        interpolated_results = []

        for timestamp, aligned_scores in window_aligned:
            # Find missing modalities
            missing_modalities = set(self.config.modality_weights.keys()) - set(
                aligned_scores.keys()
            )

            for missing_modality in missing_modalities:
                # Try to interpolate from nearby values
                interpolated_score = await self._interpolate_score(
                    missing_modality,
                    timestamp,
                    results_by_modality.get(missing_modality, []),
                )

                if interpolated_score is not None:
                    aligned_scores[missing_modality] = interpolated_score

            interpolated_results.append((timestamp, aligned_scores))

        return interpolated_results

    async def _interpolate_score(
        self,
        modality: ModalityType,
        target_timestamp: float,
        results: List[DetectionResult],
    ) -> Optional[ModalityScore]:
        """Interpolate score for missing modality."""
        if not results:
            return None

        # Find nearest results before and after target timestamp
        before_result = None
        after_result = None

        for result in results:
            result_timestamp = (
                result.metadata.get("start_time", 0)
                + result.metadata.get("end_time", 0)
            ) / 2

            if result_timestamp <= target_timestamp:
                if (
                    before_result is None
                    or result_timestamp
                    > (
                        before_result.metadata.get("start_time", 0)
                        + before_result.metadata.get("end_time", 0)
                    )
                    / 2
                ):
                    before_result = result
            else:
                if (
                    after_result is None
                    or result_timestamp
                    < (
                        after_result.metadata.get("start_time", 0)
                        + after_result.metadata.get("end_time", 0)
                    )
                    / 2
                ):
                    after_result = result

        # Check if interpolation is possible
        if before_result is None and after_result is None:
            return None

        # Use single result if only one available
        if before_result is None:
            source_result = after_result
            interpolated_score = after_result.score * 0.5  # Reduce confidence
            interpolated_confidence = after_result.confidence * 0.5
        elif after_result is None:
            source_result = before_result
            interpolated_score = before_result.score * 0.5
            interpolated_confidence = before_result.confidence * 0.5
        else:
            # Linear interpolation between two results
            before_timestamp = (
                before_result.metadata.get("start_time", 0)
                + before_result.metadata.get("end_time", 0)
            ) / 2
            after_timestamp = (
                after_result.metadata.get("start_time", 0)
                + after_result.metadata.get("end_time", 0)
            ) / 2

            if (
                abs(after_timestamp - before_timestamp)
                > self.config.max_interpolation_gap
            ):
                return None  # Gap too large for interpolation

            # Linear interpolation weights
            total_distance = after_timestamp - before_timestamp
            if total_distance == 0:
                weight_after = 0.5
            else:
                weight_after = (target_timestamp - before_timestamp) / total_distance
            weight_before = 1.0 - weight_after

            interpolated_score = (
                weight_before * before_result.score + weight_after * after_result.score
            )
            interpolated_confidence = (
                weight_before * before_result.confidence
                + weight_after * after_result.confidence
            ) * 0.8  # Reduce confidence for interpolated values

            source_result = before_result  # Use before result as template

        return ModalityScore(
            timestamp=target_timestamp,
            modality=modality,
            score=interpolated_score,
            confidence=interpolated_confidence,
            detection_result=source_result,
            weight=self.config.normalized_weights[modality],
            aligned=False,
            interpolated=True,
        )

    async def _correlation_alignment(
        self,
        results_by_modality: Dict[ModalityType, List[DetectionResult]],
        timestamps: List[float],
    ) -> List[Tuple[float, Dict[ModalityType, ModalityScore]]]:
        """Correlation-based alignment to maximize temporal agreement."""
        # Start with window alignment
        window_aligned = await self._window_alignment(results_by_modality, timestamps)

        # Calculate correlation between modalities
        _correlations = await self._calculate_modality_correlations(window_aligned)

        # Adjust alignment based on correlations
        # This is a simplified implementation - could be more sophisticated
        return window_aligned

    async def _calculate_modality_correlations(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> Dict[Tuple[ModalityType, ModalityType], float]:
        """Calculate correlations between modality pairs."""
        correlations = {}
        modalities = list(self.config.modality_weights.keys())

        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i + 1 :], i + 1):
                scores1 = []
                scores2 = []

                for timestamp, aligned_scores in aligned_results:
                    if mod1 in aligned_scores and mod2 in aligned_scores:
                        scores1.append(aligned_scores[mod1].score)
                        scores2.append(aligned_scores[mod2].score)

                if len(scores1) >= 3:  # Need minimum samples for correlation
                    try:
                        corr, _ = pearsonr(scores1, scores2)
                        correlations[(mod1, mod2)] = corr if not np.isnan(corr) else 0.0
                    except Exception:
                        correlations[(mod1, mod2)] = 0.0
                else:
                    correlations[(mod1, mod2)] = 0.0

        return correlations


class FusionScorer:
    """
    Multi-modal fusion scorer for combining detection results.

    Implements sophisticated algorithms for fusing scores from
    different modalities into unified highlight candidates with
    confidence estimation and quality assessment.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize the fusion scorer.

        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig()
        self.temporal_aligner = TemporalAligner(self.config)
        self.logger = logging.getLogger(f"{__name__}.FusionScorer")

        # Adaptive learning state
        self.adaptive_weights = self.config.normalized_weights.copy()
        self.correlation_history = {}

    async def fuse_results(
        self, results_by_modality: Dict[ModalityType, List[DetectionResult]]
    ) -> List[HighlightCandidate]:
        """
        Fuse detection results from multiple modalities.

        Args:
            results_by_modality: Detection results grouped by modality

        Returns:
            List of highlight candidates with fused scores
        """
        if not results_by_modality:
            return []

        self.logger.info(f"Fusing results from {len(results_by_modality)} modalities")

        # Temporal alignment
        aligned_results = await self.temporal_aligner.align_results(results_by_modality)

        if not aligned_results:
            self.logger.warning("No aligned results found")
            return []

        # Apply fusion method
        if self.config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            candidates = await self._weighted_average_fusion(aligned_results)
        elif self.config.fusion_method == FusionMethod.LEARNED_WEIGHTS:
            candidates = await self._learned_weights_fusion(aligned_results)
        elif self.config.fusion_method == FusionMethod.TEMPORAL_CORRELATION:
            candidates = await self._temporal_correlation_fusion(aligned_results)
        elif self.config.fusion_method == FusionMethod.CONFIDENCE_WEIGHTED:
            candidates = await self._confidence_weighted_fusion(aligned_results)
        elif self.config.fusion_method == FusionMethod.ADAPTIVE_FUSION:
            candidates = await self._adaptive_fusion(aligned_results)
        else:
            candidates = await self._weighted_average_fusion(aligned_results)

        # Filter by quality thresholds
        filtered_candidates = self._filter_candidates(candidates)

        self.logger.info(f"Generated {len(filtered_candidates)} highlight candidates")

        return filtered_candidates

    async def _weighted_average_fusion(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> List[HighlightCandidate]:
        """Simple weighted average fusion with optional chat bonus."""
        candidates = []

        for timestamp, aligned_scores in aligned_results:
            # Separate core modalities (video/audio) from chat
            core_scores = {}
            chat_score = None
            
            for modality, modality_score in aligned_scores.items():
                if modality == ModalityType.CHAT:
                    chat_score = modality_score
                else:
                    core_scores[modality] = modality_score
            
            # Calculate core score from video/audio only
            total_weighted_score = 0.0
            total_weight = 0.0
            confidences = []
            modality_results = []

            for modality, modality_score in core_scores.items():
                weight = modality_score.weight
                if modality_score.confidence >= self.config.min_modality_confidence:
                    total_weighted_score += modality_score.score * weight
                    total_weight += weight
                    confidences.append(modality_score.confidence)
                    modality_results.append(modality_score.detection_result)

            if total_weight > 0:
                fused_score = total_weighted_score / total_weight
                
                # Apply chat bonus if available
                if (chat_score is not None and 
                    chat_score.confidence >= self.config.min_modality_confidence):
                    chat_bonus = chat_score.score * self.config.chat_weight
                    fused_score = min(1.0, fused_score + chat_bonus)
                    confidences.append(chat_score.confidence)
                    modality_results.append(chat_score.detection_result)
                
                fused_confidence = calculate_confidence(
                    confidences, method="consistency"
                )

                # Apply penalties (modified to not penalize missing chat)
                fused_score = self._apply_penalties_modified(fused_score, aligned_scores)

                candidate = HighlightCandidate(
                    start_time=timestamp - 15.0,  # Default 30-second highlight
                    end_time=timestamp + 15.0,
                    score=fused_score,
                    confidence=fused_confidence,
                    modality_results=modality_results,
                    features={
                        "fusion_method": "weighted_average",
                        "modality_count": len(aligned_scores),
                        "has_chat": chat_score is not None,
                        "total_weight": total_weight,
                        "timestamp": timestamp,
                    },
                )
                candidates.append(candidate)

        return candidates

    async def _confidence_weighted_fusion(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> List[HighlightCandidate]:
        """Confidence-weighted fusion with optional chat bonus."""
        candidates = []

        for timestamp, aligned_scores in aligned_results:
            # Separate core modalities from chat
            core_scores = {}
            chat_score = None
            
            for modality, modality_score in aligned_scores.items():
                if modality == ModalityType.CHAT:
                    chat_score = modality_score
                else:
                    core_scores[modality] = modality_score
            
            # Calculate confidence-weighted score for core modalities
            total_score = 0.0
            total_confidence_weight = 0.0
            confidences = []
            modality_results = []

            for modality, modality_score in core_scores.items():
                if modality_score.confidence >= self.config.min_modality_confidence:
                    # Weight by both modality weight and confidence
                    confidence_weight = (
                        modality_score.weight * modality_score.confidence
                    )
                    total_score += modality_score.score * confidence_weight
                    total_confidence_weight += confidence_weight
                    confidences.append(modality_score.confidence)
                    modality_results.append(modality_score.detection_result)

            if total_confidence_weight > 0:
                fused_score = total_score / total_confidence_weight
                
                # Apply chat bonus if available
                if (chat_score is not None and 
                    chat_score.confidence >= self.config.min_modality_confidence):
                    chat_bonus = chat_score.score * chat_score.confidence * self.config.chat_weight
                    fused_score = min(1.0, fused_score + chat_bonus)
                    confidences.append(chat_score.confidence)
                    modality_results.append(chat_score.detection_result)
                
                fused_confidence = calculate_confidence(
                    confidences, method="consistency"
                )

                # Apply penalties
                fused_score = self._apply_penalties_modified(fused_score, aligned_scores)

                candidate = HighlightCandidate(
                    start_time=timestamp - 15.0,
                    end_time=timestamp + 15.0,
                    score=fused_score,
                    confidence=fused_confidence,
                    modality_results=modality_results,
                    features={
                        "fusion_method": "confidence_weighted",
                        "modality_count": len(aligned_scores),
                        "has_chat": chat_score is not None,
                        "confidence_weight": total_confidence_weight,
                        "timestamp": timestamp,
                    },
                )
                candidates.append(candidate)

        return candidates

    async def _adaptive_fusion(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> List[HighlightCandidate]:
        """Adaptive fusion with optional chat bonus."""
        # Update adaptive weights based on correlation
        await self._update_adaptive_weights(aligned_results)

        candidates = []

        for timestamp, aligned_scores in aligned_results:
            # Separate core modalities from chat
            core_scores = {}
            chat_score = None
            
            for modality, modality_score in aligned_scores.items():
                if modality == ModalityType.CHAT:
                    chat_score = modality_score
                else:
                    core_scores[modality] = modality_score
            
            # Use adaptive weights for core modalities
            total_weighted_score = 0.0
            total_weight = 0.0
            confidences = []
            modality_results = []

            for modality, modality_score in core_scores.items():
                if modality_score.confidence >= self.config.min_modality_confidence:
                    adaptive_weight = self.adaptive_weights.get(modality, 0.0)
                    total_weighted_score += modality_score.score * adaptive_weight
                    total_weight += adaptive_weight
                    confidences.append(modality_score.confidence)
                    modality_results.append(modality_score.detection_result)

            if total_weight > 0:
                fused_score = total_weighted_score / total_weight
                
                # Apply chat bonus if available
                if (chat_score is not None and 
                    chat_score.confidence >= self.config.min_modality_confidence):
                    # Use adaptive weight for chat if available, otherwise use config
                    chat_weight = self.adaptive_weights.get(ModalityType.CHAT, self.config.chat_weight)
                    chat_bonus = chat_score.score * chat_weight
                    fused_score = min(1.0, fused_score + chat_bonus)
                    confidences.append(chat_score.confidence)
                    modality_results.append(chat_score.detection_result)
                
                fused_confidence = calculate_confidence(
                    confidences, method="consistency"
                )

                # Apply penalties
                fused_score = self._apply_penalties_modified(fused_score, aligned_scores)

                candidate = HighlightCandidate(
                    start_time=timestamp - 15.0,
                    end_time=timestamp + 15.0,
                    score=fused_score,
                    confidence=fused_confidence,
                    modality_results=modality_results,
                    features={
                        "fusion_method": "adaptive",
                        "modality_count": len(aligned_scores),
                        "has_chat": chat_score is not None,
                        "adaptive_weights": self.adaptive_weights.copy(),
                        "timestamp": timestamp,
                    },
                )
                candidates.append(candidate)

        return candidates

    async def _temporal_correlation_fusion(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> List[HighlightCandidate]:
        """Fusion based on temporal correlation between modalities."""
        # Calculate temporal correlations
        correlations = await self._calculate_temporal_correlations(aligned_results)

        candidates = []

        for timestamp, aligned_scores in aligned_results:
            # Weight by correlation agreement
            correlation_weights = {}
            for modality in aligned_scores.keys():
                # Average correlation with other modalities
                corr_sum = 0.0
                corr_count = 0

                for other_modality in aligned_scores.keys():
                    if modality != other_modality:
                        corr_key = tuple(sorted([modality, other_modality]))
                        corr = correlations.get(corr_key, 0.0)
                        corr_sum += max(0.0, corr)  # Only positive correlations
                        corr_count += 1

                correlation_weights[modality] = (
                    corr_sum / max(1, corr_count) if corr_count > 0 else 0.0
                )

            # Calculate correlation-weighted score
            total_score = 0.0
            total_weight = 0.0
            confidences = []
            modality_results = []

            for modality, modality_score in aligned_scores.items():
                if modality_score.confidence >= self.config.min_modality_confidence:
                    corr_weight = correlation_weights.get(modality, 0.0)
                    base_weight = self.config.normalized_weights.get(modality, 0.0)
                    combined_weight = base_weight * (1.0 + corr_weight)

                    total_score += modality_score.score * combined_weight
                    total_weight += combined_weight
                    confidences.append(modality_score.confidence)
                    modality_results.append(modality_score.detection_result)

            if total_weight > 0:
                fused_score = total_score / total_weight
                fused_confidence = calculate_confidence(
                    confidences, method="consistency"
                )

                # Apply penalties
                fused_score = self._apply_penalties(fused_score, aligned_scores)

                candidate = HighlightCandidate(
                    start_time=timestamp - 15.0,
                    end_time=timestamp + 15.0,
                    score=fused_score,
                    confidence=fused_confidence,
                    modality_results=modality_results,
                    features={
                        "fusion_method": "temporal_correlation",
                        "modality_count": len(aligned_scores),
                        "correlations": correlations,
                        "correlation_weights": correlation_weights,
                        "timestamp": timestamp,
                    },
                )
                candidates.append(candidate)

        return candidates

    async def _learned_weights_fusion(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> List[HighlightCandidate]:
        """Fusion with learned optimal weights (placeholder implementation)."""
        # This would require training data and optimization
        # For now, fall back to adaptive fusion
        return await self._adaptive_fusion(aligned_results)

    def _apply_penalties(
        self, score: float, aligned_scores: Dict[ModalityType, ModalityScore]
    ) -> float:
        """Apply penalties for missing modalities and low confidence."""
        penalized_score = score

        # Missing modality penalty
        expected_modalities = set(self.config.modality_weights.keys())
        present_modalities = set(aligned_scores.keys())
        missing_count = len(expected_modalities - present_modalities)

        if missing_count > 0:
            penalty_factor = 1.0 - (
                missing_count * self.config.missing_modality_penalty
            )
            penalized_score *= max(0.0, penalty_factor)

        # Low confidence penalty
        confidences = [ms.confidence for ms in aligned_scores.values()]
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.5:
                confidence_penalty = 1.0 - (
                    (0.5 - avg_confidence) * self.config.confidence_penalty_factor
                )
                penalized_score *= max(0.0, confidence_penalty)

        return min(1.0, penalized_score)
    
    def _apply_penalties_modified(
        self, score: float, aligned_scores: Dict[ModalityType, ModalityScore]
    ) -> float:
        """Apply penalties for missing core modalities and low confidence (chat optional)."""
        penalized_score = score

        # Missing modality penalty (only for video/audio)
        core_modalities = {ModalityType.VIDEO, ModalityType.AUDIO}
        present_core_modalities = core_modalities.intersection(aligned_scores.keys())
        missing_core_count = len(core_modalities - present_core_modalities)

        if missing_core_count > 0:
            penalty_factor = 1.0 - (
                missing_core_count * self.config.missing_modality_penalty
            )
            penalized_score *= max(0.0, penalty_factor)

        # Low confidence penalty (only for present modalities)
        confidences = [ms.confidence for ms in aligned_scores.values()]
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.5:
                confidence_penalty = 1.0 - (
                    (0.5 - avg_confidence) * self.config.confidence_penalty_factor
                )
                penalized_score *= max(0.0, confidence_penalty)

        return min(1.0, penalized_score)

    async def _update_adaptive_weights(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> None:
        """Update adaptive weights based on modality agreement."""
        # Calculate performance metrics for each modality
        modality_scores = defaultdict(list)

        for timestamp, aligned_scores in aligned_results:
            for modality, modality_score in aligned_scores.items():
                modality_scores[modality].append(modality_score.score)

        # Calculate correlation-based performance
        correlations = await self._calculate_temporal_correlations(aligned_results)

        # Update weights based on correlation performance
        for modality in self.adaptive_weights.keys():
            # Calculate average correlation with other modalities
            corr_sum = 0.0
            corr_count = 0

            for other_modality in self.adaptive_weights.keys():
                if modality != other_modality:
                    corr_key = tuple(sorted([modality, other_modality]))
                    corr = correlations.get(corr_key, 0.0)
                    if corr > self.config.correlation_threshold:
                        corr_sum += corr
                        corr_count += 1

            # Update adaptive weight
            if corr_count > 0:
                avg_correlation = corr_sum / corr_count
                current_weight = self.adaptive_weights[modality]

                # Adjust weight based on correlation performance
                learning_rate = self.config.adaptive_learning_rate
                new_weight = current_weight + learning_rate * (avg_correlation - 0.5)
                self.adaptive_weights[modality] = max(0.0, min(1.0, new_weight))

        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for modality in self.adaptive_weights.keys():
                self.adaptive_weights[modality] /= total_weight

    async def _calculate_temporal_correlations(
        self, aligned_results: List[Tuple[float, Dict[ModalityType, ModalityScore]]]
    ) -> Dict[Tuple[ModalityType, ModalityType], float]:
        """Calculate temporal correlations between modality pairs."""
        correlations = {}
        modalities = list(self.config.modality_weights.keys())

        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i + 1 :], i + 1):
                timestamps = []
                scores1 = []
                scores2 = []

                for timestamp, aligned_scores in aligned_results:
                    if mod1 in aligned_scores and mod2 in aligned_scores:
                        timestamps.append(timestamp)
                        scores1.append(aligned_scores[mod1].score)
                        scores2.append(aligned_scores[mod2].score)

                if len(scores1) >= 3:
                    # Calculate both Pearson correlation and temporal correlation
                    try:
                        pearson_corr, _ = pearsonr(scores1, scores2)
                        temporal_corr = calculate_temporal_correlation(
                            timestamps, scores1
                        )

                        # Combine correlations
                        combined_corr = (pearson_corr + temporal_corr) / 2
                        correlations[(mod1, mod2)] = (
                            combined_corr if not np.isnan(combined_corr) else 0.0
                        )
                    except Exception:
                        correlations[(mod1, mod2)] = 0.0
                else:
                    correlations[(mod1, mod2)] = 0.0

        return correlations

    def _filter_candidates(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Filter candidates based on quality thresholds."""
        filtered = []

        for candidate in candidates:
            # Check minimum score threshold
            if candidate.score < self.config.min_fusion_score:
                continue

            # Check minimum confidence threshold
            if candidate.confidence < self.config.min_fusion_confidence:
                continue

            # Check multiple modality requirement (but only for core modalities)
            if self.config.require_multiple_modalities:
                # Count only video and audio modalities
                core_modalities = sum(
                    1 for result in candidate.modality_results
                    if result.modality in {ModalityType.VIDEO, ModalityType.AUDIO}
                )
                if core_modalities < 2:
                    continue

            filtered.append(candidate)

        return filtered

    def get_fusion_metrics(self) -> Dict[str, Any]:
        """Get fusion performance metrics."""
        return {
            "fusion_method": self.config.fusion_method,
            "adaptive_weights": self.adaptive_weights.copy(),
            "correlation_history": self.correlation_history.copy(),
            "config": self.config.dict(),
        }
