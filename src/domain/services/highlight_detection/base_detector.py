"""
Base classes and interfaces for the AI highlight detection system.

This module defines the core abstractions used by all detection algorithms,
including the strategy pattern implementation and data structures.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of content modalities for highlight detection."""

    VIDEO = "video"
    AUDIO = "audio"
    CHAT = "chat"
    METADATA = "metadata"


class ConfidenceLevel(str, Enum):
    """Confidence levels for highlight candidates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ContentSegment:
    """
    Represents a segment of content for analysis.

    Attributes:
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        data: Raw content data (video frames, audio samples, chat messages)
        metadata: Additional segment metadata
        segment_id: Unique identifier for the segment
        source_type: Type of source (stream, batch)
        timestamp_offset: Offset from original content start
    """

    start_time: float
    end_time: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    segment_id: str = field(default_factory=lambda: str(uuid4()))
    source_type: str = "unknown"
    timestamp_offset: float = 0.0

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        """Get segment midpoint timestamp."""
        return (self.start_time + self.end_time) / 2

    def overlaps_with(self, other: "ContentSegment", threshold: float = 0.0) -> bool:
        """Check if this segment overlaps with another."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)

        return overlap_duration > threshold

    def intersection(self, other: "ContentSegment") -> Optional["ContentSegment"]:
        """Get intersection segment with another segment."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)

        if overlap_start >= overlap_end:
            return None

        return ContentSegment(
            start_time=overlap_start,
            end_time=overlap_end,
            data=None,  # No combined data
            metadata={"intersection_of": [self.segment_id, other.segment_id]},
        )


class DetectionResult(BaseModel):
    """
    Result of a detection algorithm run.

    Contains scores, confidence metrics, and metadata about
    detected potential highlights.
    """

    segment_id: str = Field(..., description="ID of the analyzed segment")
    modality: ModalityType = Field(..., description="Content modality analyzed")
    score: float = Field(..., ge=0.0, le=1.0, description="Detection score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in score")
    features: Dict[str, float] = Field(
        default_factory=dict, description="Extracted features"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    algorithm_version: str = Field(default="1.0", description="Algorithm version used")
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )

    @field_validator("score", "confidence")
    def validate_range(cls, v):
        """Ensure scores are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score and confidence must be between 0.0 and 1.0")
        return v

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    @property
    def weighted_score(self) -> float:
        """Get confidence-weighted score."""
        return self.score * self.confidence

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Check if result meets significance threshold."""
        return self.weighted_score >= threshold


# HighlightCandidate is now consolidated in src.domain.entities.highlight
# This module will be simplified as part of the streamlining process


class DetectionConfig(BaseModel):
    """
    Base configuration for detection algorithms.

    Provides common settings that can be extended by
    specific detector implementations.
    """

    enabled: bool = Field(default=True, description="Enable this detector")
    weight: float = Field(default=1.0, ge=0.0, description="Weight in fusion scoring")
    min_confidence: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    min_score: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum score threshold"
    )
    sensitivity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Detection sensitivity"
    )
    window_size_seconds: float = Field(
        default=30.0, gt=0.0, description="Analysis window size"
    )
    overlap_ratio: float = Field(
        default=0.5, ge=0.0, lt=1.0, description="Window overlap ratio"
    )
    max_highlights_per_window: int = Field(
        default=5, ge=1, description="Max highlights per window"
    )
    algorithm_params: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific parameters"
    )

    @field_validator("min_confidence", "min_score", "sensitivity")
    def validate_thresholds(cls, v):
        """Ensure thresholds are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v


class DetectionStrategy(Protocol):
    """Protocol for detection strategy implementations."""

    async def detect(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """Detect highlights in a content segment."""
        ...

    def get_algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        ...


class BaseDetector(ABC):
    """
    Abstract base class for all highlight detection algorithms.

    Provides the common interface and functionality that all
    detectors must implement, following the strategy pattern.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the detector with configuration.

        Args:
            config: Detection configuration settings
        """
        self.config = config or DetectionConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._strategy: Optional[DetectionStrategy] = None
        self._metrics = {
            "segments_processed": 0,
            "highlights_detected": 0,
            "processing_time_total": 0.0,
            "errors": 0,
        }

    @property
    @abstractmethod
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        pass

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        pass

    @property
    @abstractmethod
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        pass

    def set_strategy(self, strategy: DetectionStrategy) -> None:
        """Set the detection strategy to use."""
        self._strategy = strategy

    async def detect_highlights(
        self, segments: List[ContentSegment], config: Optional[DetectionConfig] = None
    ) -> List[DetectionResult]:
        """
        Detect highlights in multiple content segments.

        Args:
            segments: List of content segments to analyze
            config: Optional configuration override

        Returns:
            List of detection results
        """
        detection_config = config or self.config
        results: List[DetectionResult] = []

        if not detection_config.enabled:
            self.logger.info(f"{self.algorithm_name} detector is disabled")
            return results

        self.logger.info(
            f"Processing {len(segments)} segments with {self.algorithm_name}"
        )

        # Process segments concurrently with rate limiting
        semaphore = asyncio.Semaphore(10)  # Limit concurrent processing

        async def process_segment(segment: ContentSegment) -> List[DetectionResult]:
            async with semaphore:
                return await self._detect_segment(segment, detection_config)

        # Execute all segment processing
        segment_results = await asyncio.gather(
            *[process_segment(segment) for segment in segments], return_exceptions=True
        )

        # Collect results and handle exceptions
        for i, segment_result in enumerate(segment_results):
            if isinstance(segment_result, Exception):
                self.logger.error(
                    f"Error processing segment {segments[i].segment_id}: {segment_result}"
                )
                self._metrics["errors"] += 1
                continue

            # Type narrowing: segment_result is List[DetectionResult] here
            results.extend(segment_result)
            self._metrics["segments_processed"] += 1
            self._metrics["highlights_detected"] += len(segment_result)

        # Filter results by thresholds
        filtered_results = self._filter_results(results, detection_config)

        self.logger.info(
            f"{self.algorithm_name} processed {len(segments)} segments, "
            f"found {len(filtered_results)} highlights"
        )

        return filtered_results

    async def _detect_segment(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect highlights in a single content segment.

        Args:
            segment: Content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results for the segment
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Validate segment
            if not self._validate_segment(segment):
                self.logger.warning(
                    f"Invalid segment {segment.segment_id} for {self.modality}"
                )
                return []

            # Use strategy if available, otherwise use abstract method
            if self._strategy:
                results = await self._strategy.detect(segment, config)
            else:
                results = await self._detect_features(segment, config)

            # Update processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._metrics["processing_time_total"] += processing_time

            # Set processing time on results
            for result in results:
                result.processing_time_ms = (
                    processing_time / len(results) if results else 0
                )
                result.algorithm_version = self.algorithm_version

            return results

        except Exception as e:
            self.logger.error(f"Error detecting in segment {segment.segment_id}: {e}")
            return []

    @abstractmethod
    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Abstract method for feature detection implementation.

        Must be implemented by concrete detector classes.

        Args:
            segment: Content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results
        """
        pass

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """
        Validate that a segment is suitable for this detector.

        Args:
            segment: Content segment to validate

        Returns:
            True if segment is valid for this modality
        """
        if segment.data is None:
            return False

        if segment.duration <= 0:
            return False

        # Additional validation can be implemented by subclasses
        return True

    def _filter_results(
        self, results: List[DetectionResult], config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Filter detection results based on configuration thresholds.

        Args:
            results: Raw detection results
            config: Detection configuration with thresholds

        Returns:
            Filtered results meeting threshold criteria
        """
        filtered = []

        for result in results:
            # Check minimum thresholds
            if result.score < config.min_score:
                continue

            if result.confidence < config.min_confidence:
                continue

            # Check significance
            if not result.is_significant(config.min_score * config.min_confidence):
                continue

            filtered.append(result)

        # Sort by weighted score (descending)
        filtered.sort(key=lambda r: r.weighted_score, reverse=True)

        return filtered

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this detector."""
        metrics = self._metrics.copy()

        # Calculate derived metrics
        if metrics["segments_processed"] > 0:
            metrics["avg_processing_time_ms"] = (
                metrics["processing_time_total"] / metrics["segments_processed"]
            )
            metrics["highlights_per_segment"] = (
                metrics["highlights_detected"] / metrics["segments_processed"]
            )
            metrics["error_rate"] = metrics["errors"] / (
                metrics["segments_processed"] + metrics["errors"]
            )
        else:
            metrics["avg_processing_time_ms"] = 0.0
            metrics["highlights_per_segment"] = 0.0
            metrics["error_rate"] = 0.0

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            "segments_processed": 0,
            "highlights_detected": 0,
            "processing_time_total": 0.0,
            "errors": 0,
        }

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"{self.__class__.__name__}("
            f"modality={self.modality}, "
            f"algorithm={self.algorithm_name}, "
            f"version={self.algorithm_version})"
        )


# Type variable for detector instances
T_Detector = TypeVar("T_Detector", bound=BaseDetector)


class DetectorRegistry:
    """
    Registry for managing detection algorithm implementations.

    Provides registration, discovery, and instantiation of
    detector classes with configuration.
    """

    def __init__(self) -> None:
        self._detectors: Dict[str, type[BaseDetector]] = {}
        self._instances: Dict[str, BaseDetector] = {}

    def register(self, name: str, detector_class: type[BaseDetector]) -> None:
        """Register a detector class."""
        if not issubclass(detector_class, BaseDetector):
            raise ValueError("Detector class must inherit from BaseDetector")

        self._detectors[name] = detector_class
        logger.info(f"Registered detector: {name}")

    def get_detector(
        self, name: str, config: Optional[DetectionConfig] = None
    ) -> BaseDetector:
        """Get detector instance by name."""
        if name not in self._detectors:
            raise ValueError(f"Unknown detector: {name}")

        # Return cached instance if config matches
        cache_key = f"{name}_{hash(str(config))}"
        if cache_key in self._instances:
            return self._instances[cache_key]

        # Create new instance
        detector_class = self._detectors[name]
        instance = detector_class(config)
        self._instances[cache_key] = instance

        return instance

    def list_detectors(self) -> List[str]:
        """List all registered detector names."""
        return list(self._detectors.keys())

    def clear_cache(self) -> None:
        """Clear cached detector instances."""
        self._instances.clear()


# Global detector registry
detector_registry = DetectorRegistry()
