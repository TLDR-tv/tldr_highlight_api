"""
Gemini-based unified highlight detector for the TL;DR Highlight API.

This module implements highlight detection using Google Gemini's native
multimodal understanding capabilities, replacing separate video, audio,
and fusion components with a unified approach.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import numpy as np
from pydantic import Field

from .base_detector import (
    BaseDetector,
    ContentSegment,
    DetectionConfig,
    DetectionResult,
    ModalityType,
)
from ...entities.highlight import HighlightCandidate
from ..content_processing.gemini_processor import (
    GeminiProcessorConfig,
    GeminiHighlight,
    ProcessingMode,
    gemini_processor,
    initialize_gemini_processor,
)

logger = logging.getLogger(__name__)


class GeminiDetectionConfig(DetectionConfig):
    """
    Configuration for Gemini-based highlight detection.

    Extends base configuration with Gemini-specific parameters
    for unified multimodal analysis.
    """

    # Gemini processor configuration
    processor_config: Optional[GeminiProcessorConfig] = Field(
        default=None, description="Configuration for Gemini processor"
    )

    # Detection thresholds
    highlight_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score for highlight acceptance",
    )
    highlight_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for highlight acceptance",
    )

    # Temporal grouping
    merge_window_seconds: float = Field(
        default=10.0, gt=0.0, description="Window for merging nearby highlights"
    )
    min_highlight_duration: float = Field(
        default=5.0, gt=0.0, description="Minimum duration for a highlight"
    )
    max_highlight_duration: float = Field(
        default=60.0, gt=0.0, description="Maximum duration for a highlight"
    )

    # Category weights for scoring
    category_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "action": 1.2,
            "emotional": 1.1,
            "informative": 1.0,
            "humorous": 1.1,
            "dramatic": 1.2,
            "general": 0.9,
        },
        description="Weights for different highlight categories",
    )

    # Quality modifiers
    enable_quality_boost: bool = Field(
        default=True, description="Enable quality-based score boosting"
    )
    quality_boost_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum boost for high-quality content",
    )

    # Processing mode preferences
    preferred_mode: Optional[ProcessingMode] = Field(
        default=None, description="Preferred processing mode (auto-detected if None)"
    )
    enable_streaming: bool = Field(
        default=True, description="Enable streaming mode for live content"
    )

    # Response enrichment
    include_transcriptions: bool = Field(
        default=True, description="Include transcriptions in results"
    )
    include_visual_descriptions: bool = Field(
        default=True, description="Include visual descriptions in results"
    )
    include_audio_descriptions: bool = Field(
        default=True, description="Include audio descriptions in results"
    )


class GeminiDetector(BaseDetector):
    """
    Unified highlight detector using Google Gemini.

    Leverages Gemini's native multimodal understanding to detect
    highlights without separate processing of video, audio, and text.
    This provides better context understanding and more accurate
    highlight detection.
    """

    def __init__(self, config: Optional[GeminiDetectionConfig] = None):
        """
        Initialize the Gemini detector.

        Args:
            config: Gemini detection configuration
        """
        self.gemini_config = config or GeminiDetectionConfig()
        super().__init__(self.gemini_config)

        # Initialize or get Gemini processor
        global gemini_processor
        if gemini_processor is None:
            initialize_gemini_processor(self.gemini_config.processor_config)

        self.processor = gemini_processor
        if self.processor is None:
            raise RuntimeError("Failed to initialize Gemini processor")

        self.logger = logging.getLogger(f"{__name__}.GeminiDetector")

        # Cache for processed segments
        self._segment_cache = {}
        self._cache_max_size = 100

    @property
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        # Gemini handles all modalities
        return ModalityType.VIDEO  # Primary modality for compatibility

    @property
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        return "GeminiUnifiedDetector"

    @property
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        return "2.0.0"

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """
        Validate that a segment contains valid data for Gemini processing.

        Args:
            segment: Content segment to validate

        Returns:
            True if segment is valid for processing
        """
        if not super()._validate_segment(segment):
            return False

        # Check if segment has appropriate data
        if segment.data is None:
            return False

        # Gemini can handle various data types
        valid_types = (str, bytes, np.ndarray, list, dict)
        if not isinstance(segment.data, valid_types):
            return False

        # Check for minimum duration
        if segment.duration < 1.0:  # At least 1 second
            return False

        return True

    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect highlights using Gemini's unified analysis.

        Args:
            segment: Content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results
        """
        gemini_config = (
            config if isinstance(config, GeminiDetectionConfig) else self.gemini_config
        )

        try:
            # Check cache
            cache_key = f"{segment.segment_id}_{segment.start_time}_{segment.end_time}"
            if cache_key in self._segment_cache:
                self.logger.debug(
                    f"Using cached results for segment {segment.segment_id}"
                )
                return self._segment_cache[cache_key]

            self.logger.debug(f"Processing segment {segment.segment_id} with Gemini")

            # Determine processing mode
            mode = gemini_config.preferred_mode
            if mode is None:
                mode = self._determine_processing_mode(segment)

            # Process with Gemini
            if isinstance(segment.data, str):
                # Assume it's a file path or URL
                result = await self.processor.process_video_file(
                    source=segment.data,
                    mode=mode,
                    start_time=segment.start_time,
                    duration=segment.duration,
                )
            else:
                # Convert segment data to processable format
                source = await self._prepare_segment_data(segment)
                result = await self.processor.process_video_file(
                    source=source,
                    mode=mode,
                    start_time=0.0,  # Already extracted segment
                    duration=segment.duration,
                )

            # Convert Gemini highlights to detection results
            detection_results = await self._convert_highlights_to_results(
                result.highlights, segment, result, gemini_config
            )

            # Cache results
            self._update_cache(cache_key, detection_results)

            self.logger.info(
                f"Gemini detection complete for segment {segment.segment_id}: "
                f"{len(detection_results)} highlights found"
            )

            return detection_results

        except Exception as e:
            self.logger.error(
                f"Error in Gemini detection for segment {segment.segment_id}: {e}"
            )
            return []

    async def _convert_highlights_to_results(
        self,
        highlights: List[GeminiHighlight],
        segment: ContentSegment,
        processing_result: Any,
        config: GeminiDetectionConfig,
    ) -> List[DetectionResult]:
        """Convert Gemini highlights to detection results."""
        results = []

        for highlight in highlights:
            # Apply category weight
            category_weight = config.category_weights.get(
                highlight.category.lower(), config.category_weights.get("general", 1.0)
            )

            # Calculate adjusted score
            adjusted_score = highlight.score * category_weight

            # Apply quality boost if enabled
            if config.enable_quality_boost and processing_result.overall_quality > 0.7:
                quality_boost = (
                    processing_result.overall_quality - 0.7
                ) * config.quality_boost_factor
                adjusted_score = min(1.0, adjusted_score + quality_boost)

            # Check thresholds
            if (
                adjusted_score < config.highlight_score_threshold
                or highlight.confidence < config.highlight_confidence_threshold
            ):
                continue

            # Create detection result
            result = DetectionResult(
                segment_id=segment.segment_id,
                modality=ModalityType.VIDEO,  # Primary modality
                score=adjusted_score,
                confidence=highlight.confidence,
                features={
                    "duration": highlight.duration,
                    "original_score": highlight.score,
                    "category_weight": category_weight,
                    "quality_score": processing_result.overall_quality,
                },
                metadata={
                    "algorithm": self.algorithm_name,
                    "version": self.algorithm_version,
                    "model": processing_result.model_used,
                    "mode": processing_result.mode_used.value,
                    "start_time": highlight.start_time,
                    "end_time": highlight.end_time,
                    "category": highlight.category,
                    "reason": highlight.reason,
                    "key_moments": highlight.key_moments,
                    "transcription": highlight.transcription
                    if config.include_transcriptions
                    else None,
                    "visual_description": highlight.visual_description
                    if config.include_visual_descriptions
                    else None,
                    "audio_description": highlight.audio_description
                    if config.include_audio_descriptions
                    else None,
                },
                algorithm_version=self.algorithm_version,
            )

            results.append(result)

        return results

    async def detect_highlights_unified(
        self,
        source: Union[str, List[ContentSegment]],
        config: Optional[GeminiDetectionConfig] = None,
    ) -> List[HighlightCandidate]:
        """
        Detect highlights using Gemini's unified approach.

        This method processes the entire content in one go, leveraging
        Gemini's ability to understand full context.

        Args:
            source: Video file path, URL, or list of content segments
            config: Detection configuration

        Returns:
            List of highlight candidates
        """
        config = config or self.gemini_config

        try:
            if isinstance(source, str):
                # Process entire file/URL
                result = await self.processor.process_video_file(
                    source=source, mode=config.preferred_mode
                )

                # Convert to highlight candidates
                candidates = []
                for highlight in result.highlights:
                    # Apply filters and scoring
                    category_weight = config.category_weights.get(
                        highlight.category.lower(),
                        config.category_weights.get("general", 1.0),
                    )

                    adjusted_score = highlight.score * category_weight

                    if (
                        adjusted_score >= config.highlight_score_threshold
                        and highlight.confidence
                        >= config.highlight_confidence_threshold
                    ):
                        candidate = HighlightCandidate(
                            start_time=highlight.start_time,
                            end_time=highlight.end_time,
                            score=adjusted_score,
                            confidence=highlight.confidence,
                            modality_results=[],  # Will be populated if needed
                            features={
                                "category": highlight.category,
                                "reason": highlight.reason,
                                "key_moments": highlight.key_moments,
                                "duration": highlight.duration,
                                "model": result.model_used,
                                "processing_mode": result.mode_used.value,
                                "transcription": highlight.transcription,
                                "visual_description": highlight.visual_description,
                                "audio_description": highlight.audio_description,
                                "content_summary": result.content_summary,
                                "overall_quality": result.overall_quality,
                            },
                        )
                        candidates.append(candidate)

                # Merge nearby highlights if configured
                if config.merge_window_seconds > 0:
                    candidates = self._merge_nearby_highlights(candidates, config)

                return candidates

            else:
                # Process segments individually and aggregate
                return await self.detect_highlights(source, config)

        except Exception as e:
            self.logger.error(f"Error in unified highlight detection: {e}")
            return []

    def _merge_nearby_highlights(
        self, candidates: List[HighlightCandidate], config: GeminiDetectionConfig
    ) -> List[HighlightCandidate]:
        """Merge highlights that are close in time."""
        if not candidates:
            return []

        # Sort by start time
        sorted_candidates = sorted(candidates, key=lambda x: x.start_time)

        merged = []
        current = sorted_candidates[0]

        for candidate in sorted_candidates[1:]:
            # Check if should merge
            if (
                candidate.start_time - current.end_time <= config.merge_window_seconds
                and candidate.features.get("category")
                == current.features.get("category")
            ):
                # Merge highlights
                current = HighlightCandidate(
                    start_time=current.start_time,
                    end_time=candidate.end_time,
                    score=max(current.score, candidate.score),
                    confidence=(current.confidence + candidate.confidence) / 2,
                    modality_results=current.modality_results
                    + candidate.modality_results,
                    features={
                        **current.features,
                        "merged": True,
                        "merge_count": current.features.get("merge_count", 1) + 1,
                    },
                    # Metadata should be stored in features
                )
            else:
                # Add current and start new
                if current.duration >= config.min_highlight_duration:
                    # Trim if too long
                    if current.duration > config.max_highlight_duration:
                        current.end_time = (
                            current.start_time + config.max_highlight_duration
                        )
                    merged.append(current)
                current = candidate

        # Add final highlight
        if current.duration >= config.min_highlight_duration:
            if current.duration > config.max_highlight_duration:
                current.end_time = current.start_time + config.max_highlight_duration
            merged.append(current)

        return merged

    def _determine_processing_mode(self, segment: ContentSegment) -> ProcessingMode:
        """Determine the best processing mode for a segment."""
        # Check segment metadata for hints
        if segment.metadata.get("is_live", False):
            return ProcessingMode.LIVE_API

        if isinstance(segment.data, str):
            if "youtube.com" in segment.data or "youtu.be" in segment.data:
                return ProcessingMode.DIRECT_URL

        return ProcessingMode.FILE_API

    async def _prepare_segment_data(self, segment: ContentSegment) -> str:
        """Prepare segment data for Gemini processing."""
        # Save segment data to temporary file
        import tempfile

        if isinstance(segment.data, bytes):
            # Binary data - save as file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(segment.data)
                return f.name

        elif isinstance(segment.data, np.ndarray):
            # Numpy array - convert to video
            # This is a placeholder - proper implementation would encode to video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                # Placeholder: save array dimensions as metadata
                np.save(f.name, segment.data)
                return f.name

        elif isinstance(segment.data, list):
            # List of frames - create video
            # This is a placeholder - proper implementation would encode frames
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(b"placeholder_video_data")
                return f.name

        else:
            raise ValueError(f"Unsupported segment data type: {type(segment.data)}")

    def _update_cache(self, key: str, results: List[DetectionResult]) -> None:
        """Update the segment cache with size limit."""
        self._segment_cache[key] = results

        # Limit cache size
        if len(self._segment_cache) > self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._segment_cache.keys())[: -self._cache_max_size]
            for k in keys_to_remove:
                del self._segment_cache[k]

    async def process_stream_with_gemini(
        self,
        video_stream: Any,
        audio_stream: Optional[Any] = None,
        config: Optional[GeminiDetectionConfig] = None,
    ) -> AsyncGenerator[HighlightCandidate, None]:
        """
        Process live stream using Gemini's Live API.

        Args:
            video_stream: Video frame stream
            audio_stream: Optional audio stream
            config: Detection configuration

        Yields:
            Highlight candidates as they are detected
        """
        config = config or self.gemini_config

        if not config.enable_streaming:
            self.logger.warning("Streaming mode disabled in configuration")
            return

        try:
            async for result in self.processor.process_video_stream(
                video_chunks=video_stream, audio_chunks=audio_stream
            ):
                # Convert Gemini results to candidates
                for highlight in result.highlights:
                    category_weight = config.category_weights.get(
                        highlight.category.lower(),
                        config.category_weights.get("general", 1.0),
                    )

                    adjusted_score = highlight.score * category_weight

                    if (
                        adjusted_score >= config.highlight_score_threshold
                        and highlight.confidence
                        >= config.highlight_confidence_threshold
                    ):
                        candidate = HighlightCandidate(
                            start_time=highlight.start_time,
                            end_time=highlight.end_time,
                            score=adjusted_score,
                            confidence=highlight.confidence,
                            modality_results=[],
                            features={
                                "category": highlight.category,
                                "reason": highlight.reason,
                                "key_moments": highlight.key_moments,
                                "is_live": True,
                                "model": result.model_used,
                                "processing_mode": "live_streaming",
                            },
                        )

                        yield candidate

        except Exception as e:
            self.logger.error(f"Error processing stream with Gemini: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for Gemini detection."""
        base_metrics = self.get_metrics()

        # Add Gemini-specific metrics
        if self.processor:
            processor_stats = asyncio.run(self.processor.get_processing_stats())
        else:
            processor_stats = {}

        gemini_metrics = {
            **base_metrics,
            "algorithm": self.algorithm_name,
            "version": self.algorithm_version,
            "processor_stats": processor_stats,
            "cache_size": len(self._segment_cache),
            "cache_max_size": self._cache_max_size,
            "config": self.gemini_config.model_dump(),
        }

        return gemini_metrics


# Create a global instance for backward compatibility
gemini_detector = None


def initialize_gemini_detector(config: Optional[GeminiDetectionConfig] = None):
    """Initialize the global Gemini detector instance."""
    global gemini_detector
    try:
        gemini_detector = GeminiDetector(config)
        logger.info("Gemini detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini detector: {e}")
        gemini_detector = None
