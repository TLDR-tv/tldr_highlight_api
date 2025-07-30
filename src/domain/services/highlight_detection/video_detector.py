"""
Video-based highlight detection for the TL;DR Highlight API.

This module implements sophisticated video analysis algorithms for identifying
exciting moments through motion detection, scene changes, and visual activity.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field

from .base_detector import (
    BaseDetector,
    ContentSegment,
    DetectionConfig,
    DetectionResult,
    ModalityType,
)
from ...utils.ml_utils import video_feature_extractor
from ...utils.scoring_utils import (
    normalize_score,
    calculate_confidence,
)

logger = logging.getLogger(__name__)


class VideoDetectionConfig(DetectionConfig):
    """
    Configuration for video-based highlight detection.

    Extends base configuration with video-specific parameters
    for motion analysis and scene change detection.
    """

    # Motion detection parameters
    motion_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Threshold for motion significance"
    )
    motion_window_size: int = Field(
        default=5, ge=1, description="Window size for motion analysis (frames)"
    )
    motion_weight: float = Field(
        default=0.4, ge=0.0, description="Weight for motion features in scoring"
    )

    # Scene change detection parameters
    scene_change_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Threshold for scene change detection"
    )
    scene_change_weight: float = Field(
        default=0.3, ge=0.0, description="Weight for scene change features in scoring"
    )

    # Activity analysis parameters
    activity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Threshold for activity level detection",
    )
    activity_weight: float = Field(
        default=0.3, ge=0.0, description="Weight for activity features in scoring"
    )

    # Frame sampling parameters
    max_frames_per_segment: int = Field(
        default=300, ge=10, description="Maximum frames to analyze per segment"
    )
    frame_skip_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Ratio of frames to skip for efficiency",
    )

    # Color analysis parameters
    color_analysis_enabled: bool = Field(
        default=True, description="Enable color-based excitement detection"
    )
    color_variance_threshold: float = Field(
        default=0.15, ge=0.0, description="Threshold for color variance excitement"
    )

    # Edge detection parameters
    edge_analysis_enabled: bool = Field(
        default=True, description="Enable edge-based activity detection"
    )
    edge_density_threshold: float = Field(
        default=0.1, ge=0.0, description="Threshold for edge density significance"
    )


@dataclass
class VideoFrameData:
    """
    Represents video frame data for analysis.

    Contains frame pixels, metadata, and analysis results.
    """

    frame_index: int
    timestamp: float
    pixels: np.ndarray
    width: int
    height: int
    channels: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis results (computed lazily)
    _motion_vectors: Optional[np.ndarray] = None
    _edge_map: Optional[np.ndarray] = None
    _color_histogram: Optional[np.ndarray] = None

    @property
    def aspect_ratio(self) -> float:
        """Get frame aspect ratio."""
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def total_pixels(self) -> int:
        """Get total pixel count."""
        return self.width * self.height

    def get_grayscale(self) -> np.ndarray:
        """Convert frame to grayscale."""
        if self.channels == 1:
            return self.pixels
        elif self.channels == 3:
            # RGB to grayscale conversion
            return np.dot(self.pixels[..., :3], [0.299, 0.587, 0.114])
        else:
            # Take first channel
            return self.pixels[..., 0]

    def compute_motion_vectors(self, previous_frame: "VideoFrameData") -> np.ndarray:
        """
        Compute motion vectors between this frame and previous frame.

        Args:
            previous_frame: Previous frame for motion estimation

        Returns:
            Motion vector field
        """
        if self._motion_vectors is not None:
            return self._motion_vectors

        # Simplified motion estimation using frame difference
        current_gray = self.get_grayscale()
        previous_gray = previous_frame.get_grayscale()

        # Ensure same dimensions
        if current_gray.shape != previous_gray.shape:
            return np.zeros((self.height // 8, self.width // 8, 2))

        # Compute frame difference
        frame_diff = np.abs(
            current_gray.astype(np.float32) - previous_gray.astype(np.float32)
        )

        # Downsample for motion estimation
        block_size = 8
        motion_vectors = np.zeros(
            (self.height // block_size, self.width // block_size, 2)
        )

        for i in range(0, self.height - block_size, block_size):
            for j in range(0, self.width - block_size, block_size):
                block_diff = frame_diff[i : i + block_size, j : j + block_size]
                motion_magnitude = np.mean(block_diff)

                # Simplified motion direction (would need proper block matching)
                motion_vectors[i // block_size, j // block_size] = [motion_magnitude, 0]

        self._motion_vectors = motion_vectors
        return motion_vectors

    def compute_edge_map(self) -> np.ndarray:
        """
        Compute edge map for the frame.

        Returns:
            Edge magnitude map
        """
        if self._edge_map is not None:
            return self._edge_map

        gray = self.get_grayscale()

        # Sobel edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Convolution (simplified)
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)

        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                region = gray[i - 1 : i + 2, j - 1 : j + 2]
                grad_x[i, j] = np.sum(region * sobel_x)
                grad_y[i, j] = np.sum(region * sobel_y)

        # Compute edge magnitude
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        self._edge_map = edge_magnitude
        return edge_magnitude

    def compute_color_histogram(self, bins: int = 16) -> np.ndarray:
        """
        Compute color histogram for the frame.

        Args:
            bins: Number of histogram bins per channel

        Returns:
            Concatenated color histogram
        """
        if self._color_histogram is not None:
            return self._color_histogram

        histograms = []

        if self.channels == 1:
            # Grayscale histogram
            hist, _ = np.histogram(self.pixels, bins=bins, range=(0, 255))
            histograms.append(hist)
        else:
            # Color histograms
            for channel in range(min(self.channels, 3)):
                hist, _ = np.histogram(
                    self.pixels[..., channel], bins=bins, range=(0, 255)
                )
                histograms.append(hist)

        # Concatenate and normalize
        combined_hist = np.concatenate(histograms)
        self._color_histogram = combined_hist / np.sum(combined_hist)
        return self._color_histogram


class VideoActivityAnalyzer:
    """
    Analyzes video content for activity and excitement indicators.

    Implements various algorithms for detecting visual activity,
    motion patterns, and scene changes.
    """

    def __init__(self, config: VideoDetectionConfig):
        """
        Initialize the video activity analyzer.

        Args:
            config: Video detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.VideoActivityAnalyzer")

    async def analyze_motion(self, frames: List[VideoFrameData]) -> Dict[str, float]:
        """
        Analyze motion activity in video frames.

        Args:
            frames: List of video frames to analyze

        Returns:
            Dictionary with motion analysis results
        """
        if len(frames) < 2:
            return {"motion_score": 0.0, "motion_consistency": 0.0}

        motion_magnitudes = []
        motion_directions = []

        # Analyze motion between consecutive frames
        for i in range(1, len(frames)):
            current_frame = frames[i]
            previous_frame = frames[i - 1]

            # Compute motion vectors
            motion_vectors = current_frame.compute_motion_vectors(previous_frame)

            # Calculate motion statistics
            motion_magnitude = np.mean(np.abs(motion_vectors[..., 0]))
            motion_magnitudes.append(motion_magnitude)

            # Motion direction consistency (simplified)
            direction_variance = np.var(motion_vectors[..., 1])
            motion_directions.append(direction_variance)

        # Calculate motion scores
        avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0.0
        motion_consistency = (
            1.0 - np.var(motion_magnitudes) if motion_magnitudes else 0.0
        )

        # Normalize scores
        motion_score = normalize_score(avg_motion / 255.0, method="sigmoid")
        motion_consistency = max(0.0, min(1.0, motion_consistency))

        return {
            "motion_score": motion_score,
            "motion_consistency": motion_consistency,
            "motion_magnitude": avg_motion,
            "frame_count": len(frames),
        }

    async def analyze_scene_changes(
        self, frames: List[VideoFrameData]
    ) -> Dict[str, float]:
        """
        Analyze scene changes in video frames.

        Args:
            frames: List of video frames to analyze

        Returns:
            Dictionary with scene change analysis results
        """
        if len(frames) < 2:
            return {"scene_change_score": 0.0, "scene_stability": 1.0}

        scene_change_scores = []
        color_differences = []

        # Analyze differences between consecutive frames
        for i in range(1, len(frames)):
            current_frame = frames[i]
            previous_frame = frames[i - 1]

            # Color histogram difference
            current_hist = current_frame.compute_color_histogram()
            previous_hist = previous_frame.compute_color_histogram()

            # Calculate histogram difference
            hist_diff = np.sum(np.abs(current_hist - previous_hist))
            color_differences.append(hist_diff)

            # Pixel difference (downsampled for efficiency)
            current_gray = current_frame.get_grayscale()
            previous_gray = previous_frame.get_grayscale()

            # Downsample frames
            if current_gray.shape == previous_gray.shape:
                step = max(1, min(current_gray.shape) // 64)
                current_sample = current_gray[::step, ::step]
                previous_sample = previous_gray[::step, ::step]

                pixel_diff = np.mean(
                    np.abs(
                        current_sample.astype(np.float32)
                        - previous_sample.astype(np.float32)
                    )
                )
                scene_change_scores.append(pixel_diff / 255.0)
            else:
                scene_change_scores.append(0.0)

        # Calculate scene change metrics
        avg_scene_change = np.mean(scene_change_scores) if scene_change_scores else 0.0
        scene_stability = (
            1.0 - np.var(scene_change_scores) if scene_change_scores else 1.0
        )

        # Normalize scores
        scene_change_score = normalize_score(avg_scene_change, method="sigmoid")
        scene_stability = max(0.0, min(1.0, scene_stability))

        return {
            "scene_change_score": scene_change_score,
            "scene_stability": scene_stability,
            "avg_color_difference": np.mean(color_differences)
            if color_differences
            else 0.0,
        }

    async def analyze_visual_complexity(
        self, frames: List[VideoFrameData]
    ) -> Dict[str, float]:
        """
        Analyze visual complexity and activity in frames.

        Args:
            frames: List of video frames to analyze

        Returns:
            Dictionary with visual complexity analysis results
        """
        if not frames:
            return {"complexity_score": 0.0, "edge_density": 0.0}

        edge_densities = []
        color_variances = []

        # Analyze each frame for complexity
        for frame in frames:
            # Edge density analysis
            if self.config.edge_analysis_enabled:
                edge_map = frame.compute_edge_map()
                edge_density = np.mean(
                    edge_map > self.config.edge_density_threshold * 255
                )
                edge_densities.append(edge_density)

            # Color variance analysis
            if self.config.color_analysis_enabled:
                if frame.channels >= 3:
                    color_var = np.var(frame.pixels.astype(np.float32))
                    color_variances.append(color_var / (255**2))  # Normalize
                else:
                    color_variances.append(0.0)

        # Calculate complexity metrics
        avg_edge_density = np.mean(edge_densities) if edge_densities else 0.0
        avg_color_variance = np.mean(color_variances) if color_variances else 0.0

        # Combine into complexity score
        complexity_components = []
        if edge_densities:
            complexity_components.append(avg_edge_density)
        if color_variances:
            complexity_components.append(avg_color_variance)

        complexity_score = (
            np.mean(complexity_components) if complexity_components else 0.0
        )

        return {
            "complexity_score": normalize_score(complexity_score, method="sigmoid"),
            "edge_density": avg_edge_density,
            "color_variance": avg_color_variance,
        }


class VideoDetector(BaseDetector):
    """
    Video-based highlight detector using motion analysis and visual activity.

    Implements sophisticated algorithms for identifying exciting moments
    in video content through motion detection, scene changes, and
    visual complexity analysis.
    """

    def __init__(self, config: Optional[VideoDetectionConfig] = None):
        """
        Initialize the video detector.

        Args:
            config: Video detection configuration
        """
        self.video_config = config or VideoDetectionConfig()
        super().__init__(self.video_config)

        self.activity_analyzer = VideoActivityAnalyzer(self.video_config)
        self.logger = logging.getLogger(f"{__name__}.VideoDetector")

    @property
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        return ModalityType.VIDEO

    @property
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        return "VideoActivityDetector"

    @property
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        return "1.0.0"

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """
        Validate that a segment contains valid video data.

        Args:
            segment: Content segment to validate

        Returns:
            True if segment contains valid video data
        """
        if not super()._validate_segment(segment):
            return False

        # Check if data is video frames
        if not isinstance(segment.data, (list, np.ndarray)):
            return False

        # Check if frames have proper structure
        if isinstance(segment.data, list):
            if not segment.data:
                return False

            # Check first frame
            first_frame = segment.data[0]
            if not isinstance(first_frame, (VideoFrameData, np.ndarray, dict)):
                return False

        return True

    def _prepare_frames(self, segment: ContentSegment) -> List[VideoFrameData]:
        """
        Prepare video frames from segment data.

        Args:
            segment: Content segment with video data

        Returns:
            List of VideoFrameData objects
        """
        frames = []

        if isinstance(segment.data, list):
            for i, frame_data in enumerate(segment.data):
                timestamp = (
                    segment.start_time + (i / len(segment.data)) * segment.duration
                )

                if isinstance(frame_data, VideoFrameData):
                    frames.append(frame_data)
                elif isinstance(frame_data, np.ndarray):
                    # Create VideoFrameData from numpy array
                    if len(frame_data.shape) == 3:
                        h, w, c = frame_data.shape
                    elif len(frame_data.shape) == 2:
                        h, w = frame_data.shape
                        c = 1
                    else:
                        continue

                    frame = VideoFrameData(
                        frame_index=i,
                        timestamp=timestamp,
                        pixels=frame_data,
                        width=w,
                        height=h,
                        channels=c,
                    )
                    frames.append(frame)
                elif isinstance(frame_data, dict):
                    # Create from dictionary
                    if "pixels" in frame_data:
                        pixels = np.array(frame_data["pixels"])
                        if len(pixels.shape) >= 2:
                            frame = VideoFrameData(
                                frame_index=i,
                                timestamp=timestamp,
                                pixels=pixels,
                                width=frame_data.get("width", pixels.shape[1]),
                                height=frame_data.get("height", pixels.shape[0]),
                                channels=frame_data.get(
                                    "channels",
                                    pixels.shape[2] if len(pixels.shape) == 3 else 1,
                                ),
                            )
                            frames.append(frame)

        # Limit frames for performance
        if len(frames) > self.video_config.max_frames_per_segment:
            # Sample frames evenly
            step = len(frames) // self.video_config.max_frames_per_segment
            frames = frames[::step]

        return frames

    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect video-based highlight features in a segment.

        Args:
            segment: Video content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results for video analysis
        """
        video_config = (
            config if isinstance(config, VideoDetectionConfig) else self.video_config
        )

        try:
            # Prepare video frames
            frames = self._prepare_frames(segment)
            if len(frames) < 2:
                self.logger.warning(
                    f"Insufficient frames ({len(frames)}) in segment {segment.segment_id}"
                )
                return []

            self.logger.debug(
                f"Analyzing {len(frames)} frames in segment {segment.segment_id}"
            )

            # Perform various analyses concurrently
            motion_task = self.activity_analyzer.analyze_motion(frames)
            scene_task = self.activity_analyzer.analyze_scene_changes(frames)
            complexity_task = self.activity_analyzer.analyze_visual_complexity(frames)

            # Wait for all analyses to complete
            motion_results, scene_results, complexity_results = await asyncio.gather(
                motion_task, scene_task, complexity_task
            )

            # Extract ML features
            try:
                # Prepare feature data for ML feature extractor
                frame_array = np.array(
                    [frame.pixels for frame in frames[:10]]
                )  # Sample frames
                ml_features = await video_feature_extractor.extract_features(
                    frame_array
                )
            except Exception as e:
                self.logger.warning(f"ML feature extraction failed: {e}")
                ml_features = np.array([])

            # Combine analysis results
            all_scores = {
                "motion": motion_results["motion_score"],
                "scene_change": scene_results["scene_change_score"],
                "complexity": complexity_results["complexity_score"],
            }

            # Calculate weighted score
            weights = {
                "motion": video_config.motion_weight,
                "scene_change": video_config.scene_change_weight,
                "complexity": video_config.activity_weight,
            }

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate final score
            final_score = sum(all_scores[k] * weights[k] for k in all_scores.keys())

            # Calculate confidence based on score consistency
            score_values = list(all_scores.values())
            confidence = calculate_confidence(score_values, method="consistency")

            # Check significance thresholds
            if (
                final_score < video_config.min_score
                or confidence < video_config.min_confidence
            ):
                return []

            # Create detection result
            result = DetectionResult(
                segment_id=segment.segment_id,
                modality=self.modality,
                score=final_score,
                confidence=confidence,
                features={
                    **all_scores,
                    "motion_magnitude": motion_results.get("motion_magnitude", 0.0),
                    "scene_stability": scene_results.get("scene_stability", 1.0),
                    "edge_density": complexity_results.get("edge_density", 0.0),
                    "color_variance": complexity_results.get("color_variance", 0.0),
                    "frame_count": len(frames),
                },
                metadata={
                    "algorithm": self.algorithm_name,
                    "version": self.algorithm_version,
                    "config": video_config.dict(),
                    "motion_analysis": motion_results,
                    "scene_analysis": scene_results,
                    "complexity_analysis": complexity_results,
                },
                algorithm_version=self.algorithm_version,
            )

            # Add ML features if available
            if ml_features.size > 0:
                result.metadata["ml_features"] = ml_features.tolist()

            self.logger.debug(
                f"Video analysis complete for segment {segment.segment_id}: "
                f"score={final_score:.3f}, confidence={confidence:.3f}"
            )

            return [result]

        except Exception as e:
            self.logger.error(
                f"Error in video detection for segment {segment.segment_id}: {e}"
            )
            return []

    async def detect_highlights_from_stream(
        self, frame_stream: List[VideoFrameData], window_size: float = 30.0
    ) -> List[DetectionResult]:
        """
        Detect highlights from a stream of video frames.

        Args:
            frame_stream: Stream of video frames
            window_size: Analysis window size in seconds

        Returns:
            List of detection results
        """
        if not frame_stream:
            return []

        # Group frames into segments
        segments = []
        current_segment_frames = []
        segment_start_time = frame_stream[0].timestamp

        for frame in frame_stream:
            if frame.timestamp - segment_start_time >= window_size:
                # Create segment from accumulated frames
                if current_segment_frames:
                    segment = ContentSegment(
                        start_time=segment_start_time,
                        end_time=current_segment_frames[-1].timestamp,
                        data=current_segment_frames,
                        metadata={"frame_count": len(current_segment_frames)},
                    )
                    segments.append(segment)

                # Start new segment
                current_segment_frames = [frame]
                segment_start_time = frame.timestamp
            else:
                current_segment_frames.append(frame)

        # Add final segment
        if current_segment_frames:
            segment = ContentSegment(
                start_time=segment_start_time,
                end_time=current_segment_frames[-1].timestamp,
                data=current_segment_frames,
                metadata={"frame_count": len(current_segment_frames)},
            )
            segments.append(segment)

        # Detect highlights in segments
        return await self.detect_highlights(segments)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for video detection."""
        base_metrics = self.get_metrics()

        # Add video-specific metrics
        video_metrics = {
            **base_metrics,
            "algorithm": self.algorithm_name,
            "version": self.algorithm_version,
            "config": self.video_config.dict(),
        }

        return video_metrics
