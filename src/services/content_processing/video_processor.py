"""
Video processing service for frame extraction and analysis.

This module provides efficient video frame extraction with configurable intervals,
quality filtering, and memory optimization for real-time streaming scenarios.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import weakref

import numpy as np
from pydantic import BaseModel, Field

from src.utils.media_utils import MediaProcessor, VideoFrame, StreamCapture

logger = logging.getLogger(__name__)


class VideoProcessorConfig(BaseModel):
    """Configuration for video processor."""

    # Frame extraction settings
    frame_interval_seconds: float = Field(
        default=1.0, description="Interval between frame extractions in seconds"
    )
    max_frames_per_window: int = Field(
        default=30, description="Maximum frames to extract per processing window"
    )
    quality_threshold: float = Field(
        default=0.3, description="Minimum quality score for frames (0.0-1.0)"
    )

    # Memory management
    max_memory_mb: int = Field(
        default=500, description="Maximum memory usage for video processing"
    )
    buffer_size: int = Field(
        default=50, description="Maximum number of frames to keep in memory"
    )

    # Frame processing
    resize_width: Optional[int] = Field(
        default=720, description="Resize frame width (maintains aspect ratio)"
    )
    enable_scene_detection: bool = Field(
        default=True, description="Enable scene change detection"
    )
    scene_change_threshold: float = Field(
        default=0.3, description="Threshold for scene change detection"
    )

    # Quality analysis
    enable_quality_analysis: bool = Field(
        default=True, description="Enable frame quality analysis"
    )
    blur_threshold: float = Field(
        default=100.0, description="Laplacian variance threshold for blur detection"
    )
    brightness_range: Tuple[float, float] = Field(
        default=(0.1, 0.9), description="Acceptable brightness range (0.0-1.0)"
    )

    # Processing profiles
    processing_profile: str = Field(
        default="balanced", description="Processing profile: fast, balanced, or quality"
    )

    # Streaming settings
    stream_buffer_size: int = Field(
        default=10, description="Buffer size for streaming frames"
    )
    stream_reconnect_attempts: int = Field(
        default=3, description="Number of reconnection attempts for streams"
    )


@dataclass
class ProcessedFrame:
    """Processed video frame with analysis results."""

    frame: VideoFrame
    analysis: Dict[str, Union[float, bool, str]]
    processing_time: float
    is_keyframe: bool = False
    scene_change: bool = False


@dataclass
class VideoProcessingResult:
    """Result of video processing operation."""

    frames: List[ProcessedFrame]
    total_frames_processed: int
    processing_time: float
    average_quality: float
    scene_changes: List[int]
    metadata: Dict[str, Union[str, float, int]]


class VideoProcessor:
    """
    Advanced video processor with real-time capabilities.

    Features:
    - Configurable frame extraction intervals
    - Quality-based frame filtering
    - Scene change detection
    - Memory-optimized processing
    - Real-time streaming support
    - Parallel processing capabilities
    """

    def __init__(self, config: Optional[VideoProcessorConfig] = None):
        self.config = config or VideoProcessorConfig()
        self.media_processor = MediaProcessor(max_memory_mb=self.config.max_memory_mb)
        self.frame_buffer: List[ProcessedFrame] = []
        self.processing_stats = {
            "total_frames_processed": 0,
            "total_processing_time": 0.0,
            "average_quality": 0.0,
            "scene_changes_detected": 0,
        }
        self.active_streams: Set[str] = set()
        self.stream_captures: Dict[str, StreamCapture] = {}
        self._processing_lock = asyncio.Lock()
        self._cleanup_tasks: Set[asyncio.Task] = set()

        # Weak reference to prevent circular references
        self._instances = weakref.WeakSet()
        self._instances.add(self)

    async def process_video_file(
        self,
        source: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> VideoProcessingResult:
        """
        Process video file and extract frames.

        Args:
            source: Path to video file
            start_time: Start time in seconds
            duration: Duration to process in seconds

        Returns:
            VideoProcessingResult with processed frames
        """
        start_processing = time.time()
        processed_frames = []
        scene_changes = []

        try:
            logger.info(f"Starting video processing for {source}")

            # Get media info
            media_info = await self.media_processor.get_media_info(source)
            if not media_info:
                raise ValueError(f"Unable to get media info for {source}")

            # Calculate processing parameters
            max_duration = duration if duration else media_info.duration - start_time
            max_frames = min(
                int(max_duration / self.config.frame_interval_seconds),
                self.config.max_frames_per_window,
            )

            logger.info(
                f"Processing {max_frames} frames over {max_duration:.2f} seconds"
            )

            # Extract frames
            frame_count = 0
            previous_frames = []

            async for video_frame in self.media_processor.extract_frames(
                source=source,
                interval_seconds=self.config.frame_interval_seconds,
                max_frames=max_frames,
                quality_threshold=self.config.quality_threshold,
                resize_width=self.config.resize_width,
            ):
                # Skip frames before start time
                if video_frame.timestamp < start_time:
                    continue

                # Stop if we've exceeded duration
                if duration and video_frame.timestamp > start_time + duration:
                    break

                # Process frame
                processed_frame = await self._process_single_frame(
                    video_frame, previous_frames
                )

                processed_frames.append(processed_frame)
                previous_frames.append(video_frame)

                # Keep only recent frames for scene detection
                if len(previous_frames) > 5:
                    previous_frames.pop(0)

                # Track scene changes
                if processed_frame.scene_change:
                    scene_changes.append(frame_count)

                frame_count += 1

                # Memory management
                if len(processed_frames) > self.config.buffer_size:
                    # Remove oldest frames from memory (keep metadata)
                    old_frame = processed_frames[0]
                    old_frame.frame.frame = None  # Free memory
                    processed_frames = processed_frames[1:]

                # Yield control to event loop
                await asyncio.sleep(0)

            # Calculate results
            processing_time = time.time() - start_processing
            average_quality = (
                np.mean([f.frame.quality_score for f in processed_frames])
                if processed_frames
                else 0.0
            )

            # Update stats
            self.processing_stats["total_frames_processed"] += len(processed_frames)
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["scene_changes_detected"] += len(scene_changes)

            result = VideoProcessingResult(
                frames=processed_frames,
                total_frames_processed=len(processed_frames),
                processing_time=processing_time,
                average_quality=average_quality,
                scene_changes=scene_changes,
                metadata={
                    "source": str(source),
                    "duration": max_duration,
                    "fps": media_info.fps,
                    "resolution": f"{media_info.width}x{media_info.height}",
                    "codec": media_info.codec,
                    "start_time": start_time,
                    "processing_profile": self.config.processing_profile,
                },
            )

            logger.info(
                f"Video processing completed: {len(processed_frames)} frames, "
                f"{processing_time:.2f}s, avg quality: {average_quality:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing video {source}: {e}")
            raise

    async def start_stream_processing(self, stream_url: str, stream_id: str) -> None:
        """
        Start processing a live stream.

        Args:
            stream_url: URL of the live stream
            stream_id: Unique identifier for the stream
        """
        if stream_id in self.active_streams:
            logger.warning(f"Stream {stream_id} is already being processed")
            return

        try:
            logger.info(f"Starting stream processing for {stream_id}: {stream_url}")

            # Create stream capture
            stream_capture = StreamCapture(
                stream_url=stream_url, buffer_size=self.config.stream_buffer_size
            )

            self.stream_captures[stream_id] = stream_capture
            self.active_streams.add(stream_id)

            # Start capture
            await stream_capture.start_capture()

            # Start processing task
            processing_task = asyncio.create_task(
                self._process_stream_frames(stream_id, stream_capture)
            )
            self._cleanup_tasks.add(processing_task)
            processing_task.add_done_callback(self._cleanup_tasks.discard)

            logger.info(f"Stream processing started for {stream_id}")

        except Exception as e:
            logger.error(f"Error starting stream processing for {stream_id}: {e}")
            if stream_id in self.active_streams:
                self.active_streams.remove(stream_id)
            raise

    async def stop_stream_processing(self, stream_id: str) -> None:
        """
        Stop processing a live stream.

        Args:
            stream_id: Stream identifier
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Stream {stream_id} is not being processed")
            return

        try:
            logger.info(f"Stopping stream processing for {stream_id}")

            # Remove from active streams
            self.active_streams.remove(stream_id)

            # Stop stream capture
            if stream_id in self.stream_captures:
                await self.stream_captures[stream_id].stop_capture()
                del self.stream_captures[stream_id]

            logger.info(f"Stream processing stopped for {stream_id}")

        except Exception as e:
            logger.error(f"Error stopping stream processing for {stream_id}: {e}")

    async def get_processed_frames(
        self, stream_id: str, limit: int = 10
    ) -> List[ProcessedFrame]:
        """
        Get recently processed frames from a stream.

        Args:
            stream_id: Stream identifier
            limit: Maximum number of frames to return

        Returns:
            List of processed frames
        """
        if stream_id not in self.active_streams:
            return []

        async with self._processing_lock:
            # Return recent frames from buffer
            return self.frame_buffer[-limit:] if self.frame_buffer else []

    async def _process_single_frame(
        self, video_frame: VideoFrame, previous_frames: List[VideoFrame]
    ) -> ProcessedFrame:
        """
        Process a single video frame with analysis.

        Args:
            video_frame: Frame to process
            previous_frames: Previous frames for context

        Returns:
            ProcessedFrame with analysis results
        """
        start_time = time.time()

        try:
            analysis = {}

            # Quality analysis
            if self.config.enable_quality_analysis:
                analysis.update(await self._analyze_frame_quality(video_frame))

            # Scene change detection
            scene_change = False
            if self.config.enable_scene_detection and previous_frames:
                scene_change = await self._detect_scene_change(
                    video_frame, previous_frames[-1]
                )

            # Determine if this is a keyframe (high quality + scene change)
            is_keyframe = (
                video_frame.quality_score > 0.7
                or scene_change
                or len(previous_frames) == 0  # First frame is always keyframe
            )

            processing_time = time.time() - start_time

            return ProcessedFrame(
                frame=video_frame,
                analysis=analysis,
                processing_time=processing_time,
                is_keyframe=is_keyframe,
                scene_change=scene_change,
            )

        except Exception as e:
            logger.error(f"Error processing frame at {video_frame.timestamp}: {e}")
            return ProcessedFrame(
                frame=video_frame,
                analysis={},
                processing_time=time.time() - start_time,
                is_keyframe=False,
                scene_change=False,
            )

    async def _analyze_frame_quality(self, video_frame: VideoFrame) -> Dict[str, float]:
        """
        Analyze frame quality metrics.

        Args:
            video_frame: Frame to analyze

        Returns:
            Dictionary of quality metrics
        """
        try:
            import cv2

            frame = video_frame.frame
            if frame is None:
                return {"quality_score": 0.0}

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / self.config.blur_threshold, 1.0)

            # Brightness analysis
            brightness = np.mean(gray) / 255.0
            brightness_ok = (
                self.config.brightness_range[0]
                <= brightness
                <= self.config.brightness_range[1]
            )

            # Contrast analysis
            contrast = gray.std() / 255.0

            # Noise estimation (using high-frequency components)
            noise_estimate = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            noise_score = max(0.0, 1.0 - noise_estimate / 50.0)

            return {
                "blur_score": blur_score,
                "brightness": brightness,
                "brightness_ok": brightness_ok,
                "contrast": contrast,
                "noise_score": noise_score,
                "laplacian_variance": laplacian_var,
                "quality_score": video_frame.quality_score,
            }

        except Exception as e:
            logger.error(f"Error analyzing frame quality: {e}")
            return {"quality_score": video_frame.quality_score}

    async def _detect_scene_change(
        self, current_frame: VideoFrame, previous_frame: VideoFrame
    ) -> bool:
        """
        Detect scene change between two frames.

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            True if scene change detected
        """
        try:
            import cv2

            if current_frame.frame is None or previous_frame.frame is None:
                return False

            # Convert frames to grayscale
            curr_gray = cv2.cvtColor(current_frame.frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(previous_frame.frame, cv2.COLOR_BGR2GRAY)

            # Calculate histograms
            curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])

            # Calculate correlation
            correlation = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CORREL)

            # Scene change if correlation is low
            return correlation < (1.0 - self.config.scene_change_threshold)

        except Exception as e:
            logger.error(f"Error detecting scene change: {e}")
            return False

    async def _process_stream_frames(
        self, stream_id: str, stream_capture: StreamCapture
    ) -> None:
        """
        Process frames from a live stream.

        Args:
            stream_id: Stream identifier
            stream_capture: StreamCapture instance
        """
        previous_frames = []

        try:
            while stream_id in self.active_streams:
                # Get next frame
                video_frame = await stream_capture.get_frame(timeout=5.0)
                if video_frame is None:
                    continue

                # Process frame
                processed_frame = await self._process_single_frame(
                    video_frame, previous_frames
                )

                # Add to buffer
                async with self._processing_lock:
                    self.frame_buffer.append(processed_frame)

                    # Maintain buffer size
                    if len(self.frame_buffer) > self.config.buffer_size:
                        old_frame = self.frame_buffer.pop(0)
                        # Free memory
                        if old_frame.frame.frame is not None:
                            old_frame.frame.frame = None

                # Update previous frames
                previous_frames.append(video_frame)
                if len(previous_frames) > 5:
                    previous_frames.pop(0)

                # Yield control to event loop
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"Stream processing cancelled for {stream_id}")
        except Exception as e:
            logger.error(f"Error processing stream frames for {stream_id}: {e}")
        finally:
            # Cleanup
            if stream_id in self.active_streams:
                self.active_streams.remove(stream_id)

    async def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get processing statistics.

        Returns:
            Dictionary of processing statistics
        """
        stats = self.processing_stats.copy()

        # Calculate derived metrics
        if stats["total_frames_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_frames_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0

        stats["active_streams"] = len(self.active_streams)
        stats["buffer_size"] = len(self.frame_buffer)

        return stats

    async def optimize_for_profile(self, profile: str) -> None:
        """
        Optimize processor settings for a specific profile.

        Args:
            profile: Processing profile (fast, balanced, quality)
        """
        if profile == "fast":
            self.config.frame_interval_seconds = 2.0
            self.config.quality_threshold = 0.2
            self.config.enable_quality_analysis = False
            self.config.resize_width = 480
        elif profile == "balanced":
            self.config.frame_interval_seconds = 1.0
            self.config.quality_threshold = 0.3
            self.config.enable_quality_analysis = True
            self.config.resize_width = 720
        elif profile == "quality":
            self.config.frame_interval_seconds = 0.5
            self.config.quality_threshold = 0.5
            self.config.enable_quality_analysis = True
            self.config.resize_width = 1080

        self.config.processing_profile = profile
        logger.info(f"Optimized video processor for {profile} profile")

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up video processor")

        # Stop all streams
        for stream_id in list(self.active_streams):
            await self.stop_stream_processing(stream_id)

        # Cancel cleanup tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()

        # Clear buffers
        self.frame_buffer.clear()

        logger.info("Video processor cleanup completed")

    def __del__(self):
        """Cleanup on deletion."""
        # Schedule cleanup if event loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running
            pass


# Global video processor instance
video_processor = VideoProcessor()
