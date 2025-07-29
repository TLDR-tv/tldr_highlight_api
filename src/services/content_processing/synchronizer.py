"""
Content synchronization service for multi-modal data alignment.

This module provides precise timestamp synchronization across video frames,
audio transcriptions, and chat messages for coherent multi-modal analysis.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import bisect

from pydantic import BaseModel, Field
import numpy as np

from .video_processor import ProcessedFrame
from .audio_processor import ProcessedAudio
from src.utils.nlp_utils import TextAnalysis

logger = logging.getLogger(__name__)


class SynchronizationConfig(BaseModel):
    """Configuration for content synchronization."""

    # Timing tolerances
    sync_window_seconds: float = Field(
        default=2.0, description="Maximum time difference for content synchronization"
    )
    audio_sync_tolerance: float = Field(
        default=0.5, description="Audio synchronization tolerance in seconds"
    )
    chat_sync_tolerance: float = Field(
        default=1.0, description="Chat synchronization tolerance in seconds"
    )

    # Buffer settings
    max_buffer_size: int = Field(
        default=1000, description="Maximum items in synchronization buffer"
    )
    max_age_seconds: float = Field(
        default=300.0, description="Maximum age of items in buffer before cleanup"
    )

    # Synchronization strategy
    strategy: str = Field(
        default="timestamp",
        description="Synchronization strategy: timestamp, sequence, or hybrid",
    )
    interpolation_enabled: bool = Field(
        default=True, description="Enable interpolation for missing data points"
    )

    # Quality settings
    min_confidence_threshold: float = Field(
        default=0.3, description="Minimum confidence for including content in sync"
    )
    enable_drift_correction: bool = Field(
        default=True, description="Enable automatic drift correction"
    )

    # Processing windows
    processing_window_seconds: float = Field(
        default=30.0, description="Duration of synchronized processing windows"
    )
    window_overlap_seconds: float = Field(
        default=5.0, description="Overlap between processing windows"
    )


@dataclass
class SyncPoint:
    """A synchronized point in time with multi-modal content."""

    timestamp: float
    video_frame: Optional[ProcessedFrame] = None
    audio_transcription: Optional[ProcessedAudio] = None
    chat_messages: List[TextAnalysis] = field(default_factory=list)
    confidence_score: float = 0.0
    interpolated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronizedWindow:
    """A synchronized window of multi-modal content."""

    start_time: float
    end_time: float
    duration: float
    sync_points: List[SyncPoint]
    video_frames: List[ProcessedFrame]
    audio_transcriptions: List[ProcessedAudio]
    chat_data: List[TextAnalysis]

    # Aggregate metrics
    total_sync_points: int
    average_confidence: float
    interpolated_points: int

    # Quality metrics
    video_coverage: float  # Percentage of window with video data
    audio_coverage: float  # Percentage of window with audio data
    chat_coverage: float  # Percentage of window with chat data
    sync_quality_score: float

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentBuffer:
    """Buffer for storing content before synchronization."""

    video_frames: deque
    audio_transcriptions: deque
    chat_messages: deque
    last_cleanup: float

    def __post_init__(self):
        if not hasattr(self, "last_cleanup"):
            self.last_cleanup = time.time()


@dataclass
class DriftCorrection:
    """Drift correction state for timestamp alignment."""

    video_offset: float = 0.0
    audio_offset: float = 0.0
    chat_offset: float = 0.0
    last_correction: float = 0.0
    correction_history: List[Tuple[float, float, float, float]] = field(
        default_factory=list
    )


class ContentSynchronizer:
    """
    Advanced content synchronizer for multi-modal data alignment.

    Features:
    - Precise timestamp-based synchronization
    - Automatic drift correction
    - Content interpolation for missing data
    - Quality-based filtering
    - Configurable synchronization windows
    - Real-time and batch processing modes
    """

    def __init__(self, config: Optional[SynchronizationConfig] = None):
        self.config = config or SynchronizationConfig()

        # Content buffers
        self.content_buffer = ContentBuffer(
            video_frames=deque(maxlen=self.config.max_buffer_size),
            audio_transcriptions=deque(maxlen=self.config.max_buffer_size),
            chat_messages=deque(maxlen=self.config.max_buffer_size),
            last_cleanup=time.time(),
        )

        # Synchronization state
        self.synchronized_windows: deque = deque(maxlen=100)
        self.drift_correction = DriftCorrection()
        self.last_sync_time = 0.0

        # Performance tracking
        self.sync_stats = {
            "total_windows_processed": 0,
            "total_sync_points_created": 0,
            "total_processing_time": 0.0,
            "interpolated_points": 0,
            "drift_corrections": 0,
            "average_sync_quality": 0.0,
        }

        # Processing locks
        self._sync_lock = asyncio.Lock()
        self._buffer_lock = asyncio.Lock()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        try:
            loop = asyncio.get_running_loop()
            # Start cleanup task
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running yet
            self._cleanup_task = None

    async def add_video_frame(self, processed_frame: ProcessedFrame) -> None:
        """
        Add a processed video frame to the synchronization buffer.

        Args:
            processed_frame: Processed video frame with timestamp
        """
        async with self._buffer_lock:
            # Apply drift correction
            corrected_timestamp = (
                processed_frame.frame.timestamp + self.drift_correction.video_offset
            )

            # Create a copy with corrected timestamp
            corrected_frame = ProcessedFrame(
                frame=processed_frame.frame,
                analysis=processed_frame.analysis,
                processing_time=processed_frame.processing_time,
                is_keyframe=processed_frame.is_keyframe,
                scene_change=processed_frame.scene_change,
            )
            corrected_frame.frame.timestamp = corrected_timestamp

            # Insert in sorted order
            self._insert_sorted(
                self.content_buffer.video_frames,
                corrected_frame,
                lambda x: x.frame.timestamp,
            )

            logger.debug(f"Added video frame at {corrected_timestamp:.2f}s")

    async def add_audio_transcription(self, processed_audio: ProcessedAudio) -> None:
        """
        Add a processed audio transcription to the synchronization buffer.

        Args:
            processed_audio: Processed audio with transcription
        """
        async with self._buffer_lock:
            # Apply drift correction
            corrected_timestamp = (
                processed_audio.chunk.timestamp + self.drift_correction.audio_offset
            )

            # Create a copy with corrected timestamp
            corrected_audio = ProcessedAudio(
                chunk=processed_audio.chunk,
                transcription=processed_audio.transcription,
                analysis=processed_audio.analysis,
                processing_time=processed_audio.processing_time,
                error=processed_audio.error,
            )
            corrected_audio.chunk.timestamp = corrected_timestamp

            # Insert in sorted order
            self._insert_sorted(
                self.content_buffer.audio_transcriptions,
                corrected_audio,
                lambda x: x.chunk.timestamp,
            )

            logger.debug(f"Added audio transcription at {corrected_timestamp:.2f}s")

    async def add_chat_messages(self, chat_analyses: List[TextAnalysis]) -> None:
        """
        Add processed chat messages to the synchronization buffer.

        Args:
            chat_analyses: List of processed chat message analyses
        """
        async with self._buffer_lock:
            for analysis in chat_analyses:
                # Apply drift correction
                corrected_timestamp = (
                    analysis.timestamp + self.drift_correction.chat_offset
                )

                # Create a copy with corrected timestamp
                corrected_analysis = TextAnalysis(
                    text=analysis.text,
                    timestamp=corrected_timestamp,
                    sentiment=analysis.sentiment,
                    emotion=analysis.emotion,
                    keywords=analysis.keywords,
                    topics=analysis.topics,
                    language=analysis.language,
                    word_count=analysis.word_count,
                    toxicity_score=analysis.toxicity_score,
                )

                # Insert in sorted order
                self._insert_sorted(
                    self.content_buffer.chat_messages,
                    corrected_analysis,
                    lambda x: x.timestamp,
                )

            logger.debug(f"Added {len(chat_analyses)} chat messages")

    async def synchronize_window(
        self, start_time: float, duration: Optional[float] = None
    ) -> SynchronizedWindow:
        """
        Synchronize content within a specific time window.

        Args:
            start_time: Start time of the window
            duration: Duration of the window (uses config default if None)

        Returns:
            SynchronizedWindow with synchronized content
        """
        if duration is None:
            duration = self.config.processing_window_seconds

        end_time = start_time + duration

        async with self._sync_lock:
            start_processing = time.time()

            try:
                logger.info(f"Synchronizing window {start_time:.2f}-{end_time:.2f}s")

                # Extract content for the window
                window_video = self._extract_content_in_window(
                    self.content_buffer.video_frames,
                    start_time,
                    end_time,
                    lambda x: x.frame.timestamp,
                )
                window_audio = self._extract_content_in_window(
                    self.content_buffer.audio_transcriptions,
                    start_time,
                    end_time,
                    lambda x: x.chunk.timestamp,
                )
                window_chat = self._extract_content_in_window(
                    self.content_buffer.chat_messages,
                    start_time,
                    end_time,
                    lambda x: x.timestamp,
                )

                # Create synchronization points
                sync_points = await self._create_sync_points(
                    window_video, window_audio, window_chat, start_time, end_time
                )

                # Calculate coverage metrics
                video_coverage = self._calculate_coverage(
                    window_video, start_time, end_time, lambda x: x.frame.timestamp
                )
                audio_coverage = self._calculate_coverage(
                    window_audio, start_time, end_time, lambda x: x.chunk.timestamp
                )
                chat_coverage = self._calculate_coverage(
                    window_chat, start_time, end_time, lambda x: x.timestamp
                )

                # Calculate quality metrics
                total_sync_points = len(sync_points)
                interpolated_points = sum(1 for sp in sync_points if sp.interpolated)
                average_confidence = (
                    np.mean([sp.confidence_score for sp in sync_points])
                    if sync_points
                    else 0.0
                )

                sync_quality_score = self._calculate_sync_quality(
                    video_coverage, audio_coverage, chat_coverage, average_confidence
                )

                # Create synchronized window
                synchronized_window = SynchronizedWindow(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    sync_points=sync_points,
                    video_frames=window_video,
                    audio_transcriptions=window_audio,
                    chat_data=window_chat,
                    total_sync_points=total_sync_points,
                    average_confidence=average_confidence,
                    interpolated_points=interpolated_points,
                    video_coverage=video_coverage,
                    audio_coverage=audio_coverage,
                    chat_coverage=chat_coverage,
                    sync_quality_score=sync_quality_score,
                    metadata={
                        "processing_time": time.time() - start_processing,
                        "sync_strategy": self.config.strategy,
                        "drift_corrected": self.config.enable_drift_correction,
                    },
                )

                # Update stats
                processing_time = time.time() - start_processing
                self.sync_stats["total_windows_processed"] += 1
                self.sync_stats["total_sync_points_created"] += total_sync_points
                self.sync_stats["total_processing_time"] += processing_time
                self.sync_stats["interpolated_points"] += interpolated_points
                self.sync_stats["average_sync_quality"] = (
                    self.sync_stats["average_sync_quality"]
                    * (self.sync_stats["total_windows_processed"] - 1)
                    + sync_quality_score
                ) / self.sync_stats["total_windows_processed"]

                # Store window
                self.synchronized_windows.append(synchronized_window)
                self.last_sync_time = end_time

                logger.info(
                    f"Window synchronized: {total_sync_points} sync points, "
                    f"quality: {sync_quality_score:.2f}, {processing_time:.2f}s"
                )

                return synchronized_window

            except Exception as e:
                logger.error(f"Error synchronizing window {start_time}-{end_time}: {e}")
                raise

    async def synchronize_realtime(self) -> Optional[SynchronizedWindow]:
        """
        Synchronize the most recent content window for real-time processing.

        Returns:
            SynchronizedWindow for the most recent complete window, or None
        """
        current_time = time.time()

        # Calculate window boundaries
        window_start = current_time - self.config.processing_window_seconds

        # Only synchronize if we have enough data
        if not self._has_sufficient_content(window_start, current_time):
            return None

        return await self.synchronize_window(window_start)

    async def _create_sync_points(
        self,
        video_frames: List[ProcessedFrame],
        audio_transcriptions: List[ProcessedAudio],
        chat_messages: List[TextAnalysis],
        start_time: float,
        end_time: float,
    ) -> List[SyncPoint]:
        """
        Create synchronized points from multi-modal content.

        Args:
            video_frames: Video frames in window
            audio_transcriptions: Audio transcriptions in window
            chat_messages: Chat messages in window
            start_time: Window start time
            end_time: Window end time

        Returns:
            List of synchronized points
        """
        sync_points = []

        try:
            # Collect all timestamps
            all_timestamps = set()

            for frame in video_frames:
                all_timestamps.add(frame.frame.timestamp)

            for audio in audio_transcriptions:
                all_timestamps.add(audio.chunk.timestamp)

            for chat in chat_messages:
                all_timestamps.add(chat.timestamp)

            # Sort timestamps
            sorted_timestamps = sorted(all_timestamps)

            # Create sync points
            for timestamp in sorted_timestamps:
                if start_time <= timestamp <= end_time:
                    sync_point = await self._create_sync_point_at_timestamp(
                        timestamp, video_frames, audio_transcriptions, chat_messages
                    )

                    # Filter by confidence
                    if (
                        sync_point.confidence_score
                        >= self.config.min_confidence_threshold
                    ):
                        sync_points.append(sync_point)

            # Add interpolated points if enabled
            if (
                self.config.interpolation_enabled and len(sync_points) < 10
            ):  # Sparse data
                interpolated_points = await self._interpolate_sync_points(
                    sync_points, start_time, end_time
                )
                sync_points.extend(interpolated_points)
                sync_points.sort(key=lambda x: x.timestamp)

            return sync_points

        except Exception as e:
            logger.error(f"Error creating sync points: {e}")
            return []

    async def _create_sync_point_at_timestamp(
        self,
        timestamp: float,
        video_frames: List[ProcessedFrame],
        audio_transcriptions: List[ProcessedAudio],
        chat_messages: List[TextAnalysis],
    ) -> SyncPoint:
        """Create a sync point at a specific timestamp."""
        # Find closest content for each modality
        closest_video = self._find_closest_content(
            video_frames,
            timestamp,
            lambda x: x.frame.timestamp,
            self.config.sync_window_seconds,
        )

        closest_audio = self._find_closest_content(
            audio_transcriptions,
            timestamp,
            lambda x: x.chunk.timestamp,
            self.config.audio_sync_tolerance,
        )

        # Find chat messages within tolerance
        sync_chat_messages = [
            chat
            for chat in chat_messages
            if abs(chat.timestamp - timestamp) <= self.config.chat_sync_tolerance
        ]

        # Calculate confidence score
        confidence_score = self._calculate_sync_confidence(
            closest_video, closest_audio, sync_chat_messages, timestamp
        )

        return SyncPoint(
            timestamp=timestamp,
            video_frame=closest_video,
            audio_transcription=closest_audio,
            chat_messages=sync_chat_messages,
            confidence_score=confidence_score,
            interpolated=False,
            metadata={
                "video_offset": abs(closest_video.frame.timestamp - timestamp)
                if closest_video
                else None,
                "audio_offset": abs(closest_audio.chunk.timestamp - timestamp)
                if closest_audio
                else None,
                "chat_count": len(sync_chat_messages),
            },
        )

    async def _interpolate_sync_points(
        self, existing_points: List[SyncPoint], start_time: float, end_time: float
    ) -> List[SyncPoint]:
        """Create interpolated sync points for sparse data."""
        interpolated_points = []

        if len(existing_points) < 2:
            return interpolated_points

        try:
            # Calculate target interval
            target_interval = (
                end_time - start_time
            ) / 20  # Aim for 20 points per window

            for i in range(len(existing_points) - 1):
                current_point = existing_points[i]
                next_point = existing_points[i + 1]

                time_gap = next_point.timestamp - current_point.timestamp

                if time_gap > target_interval * 2:  # Large gap, interpolate
                    num_interpolated = int(time_gap / target_interval) - 1

                    for j in range(1, num_interpolated + 1):
                        interp_timestamp = current_point.timestamp + (
                            time_gap * j / (num_interpolated + 1)
                        )

                        # Create interpolated point
                        interp_point = SyncPoint(
                            timestamp=interp_timestamp,
                            video_frame=None,  # No actual content
                            audio_transcription=None,
                            chat_messages=[],
                            confidence_score=0.3,  # Lower confidence for interpolated
                            interpolated=True,
                            metadata={
                                "interpolated_between": [
                                    current_point.timestamp,
                                    next_point.timestamp,
                                ]
                            },
                        )

                        interpolated_points.append(interp_point)

            return interpolated_points

        except Exception as e:
            logger.error(f"Error interpolating sync points: {e}")
            return []

    def _find_closest_content(
        self,
        content_list: List,
        target_timestamp: float,
        timestamp_func,
        tolerance: float,
    ):
        """Find content closest to target timestamp within tolerance."""
        if not content_list:
            return None

        closest_content = None
        min_distance = float("inf")

        for content in content_list:
            content_timestamp = timestamp_func(content)
            distance = abs(content_timestamp - target_timestamp)

            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                closest_content = content

        return closest_content

    def _calculate_sync_confidence(
        self,
        video_frame: Optional[ProcessedFrame],
        audio_transcription: Optional[ProcessedAudio],
        chat_messages: List[TextAnalysis],
        timestamp: float,
    ) -> float:
        """Calculate confidence score for a sync point."""
        confidence = 0.0

        # Video contribution
        if video_frame:
            video_confidence = video_frame.frame.quality_score * 0.4
            confidence += video_confidence

        # Audio contribution
        if audio_transcription and audio_transcription.transcription:
            audio_confidence = audio_transcription.transcription.confidence * 0.4
            confidence += audio_confidence

        # Chat contribution
        if chat_messages:
            chat_confidence = (
                min(len(chat_messages) / 5.0, 1.0) * 0.2
            )  # More messages = higher confidence
            confidence += chat_confidence

        return min(confidence, 1.0)

    def _calculate_coverage(
        self, content_list: List, start_time: float, end_time: float, timestamp_func
    ) -> float:
        """Calculate temporal coverage of content in window."""
        if not content_list:
            return 0.0

        window_duration = end_time - start_time
        if window_duration <= 0:
            return 0.0

        # Create coverage intervals
        covered_intervals = []

        for content in content_list:
            content_timestamp = timestamp_func(content)
            if start_time <= content_timestamp <= end_time:
                # Assume each content item covers a small interval
                interval_start = max(content_timestamp - 0.5, start_time)
                interval_end = min(content_timestamp + 0.5, end_time)
                covered_intervals.append((interval_start, interval_end))

        # Merge overlapping intervals
        if not covered_intervals:
            return 0.0

        covered_intervals.sort()
        merged_intervals = [covered_intervals[0]]

        for start, end in covered_intervals[1:]:
            if start <= merged_intervals[-1][1]:
                # Overlapping, merge
                merged_intervals[-1] = (
                    merged_intervals[-1][0],
                    max(merged_intervals[-1][1], end),
                )
            else:
                # Non-overlapping, add new interval
                merged_intervals.append((start, end))

        # Calculate total covered duration
        total_covered = sum(end - start for start, end in merged_intervals)
        return min(total_covered / window_duration, 1.0)

    def _calculate_sync_quality(
        self,
        video_coverage: float,
        audio_coverage: float,
        chat_coverage: float,
        average_confidence: float,
    ) -> float:
        """Calculate overall synchronization quality score."""
        # Weighted average of coverage and confidence
        coverage_score = (
            video_coverage * 0.4 + audio_coverage * 0.4 + chat_coverage * 0.2
        )
        quality_score = coverage_score * 0.6 + average_confidence * 0.4
        return min(quality_score, 1.0)

    def _extract_content_in_window(
        self, content_deque: deque, start_time: float, end_time: float, timestamp_func
    ) -> List:
        """Extract content within a time window."""
        result = []

        for content in content_deque:
            content_timestamp = timestamp_func(content)
            if start_time <= content_timestamp <= end_time:
                result.append(content)

        return result

    def _has_sufficient_content(self, start_time: float, end_time: float) -> bool:
        """Check if there's sufficient content for synchronization."""
        video_count = sum(
            1
            for frame in self.content_buffer.video_frames
            if start_time <= frame.frame.timestamp <= end_time
        )

        audio_count = sum(
            1
            for audio in self.content_buffer.audio_transcriptions
            if start_time <= audio.chunk.timestamp <= end_time
        )

        chat_count = sum(
            1
            for chat in self.content_buffer.chat_messages
            if start_time <= chat.timestamp <= end_time
        )

        # Require at least some content in each modality
        return video_count >= 1 or audio_count >= 1 or chat_count >= 5

    def _insert_sorted(self, deque_obj: deque, item, key_func):
        """Insert item into deque maintaining sorted order."""
        # Convert to list, insert, and reconstruct deque
        items_list = list(deque_obj)

        # Find insertion point
        timestamps = [key_func(x) for x in items_list]
        insertion_point = bisect.bisect_left(timestamps, key_func(item))

        # Insert item
        items_list.insert(insertion_point, item)

        # Reconstruct deque
        deque_obj.clear()
        deque_obj.extend(items_list)

    async def _periodic_cleanup(self):
        """Periodic cleanup of old content."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_old_content()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _cleanup_old_content(self):
        """Remove old content from buffers."""
        current_time = time.time()
        cutoff_time = current_time - self.config.max_age_seconds

        async with self._buffer_lock:
            # Clean video frames
            while (
                self.content_buffer.video_frames
                and self.content_buffer.video_frames[0].frame.timestamp < cutoff_time
            ):
                self.content_buffer.video_frames.popleft()

            # Clean audio transcriptions
            while (
                self.content_buffer.audio_transcriptions
                and self.content_buffer.audio_transcriptions[0].chunk.timestamp
                < cutoff_time
            ):
                self.content_buffer.audio_transcriptions.popleft()

            # Clean chat messages
            while (
                self.content_buffer.chat_messages
                and self.content_buffer.chat_messages[0].timestamp < cutoff_time
            ):
                self.content_buffer.chat_messages.popleft()

            self.content_buffer.last_cleanup = current_time

        logger.debug("Completed periodic cleanup of old content")

    async def detect_and_correct_drift(self):
        """Detect and correct timestamp drift between modalities."""
        if not self.config.enable_drift_correction:
            return

        try:
            # Analyze recent sync points for drift patterns
            recent_windows = list(self.synchronized_windows)[-5:]  # Last 5 windows

            if len(recent_windows) < 3:
                return  # Need more data

            # Calculate average offsets for each modality
            video_offsets = []
            audio_offsets = []

            for window in recent_windows:
                for sync_point in window.sync_points:
                    if sync_point.video_frame and sync_point.metadata.get(
                        "video_offset"
                    ):
                        video_offsets.append(sync_point.metadata["video_offset"])

                    if sync_point.audio_transcription and sync_point.metadata.get(
                        "audio_offset"
                    ):
                        audio_offsets.append(sync_point.metadata["audio_offset"])

            # Detect systematic drift
            if video_offsets:
                avg_video_offset = np.mean(video_offsets)
                if abs(avg_video_offset) > 0.5:  # Significant drift
                    self.drift_correction.video_offset -= avg_video_offset
                    logger.info(
                        f"Applied video drift correction: {-avg_video_offset:.2f}s"
                    )
                    self.sync_stats["drift_corrections"] += 1

            if audio_offsets:
                avg_audio_offset = np.mean(audio_offsets)
                if abs(avg_audio_offset) > 0.3:  # Significant drift
                    self.drift_correction.audio_offset -= avg_audio_offset
                    logger.info(
                        f"Applied audio drift correction: {-avg_audio_offset:.2f}s"
                    )
                    self.sync_stats["drift_corrections"] += 1

            # Record correction
            self.drift_correction.last_correction = time.time()
            self.drift_correction.correction_history.append(
                (
                    time.time(),
                    self.drift_correction.video_offset,
                    self.drift_correction.audio_offset,
                    self.drift_correction.chat_offset,
                )
            )

            # Keep only recent history
            if len(self.drift_correction.correction_history) > 20:
                self.drift_correction.correction_history = (
                    self.drift_correction.correction_history[-20:]
                )

        except Exception as e:
            logger.error(f"Error in drift correction: {e}")

    async def get_recent_windows(self, limit: int = 10) -> List[SynchronizedWindow]:
        """Get recently synchronized windows."""
        return list(self.synchronized_windows)[-limit:]

    async def get_sync_stats(self) -> Dict[str, Union[int, float]]:
        """Get synchronization statistics."""
        stats = self.sync_stats.copy()

        # Add derived metrics
        if stats["total_windows_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_windows_processed"]
            )
            stats["average_sync_points_per_window"] = (
                stats["total_sync_points_created"] / stats["total_windows_processed"]
            )
            stats["interpolation_rate"] = (
                stats["interpolated_points"] / stats["total_sync_points_created"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["average_sync_points_per_window"] = 0.0
            stats["interpolation_rate"] = 0.0

        # Add buffer status
        stats["video_buffer_size"] = len(self.content_buffer.video_frames)
        stats["audio_buffer_size"] = len(self.content_buffer.audio_transcriptions)
        stats["chat_buffer_size"] = len(self.content_buffer.chat_messages)
        stats["synchronized_windows"] = len(self.synchronized_windows)

        return stats

    async def cleanup(self):
        """Clean up synchronizer resources."""
        logger.info("Cleaning up content synchronizer")

        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear buffers
        self.content_buffer.video_frames.clear()
        self.content_buffer.audio_transcriptions.clear()
        self.content_buffer.chat_messages.clear()
        self.synchronized_windows.clear()

        logger.info("Content synchronizer cleanup completed")


# Global content synchronizer instance
content_synchronizer = ContentSynchronizer()
