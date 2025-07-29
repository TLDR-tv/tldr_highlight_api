"""
Buffer management service for processing windows and memory optimization.

This module provides efficient buffering and windowing for multi-modal content
processing with configurable sliding windows, memory management, and performance
optimization for real-time streaming scenarios.
"""

import asyncio
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Any,
)
from enum import Enum

from pydantic import BaseModel, Field
import numpy as np

from .synchronizer import SynchronizedWindow, ContentSynchronizer
from .video_processor import ProcessedFrame
from .audio_processor import ProcessedAudio
from src.utils.nlp_utils import TextAnalysis

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Types of processing windows."""

    FIXED = "fixed"  # Fixed-size windows
    SLIDING = "sliding"  # Sliding windows with overlap
    ADAPTIVE = "adaptive"  # Adaptive size based on content
    EVENT_DRIVEN = "event"  # Triggered by specific events


class BufferPriority(Enum):
    """Buffer priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class BufferConfig(BaseModel):
    """Configuration for buffer management."""

    # Window settings
    window_duration_seconds: float = Field(
        default=30.0, description="Duration of processing windows in seconds"
    )
    window_overlap_seconds: float = Field(
        default=5.0, description="Overlap between consecutive windows"
    )
    window_type: WindowType = Field(
        default=WindowType.SLIDING, description="Type of windowing strategy"
    )

    # Memory management
    max_memory_mb: int = Field(
        default=1000, description="Maximum memory usage for buffers in MB"
    )
    max_windows_in_memory: int = Field(
        default=50, description="Maximum number of windows to keep in memory"
    )
    gc_threshold_ratio: float = Field(
        default=0.8, description="Memory usage ratio to trigger garbage collection"
    )

    # Buffer sizes
    video_buffer_size: int = Field(
        default=200, description="Maximum video frames in buffer"
    )
    audio_buffer_size: int = Field(
        default=100, description="Maximum audio chunks in buffer"
    )
    chat_buffer_size: int = Field(
        default=1000, description="Maximum chat messages in buffer"
    )

    # Processing settings
    processing_batch_size: int = Field(
        default=10, description="Number of items to process in each batch"
    )
    max_concurrent_windows: int = Field(
        default=5, description="Maximum concurrent window processing"
    )

    # Quality settings
    enable_content_filtering: bool = Field(
        default=True, description="Enable quality-based content filtering"
    )
    min_content_quality: float = Field(
        default=0.3, description="Minimum quality score for content inclusion"
    )

    # Adaptive settings
    adaptive_min_duration: float = Field(
        default=10.0, description="Minimum window duration for adaptive mode"
    )
    adaptive_max_duration: float = Field(
        default=120.0, description="Maximum window duration for adaptive mode"
    )
    content_density_threshold: float = Field(
        default=0.5, description="Content density threshold for adaptive windows"
    )


@dataclass
class ProcessingWindow:
    """A processing window with multi-modal content."""

    window_id: str
    start_time: float
    end_time: float
    duration: float
    window_type: WindowType
    priority: BufferPriority = BufferPriority.MEDIUM

    # Content
    video_frames: List[ProcessedFrame] = field(default_factory=list)
    audio_transcriptions: List[ProcessedAudio] = field(default_factory=list)
    chat_messages: List[TextAnalysis] = field(default_factory=list)
    synchronized_data: Optional[SynchronizedWindow] = None

    # Processing state
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    is_processing: bool = False
    is_complete: bool = False

    # Metrics
    content_density: float = 0.0
    quality_score: float = 0.0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BufferMetrics:
    """Buffer performance and usage metrics."""

    timestamp: float

    # Memory usage
    total_memory_mb: float
    video_memory_mb: float
    audio_memory_mb: float
    chat_memory_mb: float
    window_memory_mb: float

    # Buffer sizes
    video_buffer_count: int
    audio_buffer_count: int
    chat_buffer_count: int
    window_buffer_count: int

    # Processing metrics
    windows_processed: int
    average_processing_time: float
    processing_throughput: float  # windows per second

    # Quality metrics
    average_content_density: float
    average_quality_score: float
    dropped_content_ratio: float


class BufferManager:
    """
    Advanced buffer manager for multi-modal content processing.

    Features:
    - Configurable sliding window processing
    - Memory-optimized buffering with automatic cleanup
    - Priority-based content management
    - Adaptive window sizing based on content density
    - Real-time performance monitoring
    - Efficient batch processing
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()

        # Processing windows
        self.active_windows: Dict[str, ProcessingWindow] = {}
        self.completed_windows: deque = deque(maxlen=self.config.max_windows_in_memory)
        self.window_counter = 0

        # Content buffers by priority
        self.video_buffer: Dict[BufferPriority, deque] = {
            priority: deque(maxlen=self.config.video_buffer_size // len(BufferPriority))
            for priority in BufferPriority
        }
        self.audio_buffer: Dict[BufferPriority, deque] = {
            priority: deque(maxlen=self.config.audio_buffer_size // len(BufferPriority))
            for priority in BufferPriority
        }
        self.chat_buffer: Dict[BufferPriority, deque] = {
            priority: deque(maxlen=self.config.chat_buffer_size // len(BufferPriority))
            for priority in BufferPriority
        }

        # Processing state
        self.last_window_start = 0.0
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_windows
        )

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=100)
        self.processing_stats = {
            "total_windows_created": 0,
            "total_windows_processed": 0,
            "total_processing_time": 0.0,
            "total_content_dropped": 0,
            "memory_cleanups": 0,
            "gc_triggers": 0,
        }

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._processing_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()

        # Weak references for memory management
        self._tracked_objects = []

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background processing tasks."""
        try:
            loop = asyncio.get_running_loop()

            # Window processing task
            task = loop.create_task(self._window_processor())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Memory management task
            task = loop.create_task(self._memory_manager())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Metrics collection task
            task = loop.create_task(self._metrics_collector())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        except RuntimeError:
            # No event loop running yet
            pass

    async def add_video_frame(
        self, frame: ProcessedFrame, priority: BufferPriority = BufferPriority.MEDIUM
    ) -> None:
        """
        Add a video frame to the buffer.

        Args:
            frame: Processed video frame
            priority: Buffer priority level
        """
        if self.config.enable_content_filtering:
            if frame.frame.quality_score < self.config.min_content_quality:
                logger.debug(
                    f"Dropping low quality video frame (quality: {frame.frame.quality_score:.2f})"
                )
                self.processing_stats["total_content_dropped"] += 1
                return

        # Add to appropriate buffer
        self.video_buffer[priority].append(frame)
        self._tracked_objects.append(frame)

        # Check if we need to create a new window
        await self._check_window_creation(frame.frame.timestamp)

        logger.debug(
            f"Added video frame at {frame.frame.timestamp:.2f}s (priority: {priority.name})"
        )

    async def add_audio_transcription(
        self, audio: ProcessedAudio, priority: BufferPriority = BufferPriority.MEDIUM
    ) -> None:
        """
        Add an audio transcription to the buffer.

        Args:
            audio: Processed audio with transcription
            priority: Buffer priority level
        """
        if self.config.enable_content_filtering:
            if (
                audio.transcription
                and audio.transcription.confidence < self.config.min_content_quality
            ):
                logger.debug(
                    f"Dropping low confidence audio transcription (confidence: {audio.transcription.confidence:.2f})"
                )
                self.processing_stats["total_content_dropped"] += 1
                return

        # Add to appropriate buffer
        self.audio_buffer[priority].append(audio)
        self._tracked_objects.append(audio)

        # Check if we need to create a new window
        await self._check_window_creation(audio.chunk.timestamp)

        logger.debug(
            f"Added audio transcription at {audio.chunk.timestamp:.2f}s (priority: {priority.name})"
        )

    async def add_chat_messages(
        self,
        messages: List[TextAnalysis],
        priority: BufferPriority = BufferPriority.LOW,
    ) -> None:
        """
        Add chat messages to the buffer.

        Args:
            messages: List of processed chat messages
            priority: Buffer priority level
        """
        filtered_messages = []

        for message in messages:
            if self.config.enable_content_filtering:
                if message.toxicity_score > 0.8:  # Filter highly toxic messages
                    logger.debug(
                        f"Dropping toxic chat message (toxicity: {message.toxicity_score:.2f})"
                    )
                    self.processing_stats["total_content_dropped"] += 1
                    continue

            filtered_messages.append(message)
            self._tracked_objects.append(message)

        # Add to appropriate buffer
        self.chat_buffer[priority].extend(filtered_messages)

        # Check if we need to create a new window (use latest message timestamp)
        if filtered_messages:
            latest_timestamp = max(msg.timestamp for msg in filtered_messages)
            await self._check_window_creation(latest_timestamp)

        logger.debug(
            f"Added {len(filtered_messages)} chat messages (priority: {priority.name})"
        )

    async def create_window(
        self,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        window_type: Optional[WindowType] = None,
        priority: BufferPriority = BufferPriority.MEDIUM,
    ) -> ProcessingWindow:
        """
        Create a new processing window.

        Args:
            start_time: Window start time (current time if None)
            duration: Window duration (config default if None)
            window_type: Window type (config default if None)
            priority: Window priority

        Returns:
            ProcessingWindow instance
        """
        if start_time is None:
            start_time = time.time() - self.config.window_duration_seconds

        if duration is None:
            duration = self.config.window_duration_seconds

        if window_type is None:
            window_type = self.config.window_type

        # Adaptive window sizing
        if window_type == WindowType.ADAPTIVE:
            duration = await self._calculate_adaptive_duration(start_time, duration)

        end_time = start_time + duration
        window_id = f"window_{self.window_counter}_{int(start_time)}"
        self.window_counter += 1

        # Create window
        window = ProcessingWindow(
            window_id=window_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            window_type=window_type,
            priority=priority,
        )

        # Populate window with content
        await self._populate_window(window)

        # Store window
        self.active_windows[window_id] = window
        self.processing_stats["total_windows_created"] += 1

        logger.info(
            f"Created {window_type.value} window {window_id}: {start_time:.2f}-{end_time:.2f}s"
        )

        return window

    async def process_window(self, window: ProcessingWindow) -> ProcessingWindow:
        """
        Process a window through the multi-modal pipeline.

        Args:
            window: Processing window to process

        Returns:
            Processed window with synchronized data
        """
        async with self.processing_semaphore:
            start_processing = time.time()
            window.is_processing = True

            try:
                logger.info(f"Processing window {window.window_id}")

                # Synchronize content using the content synchronizer
                synchronizer = ContentSynchronizer()

                # Add content to synchronizer
                for frame in window.video_frames:
                    await synchronizer.add_video_frame(frame)

                for audio in window.audio_transcriptions:
                    await synchronizer.add_audio_transcription(audio)

                if window.chat_messages:
                    await synchronizer.add_chat_messages(window.chat_messages)

                # Synchronize the window
                synchronized_window = await synchronizer.synchronize_window(
                    window.start_time, window.duration
                )

                window.synchronized_data = synchronized_window

                # Calculate metrics
                window.content_density = self._calculate_content_density(window)
                window.quality_score = synchronized_window.sync_quality_score
                window.processing_time = time.time() - start_processing
                window.memory_usage_mb = self._estimate_window_memory(window)

                # Mark as complete
                window.is_processing = False
                window.is_complete = True
                window.processed_at = time.time()

                # Update stats
                self.processing_stats["total_windows_processed"] += 1
                self.processing_stats["total_processing_time"] += window.processing_time

                logger.info(
                    f"Window {window.window_id} processed: "
                    f"density={window.content_density:.2f}, "
                    f"quality={window.quality_score:.2f}, "
                    f"time={window.processing_time:.2f}s"
                )

                return window

            except Exception as e:
                logger.error(f"Error processing window {window.window_id}: {e}")
                window.is_processing = False
                raise
            finally:
                # Move to completed windows
                if window.window_id in self.active_windows:
                    del self.active_windows[window.window_id]
                self.completed_windows.append(window)

    async def windows_ready_for_processing(
        self, limit: int = 10, priority: Optional[BufferPriority] = None
    ) -> List[ProcessingWindow]:
        """
        Get windows ready for processing.

        Args:
            limit: Maximum number of windows to return
            priority: Filter by priority level

        Returns:
            List of windows ready for processing
        """
        current_time = time.time()
        ready_windows = []

        for window in self.active_windows.values():
            if window.is_processing or window.is_complete:
                continue

            # Check if window is ready (end time has passed)
            if current_time >= window.end_time:
                if priority is None or window.priority == priority:
                    ready_windows.append(window)

        # Sort by priority and start time
        ready_windows.sort(key=lambda w: (w.priority.value, w.start_time), reverse=True)

        return ready_windows[:limit]

    async def process_realtime(self) -> AsyncGenerator[ProcessingWindow, None]:
        """
        Process windows in real-time as they become ready.

        Yields:
            Completed processing windows
        """
        while True:
            try:
                # Get ready windows
                ready_windows = await self.windows_ready_for_processing(
                    limit=self.config.max_concurrent_windows
                )

                if ready_windows:
                    # Process windows concurrently
                    tasks = [self.process_window(window) for window in ready_windows]
                    completed_windows = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )

                    for result in completed_windows:
                        if not isinstance(result, Exception):
                            yield result

                # Wait before checking again
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in real-time processing: {e}")
                await asyncio.sleep(5.0)

    async def _check_window_creation(self, content_timestamp: float) -> None:
        """Check if a new window should be created based on content timestamp."""
        if self.config.window_type == WindowType.FIXED:
            # Create windows at fixed intervals
            if (
                content_timestamp
                >= self.last_window_start + self.config.window_duration_seconds
            ):
                await self.create_window(
                    start_time=self.last_window_start
                    + self.config.window_duration_seconds
                )
                self.last_window_start += self.config.window_duration_seconds

        elif self.config.window_type == WindowType.SLIDING:
            # Create overlapping sliding windows
            if content_timestamp >= self.last_window_start + (
                self.config.window_duration_seconds - self.config.window_overlap_seconds
            ):
                new_start = self.last_window_start + (
                    self.config.window_duration_seconds
                    - self.config.window_overlap_seconds
                )
                await self.create_window(start_time=new_start)
                self.last_window_start = new_start

    async def _populate_window(self, window: ProcessingWindow) -> None:
        """Populate window with content from buffers."""
        # Collect content from all priority levels
        for priority in BufferPriority:
            # Video frames
            video_frames = [
                frame
                for frame in self.video_buffer[priority]
                if window.start_time <= frame.frame.timestamp <= window.end_time
            ]
            window.video_frames.extend(video_frames)

            # Audio transcriptions
            audio_transcriptions = [
                audio
                for audio in self.audio_buffer[priority]
                if window.start_time <= audio.chunk.timestamp <= window.end_time
            ]
            window.audio_transcriptions.extend(audio_transcriptions)

            # Chat messages
            chat_messages = [
                msg
                for msg in self.chat_buffer[priority]
                if window.start_time <= msg.timestamp <= window.end_time
            ]
            window.chat_messages.extend(chat_messages)

        # Sort content by timestamp
        window.video_frames.sort(key=lambda x: x.frame.timestamp)
        window.audio_transcriptions.sort(key=lambda x: x.chunk.timestamp)
        window.chat_messages.sort(key=lambda x: x.timestamp)

        logger.debug(
            f"Populated window {window.window_id}: "
            f"{len(window.video_frames)} video, "
            f"{len(window.audio_transcriptions)} audio, "
            f"{len(window.chat_messages)} chat"
        )

    async def _calculate_adaptive_duration(
        self, start_time: float, base_duration: float
    ) -> float:
        """Calculate adaptive window duration based on content density."""
        # Sample content in the time range
        sample_end = start_time + base_duration

        total_content = 0
        for priority in BufferPriority:
            total_content += sum(
                1
                for frame in self.video_buffer[priority]
                if start_time <= frame.frame.timestamp <= sample_end
            )
            total_content += sum(
                1
                for audio in self.audio_buffer[priority]
                if start_time <= audio.chunk.timestamp <= sample_end
            )
            total_content += sum(
                1
                for msg in self.chat_buffer[priority]
                if start_time <= msg.timestamp <= sample_end
            )

        # Calculate content density
        content_density = total_content / base_duration

        # Adjust duration based on density
        if content_density > self.config.content_density_threshold * 2:
            # High density, shorter window
            duration = max(self.config.adaptive_min_duration, base_duration * 0.7)
        elif content_density < self.config.content_density_threshold * 0.5:
            # Low density, longer window
            duration = min(self.config.adaptive_max_duration, base_duration * 1.5)
        else:
            duration = base_duration

        return duration

    def _calculate_content_density(self, window: ProcessingWindow) -> float:
        """Calculate content density for a window."""
        total_items = (
            len(window.video_frames)
            + len(window.audio_transcriptions)
            + len(window.chat_messages)
        )

        return total_items / window.duration if window.duration > 0 else 0.0

    def _estimate_window_memory(self, window: ProcessingWindow) -> float:
        """Estimate memory usage of a window in MB."""
        # Rough estimates based on typical content sizes
        video_memory = len(window.video_frames) * 0.5  # ~0.5MB per frame
        audio_memory = (
            len(window.audio_transcriptions) * 0.1
        )  # ~0.1MB per transcription
        chat_memory = len(window.chat_messages) * 0.001  # ~1KB per message

        return video_memory + audio_memory + chat_memory

    async def _window_processor(self):
        """Background task for processing windows."""
        while True:
            try:
                # Process ready windows
                ready_windows = await self.windows_ready_for_processing(limit=1)

                if ready_windows:
                    window = ready_windows[0]
                    await self.process_window(window)
                else:
                    await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in window processor: {e}")
                await asyncio.sleep(5.0)

    async def _memory_manager(self):
        """Background task for memory management."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_memory = await self._calculate_memory_usage()

                if (
                    current_memory
                    > self.config.max_memory_mb * self.config.gc_threshold_ratio
                ):
                    logger.info(
                        f"Memory usage high ({current_memory:.1f}MB), triggering cleanup"
                    )
                    await self._cleanup_memory()
                    self.processing_stats["gc_triggers"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory manager: {e}")

    async def _metrics_collector(self):
        """Background task for collecting performance metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute

                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                logger.debug(
                    f"Collected metrics: {metrics.windows_processed} windows processed"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")

    async def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in MB."""
        total_memory = 0.0

        # Calculate buffer memory
        for priority in BufferPriority:
            total_memory += len(self.video_buffer[priority]) * 0.5  # Video frames
            total_memory += len(self.audio_buffer[priority]) * 0.1  # Audio
            total_memory += len(self.chat_buffer[priority]) * 0.001  # Chat

        # Calculate window memory
        for window in self.active_windows.values():
            total_memory += self._estimate_window_memory(window)

        for window in self.completed_windows:
            total_memory += self._estimate_window_memory(window)

        return total_memory

    async def _cleanup_memory(self):
        """Clean up memory by removing old content."""
        async with self._memory_lock:
            current_time = time.time()
            cutoff_time = current_time - (
                self.config.window_duration_seconds * 3
            )  # Keep 3 windows worth

            # Clean buffers
            for priority in BufferPriority:
                # Clean video buffer
                while (
                    self.video_buffer[priority]
                    and self.video_buffer[priority][0].frame.timestamp < cutoff_time
                ):
                    self.video_buffer[priority].popleft()

                # Clean audio buffer
                while (
                    self.audio_buffer[priority]
                    and self.audio_buffer[priority][0].chunk.timestamp < cutoff_time
                ):
                    self.audio_buffer[priority].popleft()

                # Clean chat buffer
                while (
                    self.chat_buffer[priority]
                    and self.chat_buffer[priority][0].timestamp < cutoff_time
                ):
                    self.chat_buffer[priority].popleft()

            # Clean old completed windows
            while (
                self.completed_windows
                and self.completed_windows[0].end_time < cutoff_time
            ):
                self.completed_windows.popleft()

            self.processing_stats["memory_cleanups"] += 1

            logger.info("Memory cleanup completed")

    async def _collect_metrics(self) -> BufferMetrics:
        """Collect current buffer metrics."""
        current_time = time.time()

        # Calculate memory usage
        video_memory = sum(len(buf) * 0.5 for buf in self.video_buffer.values())
        audio_memory = sum(len(buf) * 0.1 for buf in self.audio_buffer.values())
        chat_memory = sum(len(buf) * 0.001 for buf in self.chat_buffer.values())
        window_memory = sum(
            self._estimate_window_memory(w) for w in self.active_windows.values()
        )
        total_memory = video_memory + audio_memory + chat_memory + window_memory

        # Calculate buffer counts
        video_count = sum(len(buf) for buf in self.video_buffer.values())
        audio_count = sum(len(buf) for buf in self.audio_buffer.values())
        chat_count = sum(len(buf) for buf in self.chat_buffer.values())
        window_count = len(self.active_windows)

        # Calculate processing metrics
        if self.processing_stats["total_windows_processed"] > 0:
            avg_processing_time = (
                self.processing_stats["total_processing_time"]
                / self.processing_stats["total_windows_processed"]
            )
        else:
            avg_processing_time = 0.0

        # Calculate throughput (windows per minute)
        recent_windows = [
            w
            for w in self.completed_windows
            if w.processed_at and current_time - w.processed_at <= 60
        ]
        throughput = len(recent_windows) / 60.0  # windows per second

        # Calculate quality metrics
        if self.completed_windows:
            avg_density = np.mean([w.content_density for w in self.completed_windows])
            avg_quality = np.mean([w.quality_score for w in self.completed_windows])
        else:
            avg_density = 0.0
            avg_quality = 0.0

        # Calculate dropped content ratio
        total_content = (
            self.processing_stats["total_windows_processed"]
            + self.processing_stats["total_content_dropped"]
        )
        dropped_ratio = self.processing_stats["total_content_dropped"] / max(
            total_content, 1
        )

        return BufferMetrics(
            timestamp=current_time,
            total_memory_mb=total_memory,
            video_memory_mb=video_memory,
            audio_memory_mb=audio_memory,
            chat_memory_mb=chat_memory,
            window_memory_mb=window_memory,
            video_buffer_count=video_count,
            audio_buffer_count=audio_count,
            chat_buffer_count=chat_count,
            window_buffer_count=window_count,
            windows_processed=self.processing_stats["total_windows_processed"],
            average_processing_time=avg_processing_time,
            processing_throughput=throughput,
            average_content_density=avg_density,
            average_quality_score=avg_quality,
            dropped_content_ratio=dropped_ratio,
        )

    async def get_recent_metrics(self, limit: int = 10) -> List[BufferMetrics]:
        """Get recent buffer metrics."""
        return list(self.metrics_history)[-limit:]

    async def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        # Add derived metrics
        if stats["total_windows_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_windows_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0

        # Add current state
        stats["active_windows"] = len(self.active_windows)
        stats["completed_windows"] = len(self.completed_windows)
        stats["tracked_objects"] = len(self._tracked_objects)

        return stats

    async def cleanup(self):
        """Clean up buffer manager resources."""
        logger.info("Cleaning up buffer manager")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Clear buffers
        for priority in BufferPriority:
            self.video_buffer[priority].clear()
            self.audio_buffer[priority].clear()
            self.chat_buffer[priority].clear()

        self.active_windows.clear()
        self.completed_windows.clear()
        self.metrics_history.clear()

        logger.info("Buffer manager cleanup completed")


# Global buffer manager instance
buffer_manager = BufferManager()
