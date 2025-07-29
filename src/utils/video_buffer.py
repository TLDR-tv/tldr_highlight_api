"""Unified video buffer implementation for multi-format stream buffering.

This module provides a high-performance, thread-safe video buffer that supports
HLS segments (YouTube/Twitch) and FLV frames (RTMP) with configurable retention
policies and memory management.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
)
from threading import RLock
import weakref

import numpy as np

logger = logging.getLogger(__name__)


class BufferFormat(str, Enum):
    """Supported video buffer formats."""
    
    HLS_SEGMENT = "hls_segment"  # MPEG-TS segments from HLS
    FLV_FRAME = "flv_frame"      # FLV tags from RTMP
    RAW_FRAME = "raw_frame"      # Raw decoded frames


class FrameType(str, Enum):
    """Video frame types."""
    
    I_FRAME = "i_frame"  # Keyframe/IDR frame
    P_FRAME = "p_frame"  # Predicted frame
    B_FRAME = "b_frame"  # Bidirectional predicted frame
    AUDIO = "audio"      # Audio frame
    METADATA = "metadata" # Metadata frame


@dataclass
class BufferConfig:
    """Configuration for video buffer."""
    
    # Memory limits
    max_memory_mb: int = 500
    max_items: int = 10000
    
    # Retention policies
    retention_seconds: float = 300.0  # 5 minutes
    min_retention_items: int = 100
    
    # Buffer behavior
    enable_keyframe_priority: bool = True
    enable_memory_pooling: bool = True
    enable_compression: bool = False
    
    # Performance tuning
    gc_interval_seconds: float = 30.0
    stats_interval_seconds: float = 60.0
    
    # Format-specific settings
    hls_segment_buffer_count: int = 20
    flv_frame_buffer_ms: int = 5000  # 5 seconds of frames
    
    # Thread safety
    enable_thread_safety: bool = True


@dataclass
class VideoFrame:
    """Represents a single video frame or segment."""
    
    # Core data
    data: bytes
    timestamp: float  # Seconds since epoch
    duration: float   # Duration in seconds
    
    # Frame information
    format: BufferFormat
    frame_type: FrameType
    is_keyframe: bool = False
    
    # Video properties
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    
    # Metadata
    sequence_number: Optional[int] = None
    pts: Optional[float] = None  # Presentation timestamp
    dts: Optional[float] = None  # Decode timestamp
    
    # Memory management
    size_bytes: int = field(init=False)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    # Quality metrics
    quality_score: float = 1.0
    importance_score: float = 1.0
    
    def __post_init__(self):
        """Calculate frame size after initialization."""
        self.size_bytes = len(self.data)
    
    def __lt__(self, other):
        """Compare frames by timestamp for sorting."""
        return self.timestamp < other.timestamp


@dataclass
class BufferSegment:
    """A segment of buffered video data."""
    
    segment_id: str
    start_time: float
    end_time: float
    frames: List[VideoFrame] = field(default_factory=list)
    
    # Segment properties
    total_size: int = 0
    frame_count: int = 0
    keyframe_count: int = 0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def add_frame(self, frame: VideoFrame) -> None:
        """Add a frame to the segment."""
        self.frames.append(frame)
        self.total_size += frame.size_bytes
        self.frame_count += 1
        if frame.is_keyframe:
            self.keyframe_count += 1
        self.last_accessed = time.time()
    
    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end_time - self.start_time
    
    @property
    def has_keyframe(self) -> bool:
        """Check if segment contains keyframes."""
        return self.keyframe_count > 0


class CircularVideoBuffer:
    """Thread-safe circular buffer for video data.
    
    Features:
    - Multi-format support (HLS, FLV, raw frames)
    - Circular buffer with configurable size limits
    - Memory-efficient storage with compression options
    - Thread-safe operations
    - Keyframe prioritization
    - Automatic garbage collection
    - Performance metrics
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """Initialize the video buffer."""
        self.config = config or BufferConfig()
        
        # Thread safety
        self._lock = RLock() if self.config.enable_thread_safety else None
        
        # Main storage
        self._frames: Deque[VideoFrame] = deque(maxlen=self.config.max_items)
        self._segments: Dict[str, BufferSegment] = {}
        self._keyframe_index: List[Tuple[float, int]] = []  # (timestamp, index)
        
        # Memory tracking
        self._total_memory_bytes = 0
        self._frame_count = 0
        self._dropped_frames = 0
        
        # Performance metrics
        self._stats = {
            "frames_added": 0,
            "frames_dropped": 0,
            "segments_created": 0,
            "segments_evicted": 0,
            "memory_cleanups": 0,
            "total_bytes_processed": 0,
            "avg_frame_size": 0,
            "keyframe_ratio": 0.0,
        }
        
        # Memory pools for reuse
        self._frame_pool: List[VideoFrame] = [] if self.config.enable_memory_pooling else None
        
        # Weak references for memory management
        self._weak_refs: List[weakref.ref] = []
        
        # Background tasks
        self._gc_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Initialized CircularVideoBuffer with max memory: {self.config.max_memory_mb}MB")
    
    def _with_lock(func):
        """Decorator for thread-safe operations."""
        def wrapper(self, *args, **kwargs):
            if self._lock:
                with self._lock:
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    
    @_with_lock
    def add_frame(self, frame: VideoFrame) -> bool:
        """Add a frame to the buffer.
        
        Args:
            frame: Video frame to add
            
        Returns:
            bool: True if frame was added, False if dropped
        """
        try:
            # Check memory limit
            if self._should_drop_frame(frame):
                self._stats["frames_dropped"] += 1
                self._dropped_frames += 1
                return False
            
            # Add to main buffer
            self._frames.append(frame)
            self._frame_count += 1
            self._total_memory_bytes += frame.size_bytes
            
            # Update keyframe index
            if frame.is_keyframe:
                self._keyframe_index.append((frame.timestamp, len(self._frames) - 1))
                # Keep index size reasonable
                if len(self._keyframe_index) > 1000:
                    self._keyframe_index = self._keyframe_index[-500:]
            
            # Update stats
            self._stats["frames_added"] += 1
            self._stats["total_bytes_processed"] += frame.size_bytes
            self._update_avg_frame_size()
            
            # Create weak reference
            if self.config.enable_memory_pooling:
                self._weak_refs.append(weakref.ref(frame))
            
            logger.debug(
                f"Added {frame.format.value} frame at {frame.timestamp:.2f}s "
                f"(keyframe: {frame.is_keyframe}, size: {frame.size_bytes} bytes)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding frame to buffer: {e}")
            return False
    
    @_with_lock
    def add_segment(self, segment_id: str, frames: List[VideoFrame]) -> bool:
        """Add a segment of frames to the buffer.
        
        Args:
            segment_id: Unique segment identifier
            frames: List of frames in the segment
            
        Returns:
            bool: True if segment was added successfully
        """
        if not frames:
            return False
        
        try:
            # Create segment
            start_time = min(f.timestamp for f in frames)
            end_time = max(f.timestamp + f.duration for f in frames)
            
            segment = BufferSegment(
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Add frames to segment and buffer
            for frame in sorted(frames, key=lambda f: f.timestamp):
                segment.add_frame(frame)
                self.add_frame(frame)
            
            # Store segment
            self._segments[segment_id] = segment
            self._stats["segments_created"] += 1
            
            # Evict old segments if needed
            self._evict_old_segments()
            
            logger.info(
                f"Added segment {segment_id} with {len(frames)} frames "
                f"({start_time:.2f}s - {end_time:.2f}s)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding segment to buffer: {e}")
            return False
    
    @_with_lock
    def get_frames(
        self,
        start_time: float,
        end_time: float,
        frame_types: Optional[List[FrameType]] = None,
        max_frames: Optional[int] = None
    ) -> List[VideoFrame]:
        """Get frames within a time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            frame_types: Optional filter for frame types
            max_frames: Maximum number of frames to return
            
        Returns:
            List of frames in the time range
        """
        frames = []
        
        for frame in self._frames:
            # Check time range
            if frame.timestamp < start_time:
                continue
            if frame.timestamp > end_time:
                break
            
            # Check frame type filter
            if frame_types and frame.frame_type not in frame_types:
                continue
            
            frames.append(frame)
            frame.access_count += 1
            frame.last_accessed = time.time()
            
            # Check max frames
            if max_frames and len(frames) >= max_frames:
                break
        
        return frames
    
    @_with_lock
    def get_segment(self, segment_id: str) -> Optional[BufferSegment]:
        """Get a specific segment by ID.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            BufferSegment or None if not found
        """
        segment = self._segments.get(segment_id)
        if segment:
            segment.access_count += 1
            segment.last_accessed = time.time()
        return segment
    
    @_with_lock
    def get_keyframes(
        self,
        start_time: float,
        end_time: float,
        max_frames: Optional[int] = None
    ) -> List[VideoFrame]:
        """Get keyframes within a time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            max_frames: Maximum number of keyframes to return
            
        Returns:
            List of keyframes in the time range
        """
        keyframes = []
        
        # Use keyframe index for efficient lookup
        for timestamp, idx in self._keyframe_index:
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            
            # Validate index
            if 0 <= idx < len(self._frames):
                frame = self._frames[idx]
                if frame.is_keyframe and frame.timestamp == timestamp:
                    keyframes.append(frame)
                    frame.access_count += 1
                    frame.last_accessed = time.time()
                    
                    if max_frames and len(keyframes) >= max_frames:
                        break
        
        return keyframes
    
    @_with_lock
    def get_latest_frames(self, count: int = 10) -> List[VideoFrame]:
        """Get the most recent frames.
        
        Args:
            count: Number of frames to return
            
        Returns:
            List of recent frames
        """
        if count >= len(self._frames):
            return list(self._frames)
        
        frames = list(self._frames)[-count:]
        for frame in frames:
            frame.access_count += 1
            frame.last_accessed = time.time()
        
        return frames
    
    @_with_lock
    def get_frame_at_timestamp(self, timestamp: float, tolerance: float = 0.1) -> Optional[VideoFrame]:
        """Get frame closest to a specific timestamp.
        
        Args:
            timestamp: Target timestamp in seconds
            tolerance: Maximum time difference in seconds
            
        Returns:
            Closest frame or None if not found within tolerance
        """
        closest_frame = None
        min_diff = float('inf')
        
        # Binary search would be more efficient for large buffers
        for frame in self._frames:
            diff = abs(frame.timestamp - timestamp)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_frame = frame
        
        if closest_frame:
            closest_frame.access_count += 1
            closest_frame.last_accessed = time.time()
        
        return closest_frame
    
    async def create_window(
        self,
        start_time: float,
        duration: float,
        overlap: float = 0.0
    ) -> AsyncIterator[List[VideoFrame]]:
        """Create sliding windows over the buffer.
        
        Args:
            start_time: Window start time
            duration: Window duration in seconds
            overlap: Overlap between windows in seconds
            
        Yields:
            Lists of frames for each window
        """
        current_start = start_time
        stride = duration - overlap
        
        while True:
            # Get frames for current window
            window_end = current_start + duration
            frames = self.get_frames(current_start, window_end)
            
            if not frames:
                # No more frames, wait for new data
                await asyncio.sleep(0.5)
                continue
            
            yield frames
            
            # Move to next window
            current_start += stride
            
            # Check if we've caught up to live edge
            latest_frame = self.get_latest_frames(1)
            if latest_frame and current_start > latest_frame[0].timestamp:
                # Wait for more data
                await asyncio.sleep(1.0)
    
    def _should_drop_frame(self, frame: VideoFrame) -> bool:
        """Determine if a frame should be dropped.
        
        Args:
            frame: Frame to evaluate
            
        Returns:
            bool: True if frame should be dropped
        """
        # Check memory limit
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        if self._total_memory_bytes + frame.size_bytes > max_memory_bytes:
            # Try to free memory first
            self._evict_frames()
            
            # Check again after eviction
            if self._total_memory_bytes + frame.size_bytes > max_memory_bytes:
                # Still over limit, check if we should drop this frame
                if not frame.is_keyframe and self.config.enable_keyframe_priority:
                    return True  # Drop non-keyframes when memory is tight
        
        return False
    
    def _evict_frames(self) -> int:
        """Evict old frames to free memory.
        
        Returns:
            Number of frames evicted
        """
        evicted = 0
        current_time = time.time()
        
        # Calculate retention cutoff
        cutoff_time = current_time - self.config.retention_seconds
        
        # Keep minimum number of frames
        while len(self._frames) > self.config.min_retention_items:
            if not self._frames:
                break
            
            oldest_frame = self._frames[0]
            
            # Check if frame is old enough to evict
            if oldest_frame.timestamp < cutoff_time:
                # Special handling for keyframes if enabled
                if oldest_frame.is_keyframe and self.config.enable_keyframe_priority:
                    # Keep keyframes longer
                    extended_cutoff = cutoff_time - 60  # Extra minute for keyframes
                    if oldest_frame.timestamp < extended_cutoff:
                        self._frames.popleft()
                        self._total_memory_bytes -= oldest_frame.size_bytes
                        evicted += 1
                else:
                    self._frames.popleft()
                    self._total_memory_bytes -= oldest_frame.size_bytes
                    evicted += 1
            else:
                break  # Frames are ordered by time, so we can stop
        
        if evicted > 0:
            logger.debug(f"Evicted {evicted} frames to free memory")
        
        return evicted
    
    def _evict_old_segments(self) -> int:
        """Evict old segments to manage memory.
        
        Returns:
            Number of segments evicted
        """
        if not self._segments:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - self.config.retention_seconds
        
        segments_to_remove = []
        for segment_id, segment in self._segments.items():
            if segment.end_time < cutoff_time:
                segments_to_remove.append(segment_id)
        
        for segment_id in segments_to_remove:
            del self._segments[segment_id]
            self._stats["segments_evicted"] += 1
        
        if segments_to_remove:
            logger.debug(f"Evicted {len(segments_to_remove)} old segments")
        
        return len(segments_to_remove)
    
    def _update_avg_frame_size(self) -> None:
        """Update average frame size statistic."""
        if self._stats["frames_added"] > 0:
            self._stats["avg_frame_size"] = (
                self._stats["total_bytes_processed"] / self._stats["frames_added"]
            )
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        try:
            loop = asyncio.get_running_loop()
            
            # Garbage collection task
            self._gc_task = loop.create_task(self._garbage_collection_loop())
            
            # Stats collection task
            self._stats_task = loop.create_task(self._stats_collection_loop())
            
        except RuntimeError:
            # No event loop running yet
            logger.debug("No event loop available for background tasks")
    
    async def _garbage_collection_loop(self) -> None:
        """Background task for garbage collection."""
        while True:
            try:
                await asyncio.sleep(self.config.gc_interval_seconds)
                
                with self._lock if self._lock else nullcontext():
                    # Evict old frames and segments
                    frames_evicted = self._evict_frames()
                    segments_evicted = self._evict_old_segments()
                    
                    # Clean up weak references
                    if self._weak_refs:
                        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
                    
                    # Update stats
                    self._stats["memory_cleanups"] += 1
                    
                    logger.debug(
                        f"GC cycle: evicted {frames_evicted} frames, "
                        f"{segments_evicted} segments"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in GC loop: {e}")
    
    async def _stats_collection_loop(self) -> None:
        """Background task for statistics collection."""
        while True:
            try:
                await asyncio.sleep(self.config.stats_interval_seconds)
                
                with self._lock if self._lock else nullcontext():
                    # Calculate keyframe ratio
                    if self._frame_count > 0:
                        keyframe_count = sum(1 for f in self._frames if f.is_keyframe)
                        self._stats["keyframe_ratio"] = keyframe_count / len(self._frames)
                    
                    # Log current stats
                    logger.info(
                        f"Buffer stats: {len(self._frames)} frames, "
                        f"{self._total_memory_bytes / 1024 / 1024:.1f}MB, "
                        f"{self._stats['frames_dropped']} dropped, "
                        f"keyframe ratio: {self._stats['keyframe_ratio']:.2%}"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")
    
    @_with_lock
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()
        stats.update({
            "current_frames": len(self._frames),
            "current_segments": len(self._segments),
            "memory_usage_mb": self._total_memory_bytes / 1024 / 1024,
            "memory_limit_mb": self.config.max_memory_mb,
            "oldest_frame_timestamp": self._frames[0].timestamp if self._frames else None,
            "newest_frame_timestamp": self._frames[-1].timestamp if self._frames else None,
            "buffer_duration_seconds": (
                self._frames[-1].timestamp - self._frames[0].timestamp
                if len(self._frames) > 1 else 0
            ),
        })
        return stats
    
    @_with_lock
    def clear(self) -> None:
        """Clear all buffered data."""
        self._frames.clear()
        self._segments.clear()
        self._keyframe_index.clear()
        self._total_memory_bytes = 0
        self._frame_count = 0
        logger.info("Cleared video buffer")
    
    async def close(self) -> None:
        """Clean up resources and stop background tasks."""
        # Cancel background tasks
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
        
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        # Clear buffers
        self.clear()
        
        logger.info("Closed video buffer")


# Context manager support
from contextlib import nullcontext


class VideoBufferManager:
    """High-level manager for video buffers across multiple streams."""
    
    def __init__(self, default_config: Optional[BufferConfig] = None):
        """Initialize the buffer manager."""
        self.default_config = default_config or BufferConfig()
        self._buffers: Dict[str, CircularVideoBuffer] = {}
        self._lock = RLock()
    
    def get_buffer(self, stream_id: str, config: Optional[BufferConfig] = None) -> CircularVideoBuffer:
        """Get or create a buffer for a stream.
        
        Args:
            stream_id: Unique stream identifier
            config: Optional buffer configuration
            
        Returns:
            Video buffer for the stream
        """
        with self._lock:
            if stream_id not in self._buffers:
                buffer_config = config or self.default_config
                self._buffers[stream_id] = CircularVideoBuffer(buffer_config)
                logger.info(f"Created video buffer for stream: {stream_id}")
            
            return self._buffers[stream_id]
    
    async def remove_buffer(self, stream_id: str) -> None:
        """Remove and close a stream buffer.
        
        Args:
            stream_id: Stream identifier
        """
        with self._lock:
            if stream_id in self._buffers:
                buffer = self._buffers.pop(stream_id)
                await buffer.close()
                logger.info(f"Removed video buffer for stream: {stream_id}")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all buffers.
        
        Returns:
            Dictionary mapping stream IDs to stats
        """
        with self._lock:
            return {
                stream_id: buffer.get_stats()
                for stream_id, buffer in self._buffers.items()
            }
    
    async def close_all(self) -> None:
        """Close all buffers."""
        with self._lock:
            for buffer in self._buffers.values():
                await buffer.close()
            self._buffers.clear()
        logger.info("Closed all video buffers")


# Global buffer manager instance
video_buffer_manager = VideoBufferManager()