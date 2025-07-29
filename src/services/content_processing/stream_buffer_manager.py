"""High-level stream buffer management for unified video processing.

This module provides the integration layer between stream adapters and the
unified buffering system, coordinating video buffers, frame synchronization,
and segment processing across different stream formats.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any, Callable

from src.utils.video_buffer import (
    BufferConfig,
    BufferFormat,
    CircularVideoBuffer,
    FrameType,
    VideoFrame,
    VideoBufferManager,
)
from src.utils.frame_synchronizer import (
    FrameSynchronizer,
    SyncConfig,
    TimestampFormat,
)
from src.utils.segment_processor import (
    ProcessedSegment,
    SegmentConfig,
    SegmentProcessor,
    SegmentStrategy,
)
from src.services.stream_adapters.base import BaseStreamAdapter

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Stream type identifiers."""
    
    YOUTUBE_HLS = "youtube_hls"
    TWITCH_HLS = "twitch_hls"
    RTMP_FLV = "rtmp_flv"
    GENERIC_HLS = "generic_hls"
    RAW_VIDEO = "raw_video"


@dataclass
class StreamBufferConfig:
    """Configuration for stream buffer management."""
    
    # Buffer configuration
    buffer_config: BufferConfig = field(default_factory=BufferConfig)
    
    # Synchronization configuration
    sync_config: SyncConfig = field(default_factory=SyncConfig)
    
    # Segmentation configuration
    segment_config: SegmentConfig = field(default_factory=SegmentConfig)
    
    # Stream-specific settings
    enable_multi_stream_sync: bool = True
    enable_format_conversion: bool = True
    enable_quality_adaptation: bool = True
    
    # Performance settings
    max_streams: int = 10
    max_total_memory_mb: int = 2000
    enable_stream_priorities: bool = True
    
    # Processing settings
    process_audio: bool = True
    process_metadata: bool = True
    extract_keyframes: bool = True


@dataclass
class StreamBufferStats:
    """Statistics for stream buffer management."""
    
    stream_id: str
    stream_type: StreamType
    
    # Buffer stats
    frames_buffered: int = 0
    segments_buffered: int = 0
    memory_usage_mb: float = 0.0
    buffer_duration_seconds: float = 0.0
    
    # Processing stats
    frames_processed: int = 0
    segments_processed: int = 0
    keyframes_extracted: int = 0
    
    # Quality metrics
    avg_frame_quality: float = 0.0
    dropped_frames: int = 0
    sync_accuracy: float = 1.0
    
    # Timing
    started_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    last_update: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class StreamBufferManager:
    """High-level manager for stream buffering and processing.
    
    This class coordinates:
    - Video buffer management across multiple streams
    - Frame synchronization between streams
    - Segment processing and windowing
    - Format conversion and adaptation
    - Multi-stream synchronization
    """
    
    def __init__(self, config: Optional[StreamBufferConfig] = None):
        """Initialize the stream buffer manager."""
        self.config = config or StreamBufferConfig()
        
        # Core components
        self.buffer_manager = VideoBufferManager(self.config.buffer_config)
        self.synchronizer = FrameSynchronizer(self.config.sync_config)
        
        # Stream tracking
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._stream_processors: Dict[str, SegmentProcessor] = {}
        self._stream_tasks: Dict[str, List[asyncio.Task]] = {}
        
        # Statistics
        self._stats: Dict[str, StreamBufferStats] = {}
        
        # Callbacks
        self._frame_callbacks: List[Callable[[str, VideoFrame], None]] = []
        self._segment_callbacks: List[Callable[[str, ProcessedSegment], None]] = []
        
        # Resource management
        self._total_memory_usage = 0.0
        self._active_streams = 0
        
        logger.info("Initialized StreamBufferManager")
    
    async def add_stream(
        self,
        stream_id: str,
        adapter: BaseStreamAdapter,
        stream_type: StreamType,
        priority: int = 5,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a stream to the buffer manager.
        
        Args:
            stream_id: Unique stream identifier
            adapter: Stream adapter instance
            stream_type: Type of stream
            priority: Stream priority (1-10, higher is more important)
            custom_config: Custom configuration overrides
            
        Returns:
            bool: True if stream was added successfully
        """
        if stream_id in self._streams:
            logger.warning(f"Stream {stream_id} already exists")
            return False
        
        if self._active_streams >= self.config.max_streams:
            logger.error(f"Maximum number of streams ({self.config.max_streams}) reached")
            return False
        
        try:
            # Create buffer for stream
            buffer = self.buffer_manager.get_buffer(stream_id, self.config.buffer_config)
            
            # Register stream with synchronizer
            timestamp_format = self._get_timestamp_format(stream_type)
            self.synchronizer.register_stream(
                stream_id,
                self._get_buffer_format(stream_type),
                timestamp_format,
                is_reference=(self._active_streams == 0)  # First stream is reference
            )
            
            # Create segment processor
            processor = SegmentProcessor(
                buffer=buffer,
                synchronizer=self.synchronizer if self.config.enable_multi_stream_sync else None,
                config=self.config.segment_config
            )
            self._stream_processors[stream_id] = processor
            
            # Store stream info
            self._streams[stream_id] = {
                "adapter": adapter,
                "type": stream_type,
                "priority": priority,
                "buffer": buffer,
                "processor": processor,
                "config": custom_config or {},
                "active": True,
            }
            
            # Initialize statistics
            self._stats[stream_id] = StreamBufferStats(
                stream_id=stream_id,
                stream_type=stream_type
            )
            
            # Start processing tasks
            tasks = [
                asyncio.create_task(self._process_stream_data(stream_id)),
                asyncio.create_task(self._process_segments(stream_id)),
            ]
            self._stream_tasks[stream_id] = tasks
            
            self._active_streams += 1
            
            logger.info(
                f"Added stream {stream_id} (type: {stream_type}, priority: {priority})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding stream {stream_id}: {e}")
            # Cleanup on error
            await self.remove_stream(stream_id)
            return False
    
    async def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream from the buffer manager.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if stream was removed successfully
        """
        if stream_id not in self._streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        try:
            # Cancel processing tasks
            if stream_id in self._stream_tasks:
                for task in self._stream_tasks[stream_id]:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*self._stream_tasks[stream_id], return_exceptions=True)
                del self._stream_tasks[stream_id]
            
            # Close segment processor
            if stream_id in self._stream_processors:
                await self._stream_processors[stream_id].close()
                del self._stream_processors[stream_id]
            
            # Remove buffer
            await self.buffer_manager.remove_buffer(stream_id)
            
            # Remove from streams
            del self._streams[stream_id]
            
            self._active_streams -= 1
            
            logger.info(f"Removed stream {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing stream {stream_id}: {e}")
            return False
    
    async def _process_stream_data(self, stream_id: str) -> None:
        """Process incoming stream data into frames and buffer them."""
        stream_info = self._streams.get(stream_id)
        if not stream_info:
            return
        
        adapter = stream_info["adapter"]
        stream_type = stream_info["type"]
        buffer = stream_info["buffer"]
        stats = self._stats[stream_id]
        
        try:
            logger.info(f"Starting data processing for stream {stream_id}")
            
            # Process based on stream type
            if stream_type in [StreamType.YOUTUBE_HLS, StreamType.TWITCH_HLS, StreamType.GENERIC_HLS]:
                await self._process_hls_stream(stream_id, adapter, buffer, stats)
            elif stream_type == StreamType.RTMP_FLV:
                await self._process_rtmp_stream(stream_id, adapter, buffer, stats)
            else:
                await self._process_generic_stream(stream_id, adapter, buffer, stats)
                
        except asyncio.CancelledError:
            logger.info(f"Stream data processing cancelled for {stream_id}")
        except Exception as e:
            logger.error(f"Error processing stream data for {stream_id}: {e}")
    
    async def _process_hls_stream(
        self,
        stream_id: str,
        adapter: BaseStreamAdapter,
        buffer: CircularVideoBuffer,
        stats: StreamBufferStats
    ) -> None:
        """Process HLS stream data."""
        segment_count = 0
        segment_start_time = datetime.now(timezone.utc).timestamp()
        
        async for chunk in adapter.get_stream_data():
            try:
                # HLS chunks are typically MPEG-TS segments
                # Create frame representation of segment
                current_time = datetime.now(timezone.utc).timestamp()
                
                frame = VideoFrame(
                    data=chunk,
                    timestamp=current_time,
                    duration=2.0,  # Typical HLS segment duration
                    format=BufferFormat.HLS_SEGMENT,
                    frame_type=FrameType.I_FRAME if segment_count % 5 == 0 else FrameType.P_FRAME,
                    is_keyframe=(segment_count % 5 == 0),
                    sequence_number=segment_count,
                )
                
                # Synchronize if multi-stream sync is enabled
                if self.config.enable_multi_stream_sync:
                    frame, sync_confidence = await self.synchronizer.synchronize_frame(
                        stream_id, frame
                    )
                    stats.sync_accuracy = sync_confidence
                
                # Add to buffer
                if buffer.add_frame(frame):
                    stats.frames_buffered += 1
                    stats.frames_processed += 1
                    
                    # Notify callbacks
                    await self._notify_frame_callbacks(stream_id, frame)
                else:
                    stats.dropped_frames += 1
                
                # Add as segment
                segment_id = f"hls_segment_{segment_count}"
                if buffer.add_segment(segment_id, [frame]):
                    stats.segments_buffered += 1
                
                segment_count += 1
                
                # Update stats
                stats.last_update = current_time
                stats.buffer_duration_seconds = current_time - segment_start_time
                
            except Exception as e:
                logger.error(f"Error processing HLS chunk for {stream_id}: {e}")
    
    async def _process_rtmp_stream(
        self,
        stream_id: str,
        adapter: BaseStreamAdapter,
        buffer: CircularVideoBuffer,
        stats: StreamBufferStats
    ) -> None:
        """Process RTMP/FLV stream data."""
        frame_count = 0
        
        async for data in adapter.get_stream_data():
            try:
                # RTMP data comes as JSON-encoded frame info
                import json
                frame_info = json.loads(data)
                
                # Create frame based on type
                if frame_info["type"] == "video":
                    frame_type = FrameType.I_FRAME if frame_info.get("is_keyframe") else FrameType.P_FRAME
                    is_keyframe = frame_info.get("is_keyframe", False)
                elif frame_info["type"] == "audio":
                    frame_type = FrameType.AUDIO
                    is_keyframe = False
                else:
                    frame_type = FrameType.METADATA
                    is_keyframe = False
                
                frame = VideoFrame(
                    data=data,  # Keep original data
                    timestamp=frame_info["timestamp"],
                    duration=1.0 / 30.0,  # Assume 30fps for now
                    format=BufferFormat.FLV_FRAME,
                    frame_type=frame_type,
                    is_keyframe=is_keyframe,
                    codec=frame_info.get("codec"),
                    sequence_number=frame_count,
                )
                
                # Synchronize if multi-stream sync is enabled
                if self.config.enable_multi_stream_sync:
                    frame, sync_confidence = await self.synchronizer.synchronize_frame(
                        stream_id, frame
                    )
                    stats.sync_accuracy = sync_confidence
                
                # Add to buffer
                if buffer.add_frame(frame):
                    stats.frames_buffered += 1
                    stats.frames_processed += 1
                    
                    if is_keyframe:
                        stats.keyframes_extracted += 1
                    
                    # Notify callbacks
                    await self._notify_frame_callbacks(stream_id, frame)
                else:
                    stats.dropped_frames += 1
                
                frame_count += 1
                
                # Update stats
                stats.last_update = datetime.now(timezone.utc).timestamp()
                
            except Exception as e:
                logger.error(f"Error processing RTMP data for {stream_id}: {e}")
    
    async def _process_generic_stream(
        self,
        stream_id: str,
        adapter: BaseStreamAdapter,
        buffer: CircularVideoBuffer,
        stats: StreamBufferStats
    ) -> None:
        """Process generic stream data."""
        frame_count = 0
        
        async for chunk in adapter.get_stream_data():
            try:
                # For generic streams, treat each chunk as a frame
                current_time = datetime.now(timezone.utc).timestamp()
                
                frame = VideoFrame(
                    data=chunk,
                    timestamp=current_time,
                    duration=1.0 / 30.0,  # Assume 30fps
                    format=BufferFormat.RAW_FRAME,
                    frame_type=FrameType.P_FRAME,
                    is_keyframe=(frame_count % 30 == 0),  # Keyframe every second
                    sequence_number=frame_count,
                )
                
                # Add to buffer
                if buffer.add_frame(frame):
                    stats.frames_buffered += 1
                    stats.frames_processed += 1
                    
                    # Notify callbacks
                    await self._notify_frame_callbacks(stream_id, frame)
                else:
                    stats.dropped_frames += 1
                
                frame_count += 1
                
                # Update stats
                stats.last_update = current_time
                
            except Exception as e:
                logger.error(f"Error processing generic data for {stream_id}: {e}")
    
    async def _process_segments(self, stream_id: str) -> None:
        """Process stream segments for analysis."""
        stream_info = self._streams.get(stream_id)
        if not stream_info:
            return
        
        processor = stream_info["processor"]
        stats = self._stats[stream_id]
        
        try:
            logger.info(f"Starting segment processing for stream {stream_id}")
            
            # Add segment callback
            processor.add_callback(
                lambda segment: asyncio.create_task(
                    self._notify_segment_callbacks(stream_id, segment)
                )
            )
            
            # Process segments
            async for segment in processor.process_stream(stream_id, real_time=True):
                stats.segments_processed += 1
                
                # Update quality metrics
                if stats.segments_processed > 0:
                    alpha = 0.1  # Smoothing factor
                    stats.avg_frame_quality = (
                        alpha * segment.quality_score +
                        (1 - alpha) * stats.avg_frame_quality
                    )
                
                logger.debug(
                    f"Processed segment for {stream_id}: "
                    f"{segment.frame_count} frames, quality: {segment.quality_score:.2f}"
                )
                
        except asyncio.CancelledError:
            logger.info(f"Segment processing cancelled for {stream_id}")
        except Exception as e:
            logger.error(f"Error processing segments for {stream_id}: {e}")
    
    async def get_synchronized_segments(
        self,
        duration: float = 10.0,
        overlap: float = 2.0
    ) -> AsyncIterator[Dict[str, ProcessedSegment]]:
        """Get synchronized segments from all active streams.
        
        Args:
            duration: Segment duration in seconds
            overlap: Overlap between segments in seconds
            
        Yields:
            Dictionary mapping stream IDs to synchronized segments
        """
        if not self.config.enable_multi_stream_sync:
            logger.warning("Multi-stream sync is disabled")
            return
        
        # Get current time as reference
        current_time = datetime.now(timezone.utc).timestamp()
        
        while True:
            segment_end = current_time + duration
            
            # Wait for segment to complete
            wait_time = segment_end - datetime.now(timezone.utc).timestamp()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Get aligned frames from all streams
            aligned_frames = await self.synchronizer.align_frames(
                current_time + duration / 2,  # Center of segment
                tolerance=duration / 2
            )
            
            # Create segments for each stream
            synchronized_segments = {}
            
            for stream_id, center_frame in aligned_frames.items():
                if not center_frame or stream_id not in self._streams:
                    continue
                
                buffer = self._streams[stream_id]["buffer"]
                
                # Get frames for segment window
                frames = buffer.get_frames(current_time, segment_end)
                
                if frames:
                    # Create synthetic segment
                    segment = ProcessedSegment(
                        segment_id=f"sync_segment_{int(current_time)}",
                        start_time=current_time,
                        end_time=segment_end,
                        duration=duration,
                        frames=frames,
                        keyframe_indices=[i for i, f in enumerate(frames) if f.is_keyframe],
                        frame_count=len(frames),
                        strategy=SegmentStrategy.SLIDING_WINDOW,
                        overlap_ratio=overlap / duration
                    )
                    
                    synchronized_segments[stream_id] = segment
            
            if synchronized_segments:
                yield synchronized_segments
            
            # Move to next segment
            current_time += duration - overlap
    
    def _get_buffer_format(self, stream_type: StreamType) -> BufferFormat:
        """Get buffer format for stream type."""
        if stream_type in [StreamType.YOUTUBE_HLS, StreamType.TWITCH_HLS, StreamType.GENERIC_HLS]:
            return BufferFormat.HLS_SEGMENT
        elif stream_type == StreamType.RTMP_FLV:
            return BufferFormat.FLV_FRAME
        else:
            return BufferFormat.RAW_FRAME
    
    def _get_timestamp_format(self, stream_type: StreamType) -> TimestampFormat:
        """Get timestamp format for stream type."""
        if stream_type in [StreamType.YOUTUBE_HLS, StreamType.TWITCH_HLS, StreamType.GENERIC_HLS]:
            return TimestampFormat.HLS_TIMESTAMP
        elif stream_type == StreamType.RTMP_FLV:
            return TimestampFormat.RTMP_TIMESTAMP
        else:
            return TimestampFormat.EPOCH_SECONDS
    
    def add_frame_callback(self, callback: Callable[[str, VideoFrame], None]) -> None:
        """Add a callback for frame events.
        
        Args:
            callback: Function to call with (stream_id, frame)
        """
        self._frame_callbacks.append(callback)
    
    def add_segment_callback(self, callback: Callable[[str, ProcessedSegment], None]) -> None:
        """Add a callback for segment events.
        
        Args:
            callback: Function to call with (stream_id, segment)
        """
        self._segment_callbacks.append(callback)
    
    async def _notify_frame_callbacks(self, stream_id: str, frame: VideoFrame) -> None:
        """Notify frame callbacks."""
        for callback in self._frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stream_id, frame)
                else:
                    callback(stream_id, frame)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}")
    
    async def _notify_segment_callbacks(self, stream_id: str, segment: ProcessedSegment) -> None:
        """Notify segment callbacks."""
        for callback in self._segment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stream_id, segment)
                else:
                    callback(stream_id, segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")
    
    def get_stream_stats(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for streams.
        
        Args:
            stream_id: Specific stream ID or None for all streams
            
        Returns:
            Statistics dictionary
        """
        if stream_id:
            if stream_id not in self._stats:
                return {}
            
            stats = self._stats[stream_id]
            buffer_stats = self.buffer_manager.get_all_stats().get(stream_id, {})
            
            return {
                "stream_id": stream_id,
                "stream_type": stats.stream_type.value,
                "frames_buffered": stats.frames_buffered,
                "segments_buffered": stats.segments_buffered,
                "frames_processed": stats.frames_processed,
                "segments_processed": stats.segments_processed,
                "keyframes_extracted": stats.keyframes_extracted,
                "dropped_frames": stats.dropped_frames,
                "avg_frame_quality": stats.avg_frame_quality,
                "sync_accuracy": stats.sync_accuracy,
                "memory_usage_mb": buffer_stats.get("memory_usage_mb", 0.0),
                "buffer_duration_seconds": buffer_stats.get("buffer_duration_seconds", 0.0),
                "uptime_seconds": datetime.now(timezone.utc).timestamp() - stats.started_at,
            }
        else:
            # Return stats for all streams
            all_stats = {}
            for sid in self._stats:
                all_stats[sid] = self.get_stream_stats(sid)
            
            # Add global stats
            all_stats["global"] = {
                "active_streams": self._active_streams,
                "total_memory_usage_mb": sum(
                    s.get("memory_usage_mb", 0.0) for s in all_stats.values()
                ),
                "sync_stats": self.synchronizer.get_sync_stats(),
            }
            
            return all_stats
    
    async def close(self) -> None:
        """Clean up all resources."""
        logger.info("Closing StreamBufferManager")
        
        # Remove all streams
        stream_ids = list(self._streams.keys())
        for stream_id in stream_ids:
            await self.remove_stream(stream_id)
        
        # Close core components
        await self.buffer_manager.close_all()
        await self.synchronizer.close()
        
        logger.info("StreamBufferManager closed")


# Global instance
stream_buffer_manager = StreamBufferManager()