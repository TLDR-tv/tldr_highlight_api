"""Segmentation and windowing logic for stream processing.

This module provides flexible segmentation strategies for creating analysis windows
from continuous video streams, supporting fixed, sliding, and adaptive windows.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Any
)

import numpy as np

from .video_buffer import VideoFrame, BufferSegment, CircularVideoBuffer, FrameType
from .frame_synchronizer import FrameSynchronizer

logger = logging.getLogger(__name__)


class SegmentStrategy(str, Enum):
    """Segmentation strategies."""
    
    FIXED_DURATION = "fixed_duration"      # Fixed-size segments
    SLIDING_WINDOW = "sliding_window"      # Overlapping sliding windows
    KEYFRAME_BASED = "keyframe_based"      # Segment at keyframes
    SCENE_BASED = "scene_based"            # Segment at scene changes
    ADAPTIVE = "adaptive"                  # Adaptive sizing based on content
    EVENT_DRIVEN = "event_driven"          # Triggered by specific events


@dataclass
class SegmentConfig:
    """Configuration for segment processing."""
    
    # Basic settings
    strategy: SegmentStrategy = SegmentStrategy.SLIDING_WINDOW
    segment_duration_seconds: float = 10.0
    overlap_seconds: float = 2.0
    
    # Keyframe settings
    min_keyframes_per_segment: int = 1
    max_segments_without_keyframe: int = 3
    force_keyframe_alignment: bool = True
    
    # Adaptive settings
    min_segment_duration: float = 5.0
    max_segment_duration: float = 30.0
    target_complexity: float = 0.5
    complexity_window_size: int = 100
    
    # Quality settings
    min_frames_per_segment: int = 30
    max_frames_per_segment: int = 900  # ~30 seconds at 30fps
    enable_quality_filtering: bool = True
    min_segment_quality: float = 0.3
    
    # Processing settings
    batch_size: int = 5
    enable_parallel_processing: bool = True
    max_concurrent_segments: int = 10
    
    # Event detection
    scene_change_threshold: float = 0.8
    motion_threshold: float = 0.5
    audio_peak_threshold: float = 0.9


@dataclass
class ProcessedSegment:
    """A processed video segment ready for analysis."""
    
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    
    # Frame data
    frames: List[VideoFrame]
    keyframe_indices: List[int]
    frame_count: int
    
    # Segment properties
    strategy: SegmentStrategy
    overlap_ratio: float = 0.0
    is_complete: bool = True
    
    # Quality metrics
    quality_score: float = 1.0
    complexity_score: float = 0.5
    motion_score: float = 0.0
    
    # Processing metadata
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    processed_at: Optional[float] = None
    processing_time: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_keyframe(self) -> bool:
        """Check if segment contains keyframes."""
        return len(self.keyframe_indices) > 0
    
    @property
    def avg_fps(self) -> float:
        """Calculate average FPS for the segment."""
        if self.duration > 0:
            return self.frame_count / self.duration
        return 0.0
    
    def get_keyframes(self) -> List[VideoFrame]:
        """Get keyframes from the segment."""
        return [self.frames[i] for i in self.keyframe_indices if i < len(self.frames)]


class ComplexityAnalyzer:
    """Analyzes content complexity for adaptive segmentation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.motion_history = deque(maxlen=window_size)
        self.scene_change_history = deque(maxlen=window_size)
        self.audio_level_history = deque(maxlen=window_size)
    
    def add_frame_metrics(
        self,
        motion_score: float,
        scene_change_score: float,
        audio_level: float
    ) -> None:
        """Add frame metrics for complexity analysis."""
        self.motion_history.append(motion_score)
        self.scene_change_history.append(scene_change_score)
        self.audio_level_history.append(audio_level)
    
    def calculate_complexity(self) -> float:
        """Calculate current content complexity (0.0 to 1.0)."""
        if not self.motion_history:
            return 0.5
        
        # Motion complexity
        motion_var = np.var(self.motion_history) if len(self.motion_history) > 1 else 0
        motion_complexity = min(motion_var * 10, 1.0)  # Normalize
        
        # Scene change frequency
        scene_changes = sum(1 for s in self.scene_change_history if s > 0.5)
        scene_complexity = scene_changes / len(self.scene_change_history)
        
        # Audio dynamics
        if self.audio_level_history:
            audio_var = np.var(self.audio_level_history)
            audio_complexity = min(audio_var * 5, 1.0)  # Normalize
        else:
            audio_complexity = 0.0
        
        # Weighted average
        complexity = (
            0.4 * motion_complexity +
            0.4 * scene_complexity +
            0.2 * audio_complexity
        )
        
        return float(np.clip(complexity, 0.0, 1.0))
    
    def predict_segment_boundary(self) -> bool:
        """Predict if current position is good for segment boundary."""
        if len(self.scene_change_history) < 3:
            return False
        
        # Check for recent scene change
        recent_scene_change = any(s > 0.8 for s in list(self.scene_change_history)[-3:])
        
        # Check for motion lull
        recent_motion = list(self.motion_history)[-5:] if len(self.motion_history) >= 5 else []
        motion_lull = all(m < 0.3 for m in recent_motion) if recent_motion else False
        
        return recent_scene_change or motion_lull


class SegmentProcessor:
    """Processes video streams into segments for analysis.
    
    Features:
    - Multiple segmentation strategies
    - Adaptive segment sizing
    - Keyframe-aware segmentation
    - Scene change detection
    - Quality-based filtering
    - Parallel segment processing
    """
    
    def __init__(
        self,
        buffer: CircularVideoBuffer,
        synchronizer: Optional[FrameSynchronizer] = None,
        config: Optional[SegmentConfig] = None
    ):
        """Initialize the segment processor.
        
        Args:
            buffer: Video buffer to read from
            synchronizer: Optional frame synchronizer
            config: Segment configuration
        """
        self.buffer = buffer
        self.synchronizer = synchronizer
        self.config = config or SegmentConfig()
        
        # Processing state
        self._segment_counter = 0
        self._last_segment_end = 0.0
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        
        # Strategy handlers
        self._strategy_handlers = {
            SegmentStrategy.FIXED_DURATION: self._create_fixed_segments,
            SegmentStrategy.SLIDING_WINDOW: self._create_sliding_segments,
            SegmentStrategy.KEYFRAME_BASED: self._create_keyframe_segments,
            SegmentStrategy.SCENE_BASED: self._create_scene_segments,
            SegmentStrategy.ADAPTIVE: self._create_adaptive_segments,
            SegmentStrategy.EVENT_DRIVEN: self._create_event_segments,
        }
        
        # Complexity analyzer for adaptive segmentation
        self._complexity_analyzer = ComplexityAnalyzer(
            window_size=self.config.complexity_window_size
        )
        
        # Processing tasks
        self._processing_tasks: List[asyncio.Task] = []
        
        # Segment callbacks
        self._segment_callbacks: List[Callable[[ProcessedSegment], None]] = []
        
        # Statistics
        self._stats = {
            "segments_created": 0,
            "segments_processed": 0,
            "frames_processed": 0,
            "keyframes_found": 0,
            "segments_filtered": 0,
            "avg_segment_duration": 0.0,
            "avg_frames_per_segment": 0.0,
        }
        
        logger.info(f"Initialized SegmentProcessor with strategy: {self.config.strategy}")
    
    async def process_stream(
        self,
        stream_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        real_time: bool = True
    ) -> AsyncIterator[ProcessedSegment]:
        """Process a stream into segments.
        
        Args:
            stream_id: Stream identifier
            start_time: Start timestamp (None for live processing)
            end_time: End timestamp (None for continuous)
            real_time: Whether to process in real-time
            
        Yields:
            Processed segments
        """
        handler = self._strategy_handlers.get(self.config.strategy)
        if not handler:
            raise ValueError(f"Unknown segmentation strategy: {self.config.strategy}")
        
        logger.info(f"Starting segment processing for stream {stream_id}")
        
        try:
            async for segment in handler(stream_id, start_time, end_time, real_time):
                # Apply quality filtering if enabled
                if self.config.enable_quality_filtering:
                    if segment.quality_score < self.config.min_segment_quality:
                        logger.debug(
                            f"Filtered low-quality segment {segment.segment_id} "
                            f"(quality: {segment.quality_score:.2f})"
                        )
                        self._stats["segments_filtered"] += 1
                        continue
                
                # Update statistics
                self._update_stats(segment)
                
                # Notify callbacks
                await self._notify_callbacks(segment)
                
                yield segment
                
        except Exception as e:
            logger.error(f"Error in segment processing: {e}")
            raise
    
    async def _create_fixed_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create fixed-duration segments."""
        current_time = start_time or datetime.now(timezone.utc).timestamp()
        
        while True:
            segment_end = current_time + self.config.segment_duration_seconds
            
            # Wait for segment to complete if real-time
            if real_time:
                wait_time = segment_end - datetime.now(timezone.utc).timestamp()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Get frames for segment
            frames = self.buffer.get_frames(current_time, segment_end)
            
            if not frames:
                if not real_time or (end_time and current_time >= end_time):
                    break
                await asyncio.sleep(1.0)
                continue
            
            # Create segment
            segment = await self._create_segment(
                frames,
                current_time,
                segment_end,
                SegmentStrategy.FIXED_DURATION
            )
            
            yield segment
            
            # Move to next segment
            current_time = segment_end
            
            if end_time and current_time >= end_time:
                break
    
    async def _create_sliding_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create sliding window segments with overlap."""
        current_time = start_time or datetime.now(timezone.utc).timestamp()
        stride = self.config.segment_duration_seconds - self.config.overlap_seconds
        
        while True:
            segment_end = current_time + self.config.segment_duration_seconds
            
            # Wait for segment to complete if real-time
            if real_time:
                wait_time = segment_end - datetime.now(timezone.utc).timestamp()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Get frames for segment
            frames = self.buffer.get_frames(current_time, segment_end)
            
            if not frames:
                if not real_time or (end_time and current_time >= end_time):
                    break
                await asyncio.sleep(1.0)
                continue
            
            # Calculate overlap ratio
            overlap_ratio = self.config.overlap_seconds / self.config.segment_duration_seconds
            
            # Create segment
            segment = await self._create_segment(
                frames,
                current_time,
                segment_end,
                SegmentStrategy.SLIDING_WINDOW,
                overlap_ratio=overlap_ratio
            )
            
            yield segment
            
            # Move to next segment (with stride)
            current_time += stride
            
            if end_time and current_time >= end_time:
                break
    
    async def _create_keyframe_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create segments based on keyframe positions."""
        current_time = start_time or datetime.now(timezone.utc).timestamp()
        segment_start = current_time
        frames_buffer = []
        segments_without_keyframe = 0
        
        while True:
            # Get next batch of frames
            batch_end = current_time + 1.0  # Process 1 second at a time
            frames = self.buffer.get_frames(current_time, batch_end)
            
            if not frames and real_time:
                await asyncio.sleep(0.5)
                continue
            elif not frames:
                # Process remaining frames
                if frames_buffer:
                    segment = await self._create_segment(
                        frames_buffer,
                        segment_start,
                        current_time,
                        SegmentStrategy.KEYFRAME_BASED
                    )
                    yield segment
                break
            
            # Add frames to buffer
            frames_buffer.extend(frames)
            
            # Check for keyframes
            keyframe_found = any(f.is_keyframe for f in frames)
            
            # Determine if we should create a segment
            should_segment = False
            
            if keyframe_found and len(frames_buffer) >= self.config.min_frames_per_segment:
                should_segment = True
                segments_without_keyframe = 0
            elif len(frames_buffer) >= self.config.max_frames_per_segment:
                should_segment = True
            elif (current_time - segment_start) >= self.config.max_segment_duration:
                should_segment = True
            elif segments_without_keyframe >= self.config.max_segments_without_keyframe:
                should_segment = True
            
            if should_segment:
                # Create segment
                segment = await self._create_segment(
                    frames_buffer,
                    segment_start,
                    current_time,
                    SegmentStrategy.KEYFRAME_BASED
                )
                
                yield segment
                
                # Reset for next segment
                frames_buffer = []
                segment_start = current_time
                
                if not keyframe_found:
                    segments_without_keyframe += 1
            
            current_time = batch_end
            
            if end_time and current_time >= end_time:
                break
    
    async def _create_scene_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create segments based on scene changes."""
        # This is a placeholder implementation
        # Real scene detection would require computer vision analysis
        logger.warning("Scene-based segmentation not fully implemented, falling back to adaptive")
        
        async for segment in self._create_adaptive_segments(
            stream_id, start_time, end_time, real_time
        ):
            segment.strategy = SegmentStrategy.SCENE_BASED
            yield segment
    
    async def _create_adaptive_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create adaptive segments based on content complexity."""
        current_time = start_time or datetime.now(timezone.utc).timestamp()
        segment_start = current_time
        frames_buffer = []
        
        while True:
            # Get next batch of frames
            batch_end = current_time + 0.5  # Process 0.5 seconds at a time
            frames = self.buffer.get_frames(current_time, batch_end)
            
            if not frames and real_time:
                await asyncio.sleep(0.2)
                continue
            elif not frames:
                # Process remaining frames
                if frames_buffer:
                    segment = await self._create_segment(
                        frames_buffer,
                        segment_start,
                        current_time,
                        SegmentStrategy.ADAPTIVE
                    )
                    yield segment
                break
            
            # Add frames to buffer
            frames_buffer.extend(frames)
            
            # Update complexity analyzer (simplified - would need actual metrics)
            for frame in frames:
                motion = frame.metadata.get("motion_score", 0.5)
                scene_change = frame.metadata.get("scene_change_score", 0.0)
                audio_level = frame.metadata.get("audio_level", 0.5)
                
                self._complexity_analyzer.add_frame_metrics(
                    motion, scene_change, audio_level
                )
            
            # Calculate current complexity
            complexity = self._complexity_analyzer.calculate_complexity()
            
            # Determine segment duration based on complexity
            if complexity > self.config.target_complexity:
                # High complexity: shorter segments
                target_duration = self.config.min_segment_duration
            else:
                # Low complexity: longer segments
                duration_scale = 1.0 - complexity / self.config.target_complexity
                target_duration = self.config.min_segment_duration + (
                    (self.config.max_segment_duration - self.config.min_segment_duration) *
                    duration_scale
                )
            
            current_duration = current_time - segment_start
            
            # Check if we should create a segment
            should_segment = False
            
            if current_duration >= target_duration:
                should_segment = True
            elif self._complexity_analyzer.predict_segment_boundary():
                should_segment = True
            elif len(frames_buffer) >= self.config.max_frames_per_segment:
                should_segment = True
            
            if should_segment and len(frames_buffer) >= self.config.min_frames_per_segment:
                # Create segment
                segment = await self._create_segment(
                    frames_buffer,
                    segment_start,
                    current_time,
                    SegmentStrategy.ADAPTIVE
                )
                
                segment.complexity_score = complexity
                yield segment
                
                # Reset for next segment
                frames_buffer = []
                segment_start = current_time
            
            current_time = batch_end
            
            if end_time and current_time >= end_time:
                break
    
    async def _create_event_segments(
        self,
        stream_id: str,
        start_time: Optional[float],
        end_time: Optional[float],
        real_time: bool
    ) -> AsyncIterator[ProcessedSegment]:
        """Create segments triggered by specific events."""
        # This is a placeholder implementation
        # Real event detection would require custom event handlers
        logger.warning("Event-driven segmentation not implemented, falling back to fixed duration")
        
        async for segment in self._create_fixed_segments(
            stream_id, start_time, end_time, real_time
        ):
            segment.strategy = SegmentStrategy.EVENT_DRIVEN
            yield segment
    
    async def _create_segment(
        self,
        frames: List[VideoFrame],
        start_time: float,
        end_time: float,
        strategy: SegmentStrategy,
        overlap_ratio: float = 0.0
    ) -> ProcessedSegment:
        """Create a processed segment from frames."""
        self._segment_counter += 1
        segment_id = f"segment_{self._segment_counter}_{int(start_time)}"
        
        # Find keyframe indices
        keyframe_indices = [
            i for i, frame in enumerate(frames)
            if frame.is_keyframe
        ]
        
        # Calculate quality score (simplified)
        quality_scores = [f.quality_score for f in frames if f.quality_score > 0]
        quality_score = np.mean(quality_scores) if quality_scores else 1.0
        
        # Create segment
        segment = ProcessedSegment(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            frames=frames,
            keyframe_indices=keyframe_indices,
            frame_count=len(frames),
            strategy=strategy,
            overlap_ratio=overlap_ratio,
            quality_score=float(quality_score)
        )
        
        # Synchronize frames if synchronizer is available
        if self.synchronizer:
            # This would synchronize timestamps across streams
            pass
        
        self._stats["segments_created"] += 1
        
        logger.debug(
            f"Created {strategy.value} segment {segment_id}: "
            f"{len(frames)} frames, {len(keyframe_indices)} keyframes, "
            f"quality: {quality_score:.2f}"
        )
        
        return segment
    
    def add_callback(self, callback: Callable[[ProcessedSegment], None]) -> None:
        """Add a callback for processed segments.
        
        Args:
            callback: Function to call with each processed segment
        """
        self._segment_callbacks.append(callback)
    
    async def _notify_callbacks(self, segment: ProcessedSegment) -> None:
        """Notify all callbacks of a processed segment."""
        for callback in self._segment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(segment)
                else:
                    callback(segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")
    
    def _update_stats(self, segment: ProcessedSegment) -> None:
        """Update processing statistics."""
        self._stats["segments_processed"] += 1
        self._stats["frames_processed"] += segment.frame_count
        self._stats["keyframes_found"] += len(segment.keyframe_indices)
        
        # Update moving averages
        alpha = 0.1  # Smoothing factor
        
        self._stats["avg_segment_duration"] = (
            alpha * segment.duration +
            (1 - alpha) * self._stats["avg_segment_duration"]
        )
        
        self._stats["avg_frames_per_segment"] = (
            alpha * segment.frame_count +
            (1 - alpha) * self._stats["avg_frames_per_segment"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()
    
    async def process_parallel(
        self,
        segments: List[ProcessedSegment],
        processor: Callable[[ProcessedSegment], Any],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Process segments in parallel.
        
        Args:
            segments: List of segments to process
            processor: Async function to process each segment
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of processing results
        """
        if not self.config.enable_parallel_processing:
            # Process sequentially
            results = []
            for segment in segments:
                result = await processor(segment)
                results.append(result)
            return results
        
        # Process in parallel with semaphore
        max_concurrent = max_concurrent or self.config.max_concurrent_segments
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(segment):
            async with semaphore:
                return await processor(segment)
        
        tasks = [process_with_semaphore(segment) for segment in segments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]
    
    async def close(self) -> None:
        """Clean up resources."""
        # Cancel any running tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        logger.info("Closed SegmentProcessor")