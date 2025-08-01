"""Stream segmenter that focuses solely on FFmpeg-based stream ingestion and segmentation.

This module provides a clean interface for ingesting streams and yielding segments,
without any content processing concerns.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, Callable, Dict, Any

from ..media.ffmpeg_segmenter import FFmpegSegmenter, SegmentConfig, SegmentInfo

logger = logging.getLogger(__name__)


@dataclass
class StreamSegmenterConfig:
    """Configuration for stream segmentation."""
    
    # Stream settings
    stream_url: str
    stream_id: str
    output_dir: Path
    
    # Segmentation settings
    segment_duration: int = 30  # seconds
    segment_format: str = "mp4"
    force_keyframe_interval: int = 2  # seconds
    
    # Codec settings
    video_codec: str = "copy"  # Don't re-encode by default
    audio_codec: str = "copy"
    
    # Cleanup settings
    delete_old_segments: bool = True
    segment_retention_count: int = 10  # Keep last N segments
    
    # Connection settings
    retry_attempts: int = 3
    retry_delay: float = 5.0
    reconnect_on_error: bool = True


class StreamSegmenter:
    """Stream segmenter that yields segments from a live stream or video file.
    
    This class focuses solely on stream ingestion and segmentation using FFmpeg.
    It does not perform any content analysis or processing.
    """
    
    def __init__(
        self,
        config: StreamSegmenterConfig,
        segment_callback: Optional[Callable[[SegmentInfo], None]] = None
    ):
        """Initialize the stream segmenter.
        
        Args:
            config: Segmentation configuration
            segment_callback: Optional callback when segments are created
        """
        self.config = config
        self.segment_callback = segment_callback
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create segment-specific directory
        self.segment_dir = self.config.output_dir / f"stream_{config.stream_id}"
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure FFmpeg segmenter
        segment_config = SegmentConfig(
            segment_duration=config.segment_duration,
            segment_format=config.segment_format,
            force_keyframe_every=config.force_keyframe_interval,
            video_codec=config.video_codec,
            audio_codec=config.audio_codec,
            delete_threshold=config.segment_retention_count if config.delete_old_segments else None
        )
        
        self._ffmpeg_segmenter = FFmpegSegmenter(
            output_dir=self.segment_dir,
            config=segment_config,
            segment_callback=self._on_segment_created
        )
        
        # State tracking
        self._segments_created = 0
        self._is_running = False
        self._start_time: Optional[float] = None
        
        logger.info(
            f"Initialized StreamSegmenter for {config.stream_id} "
            f"with {config.segment_duration}s segments"
        )
    
    async def start(self) -> None:
        """Start stream segmentation."""
        if self._is_running:
            logger.warning("Segmenter is already running")
            return
        
        self._is_running = True
        self._start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting segmentation for stream: {self.config.stream_url}")
        
        # Start FFmpeg segmentation with retry logic
        for attempt in range(self.config.retry_attempts):
            try:
                await self._ffmpeg_segmenter.start(self.config.stream_url)
                break
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(
                        f"Failed to start segmentation (attempt {attempt + 1}/"
                        f"{self.config.retry_attempts}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to start segmentation after {self.config.retry_attempts} attempts")
                    self._is_running = False
                    raise
    
    async def stop(self) -> None:
        """Stop stream segmentation."""
        if not self._is_running:
            return
        
        logger.info("Stopping stream segmentation")
        self._is_running = False
        
        await self._ffmpeg_segmenter.stop()
        
        duration = asyncio.get_event_loop().time() - self._start_time if self._start_time else 0
        logger.info(
            f"Segmentation stopped. Created {self._segments_created} segments "
            f"in {duration:.1f} seconds"
        )
    
    async def get_segments(self) -> AsyncIterator[SegmentInfo]:
        """Yield segments as they become available.
        
        This is the main interface for consuming segments. It yields
        SegmentInfo objects as FFmpeg creates them.
        
        Yields:
            SegmentInfo: Information about each created segment
        """
        if not self._is_running:
            raise RuntimeError("Segmenter must be started before getting segments")
        
        try:
            async for segment in self._ffmpeg_segmenter.get_segments():
                if not self._is_running:
                    break
                yield segment
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            if self.config.reconnect_on_error and self._is_running:
                logger.info("Attempting to reconnect...")
                await self._reconnect()
            else:
                raise
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect to the stream."""
        # Stop current segmentation
        await self._ffmpeg_segmenter.stop()
        
        # Wait before reconnecting
        await asyncio.sleep(self.config.retry_delay)
        
        # Restart segmentation
        if self._is_running:
            await self.start()
    
    def _on_segment_created(self, segment: SegmentInfo) -> None:
        """Handle segment creation event."""
        self._segments_created += 1
        
        logger.debug(
            f"Segment {segment.index} created: {segment.filename} "
            f"({segment.duration:.1f}s)"
        )
        
        # Call user callback if provided
        if self.segment_callback:
            try:
                self.segment_callback(segment)
            except Exception as e:
                logger.error(f"Error in segment callback: {e}")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Segmentation statistics."""
        duration = asyncio.get_event_loop().time() - self._start_time if self._start_time else 0
        
        return {
            "segments_created": self._segments_created,
            "is_running": self._is_running,
            "uptime_seconds": duration,
            "segments_per_minute": (self._segments_created / duration * 60) if duration > 0 else 0,
            "stream_id": self.config.stream_id,
            "segment_duration": self.config.segment_duration
        }
    
    @property
    def is_running(self) -> bool:
        """Check if segmenter is running."""
        return self._is_running and self._ffmpeg_segmenter.is_running
    
    @property
    def segment_directory(self) -> Path:
        """Get the directory where segments are stored."""
        return self.segment_dir