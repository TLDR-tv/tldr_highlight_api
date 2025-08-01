"""Stream ingestion pipeline - clean Pythonic implementation.

This module handles video stream ingestion and processing
for the B2B AI highlighting agent.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional
from dataclasses import dataclass, field

import logfire

from ..media.ffmpeg_segmenter import FFmpegSegmenter, SegmentInfo
from ...domain.entities.stream import Stream
from ...domain.entities.video_segment import VideoSegment
from ...domain.entities.dimension_set import DimensionSet
from ...domain.services.highlight_analyzer import HighlightAnalyzer


@dataclass
class StreamPipeline:
    """Ingests and processes video streams for highlight detection.
    
    Simple, efficient pipeline that segments streams and analyzes
    each segment for highlights.
    """
    
    stream: Stream
    dimension_set: DimensionSet
    analyzer: HighlightAnalyzer
    
    # Configuration
    segment_duration: int = 30
    max_concurrent: int = 3
    temp_dir: Optional[Path] = None
    
    # State
    _segmenter: Optional[FFmpegSegmenter] = None
    _temp_dir_obj: Optional[Any] = None
    _segments_processed: int = 0
    _active_tasks: Dict[int, asyncio.Task] = field(default_factory=dict)
    
    def __post_init__(self):
        self.logger = logfire.get_logger(__name__)
        
        # Set up temp directory
        if not self.temp_dir:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self._temp_dir_obj.name)
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self) -> AsyncIterator[Dict[str, Any]]:
        """Start processing the stream.
        
        Yields progress updates as segments are processed.
        """
        try:
            # Initialize segmenter
            self._segmenter = FFmpegSegmenter(
                stream_url=self.stream.url.value,
                output_dir=self.temp_dir,
                segment_duration=self.segment_duration,
            )
            
            # Start segmentation
            await self._segmenter.start()
            
            # Process segments as they arrive
            async for segment in self._segmenter.get_segments():
                result = await self._process_segment(segment)
                
                yield {
                    "event": "segment_processed",
                    "segment_index": segment.index,
                    "highlights_found": len(result.get("highlights", [])),
                    "error": result.get("error"),
                    "stats": self.stats,
                }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            yield {
                "event": "error",
                "error": str(e),
                "stats": self.stats,
            }
        finally:
            await self.cleanup()
    
    async def stop(self):
        """Stop the pipeline gracefully."""
        if self._segmenter:
            await self._segmenter.stop()
        
        # Wait for active tasks
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop()
        
        # Clean up temp directory
        if self._temp_dir_obj:
            self._temp_dir_obj.cleanup()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Pipeline statistics."""
        return {
            "segments_processed": self._segments_processed,
            "active_tasks": len(self._active_tasks),
            "stream_id": self.stream.id,
            "is_running": self._segmenter and self._segmenter.is_running,
        }
    
    # Private methods
    
    async def _process_segment(self, segment_info: SegmentInfo) -> Dict[str, Any]:
        """Process a single segment."""
        # Limit concurrent processing
        while len(self._active_tasks) >= self.max_concurrent:
            # Wait for a task to complete
            done, pending = await asyncio.wait(
                self._active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            # Clean up completed tasks
            for task in done:
                for idx, t in list(self._active_tasks.items()):
                    if t == task:
                        del self._active_tasks[idx]
        
        # Create processing task
        task = asyncio.create_task(
            self._analyze_segment(segment_info)
        )
        self._active_tasks[segment_info.index] = task
        
        # Wait for result
        result = await task
        
        # Clean up
        del self._active_tasks[segment_info.index]
        self._segments_processed += 1
        
        return result
    
    async def _analyze_segment(self, segment_info: SegmentInfo) -> Dict[str, Any]:
        """Analyze a segment for highlights."""
        try:
            # Create video segment entity
            video_segment = VideoSegment.create(
                stream_id=self.stream.id,
                segment_index=segment_info.index,
                file_path=segment_info.path,
                start_time=segment_info.start_time,
                end_time=segment_info.end_time,
                metadata={
                    "filename": segment_info.filename,
                    "duration": segment_info.duration,
                }
            )
            
            # Analyze with domain service
            highlights = await self.analyzer.analyze_segment(
                self.stream,
                video_segment,
                self.dimension_set,
                agent_config=None  # Could be passed in
            )
            
            # Delete segment file if configured
            try:
                segment_info.path.unlink()
            except Exception:
                pass
            
            return {
                "highlights": highlights,
                "segment_index": segment_info.index,
            }
            
        except Exception as e:
            self.logger.error(
                f"Error analyzing segment {segment_info.index}: {e}",
                exc_info=True
            )
            return {
                "error": str(e),
                "segment_index": segment_info.index,
            }


@dataclass 
class SimplifiedIngestionConfig:
    """Simplified configuration for stream ingestion."""
    
    segment_duration: int = 30
    max_concurrent_processing: int = 3
    cleanup_segments: bool = True
    
    # Advanced options
    force_keyframe_interval: int = 2
    segment_format: str = "mp4"
    audio_codec: str = "aac"
    video_codec: str = "libx264"


async def ingest_stream(
    stream: Stream,
    dimension_set: DimensionSet,
    analyzer: HighlightAnalyzer,
    config: Optional[SimplifiedIngestionConfig] = None
) -> AsyncIterator[Dict[str, Any]]:
    """Simple function to ingest and process a stream.
    
    This is a convenience function that creates and runs a pipeline.
    """
    config = config or SimplifiedIngestionConfig()
    
    pipeline = StreamPipeline(
        stream=stream,
        dimension_set=dimension_set,
        analyzer=analyzer,
        segment_duration=config.segment_duration,
        max_concurrent=config.max_concurrent_processing,
    )
    
    async for update in pipeline.start():
        yield update