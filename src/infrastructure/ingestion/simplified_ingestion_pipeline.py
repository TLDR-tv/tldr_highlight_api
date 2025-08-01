"""Simplified stream ingestion pipeline using FFmpeg segmentation.

This module provides a streamlined pipeline that uses FFmpeg's segment muxer
directly, eliminating the need for complex in-memory frame buffering.
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Any, Callable

from ..media.ffmpeg_segmenter import FFmpegSegmenter, SegmentConfig, SegmentInfo
from ..content_processing.gemini_video_processor import GeminiVideoProcessor
from ...domain.entities.dimension_set_aggregate import DimensionSetAggregate
from ...domain.entities.highlight_agent_config import HighlightAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedIngestionConfig:
    """Configuration for simplified ingestion pipeline."""
    
    # Stream settings
    stream_url: str
    stream_id: str
    
    # Segmentation settings
    segment_duration: int = 30  # seconds
    segment_overlap: int = 5   # seconds for overlap processing
    
    # Processing settings
    enable_audio_processing: bool = True
    delete_segments_after_processing: bool = True
    max_concurrent_processing: int = 3
    
    # Output settings
    temp_dir: Optional[Path] = None
    keep_processed_segments: bool = False
    
    # Retry settings
    retry_attempts: int = 3
    retry_delay: float = 5.0


@dataclass
class ProcessingResult:
    """Result from processing a segment."""
    
    segment_index: int
    segment_path: Path
    start_time: float
    end_time: float
    highlights: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


class SimplifiedIngestionPipeline:
    """Simplified pipeline using FFmpeg's segment muxer.
    
    This pipeline eliminates complex in-memory buffering by using FFmpeg's
    built-in segmentation capabilities, making the system more robust and
    easier to maintain.
    """
    
    def __init__(
        self,
        config: SimplifiedIngestionConfig,
        gemini_processor: GeminiVideoProcessor,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the simplified pipeline.
        
        Args:
            config: Pipeline configuration
            gemini_processor: Gemini processor for video analysis
            dimension_set: Dimension set for highlight detection
            agent_config: Optional agent configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.gemini_processor = gemini_processor
        self.dimension_set = dimension_set
        self.agent_config = agent_config
        self.progress_callback = progress_callback
        
        # Set up temp directory
        if config.temp_dir:
            self.temp_dir = config.temp_dir
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self._temp_dir_obj.name)
        
        # Create segment output directory
        self.segment_dir = self.temp_dir / f"stream_{config.stream_id}"
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize segmenter
        segment_config = SegmentConfig(
            segment_duration=config.segment_duration,
            force_keyframe_every=2,  # Force keyframe every 2 seconds
            video_codec="copy",      # Don't re-encode
            audio_codec="copy",
            delete_threshold=10 if config.delete_segments_after_processing else None
        )
        
        self.segmenter = FFmpegSegmenter(
            output_dir=self.segment_dir,
            config=segment_config,
            segment_callback=self._on_segment_created
        )
        
        # Processing state
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        self._processing_tasks: List[asyncio.Task] = []
        self._results: List[ProcessingResult] = []
        self._shutdown = False
        
        # Statistics
        self.stats = {
            "segments_created": 0,
            "segments_processed": 0,
            "highlights_found": 0,
            "processing_errors": 0,
            "total_processing_time": 0.0,
            "start_time": None,
            "end_time": None
        }
        
        logger.info(f"Initialized SimplifiedIngestionPipeline for stream {config.stream_id}")
    
    async def start(self) -> AsyncIterator[ProcessingResult]:
        """Start the ingestion pipeline and yield results."""
        try:
            self.stats["start_time"] = datetime.now(timezone.utc)
            logger.info(f"Starting ingestion for stream: {self.config.stream_url}")
            
            # Start FFmpeg segmentation
            await self.segmenter.start(self.config.stream_url)
            
            # Process segments as they arrive
            async for segment in self.segmenter.get_segments():
                if self._shutdown:
                    break
                
                # Process segment in background
                task = asyncio.create_task(self._process_segment_wrapper(segment))
                self._processing_tasks.append(task)
                
                # Clean up completed tasks
                self._processing_tasks = [t for t in self._processing_tasks if not t.done()]
                
                # Yield any completed results
                for task in [t for t in self._processing_tasks if t.done()]:
                    try:
                        result = await task
                        if result:
                            yield result
                    except Exception as e:
                        logger.error(f"Error getting task result: {e}")
            
            # Wait for remaining tasks
            if self._processing_tasks:
                results = await asyncio.gather(*self._processing_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, ProcessingResult):
                        yield result
                        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        logger.info("Stopping ingestion pipeline")
        self._shutdown = True
        self.stats["end_time"] = datetime.now(timezone.utc)
        
        # Stop segmenter
        await self.segmenter.stop()
        
        # Cancel processing tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Cleanup temp directory if we created it
        if hasattr(self, '_temp_dir_obj'):
            self._temp_dir_obj.cleanup()
        
        logger.info(f"Pipeline stopped. Stats: {self.get_stats()}")
    
    def _on_segment_created(self, segment: SegmentInfo) -> None:
        """Callback when a new segment is created."""
        self.stats["segments_created"] += 1
        logger.debug(f"Segment created: {segment.filename} (index: {segment.index})")
        
        if self.progress_callback:
            self.progress_callback({
                "event": "segment_created",
                "segment_index": segment.index,
                "segment_file": segment.filename,
                "stats": self.get_stats()
            })
    
    async def _process_segment_wrapper(self, segment: SegmentInfo) -> Optional[ProcessingResult]:
        """Wrapper to process segment with semaphore."""
        async with self._processing_semaphore:
            return await self._process_segment(segment)
    
    async def _process_segment(self, segment: SegmentInfo) -> Optional[ProcessingResult]:
        """Process a video segment with Gemini."""
        start_time = datetime.now()
        
        result = ProcessingResult(
            segment_index=segment.index,
            segment_path=segment.path,
            start_time=segment.start_time,
            end_time=segment.end_time
        )
        
        try:
            logger.info(f"Processing segment {segment.index}: {segment.filename}")
            
            # Prepare segment info for Gemini
            segment_info = {
                "segment_id": f"segment_{segment.index}",
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "file_path": str(segment.path)
            }
            
            # Process with Gemini
            highlights = await self.gemini_processor.analyze_video_with_dimensions(
                video_path=str(segment.path),
                segment_info=segment_info,
                dimension_set=self.dimension_set,
                agent_config=self.agent_config
            )
            
            # Convert highlights to dictionaries
            result.highlights = [
                {
                    "start_time": h.start_time,
                    "end_time": h.end_time,
                    "confidence_score": h.confidence_score,
                    "dimension_scores": h.dimension_scores,
                    "highlight_types": h.highlight_types,
                    "title": h.title,
                    "description": h.description
                }
                for h in highlights
            ]
            
            # Update statistics
            self.stats["segments_processed"] += 1
            self.stats["highlights_found"] += len(highlights)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            self.stats["total_processing_time"] += processing_time
            
            logger.info(
                f"Processed segment {segment.index}: "
                f"found {len(highlights)} highlights in {processing_time:.2f}s"
            )
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback({
                    "event": "segment_processed",
                    "segment_index": segment.index,
                    "highlights_found": len(highlights),
                    "processing_time": processing_time,
                    "stats": self.get_stats()
                })
            
            # Delete segment if configured
            if self.config.delete_segments_after_processing and not self.config.keep_processed_segments:
                try:
                    segment.path.unlink()
                    logger.debug(f"Deleted processed segment: {segment.filename}")
                except Exception as e:
                    logger.error(f"Error deleting segment: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing segment {segment.index}: {e}")
            result.error = str(e)
            self.stats["processing_errors"] += 1
            
            if self.progress_callback:
                self.progress_callback({
                    "event": "segment_error",
                    "segment_index": segment.index,
                    "error": str(e),
                    "stats": self.get_stats()
                })
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["segments_processed"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["segments_processed"]
            )
            stats["highlights_per_segment"] = (
                stats["highlights_found"] / stats["segments_processed"]
            )
        else:
            stats["avg_processing_time"] = 0.0
            stats["highlights_per_segment"] = 0.0
        
        # Calculate uptime
        if stats["start_time"]:
            end_time = stats["end_time"] or datetime.now(timezone.utc)
            stats["uptime_seconds"] = (end_time - stats["start_time"]).total_seconds()
        else:
            stats["uptime_seconds"] = 0.0
        
        return stats
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.segmenter.is_running and not self._shutdown