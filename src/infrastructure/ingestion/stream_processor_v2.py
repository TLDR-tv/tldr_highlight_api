"""Stream processor that delegates to domain services for analysis.

This updated processor properly separates infrastructure concerns
(file handling, concurrency) from domain logic (highlight analysis).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..media.ffmpeg_segmenter import SegmentInfo
from ...domain.entities.video_segment import VideoSegment
from ...domain.entities.detected_highlight import DetectedHighlight
from ...domain.services.highlight_analysis_service import HighlightAnalysisService
from ...domain.entities.dimension_set_aggregate import DimensionSetAggregate
from ...domain.entities.highlight_agent_config import HighlightAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing a segment."""
    
    segment_index: int
    segment_path: Path
    start_time: float
    end_time: float
    highlights: List[DetectedHighlight] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class StreamProcessorConfig:
    """Configuration for stream processing."""
    
    # Processing settings
    max_concurrent_processing: int = 3
    
    # File management
    delete_after_processing: bool = True
    keep_failed_segments: bool = False
    
    # Processing options
    min_segment_duration: float = 5.0  # Skip segments shorter than this
    processing_timeout: float = 120.0  # Timeout per segment


class StreamProcessorV2:
    """Stream processor that uses domain services for highlight analysis.
    
    This processor handles the infrastructure concerns of processing
    video segments while delegating the actual analysis to the domain layer.
    """
    
    def __init__(
        self,
        config: StreamProcessorConfig,
        highlight_service: HighlightAnalysisService,
        stream_id: int
    ):
        """Initialize the stream processor.
        
        Args:
            config: Processing configuration
            highlight_service: Domain service for highlight analysis
            stream_id: ID of the stream being processed
        """
        self.config = config
        self.highlight_service = highlight_service
        self.stream_id = stream_id
        
        # Processing state
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_processing)
        self._active_tasks: Dict[int, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "segments_processed": 0,
            "segments_failed": 0,
            "highlights_found": 0,
            "total_processing_time": 0.0,
            "start_time": None
        }
        
        logger.info(
            f"Initialized StreamProcessorV2 with {config.max_concurrent_processing} "
            f"concurrent workers"
        )
    
    async def process_segment(
        self,
        segment_info: SegmentInfo,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> ProcessingResult:
        """Process a single video segment using domain services.
        
        Args:
            segment_info: Segment information from the segmenter
            dimension_set: Dimension set for analysis
            agent_config: Optional agent configuration
            
        Returns:
            ProcessingResult containing highlights and metadata
        """
        start_time = datetime.now()
        
        result = ProcessingResult(
            segment_index=segment_info.index,
            segment_path=segment_info.path,
            start_time=segment_info.start_time,
            end_time=segment_info.end_time
        )
        
        # Skip short segments
        if segment_info.duration < self.config.min_segment_duration:
            logger.warning(
                f"Skipping segment {segment_info.index}: duration {segment_info.duration:.1f}s "
                f"is below minimum {self.config.min_segment_duration}s"
            )
            result.metadata["skipped"] = True
            result.metadata["skip_reason"] = "duration_too_short"
            return result
        
        try:
            logger.info(f"Processing segment {segment_info.index}: {segment_info.filename}")
            
            # Create domain video segment entity
            video_segment = VideoSegment.create(
                stream_id=self.stream_id,
                segment_index=segment_info.index,
                file_path=segment_info.path,
                start_time=segment_info.start_time,
                end_time=segment_info.end_time,
                metadata={
                    "filename": segment_info.filename,
                    "format": self.config.__dict__.get("segment_format", "mp4")
                }
            )
            
            # Process with timeout using domain service
            process_task = self.highlight_service.analyze_segment(
                segment=video_segment,
                dimension_set=dimension_set,
                agent_config=agent_config
            )
            
            highlights = await asyncio.wait_for(
                process_task,
                timeout=self.config.processing_timeout
            )
            
            result.highlights = highlights
            
            # Update statistics
            self.stats["segments_processed"] += 1
            self.stats["highlights_found"] += len(highlights)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            self.stats["total_processing_time"] += processing_time
            
            logger.info(
                f"Processed segment {segment_info.index}: "
                f"found {len(highlights)} highlights in {processing_time:.2f}s"
            )
            
            # Delete segment if configured
            if self.config.delete_after_processing:
                try:
                    segment_info.path.unlink()
                    logger.debug(f"Deleted processed segment: {segment_info.filename}")
                except Exception as e:
                    logger.error(f"Error deleting segment: {e}")
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Processing timeout after {self.config.processing_timeout}s"
            logger.error(f"Segment {segment_info.index}: {error_msg}")
            result.error = error_msg
            self.stats["segments_failed"] += 1
            
            # Handle failed segment file
            self._handle_failed_segment(segment_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_info.index}: {e}")
            result.error = str(e)
            self.stats["segments_failed"] += 1
            
            # Handle failed segment file
            self._handle_failed_segment(segment_info)
            
            return result
    
    async def process_segment_async(
        self,
        segment_info: SegmentInfo,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> asyncio.Task:
        """Process a segment asynchronously with concurrency control.
        
        Args:
            segment_info: Segment to process
            dimension_set: Dimension set for analysis
            agent_config: Optional agent configuration
            
        Returns:
            Task that will complete with ProcessingResult
        """
        async def _process_with_semaphore():
            async with self._processing_semaphore:
                return await self.process_segment(
                    segment_info, dimension_set, agent_config
                )
        
        # Create task and track it
        task = asyncio.create_task(_process_with_semaphore())
        self._active_tasks[segment_info.index] = task
        
        # Clean up when done
        def _cleanup(_):
            self._active_tasks.pop(segment_info.index, None)
        task.add_done_callback(_cleanup)
        
        return task
    
    async def wait_for_completion(self) -> List[ProcessingResult]:
        """Wait for all active processing tasks to complete.
        
        Returns:
            List of all processing results
        """
        if not self._active_tasks:
            return []
        
        logger.info(f"Waiting for {len(self._active_tasks)} processing tasks to complete")
        
        tasks = list(self._active_tasks.values())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ProcessingResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
        
        return valid_results
    
    def _handle_failed_segment(self, segment_info: SegmentInfo) -> None:
        """Handle a segment that failed processing."""
        if self.config.delete_after_processing and not self.config.keep_failed_segments:
            try:
                segment_info.path.unlink()
                logger.debug(f"Deleted failed segment: {segment_info.filename}")
            except Exception as e:
                logger.error(f"Error deleting failed segment: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        total_segments = stats["segments_processed"] + stats["segments_failed"]
        if total_segments > 0:
            stats["success_rate"] = stats["segments_processed"] / total_segments
        else:
            stats["success_rate"] = 0.0
        
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
        
        stats["active_tasks"] = len(self._active_tasks)
        
        return stats
    
    @property
    def active_task_count(self) -> int:
        """Get number of active processing tasks."""
        return len(self._active_tasks)