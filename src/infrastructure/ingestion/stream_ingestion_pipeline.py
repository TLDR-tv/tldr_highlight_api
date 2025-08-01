"""Stream ingestion pipeline that orchestrates segmentation and processing.

This module combines the StreamSegmenter and StreamProcessor to provide
a complete stream analysis pipeline.
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, Optional, Any, Callable, List

from .stream_segmenter import StreamSegmenter, StreamSegmenterConfig
from .stream_processor import StreamProcessor, StreamProcessorConfig, ProcessingResult
from ..content_processing.gemini_video_processor import GeminiVideoProcessor
from ...domain.entities.dimension_set_aggregate import DimensionSetAggregate
from ...domain.entities.highlight_agent_config import HighlightAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamIngestionConfig:
    """Configuration for stream ingestion pipeline."""
    
    # Stream settings
    stream_url: str
    stream_id: str
    
    # Segmentation settings
    segment_duration: int = 30  # seconds
    segment_format: str = "mp4"
    force_keyframe_interval: int = 2
    
    # Processing settings
    enable_audio_processing: bool = True
    delete_segments_after_processing: bool = True
    max_concurrent_processing: int = 3
    
    # Output settings
    temp_dir: Optional[Path] = None
    segment_retention_count: int = 10
    
    # Connection settings
    retry_attempts: int = 3
    retry_delay: float = 5.0
    reconnect_on_error: bool = True


class StreamIngestionPipeline:
    """Orchestrates stream segmentation and processing.
    
    This pipeline combines StreamSegmenter for ingesting and segmenting streams
    with StreamProcessor for analyzing segments and detecting highlights.
    """
    
    def __init__(
        self,
        config: StreamIngestionConfig,
        gemini_processor: GeminiVideoProcessor,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the stream ingestion pipeline.
        
        Args:
            config: Pipeline configuration
            gemini_processor: Gemini processor for video analysis
            dimension_set: Dimension set for highlight detection
            agent_config: Optional agent configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
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
        
        # Initialize segmenter
        segmenter_config = StreamSegmenterConfig(
            stream_url=config.stream_url,
            stream_id=config.stream_id,
            output_dir=self.temp_dir,
            segment_duration=config.segment_duration,
            segment_format=config.segment_format,
            force_keyframe_interval=config.force_keyframe_interval,
            delete_old_segments=config.delete_segments_after_processing,
            segment_retention_count=config.segment_retention_count,
            retry_attempts=config.retry_attempts,
            retry_delay=config.retry_delay,
            reconnect_on_error=config.reconnect_on_error
        )
        
        self.segmenter = StreamSegmenter(
            config=segmenter_config,
            segment_callback=self._on_segment_created
        )
        
        # Initialize domain services
        from ...domain.services.highlight_analysis_service import HighlightAnalysisService
        from ..ai_adapters.gemini_analyzer import GeminiAIAnalyzer
        
        # Create AI analyzer adapter
        ai_analyzer = GeminiAIAnalyzer(
            api_key=gemini_processor.api_key,
            model_name=gemini_processor.model_name
        )
        
        # Create domain highlight analysis service
        highlight_service = HighlightAnalysisService(
            ai_analyzer=ai_analyzer,
            processing_options=None  # Use defaults
        )
        
        # Initialize processor with domain service
        processor_config = StreamProcessorConfig(
            max_concurrent_processing=config.max_concurrent_processing,
            delete_after_processing=config.delete_segments_after_processing
        )
        
        self.processor = StreamProcessor(
            config=processor_config,
            highlight_service=highlight_service,
            stream_id=int(config.stream_id.split('_')[-1])  # Extract numeric ID
        )
        
        # Pipeline state
        self._shutdown = False
        self._processing_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "pipeline_started": None,
            "pipeline_stopped": None
        }
        
        logger.info(f"Initialized StreamIngestionPipeline for stream {config.stream_id}")
    
    async def start(self) -> AsyncIterator[ProcessingResult]:
        """Start the ingestion pipeline and yield results."""
        try:
            self.stats["pipeline_started"] = datetime.now(timezone.utc)
            logger.info(f"Starting ingestion pipeline for stream: {self.config.stream_url}")
            
            # Start stream segmentation
            await self.segmenter.start()
            
            # Process segments as they arrive
            async for segment in self.segmenter.get_segments():
                if self._shutdown:
                    break
                
                # Create processing task
                task = await self.processor.process_segment_async(
                    segment, self.dimension_set, self.agent_config
                )
                self._processing_tasks.append(task)
                
                # Clean up completed tasks and yield results
                completed_tasks = []
                for task in self._processing_tasks:
                    if task.done():
                        completed_tasks.append(task)
                        try:
                            result = await task
                            if result:
                                yield result
                                self._notify_progress(result)
                        except Exception as e:
                            logger.error(f"Error getting task result: {e}")
                
                # Remove completed tasks
                for task in completed_tasks:
                    self._processing_tasks.remove(task)
            
            # Wait for remaining tasks
            if self._processing_tasks:
                logger.info(f"Waiting for {len(self._processing_tasks)} remaining tasks")
                results = await asyncio.gather(*self._processing_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, ProcessingResult):
                        yield result
                        self._notify_progress(result)
                        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        logger.info("Stopping ingestion pipeline")
        self._shutdown = True
        self.stats["pipeline_stopped"] = datetime.now(timezone.utc)
        
        # Stop segmenter
        await self.segmenter.stop()
        
        # Wait for processing to complete
        await self.processor.wait_for_completion()
        
        # Cleanup temp directory if we created it
        if hasattr(self, '_temp_dir_obj'):
            self._temp_dir_obj.cleanup()
        
        logger.info(f"Pipeline stopped. Stats: {self.get_stats()}")
    
    def _on_segment_created(self, segment) -> None:
        """Callback when a new segment is created."""
        logger.debug(f"Segment created: {segment.filename} (index: {segment.index})")
        
        if self.progress_callback:
            self.progress_callback({
                "event": "segment_created",
                "segment_index": segment.index,
                "segment_file": segment.filename,
                "duration": segment.duration,
                "stats": self.get_stats()
            })
    
    def _notify_progress(self, result: ProcessingResult) -> None:
        """Notify progress callback with processing result."""
        if self.progress_callback:
            event = "segment_processed" if not result.error else "segment_error"
            self.progress_callback({
                "event": event,
                "segment_index": result.segment_index,
                "highlights_found": len(result.highlights),
                "processing_time": result.processing_time,
                "error": result.error,
                "stats": self.get_stats()
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined pipeline statistics."""
        # Get stats from both components
        segmenter_stats = self.segmenter.get_stats()
        processor_stats = self.processor.get_stats()
        
        # Combine stats
        stats = {
            "pipeline_started": self.stats["pipeline_started"],
            "pipeline_stopped": self.stats["pipeline_stopped"],
            **segmenter_stats,
            **processor_stats
        }
        
        # Calculate pipeline uptime
        if stats["pipeline_started"]:
            end_time = stats["pipeline_stopped"] or datetime.now(timezone.utc)
            stats["pipeline_uptime_seconds"] = (end_time - stats["pipeline_started"]).total_seconds()
        else:
            stats["pipeline_uptime_seconds"] = 0.0
        
        return stats
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.segmenter.is_running and not self._shutdown
    
    @property
    def segment_directory(self) -> Path:
        """Get the directory where segments are stored."""
        return self.segmenter.segment_directory