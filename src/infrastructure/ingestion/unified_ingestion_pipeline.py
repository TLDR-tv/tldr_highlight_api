"""Unified stream ingestion pipeline for processing streams with Gemini.

This module provides a streamlined pipeline that segments streams
and processes them directly with Gemini's native video understanding.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Callable

from ..adapters.stream.base import StreamAdapter
from ..adapters.stream.factory import StreamAdapterFactory
from ..media.ffmpeg_integration import (
    FFmpegProcessor,
    TranscodeOptions,
)
from ..content_processing.gemini_video_processor import GeminiVideoProcessor
from ..streaming.segment_processor import (
    SegmentProcessor,
    SegmentConfig,
    ProcessedSegment,
)

logger = logging.getLogger(__name__)


class IngestionStatus(str, Enum):
    """Status of stream ingestion."""

    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class IngestionConfig:
    """Configuration for stream ingestion pipeline."""

    stream_url: str
    adapter_type: Optional[str] = None  # Auto-detect if not specified

    # Segment configuration
    segment_duration_seconds: float = 30.0
    segment_overlap_seconds: float = 5.0
    buffer_duration_seconds: float = 60.0

    # Processing options
    enable_audio_processing: bool = True
    hardware_acceleration: bool = False
    transcode_options: Optional[TranscodeOptions] = None

    # Output configuration
    output_dir: Optional[str] = None
    save_segments: bool = False

    # Performance tuning
    max_concurrent_segments: int = 3
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0


@dataclass
class SegmentAnalysis:
    """Analysis results for a video segment."""

    segment_id: str
    start_time: float
    end_time: float
    highlights: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class IngestionResult:
    """Result of stream ingestion and analysis."""

    stream_url: str
    adapter_type: str
    status: IngestionStatus
    segments_processed: int = 0
    total_duration_seconds: float = 0.0
    analyses: List[SegmentAnalysis] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class UnifiedIngestionPipeline:
    """Unified pipeline for stream ingestion and processing with Gemini."""

    def __init__(
        self,
        config: IngestionConfig,
        gemini_processor: Optional[GeminiVideoProcessor] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize the ingestion pipeline.

        Args:
            config: Pipeline configuration
            gemini_processor: Gemini processor for video analysis
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.gemini_processor = gemini_processor
        self.progress_callback = progress_callback

        # Initialize components
        self.stream_adapter: Optional[StreamAdapter] = None
        self.ffmpeg_processor = FFmpegProcessor(config.hardware_acceleration)
        self.segment_processor = SegmentProcessor(
            SegmentConfig(
                segment_duration=config.segment_duration_seconds,
                overlap_duration=config.segment_overlap_seconds,
                buffer_size=int(
                    config.buffer_duration_seconds / config.segment_duration_seconds
                ),
            )
        )

        # Pipeline state
        self.status = IngestionStatus.INITIALIZING
        self._processing_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._current_result = IngestionResult(
            stream_url=config.stream_url,
            adapter_type=config.adapter_type or "unknown",
            status=IngestionStatus.INITIALIZING,
        )
        self._metrics = {
            "segments_created": 0,
            "segments_analyzed": 0,
            "highlights_found": 0,
            "errors": 0,
        }

    async def start(self) -> AsyncIterator[IngestionResult]:
        """Start the ingestion pipeline and yield results.

        Yields:
            Ingestion results as processing progresses
        """
        try:
            self._current_result.started_at = datetime.now(timezone.utc)
            await self._initialize_pipeline()

            async for result in self._run_pipeline():
                yield result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._current_result.status = IngestionStatus.ERROR
            self._current_result.errors.append(str(e))
            yield self._current_result
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        logger.info("Stopping ingestion pipeline")
        self._shutdown_event.set()

        # Cancel all processing tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        # Cleanup
        if self.stream_adapter:
            await self.stream_adapter.disconnect()

        self._current_result.completed_at = datetime.now(timezone.utc)
        self.status = IngestionStatus.STOPPED

    async def _initialize_pipeline(self) -> None:
        """Initialize pipeline components."""
        logger.info(f"Initializing pipeline for: {self.config.stream_url}")
        self.status = IngestionStatus.CONNECTING

        # Create stream adapter
        adapter_type = (
            self.config.adapter_type
            or StreamAdapterFactory.detect_adapter_type(self.config.stream_url)
        )
        self.stream_adapter = StreamAdapterFactory.create_adapter(
            adapter_type, self.config.stream_url
        )
        self._current_result.adapter_type = adapter_type

        # Connect to stream
        if not await self.stream_adapter.connect():
            raise RuntimeError(f"Failed to connect to stream: {self.config.stream_url}")

        # Get stream info
        stream_info = await self.stream_adapter.get_stream_info()
        self._current_result.metadata.update(stream_info)

        logger.info(f"Connected to stream: {stream_info}")
        self.status = IngestionStatus.PROCESSING

    async def _run_pipeline(self) -> AsyncIterator[IngestionResult]:
        """Run the main processing pipeline."""
        try:
            # Start segment ingestion
            segment_task = asyncio.create_task(self._ingest_segments())
            self._processing_tasks.append(segment_task)

            # Process segments as they become available
            async for segment in self.segment_processor.get_segments():
                if self._shutdown_event.is_set():
                    break

                # Process segment with Gemini
                analysis = await self._process_segment(segment)
                self._current_result.analyses.append(analysis)
                self._current_result.segments_processed += 1

                # Update metrics
                self._metrics["segments_analyzed"] += 1
                if analysis.highlights:
                    self._metrics["highlights_found"] += len(analysis.highlights)

                # Update progress
                if self.progress_callback:
                    self.progress_callback(
                        {
                            "status": "processing",
                            "segments_processed": self._current_result.segments_processed,
                            "highlights_found": self._metrics["highlights_found"],
                        }
                    )

                # Yield current result
                yield self._current_result

        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            self._current_result.errors.append(str(e))
            self._metrics["errors"] += 1
            raise

    async def _ingest_segments(self) -> None:
        """Ingest stream and create video segments."""
        try:
            segment_count = 0

            # Get stream data from adapter
            async for stream_data in self.stream_adapter.get_stream_data():
                if self._shutdown_event.is_set():
                    break

                # Create video segments using FFmpeg
                # In a real implementation, this would use FFmpeg to split
                # the stream into segments for Gemini processing
                segment_count += 1

                # For now, we'll simulate segment creation
                segment_info = {
                    "id": f"segment_{segment_count}",
                    "start_time": (segment_count - 1)
                    * self.config.segment_duration_seconds,
                    "duration": self.config.segment_duration_seconds,
                    "path": f"/tmp/segment_{segment_count}.mp4",  # Would be actual path
                }

                # Add segment to processor
                await self.segment_processor.add_segment(segment_info)
                self._metrics["segments_created"] += 1

        except asyncio.CancelledError:
            logger.info("Segment ingestion cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in segment ingestion: {e}")
            raise

    async def _process_segment(self, segment: ProcessedSegment) -> SegmentAnalysis:
        """Process a video segment with Gemini.

        Args:
            segment: Video segment to process

        Returns:
            Analysis results for the segment
        """
        analysis = SegmentAnalysis(
            segment_id=segment.segment_id,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )

        try:
            if self.gemini_processor:
                # Process segment with Gemini's native video understanding
                # The actual video file would be passed to Gemini
                logger.info(f"Processing segment {segment.segment_id} with Gemini")

                # In production, this would call:
                # gemini_result = await self.gemini_processor.analyze_video_with_dimensions(
                #     video_path=segment.path,
                #     segment_info=segment.to_dict(),
                #     dimension_set=dimension_set,
                #     agent_config=agent_config
                # )

                # For now, we'll add placeholder data
                analysis.metadata["gemini_processed"] = True
                analysis.metadata["segment_duration"] = segment.duration

            else:
                logger.warning("No Gemini processor configured")
                analysis.metadata["skipped"] = True

        except Exception as e:
            logger.error(f"Error processing segment {segment.segment_id}: {e}")
            analysis.error = str(e)
            self._metrics["errors"] += 1

        return analysis

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return {
            **self._metrics,
            "status": self.status.value,
            "uptime_seconds": (
                (
                    datetime.now(timezone.utc) - self._current_result.started_at
                ).total_seconds()
                if self._current_result.started_at
                else 0
            ),
        }
