"""Unified stream ingestion pipeline integrating FFmpeg with stream adapters.

This module provides a comprehensive pipeline that combines stream adapters,
FFmpeg processing, and content analysis for highlight detection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Callable

from ..adapters.stream.base import StreamAdapter
from ..adapters.stream.factory import get_stream_adapter
from ..media.ffmpeg_integration import (
    FFmpegProcessor,
    VideoFrameExtractor,
    TranscodeOptions,
    VideoCodec,
    AudioCodec,
)
from ..content_processing.video_processor import VideoProcessor, VideoProcessorConfig
from ..content_processing.audio_processor import AudioProcessor, AudioProcessorConfig
from ..content_processing.gemini_processor import GeminiProcessor, GeminiProcessorConfig
from ..streaming.segment_processor import (
    SegmentProcessor,
    SegmentConfig,
    ProcessedSegment,
)
from ..streaming.video_buffer import CircularVideoBuffer

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

    # Stream settings
    stream_url: str
    platform: Optional[str] = None

    # Processing options
    enable_video_processing: bool = True
    enable_audio_processing: bool = True
    enable_ai_analysis: bool = True
    enable_real_time: bool = True

    # FFmpeg settings
    transcode_options: Optional[TranscodeOptions] = None
    hardware_acceleration: bool = False
    frame_extraction_interval: float = 1.0

    # Buffer settings
    buffer_duration_seconds: float = 300.0  # 5 minutes
    segment_duration_seconds: float = 10.0

    # Quality thresholds
    min_video_quality: float = 0.3
    min_audio_quality: float = 0.3

    # Performance settings
    max_concurrent_segments: int = 5
    processing_timeout: float = 30.0

    # AI settings
    gemini_config: Optional[GeminiProcessorConfig] = None

    def __post_init__(self) -> None:
        """Initialize default configurations."""
        if self.gemini_config is None:
            self.gemini_config = GeminiProcessorConfig()

        if self.transcode_options is None:
            self.transcode_options = TranscodeOptions(
                video_codec=VideoCodec.H264,
                audio_codec=AudioCodec.AAC,
                quality="fast",
                hardware_acceleration=self.hardware_acceleration,
            )


@dataclass
class IngestionResult:
    """Result from stream ingestion processing."""

    timestamp: float
    segment: ProcessedSegment
    video_analysis: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    highlight_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamIngestionPipeline:
    """Unified pipeline for stream ingestion and processing.

    Integrates:
    - Stream adapters for platform-specific connection
    - FFmpeg for media processing and transcoding
    - Video/audio processors for content analysis
    - Gemini AI for intelligent highlight detection
    - Segment processing for windowed analysis
    """

    def __init__(self, config: IngestionConfig):
        """Initialize the ingestion pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.status = IngestionStatus.INITIALIZING

        # Core components
        self.stream_adapter: Optional[StreamAdapter] = None
        self.ffmpeg_processor = FFmpegProcessor(config.hardware_acceleration)
        self.frame_extractor = VideoFrameExtractor(config.hardware_acceleration)

        # Processing components
        self.video_processor: Optional[VideoProcessor] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.gemini_processor: Optional[GeminiProcessor] = None

        # Segmentation and buffering
        self.video_buffer: Optional[CircularVideoBuffer] = None
        self.segment_processor: Optional[SegmentProcessor] = None

        # Pipeline state
        self._processing_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._result_callbacks: List[Callable[[IngestionResult], None]] = []

        # Metrics
        self._metrics = {
            "segments_processed": 0,
            "frames_analyzed": 0,
            "audio_chunks_processed": 0,
            "ai_analyses": 0,
            "highlights_detected": 0,
            "errors": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
        }

        logger.info(f"Initialized StreamIngestionPipeline for {config.stream_url}")

    async def start(self) -> None:
        """Start the ingestion pipeline."""
        try:
            self.status = IngestionStatus.CONNECTING

            # Initialize stream adapter
            self.stream_adapter = await get_stream_adapter(
                self.config.stream_url, platform=self.config.platform
            )

            # Initialize processing components
            await self._initialize_processors()

            # Initialize buffer and segmentation
            await self._initialize_buffering()

            # Start stream connection
            await self.stream_adapter.start()

            # Start processing pipeline
            await self._start_processing_pipeline()

            self.status = IngestionStatus.PROCESSING
            logger.info("Stream ingestion pipeline started successfully")

        except Exception as e:
            self.status = IngestionStatus.ERROR
            logger.error(f"Failed to start ingestion pipeline: {e}")
            raise

    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        logger.info("Stopping stream ingestion pipeline")

        self.status = IngestionStatus.STOPPED
        self._shutdown_event.set()

        # Stop processing tasks
        for task in self._processing_tasks:
            task.cancel()

        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        # Stop stream adapter
        if self.stream_adapter:
            await self.stream_adapter.stop()

        # Cleanup processors
        if self.video_processor:
            await self.video_processor.cleanup()
        if self.audio_processor:
            await self.audio_processor.cleanup()
        if self.gemini_processor:
            await self.gemini_processor.cleanup()

        logger.info("Stream ingestion pipeline stopped")

    async def process_stream(self) -> AsyncIterator[IngestionResult]:
        """Process the stream and yield ingestion results.

        Yields:
            IngestionResult: Processing results with highlight analysis
        """
        if self.status != IngestionStatus.PROCESSING:
            raise RuntimeError("Pipeline not started or in error state")

        try:
            # Process segments from the segment processor
            async for segment in self.segment_processor.process_stream(
                stream_id=self.config.stream_url, real_time=self.config.enable_real_time
            ):
                start_time = asyncio.get_event_loop().time()

                # Process the segment through all modalities
                result = await self._process_segment(segment)

                # Update metrics
                processing_time = asyncio.get_event_loop().time() - start_time
                self._update_metrics(processing_time)

                # Notify callbacks
                await self._notify_callbacks(result)

                yield result

        except asyncio.CancelledError:
            logger.info("Stream processing cancelled")
            raise
        except Exception as e:
            self.status = IngestionStatus.ERROR
            self._metrics["errors"] += 1
            logger.error(f"Error in stream processing: {e}")
            raise

    async def _initialize_processors(self) -> None:
        """Initialize processing components."""
        # Video processor
        if self.config.enable_video_processing:
            video_config = VideoProcessorConfig(
                frame_interval_seconds=self.config.frame_extraction_interval,
                quality_threshold=self.config.min_video_quality,
            )
            self.video_processor = VideoProcessor(video_config)

        # Audio processor
        if self.config.enable_audio_processing:
            audio_config = AudioProcessorConfig(
                enable_transcription=True,
                enable_event_detection=True,
            )
            self.audio_processor = AudioProcessor(audio_config)

        # Gemini AI processor
        if self.config.enable_ai_analysis:
            self.gemini_processor = GeminiProcessor(self.config.gemini_config)

    async def _initialize_buffering(self) -> None:
        """Initialize buffering and segmentation."""
        # Create video buffer
        buffer_frames = int(self.config.buffer_duration_seconds * 30)  # Assume 30 FPS
        self.video_buffer = CircularVideoBuffer(
            capacity=buffer_frames, stream_id=self.config.stream_url
        )

        # Create segment processor
        segment_config = SegmentConfig(
            segment_duration_seconds=self.config.segment_duration_seconds,
            max_concurrent_segments=self.config.max_concurrent_segments,
        )
        self.segment_processor = SegmentProcessor(
            buffer=self.video_buffer, config=segment_config
        )

    async def _start_processing_pipeline(self) -> None:
        """Start the processing pipeline tasks."""
        # Start frame ingestion task
        frame_task = asyncio.create_task(self._ingest_frames())
        self._processing_tasks.append(frame_task)

        # Start audio ingestion task if enabled
        if self.config.enable_audio_processing:
            audio_task = asyncio.create_task(self._ingest_audio())
            self._processing_tasks.append(audio_task)

    async def _ingest_frames(self) -> None:
        """Ingest video frames from stream into buffer."""
        try:
            async for frame_data in self.stream_adapter.get_stream_data():
                if self._shutdown_event.is_set():
                    break

                # Extract frames using FFmpeg
                # This is a simplified version - real implementation would
                # process the stream data into actual video frames
                timestamp = datetime.now(timezone.utc).timestamp()

                # Add to buffer (mock frame for now)
                from ..streaming.video_buffer import VideoFrame

                frame = VideoFrame(
                    timestamp=timestamp,
                    data=frame_data,
                    width=1920,
                    height=1080,
                    format="rgb24",
                    is_keyframe=True,  # Simplified
                    metadata={"source": "stream_ingestion"},
                )

                await self.video_buffer.add_frame(frame)

        except asyncio.CancelledError:
            logger.info("Frame ingestion cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in frame ingestion: {e}")
            raise

    async def _ingest_audio(self) -> None:
        """Ingest audio data from stream."""
        if not self.audio_processor:
            return

        try:
            # Process audio stream
            async for audio_result in self.audio_processor.process_audio_stream(
                self.config.stream_url,
                duration_seconds=None,  # Continuous
            ):
                if self._shutdown_event.is_set():
                    break

                # Store audio analysis for segment correlation
                # This would be stored in a synchronized manner with video frames
                logger.debug(f"Processed audio chunk: {audio_result.timestamp}")

        except asyncio.CancelledError:
            logger.info("Audio ingestion cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in audio ingestion: {e}")
            raise

    async def _process_segment(self, segment: ProcessedSegment) -> IngestionResult:
        """Process a video segment through all modalities.

        Args:
            segment: Video segment to process

        Returns:
            Complete ingestion result with analysis
        """
        result = IngestionResult(
            timestamp=segment.start_time,
            segment=segment,
            metadata={"segment_id": segment.segment_id},
        )

        # Video analysis
        if self.config.enable_video_processing and self.video_processor:
            try:
                # Extract key frames for analysis
                keyframes = segment.get_keyframes()
                if keyframes:
                    # Simplified video analysis
                    result.video_analysis = {
                        "frame_count": len(keyframes),
                        "keyframe_count": len(segment.keyframe_indices),
                        "quality_score": segment.quality_score,
                        "motion_score": segment.motion_score,
                    }
                    self._metrics["frames_analyzed"] += len(keyframes)

            except Exception as e:
                logger.error(f"Video analysis error: {e}")
                result.metadata["video_error"] = str(e)

        # Audio analysis (simplified - would correlate with segment timing)
        if self.config.enable_audio_processing and self.audio_processor:
            try:
                # Mock audio analysis for segment
                result.audio_analysis = {
                    "transcription": f"Audio content for segment {segment.segment_id}",
                    "volume_level": 0.7,
                    "detected_events": ["speech", "background_music"],
                }
                self._metrics["audio_chunks_processed"] += 1

            except Exception as e:
                logger.error(f"Audio analysis error: {e}")
                result.metadata["audio_error"] = str(e)

        # AI analysis using Gemini
        if self.config.enable_ai_analysis and self.gemini_processor:
            try:
                # Multimodal analysis
                keyframes = segment.get_keyframes()
                if keyframes and result.audio_analysis:
                    # Get first keyframe as representative
                    frame_data = keyframes[0].data
                    audio_transcript = result.audio_analysis.get("transcription", "")

                    ai_result = await self.gemini_processor.analyze_multimodal(
                        video_frame=frame_data,
                        audio_transcript=audio_transcript,
                        chat_messages=[],  # Would need chat integration
                        timestamp=segment.start_time,
                    )

                    result.ai_analysis = {
                        "analysis_text": ai_result.analysis_text,
                        "highlight_score": ai_result.highlight_score,
                        "detected_elements": ai_result.detected_elements,
                        "suggested_tags": ai_result.suggested_tags,
                        "confidence": ai_result.confidence,
                    }

                    result.highlight_score = ai_result.highlight_score
                    self._metrics["ai_analyses"] += 1

                    if ai_result.highlight_score > 0.7:
                        self._metrics["highlights_detected"] += 1

            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                result.metadata["ai_error"] = str(e)

        self._metrics["segments_processed"] += 1
        return result

    def add_result_callback(self, callback: Callable[[IngestionResult], None]) -> None:
        """Add callback for processing results.

        Args:
            callback: Function to call with each result
        """
        self._result_callbacks.append(callback)

    async def _notify_callbacks(self, result: IngestionResult) -> None:
        """Notify result callbacks."""
        for callback in self._result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")

    def _update_metrics(self, processing_time: float) -> None:
        """Update processing metrics."""
        self._metrics["total_processing_time"] += processing_time

        if self._metrics["segments_processed"] > 0:
            self._metrics["avg_processing_time"] = (
                self._metrics["total_processing_time"]
                / self._metrics["segments_processed"]
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline processing metrics."""
        metrics = self._metrics.copy()

        # Add component metrics
        if self.stream_adapter:
            metrics["stream_adapter"] = self.stream_adapter.get_metrics_summary()

        if self.video_processor:
            metrics["video_processor"] = self.video_processor.get_processing_stats()

        if self.audio_processor:
            metrics["audio_processor"] = self.audio_processor.get_processing_stats()

        if self.gemini_processor:
            metrics["gemini_processor"] = self.gemini_processor.get_processing_stats()

        return metrics

    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health information."""
        health = {
            "status": self.status.value,
            "stream_connected": self.stream_adapter.is_connected
            if self.stream_adapter
            else False,
            "stream_healthy": self.stream_adapter.is_healthy
            if self.stream_adapter
            else False,
            "buffer_health": "healthy",  # Would check buffer status
            "processing_health": "healthy"
            if self.status == IngestionStatus.PROCESSING
            else "unhealthy",
            "error_rate": self._metrics["errors"]
            / max(self._metrics["segments_processed"], 1),
        }

        return health
