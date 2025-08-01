"""Streamlined Celery tasks for FFmpeg stream ingestion and AI highlight detection.

This module provides the core Celery tasks for the B2B highlight detection pipeline:
1. FFmpeg-based stream ingestion and chunking
2. AI-powered highlight detection using B2BStreamAgent
"""

import asyncio
import tempfile
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from celery import Task
import logfire

from src.infrastructure.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.async_processing.celery_app import celery_app
from src.infrastructure.media.clip_generator import ClipGenerator
from src.infrastructure.media.thumbnail_generator import ThumbnailGenerator
from src.infrastructure.media.ffmpeg_integration import (
    FFmpegProcessor,
    VideoFrameExtractor,
    FFmpegProbe,
    TranscodeOptions,
    VideoCodec,
    AudioCodec,
    ContainerFormat,
)
from src.infrastructure.storage.s3_storage import S3Storage
from src.domain.services.b2b_stream_agent import B2BStreamAgent
from src.infrastructure.async_processing.error_handler import ErrorHandler
from src.infrastructure.async_processing.progress_tracker import (
    ProgressTracker,
    ProgressEvent,
)
from src.infrastructure.async_processing.webhook_dispatcher import (
    WebhookDispatcher,
    WebhookEvent,
)
from src.infrastructure.observability import (
    metrics,
    traced_background_task,
)

logger = structlog.get_logger(__name__)


class StreamProcessingTask(Task):
    """Base task class for stream processing with B2B agent integration."""

    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60
    retry_backoff = True
    retry_jitter = True

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.progress_tracker = ProgressTracker()
        self.webhook_dispatcher = WebhookDispatcher()
        self.ffmpeg_processor = FFmpegProcessor(hardware_acceleration=True)
        self.frame_extractor = VideoFrameExtractor(use_hardware_acceleration=True)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with B2B agent cleanup."""
        logger.error(
            "Stream processing task failed",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            args=args,
            kwargs=kwargs,
        )

        if args and len(args) > 0:
            stream_id = args[0]
            try:
                # Update progress with failure
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=0,
                    status="failed",
                    event_type=ProgressEvent.ERROR,
                    details={"error": str(exc), "task": self.name},
                )

                # Send failure webhook
                asyncio.create_task(
                    self.webhook_dispatcher.dispatch_webhook(
                        stream_id=stream_id,
                        event=WebhookEvent.ERROR_OCCURRED,
                        data={
                            "error": str(exc),
                            "task": self.name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
            except Exception as e:
                logger.error("Failed to handle task failure", error=str(e))


@celery_app.task(bind=True, base=StreamProcessingTask, name="ingest_stream_with_ffmpeg")
@traced_background_task(name="ingest_stream_with_ffmpeg")
def ingest_stream_with_ffmpeg(
    self,
    stream_id: int,
    chunk_duration: int = 30,
    agent_config_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ingest stream using FFmpeg and create segments for AI analysis.

    This task:
    1. Uses FFmpeg to connect to the stream source
    2. Creates video/audio segments for processing
    3. Extracts keyframes for visual analysis
    4. Prepares data for AI highlight detection

    Args:
        stream_id: Stream ID to process
        chunk_duration: Duration of each chunk in seconds
        agent_config_id: Optional B2B agent configuration ID

    Returns:
        Dict with ingestion results and segment information
    """
    with logfire.span("ingest_stream_with_ffmpeg.start") as span:
        span.set_attribute("stream.id", stream_id)
        span.set_attribute("chunk_duration", chunk_duration)
        span.set_attribute("agent_config_id", agent_config_id)

        logger.info(
            "Starting FFmpeg stream ingestion",
            stream_id=stream_id,
            chunk_duration=chunk_duration,
        )

        # Track task start
        metrics.increment_task_executed(
            task_name="ingest_stream_with_ffmpeg",
            organization_id=None,  # Will be set after we fetch stream
            success=False,  # Will update on success
        )

    try:
        with get_db_session() as db:
            # Get stream with observability
            with logfire.span("fetch_stream_details") as span:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if not stream:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", f"Stream {stream_id} not found")
                    raise ValueError(f"Stream {stream_id} not found")

                # Add stream context
                span.set_attribute("stream.platform", stream.platform)
                span.set_attribute(
                    "stream.source_url",
                    stream.source_url[:50] + "..."
                    if len(stream.source_url) > 50
                    else stream.source_url,
                )
                span.set_attribute("stream.organization_id", stream.organization_id)

                # Now we can track with organization context
                logfire.set_attribute("organization.id", stream.organization_id)
                metrics.increment_stream_started(
                    platform=stream.platform,
                    organization_id=str(stream.organization_id),
                    stream_type="live",
                )

            # Update status to processing
            with logfire.span("update_stream_status"):
                stream.status = StreamStatus.PROCESSING
                db.commit()

            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=10,
                status="processing",
                event_type=ProgressEvent.STARTED,
                details={
                    "task": "ffmpeg_stream_ingestion",
                    "chunk_duration": chunk_duration,
                },
            )

            # Probe stream first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            with logfire.span("probe_stream") as probe_span:
                probe_span.set_attribute("probe.timeout", 15)
                probe_start_time = datetime.now(timezone.utc)

                try:
                    media_info = loop.run_until_complete(
                        FFmpegProbe.probe_stream(stream.source_url, timeout=15)
                    )

                    probe_duration = (
                        datetime.now(timezone.utc) - probe_start_time
                    ).total_seconds()
                    probe_span.set_attribute("probe.duration_seconds", probe_duration)
                    probe_span.set_attribute("probe.format", media_info.format_name)
                    probe_span.set_attribute(
                        "probe.video_streams", len(media_info.video_streams)
                    )
                    probe_span.set_attribute(
                        "probe.audio_streams", len(media_info.audio_streams)
                    )
                    probe_span.set_attribute("probe.success", True)

                    logger.info(
                        "Stream probed successfully",
                        format=media_info.format_name,
                        video_streams=len(media_info.video_streams),
                        audio_streams=len(media_info.audio_streams),
                    )

                    # Track probe metrics
                    metrics.record_highlight_processing_time(
                        duration_seconds=probe_duration,
                        stage="stream_probe",
                        platform=stream.platform,
                    )

                except Exception as e:
                    probe_span.set_attribute("probe.success", False)
                    probe_span.set_attribute("probe.error", str(e))
                    logger.error(f"Failed to probe stream: {e}")
                    raise ValueError(f"Cannot access stream: {e}")

            # Create temporary working directory
            temp_dir = tempfile.mkdtemp(prefix=f"stream_{stream_id}_")

            try:
                # Start stream ingestion with chunking
                with logfire.span("ingest_stream_segments") as segment_span:
                    segment_span.set_attribute(
                        "segments.chunk_duration", chunk_duration
                    )
                    segment_span.set_attribute("segments.temp_dir", temp_dir)
                    segment_start_time = datetime.now(timezone.utc)

                    segments = loop.run_until_complete(
                        self._ingest_stream_segments(
                            stream.source_url, temp_dir, chunk_duration, media_info
                        )
                    )

                    segment_duration = (
                        datetime.now(timezone.utc) - segment_start_time
                    ).total_seconds()
                    segment_span.set_attribute("segments.count", len(segments))
                    segment_span.set_attribute(
                        "segments.duration_seconds", segment_duration
                    )
                    segment_span.set_attribute("segments.success", True)

                    # Track segment creation metrics
                    metrics.record_highlight_processing_time(
                        duration_seconds=segment_duration,
                        stage="segment_creation",
                        platform=stream.platform,
                    )

                # Update progress
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=40,
                    status="processing",
                    event_type=ProgressEvent.PROGRESS_UPDATE,
                    details={
                        "task": "stream_ingestion_complete",
                        "segments_created": len(segments),
                        "temp_dir": temp_dir,
                    },
                )

                # Trigger AI highlight detection task
                with logfire.span("trigger_ai_detection") as trigger_span:
                    trigger_span.set_attribute("next_task", "detect_highlights_with_ai")

                    detect_highlights_with_ai.delay(
                        stream_id=stream_id,
                        ingestion_data={
                            "segments": segments,
                            "temp_dir": temp_dir,
                            "media_info": {
                                "format": media_info.format_name,
                                "duration": media_info.duration,
                                "video_streams": len(media_info.video_streams),
                                "audio_streams": len(media_info.audio_streams),
                            },
                        },
                        agent_config_id=agent_config_id,
                    )

                    trigger_span.set_attribute("trigger.success", True)

                # Calculate total task duration
                task_end_time = datetime.now(timezone.utc)
                total_duration = (
                    task_end_time - datetime.now(timezone.utc)
                ).total_seconds()

                # Track successful completion
                metrics.increment_task_executed(
                    task_name="ingest_stream_with_ffmpeg",
                    organization_id=str(stream.organization_id),
                    success=True,
                )

                metrics.record_task_duration(
                    duration_seconds=total_duration,
                    task_name="ingest_stream_with_ffmpeg",
                    organization_id=str(stream.organization_id),
                )

                result = {
                    "stream_id": stream_id,
                    "status": "ingestion_complete",
                    "segments_created": len(segments),
                    "media_info": {
                        "format": media_info.format_name,
                        "duration": media_info.duration,
                        "video_streams": len(media_info.video_streams),
                        "audio_streams": len(media_info.audio_streams),
                    },
                    "temp_dir": temp_dir,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                logfire.info(
                    "Stream ingestion completed successfully",
                    stream_id=stream_id,
                    segments_created=len(segments),
                    platform=stream.platform,
                    organization_id=stream.organization_id,
                )

                return result

            except Exception as e:
                # Clean up temp directory on failure
                import shutil

                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise e
            finally:
                loop.close()

    except Exception as exc:
        logfire.error(
            "FFmpeg stream ingestion failed",
            stream_id=stream_id,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )

        # Track task failure
        if "stream" in locals() and hasattr(stream, "organization_id"):
            metrics.increment_task_executed(
                task_name="ingest_stream_with_ffmpeg",
                organization_id=str(stream.organization_id),
                success=False,
            )

            metrics.increment_stream_completed(
                platform=stream.platform,
                organization_id=str(stream.organization_id),
                stream_type="live",
                success=False,
            )

        # Update stream status to failed
        try:
            with get_db_session() as db:
                with logfire.span("update_failed_stream_status"):
                    stream = db.query(Stream).filter(Stream.id == stream_id).first()
                    if stream:
                        stream.status = StreamStatus.FAILED
                        stream.error_message = str(exc)
                        db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))

        raise exc

    async def _ingest_stream_segments(
        self, stream_url: str, temp_dir: str, chunk_duration: int, media_info
    ) -> List[Dict[str, Any]]:
        """Ingest stream and create segments using FFmpeg."""
        segments = []

        try:
            # Create segments directory
            segments_dir = os.path.join(temp_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)

            # Determine optimal transcoding options based on stream
            video_info = (
                media_info.video_streams[0] if media_info.video_streams else None
            )
            audio_info = (
                media_info.audio_streams[0] if media_info.audio_streams else None
            )

            transcode_options = TranscodeOptions(
                video_codec=VideoCodec.H264,
                audio_codec=AudioCodec.AAC,
                video_bitrate=2000,  # 2Mbps
                audio_bitrate=128,  # 128kbps
                quality="fast",
                container=ContainerFormat.MP4,
            )

            # For now, create one segment for the entire stream
            # In production, you'd implement proper chunking
            segment_path = os.path.join(segments_dir, "segment_001.mp4")

            success = await self.ffmpeg_processor.transcode_stream(
                input_source=stream_url,
                output_path=segment_path,
                options=transcode_options,
            )

            if success and os.path.exists(segment_path):
                segment_info = {
                    "id": "segment_001",
                    "path": segment_path,
                    "start_time": 0,
                    "duration": chunk_duration,
                    "video_info": {
                        "width": video_info.width if video_info else 0,
                        "height": video_info.height if video_info else 0,
                        "fps": video_info.fps if video_info else 0,
                        "codec": video_info.codec if video_info else "unknown",
                    },
                    "audio_info": {
                        "sample_rate": audio_info.sample_rate if audio_info else 0,
                        "channels": audio_info.channels if audio_info else 0,
                        "codec": audio_info.codec if audio_info else "unknown",
                    },
                }
                segments.append(segment_info)

                logger.info(f"Created segment: {segment_path}")

            return segments

        except Exception as e:
            logger.error(f"Failed to create stream segments: {e}")
            raise

    


@celery_app.task(bind=True, base=StreamProcessingTask, name="detect_highlights_with_ai")
@traced_background_task(name="detect_highlights_with_ai")
def detect_highlights_with_ai(
    self,
    stream_id: int,
    ingestion_data: Dict[str, Any],
    agent_config_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect highlights using B2B AI agent with consumer-specific configuration.

    This task:
    1. Initializes the B2BStreamAgent with appropriate configuration
    2. Processes each segment using the agent's AI analysis
    3. Creates highlights based on agent's recommendations
    4. Stores results and triggers completion notifications

    Args:
        stream_id: Stream ID to process
        ingestion_data: Results from FFmpeg ingestion task
        agent_config_id: Optional B2B agent configuration ID

    Returns:
        Dict with highlight detection results
    """
    with logfire.span("detect_highlights_with_ai.start") as span:
        span.set_attribute("stream.id", stream_id)
        span.set_attribute("agent_config_id", agent_config_id)
        span.set_attribute("segments.count", len(ingestion_data.get("segments", [])))
        span.set_attribute("keyframes.count", len(ingestion_data.get("keyframes", [])))

        logger.info(
            "Starting AI highlight detection",
            stream_id=stream_id,
            agent_config_id=agent_config_id,
        )

        # Track task start
        metrics.increment_task_executed(
            task_name="detect_highlights_with_ai",
            organization_id=None,  # Will be set after we fetch stream
            success=False,  # Will update on success
        )

    try:
        with get_db_session() as db:
            # Get stream with observability
            with logfire.span("fetch_stream_for_ai_detection") as span:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if not stream:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", f"Stream {stream_id} not found")
                    raise ValueError(f"Stream {stream_id} not found")

                # Add stream context
                span.set_attribute("stream.platform", stream.platform)
                span.set_attribute("stream.organization_id", stream.organization_id)

                # Set organization context for metrics
                logfire.set_attribute("organization.id", stream.organization_id)

            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=50,
                status="processing",
                event_type=ProgressEvent.PROGRESS_UPDATE,
                details={"task": "ai_highlight_detection_started"},
            )

            # Initialize B2B agent configuration (simplified)
            from src.domain.entities.stream_processing_config import StreamProcessingConfig

            # Use default config if none specified
            if agent_config_id:
                # In real implementation, fetch from repository
                # For now, create a simplified config for streamlined processing
                agent_config = StreamProcessingConfig.create_default(
                    organization_id=stream.organization_id,
                    user_id=stream.user_id,
                    dimension_set_id=1,  # Would be fetched from config
                    name="Default Stream Processing"
                )
            else:
                # Create default simplified config
                agent_config = StreamProcessingConfig.create_default(
                    organization_id=stream.organization_id,
                    user_id=stream.user_id,
                    dimension_set_id=1,  # Would be fetched from config
                    name="Default Stream Processing"
                )

            # Initialize Gemini processor (now primary method)
            gemini_processor = None
            dimension_set = None
            from src.infrastructure.config import settings

            if hasattr(settings, "gemini_api_key") and settings.gemini_api_key:
                with logfire.span("initialize_gemini_processor") as gemini_span:
                    from src.infrastructure.content_processing import (
                        GeminiVideoProcessor,
                    )
                    from src.application.services.dimension_set_service import DimensionSetService
                    from src.infrastructure.persistence.repositories.dimension_set_repository import DimensionSetRepository
                    from src.infrastructure.database import get_async_session

                    gemini_processor = GeminiVideoProcessor(
                        api_key=settings.gemini_api_key,
                        model_name=getattr(
                            settings, "gemini_model", "gemini-2.0-flash-exp"
                        ),
                    )
                    gemini_span.set_attribute("gemini.enabled", True)
                    gemini_span.set_attribute(
                        "gemini.model", gemini_processor.model_name
                    )

                    # Get dimension set through application service
                    async with get_async_session() as session:
                        dimension_set_repo = DimensionSetRepository(session)
                        dimension_set_service = DimensionSetService(dimension_set_repo)
                        
                        dimension_set = await dimension_set_service.get_dimension_set_for_stream(
                            dimension_set_id=stream.dimension_set_id,
                            organization_id=stream.organization_id or 1,
                            user_id=stream.user_id,
                        )
                    
                    gemini_span.set_attribute("dimension_set.name", dimension_set.name)
                    gemini_span.set_attribute(
                        "dimension_count", len(dimension_set.dimensions)
                    )
            else:
                logfire.error(
                    "Gemini API key not configured - highlight detection will fail"
                )
                raise ValueError("Gemini API key is required for highlight detection")

            # Create B2B agent with observability
                with logfire.span("initialize_b2b_agent") as agent_span:
                    agent_span.set_attribute("agent_config_id", agent_config_id)
                    agent_span.set_attribute("agent_config.type", agent_config.config_type)
                    agent_span.set_attribute("gemini.enabled", gemini_processor is not None)

                    b2b_agent = B2BStreamAgent(
                        stream=stream,
                        agent_config=agent_config,
                        gemini_processor=gemini_processor,
                        dimension_set=dimension_set,
                    )

                    agent_span.set_attribute("agent.initialized", True)

                # Initialize clip, thumbnail, and caption generators
                clip_generator = ClipGenerator()
                thumbnail_generator = ThumbnailGenerator()
                # Caption generation is now handled by GeminiVideoProcessor
                s3_storage = S3Storage(
                    access_key_id=settings.aws_access_key_id,
                    secret_access_key=settings.aws_secret_access_key,
                    bucket_name=settings.aws_s3_bucket,
                    endpoint_url=settings.aws_s3_endpoint_url,
                    region_name=settings.aws_region,
                )

                # Start the agent
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    with logfire.span("start_b2b_agent"):
                        loop.run_until_complete(b2b_agent.start())
                        logfire.info("B2B agent started successfully", stream_id=stream_id)

                    # Process each segment
                    all_highlights = []
                    segments = ingestion_data.get("segments", [])

                    for i, segment in enumerate(segments):
                        with logfire.span(f"process_segment_{i}") as segment_span:
                            segment_span.set_attribute("segment.index", i)
                            segment_span.set_attribute("segment.id", segment["id"])
                            segment_span.set_attribute(
                                "segment.start_time", segment["start_time"]
                            )
                            segment_span.set_attribute(
                                "segment.duration", segment["duration"]
                            )

                            # Create segment data for analysis
                            segment_data = {
                                "id": segment["id"],
                                "path": segment["path"],
                                "start_time": segment["start_time"],
                                "duration": segment["duration"],
                                "video_info": segment["video_info"],
                                "audio_info": segment["audio_info"],
                            }

                            # Process segment with Gemini (now primary method)
                            candidates = []

                            # Use Gemini analysis for all video segments
                            if segment.get("path"):
                                with logfire.span(
                                    "analyze_video_segment_gemini"
                                ) as analyze_span:
                                    analyze_span.set_attribute(
                                        "method", "gemini_dimensions"
                                    )
                                    analyze_span.set_attribute(
                                        "dimension_set",
                                        dimension_set.name if dimension_set else "none",
                                    )
                                    analyze_start = datetime.now(timezone.utc)

                                    try:
                                        candidates = loop.run_until_complete(
                                            b2b_agent.analyze_video_segment_with_gemini(
                                                segment_data
                                            )
                                        )

                                        analyze_duration = (
                                            datetime.now(timezone.utc) - analyze_start
                                        ).total_seconds()
                                        analyze_span.set_attribute(
                                            "candidates.count", len(candidates)
                                        )
                                        analyze_span.set_attribute(
                                            "analysis.duration_seconds", analyze_duration
                                        )
                                        analyze_span.set_attribute("success", True)

                                        # Track Gemini analysis metrics
                                        metrics.record_highlight_processing_time(
                                            duration_seconds=analyze_duration,
                                            stage="gemini_video_analysis",
                                            platform=stream.platform,
                                        )
                                    except Exception as gemini_error:
                                        analyze_span.set_attribute("success", False)
                                        analyze_span.set_attribute(
                                            "error", str(gemini_error)
                                        )
                                        logger.error(
                                            f"Gemini analysis failed for segment {segment_data.get('id')}: {gemini_error}",
                                            exc_info=True,
                                        )
                                        # Re-raise to fail the task - Gemini is now required
                                        raise

                            # Create highlights from candidates
                            segment_highlights = []
                            for candidate in candidates:
                                with logfire.span("evaluate_highlight_candidate"):
                                    should_create = loop.run_until_complete(
                                        b2b_agent.should_create_highlight(candidate)
                                    )

                                    if should_create:
                                        highlight = loop.run_until_complete(
                                            b2b_agent.create_highlight(candidate)
                                        )
                                        if highlight:
                                            # Generate clip, thumbnail, and caption
                                            clip_path = await clip_generator.generate_clip(
                                                source_path=segment["path"],
                                                output_dir=ingestion_data["temp_dir"],
                                                start_time=highlight.start_time,
                                                duration=highlight.duration,
                                            )
                                            thumbnail_path = await thumbnail_generator.generate_thumbnail(
                                                video_path=clip_path,
                                                output_dir=ingestion_data["temp_dir"],
                                            )
                                            caption = await caption_generator.generate_caption(
                                                video_path=clip_path
                                            )

                                            # Upload to S3
                                            clip_url = await s3_storage.upload_file(
                                                file_path=clip_path,
                                                object_name=f"clips/{highlight.id}.mp4",
                                            )
                                            thumbnail_url = await s3_storage.upload_file(
                                                file_path=thumbnail_path,
                                                object_name=f"thumbnails/{highlight.id}.jpg",
                                            )

                                            # Update highlight with URLs and caption
                                            highlight.video_url = clip_url
                                            highlight.thumbnail_url = thumbnail_url
                                            highlight.caption = caption

                                            all_highlights.append(highlight)
                                            segment_highlights.append(highlight)

                            segment_span.set_attribute(
                                "highlights.created", len(segment_highlights)
                            )

                        # Update progress
                        progress = 50 + (i + 1) * 30 / len(segments)
                        self.progress_tracker.update_progress(
                            stream_id=stream_id,
                            progress_percentage=progress,
                            status="processing",
                            event_type=ProgressEvent.PROGRESS_UPDATE,
                            details={
                                "task": "segment_analysis_complete",
                                "segment": i + 1,
                                "total_segments": len(segments),
                                "highlights_so_far": len(all_highlights),
                            },
                        )

                    # Stop the agent
                    with logfire.span("stop_b2b_agent"):
                        loop.run_until_complete(b2b_agent.stop())
                        logfire.info("B2B agent stopped successfully", stream_id=stream_id)

                    # Update stream status to completed
                    with logfire.span("update_stream_completed"):
                        stream.status = StreamStatus.COMPLETED
                        stream.completed_at = datetime.now(timezone.utc)
                        db.commit()

                        # Track completion metrics
                        metrics.increment_highlights_detected(
                            count=len(all_highlights),
                            platform=stream.platform,
                            organization_id=str(stream.organization_id),
                            detection_method="b2b_agent",
                        )

                        metrics.increment_stream_completed(
                            platform=stream.platform,
                            organization_id=str(stream.organization_id),
                            stream_type="live",
                            success=True,
                        )

                    # Clean up Gemini files if processor was used
                    if gemini_processor:
                        with logfire.span("cleanup_gemini_files"):
                            try:
                                loop.run_until_complete(
                                    gemini_processor.cleanup_all_files()
                                )
                                logfire.info("Cleaned up all Gemini uploaded files")
                            except Exception as cleanup_error:
                                logger.error(
                                    f"Failed to cleanup Gemini files: {cleanup_error}"
                                )

                    # Clean up temporary files
                    with logfire.span("cleanup_temp_files"):
                        temp_dir = ingestion_data.get("temp_dir")
                        if temp_dir and os.path.exists(temp_dir):
                            import shutil

                            shutil.rmtree(temp_dir)
                            logfire.info("Cleaned up temporary files", temp_dir=temp_dir)

                    # Final progress update
                    self.progress_tracker.update_progress(
                        stream_id=stream_id,
                        progress_percentage=100,
                        status="completed",
                        event_type=ProgressEvent.COMPLETED,
                        details={
                            "task": "ai_highlight_detection_complete",
                            "total_highlights": len(all_highlights),
                            "agent_metrics": b2b_agent.get_performance_metrics(),
                        },
                )

                    # Send completion webhook
                    asyncio.create_task(
                        self.webhook_dispatcher.dispatch_webhook(
                            stream_id=stream_id,
                            event=WebhookEvent.PROCESSING_COMPLETE,
                            data={
                                "stream_id": stream_id,
                                "status": "completed",
                                "highlights_count": len(all_highlights),
                                "agent_metrics": b2b_agent.get_performance_metrics(),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )

                    # Track successful task completion
                    metrics.increment_task_executed(
                        task_name="detect_highlights_with_ai",
                        organization_id=str(stream.organization_id),
                        success=True,
                    )

                    # Calculate total processing time
                    task_duration = (
                        datetime.now(timezone.utc) - datetime.now(timezone.utc)
                    ).total_seconds()
                    metrics.record_task_duration(
                        duration_seconds=task_duration,
                        task_name="detect_highlights_with_ai",
                        organization_id=str(stream.organization_id),
                    )

                    result = {
                        "stream_id": stream_id,
                        "status": "completed",
                        "highlights_created": len(all_highlights),
                        "agent_config_used": agent_config.name,
                        "agent_metrics": b2b_agent.get_performance_metrics(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    logfire.info(
                        "AI highlight detection completed successfully",
                        stream_id=stream_id,
                        highlights_created=len(all_highlights),
                        platform=stream.platform,
                        organization_id=stream.organization_id,
                        agent_config=agent_config.name,
                    )

                    return result
                finally:
                    loop.close()

    except Exception as exc:
        logfire.error(
            "AI highlight detection failed",
            stream_id=stream_id,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )

        # Track task failure
        if "stream" in locals() and hasattr(stream, "organization_id"):
            metrics.increment_task_executed(
                task_name="detect_highlights_with_ai",
                organization_id=str(stream.organization_id),
                success=False,
            )

            metrics.increment_stream_completed(
                platform=stream.platform,
                organization_id=str(stream.organization_id),
                stream_type="live",
                success=False,
            )

        # Update stream status to failed
        try:
            with get_db_session() as db:
                with logfire.span("update_failed_stream_status_ai"):
                    stream = db.query(Stream).filter(Stream.id == stream_id).first()
                    if stream:
                        stream.status = StreamStatus.FAILED
                        stream.error_message = str(exc)
                        db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))

        raise exc


@celery_app.task(bind=True, name="cleanup_stream_resources")
def cleanup_stream_resources(self, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up temporary files and resources from stream processing.

    Args:
        max_age_hours: Maximum age in hours for resources to keep

    Returns:
        Dict with cleanup results
    """
    logger.info("Starting stream resource cleanup", max_age_hours=max_age_hours)

    try:
        import tempfile
        import shutil
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        temp_files_cleaned = 0

        # Clean up temporary directories
        temp_root = tempfile.gettempdir()
        for item in os.listdir(temp_root):
            if item.startswith("stream_"):
                item_path = os.path.join(temp_root, item)
                if os.path.isdir(item_path):
                    try:
                        # Check if directory is old enough
                        mod_time = datetime.fromtimestamp(
                            os.path.getmtime(item_path), tz=timezone.utc
                        )
                        if mod_time < cutoff_time:
                            shutil.rmtree(item_path)
                            temp_files_cleaned += 1
                            logger.debug(f"Cleaned up temp directory: {item_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {item_path}: {e}")

        # Clean up completed streams from database
        with get_db_session() as db:
            old_streams = (
                db.query(Stream)
                .filter(
                    Stream.status == StreamStatus.COMPLETED,
                    Stream.completed_at < cutoff_time,
                )
                .count()
            )

            # Archive rather than delete (or implement retention policy)
            # For now, just log the count
            logger.info(
                f"Found {old_streams} completed streams older than {max_age_hours} hours"
            )

        cleanup_result = {
            "temp_directories_cleaned": temp_files_cleaned,
            "old_streams_found": old_streams,
            "cutoff_time": cutoff_time.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Stream resource cleanup completed", **cleanup_result)
        return cleanup_result

    except Exception as exc:
        logger.error("Failed to cleanup stream resources", error=str(exc))
        raise exc
