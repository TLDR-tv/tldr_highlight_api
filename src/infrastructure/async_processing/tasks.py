"""
Core Celery task definitions for stream processing workflow.

This module defines all the Celery tasks that comprise the stream processing
pipeline, including data ingestion, content processing, AI-powered highlight
detection, finalization, and notification delivery.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import structlog
from celery import Task

from src.core.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.persistence.models.highlight import Highlight
from src.services.async_processing.celery_app import celery_app
from src.services.async_processing.error_handler import ErrorHandler
from src.services.async_processing.progress_tracker import (
    ProgressTracker,
    ProgressEvent,
)
from src.services.async_processing.webhook_dispatcher import (
    WebhookDispatcher,
    WebhookEvent,
)


logger = structlog.get_logger(__name__)


class BaseStreamTask(Task):
    """Base task class with common functionality for stream processing tasks."""

    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60
    retry_backoff = True
    retry_jitter = True

    def __init__(self):
        """Initialize base task with error handler and progress tracker."""
        self.error_handler = ErrorHandler()
        self.progress_tracker = ProgressTracker()
        self.webhook_dispatcher = WebhookDispatcher()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with comprehensive error logging and notification."""
        logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            args=args,
            kwargs=kwargs,
            traceback=einfo.traceback,
        )

        # Update progress with failure
        if args and len(args) > 0:
            stream_id = args[0]
            try:
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
                logger.error("Failed to update progress on task failure", error=str(e))

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry with logging and progress update."""
        logger.warning(
            "Task retry",
            task_id=task_id,
            task_name=self.name,
            exception=str(exc),
            retry_count=self.request.retries,
            max_retries=self.max_retries,
        )

        # Update progress with retry information
        if args and len(args) > 0:
            stream_id = args[0]
            try:
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=None,  # Don't change progress
                    status="processing",
                    event_type=ProgressEvent.RETRY,
                    details={
                        "retry_count": self.request.retries,
                        "max_retries": self.max_retries,
                        "error": str(exc),
                    },
                )
            except Exception as e:
                logger.error("Failed to update progress on task retry", error=str(e))

    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion."""
        logger.info(
            "Task completed successfully",
            task_id=task_id,
            task_name=self.name,
            args=args,
            kwargs=kwargs,
            result=retval,
        )


@celery_app.task(bind=True, base=BaseStreamTask, name="start_stream_processing")
def start_stream_processing(
    self, stream_id: int, options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Entry point task for stream processing workflow.

    This task initializes the stream processing pipeline, validates the stream,
    sets up resources, and triggers the subsequent processing tasks.

    Args:
        stream_id: ID of the stream to process
        options: Processing options and configuration

    Returns:
        Dict[str, Any]: Task result with stream information and next steps
    """
    logger.info("Starting stream processing", stream_id=stream_id, options=options)

    try:
        with get_db_session() as db:
            # Get stream from database
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")

            # Update stream status to processing
            stream.status = StreamStatus.PROCESSING
            db.commit()

            # Initialize progress tracking
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=5,
                status="processing",
                event_type=ProgressEvent.STARTED,
                details={"task": "stream_processing_started"},
            )

            # Send started webhook
            asyncio.create_task(
                self.webhook_dispatcher.dispatch_webhook(
                    stream_id=stream_id,
                    event=WebhookEvent.STREAM_STARTED,
                    data={
                        "stream_id": stream_id,
                        "platform": stream.platform,
                        "source_url": stream.source_url,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

            # Validate stream and prepare for processing
            # TODO: Implement stream validation
            validation_result = {
                "valid": True,
                "platform": stream.platform,
                "source_url": stream.source_url,
                "options": options or {},
            }

            return {
                "stream_id": stream_id,
                "status": "started",
                "platform": stream.platform,
                "validation": validation_result,
                "next_task": "ingest_stream_data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as exc:
        logger.error(
            "Failed to start stream processing", stream_id=stream_id, error=str(exc)
        )

        # Update stream status to failed
        try:
            with get_db_session() as db:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if stream:
                    stream.status = StreamStatus.FAILED
                    db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))

        raise exc

    def _validate_stream(
        self, stream: Stream, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate stream configuration and accessibility."""
        try:
            # Basic validation
            if not stream.source_url:
                return {"valid": False, "error": "Source URL is required"}

            if not stream.platform:
                return {"valid": False, "error": "Platform is required"}

            # Platform-specific validation would go here
            # For now, return valid
            return {
                "valid": True,
                "platform": stream.platform,
                "estimated_duration": options.get("estimated_duration", "unknown"),
                "processing_options": options,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


@celery_app.task(bind=True, base=BaseStreamTask, name="ingest_stream_data")
def ingest_stream_data(self, stream_id: int, chunk_size: int = 30) -> Dict[str, Any]:
    """
    Ingest stream data in configurable chunks for processing.

    This task handles the actual data ingestion from the streaming platform,
    breaking the stream into manageable chunks for subsequent processing.

    Args:
        stream_id: ID of the stream to ingest
        chunk_size: Size of each chunk in seconds

    Returns:
        Dict[str, Any]: Ingestion results with chunk information
    """
    logger.info("Ingesting stream data", stream_id=stream_id, chunk_size=chunk_size)

    try:
        with get_db_session() as db:
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")

            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=15,
                status="processing",
                event_type=ProgressEvent.PROGRESS_UPDATE,
                details={"task": "ingesting_stream_data", "chunk_size": chunk_size},
            )

            # Simulate stream ingestion (in real implementation, this would
            # connect to the actual streaming platform API)
            # TODO: Implement platform-specific ingestion
            ingestion_result = {
                "status": "success",
                "chunks_created": 10,
                "total_duration": 300,
                "platform": stream.platform,
            }

            # Update progress
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=30,
                status="processing",
                event_type=ProgressEvent.PROGRESS_UPDATE,
                details={
                    "task": "stream_data_ingested",
                    "chunks_created": ingestion_result["chunks_created"],
                    "total_duration": ingestion_result["total_duration"],
                },
            )

            return ingestion_result

    except Exception as exc:
        logger.error(
            "Failed to ingest stream data", stream_id=stream_id, error=str(exc)
        )
        raise exc

    def _ingest_from_platform(self, stream: Stream, chunk_size: int) -> Dict[str, Any]:
        """Ingest data from the specific streaming platform."""
        # Platform-specific ingestion logic would go here
        # For now, simulate the ingestion process

        # TODO: Implement platform-specific handlers
        platform_handlers = {
            "twitch": lambda s, c: {"video_chunks": [], "audio_chunks": []},
            "youtube": lambda s, c: {"video_chunks": [], "audio_chunks": []},
            "rtmp": lambda s, c: {"video_chunks": [], "audio_chunks": []},
        }

        handler = platform_handlers.get(stream.platform)
        if not handler:
            raise ValueError(f"Unsupported platform: {stream.platform}")

        return handler(stream, chunk_size)

    def _ingest_twitch_stream(self, stream: Stream, chunk_size: int) -> Dict[str, Any]:
        """Handle Twitch stream ingestion."""
        # Simulated ingestion result
        return {
            "platform": "twitch",
            "chunks_created": 10,
            "total_duration": 300,  # 5 minutes
            "video_chunks": ["chunk_1.mp4", "chunk_2.mp4"],
            "audio_chunks": ["chunk_1.wav", "chunk_2.wav"],
            "metadata": {"quality": "1080p", "fps": 60},
        }

    def _ingest_youtube_stream(self, stream: Stream, chunk_size: int) -> Dict[str, Any]:
        """Handle YouTube stream ingestion."""
        return {
            "platform": "youtube",
            "chunks_created": 8,
            "total_duration": 240,  # 4 minutes
            "video_chunks": ["yt_chunk_1.mp4", "yt_chunk_2.mp4"],
            "audio_chunks": ["yt_chunk_1.wav", "yt_chunk_2.wav"],
            "metadata": {"quality": "720p", "fps": 30},
        }

    def _ingest_rtmp_stream(self, stream: Stream, chunk_size: int) -> Dict[str, Any]:
        """Handle RTMP stream ingestion."""
        return {
            "platform": "rtmp",
            "chunks_created": 12,
            "total_duration": 360,  # 6 minutes
            "video_chunks": ["rtmp_chunk_1.mp4", "rtmp_chunk_2.mp4"],
            "audio_chunks": ["rtmp_chunk_1.wav", "rtmp_chunk_2.wav"],
            "metadata": {"quality": "480p", "fps": 24},
        }


@celery_app.task(bind=True, base=BaseStreamTask, name="process_multimodal_content")
def process_multimodal_content(
    self, stream_id: int, ingestion_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process multimodal content (video, audio, chat) for highlight detection.

    This task analyzes the ingested stream data across multiple modalities
    to prepare it for AI-powered highlight detection.

    Args:
        stream_id: ID of the stream being processed
        ingestion_data: Data from the ingestion task

    Returns:
        Dict[str, Any]: Processed content ready for highlight detection
    """
    logger.info("Processing multimodal content", stream_id=stream_id)

    try:
        # Update progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=45,
            status="processing",
            event_type=ProgressEvent.PROGRESS_UPDATE,
            details={"task": "processing_multimodal_content"},
        )

        # Process different modalities
        # TODO: Implement feature extraction
        video_features = {
            "fps": 30,
            "resolution": "1920x1080",
            "duration": ingestion_data.get("total_duration", 300),
            "keyframes": 150,
            "scenes": 20,
        }

        audio_features = {
            "sample_rate": 44100,
            "channels": 2,
            "duration": ingestion_data.get("total_duration", 300),
            "volume_peaks": 15,
            "silence_segments": 5,
        }

        # Try to analyze chat, but don't fail if not available
        chat_analysis = {
            "available": False,
            "reason": "Chat analysis not implemented yet",
            "sentiment_score": 0.0,
            "message_count": 0,
        }

        # Build processed content with available modalities
        modalities_available = ["video", "audio"]
        if chat_analysis and chat_analysis.get("available", True):
            modalities_available.append("chat")

        processed_content = {
            "stream_id": stream_id,
            "video_features": video_features,
            "audio_features": audio_features,
            "chat_analysis": chat_analysis,
            "processing_metadata": {
                "total_chunks": len(ingestion_data["video_chunks"]),
                "processing_time": 45.5,  # Simulated processing time
                "quality_score": 0.85,
                "modalities_available": modalities_available,
            },
        }

        # Update progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=60,
            status="processing",
            event_type=ProgressEvent.PROGRESS_UPDATE,
            details={
                "task": "multimodal_content_processed",
                "modalities": processed_content["processing_metadata"][
                    "modalities_available"
                ],
                "quality_score": processed_content["processing_metadata"][
                    "quality_score"
                ],
            },
        )

        return processed_content

    except Exception as exc:
        logger.error(
            "Failed to process multimodal content", stream_id=stream_id, error=str(exc)
        )
        raise exc

    def _extract_video_features(self, video_chunks: List[str]) -> Dict[str, Any]:
        """Extract features from video chunks."""
        # Simulated video feature extraction
        return {
            "scene_changes": [10.5, 25.3, 40.1],
            "motion_intensity": [0.7, 0.9, 0.6, 0.8],
            "visual_complexity": 0.75,
            "dominant_colors": ["#FF0000", "#00FF00", "#0000FF"],
            "object_detection": {"people": 3, "vehicles": 1, "animals": 0},
        }

    def _extract_audio_features(self, audio_chunks: List[str]) -> Dict[str, Any]:
        """Extract features from audio chunks."""
        return {
            "transcription": "This is a simulated transcription of the audio content",
            "sentiment_scores": [0.6, 0.8, 0.4, 0.9],
            "volume_levels": [0.7, 0.8, 0.6, 0.9],
            "speech_segments": [
                {"start": 0, "end": 15, "speaker": "host"},
                {"start": 15, "end": 30, "speaker": "guest"},
            ],
            "music_detection": False,
            "sound_effects": ["applause", "laughter"],
        }

    def _analyze_chat_sentiment(self, stream_id: int) -> Dict[str, Any]:
        """Analyze chat sentiment and engagement."""
        return {
            "total_messages": 150,
            "sentiment_distribution": {
                "positive": 0.6,
                "neutral": 0.3,
                "negative": 0.1,
            },
            "engagement_spikes": [12.5, 28.7, 41.2],
            "top_keywords": ["awesome", "amazing", "wow", "great"],
            "emoji_usage": {"😍": 25, "👏": 18, "🔥": 12},
        }


@celery_app.task(bind=True, base=BaseStreamTask, name="detect_highlights")
def detect_highlights(
    self, stream_id: int, processed_content: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use AI to detect highlights from processed multimodal content.

    This task applies AI models to identify the most interesting moments
    in the stream based on multimodal analysis.

    Args:
        stream_id: ID of the stream being processed
        processed_content: Output from multimodal content processing

    Returns:
        Dict[str, Any]: Detected highlights with timestamps and metadata
    """
    logger.info("Detecting highlights using AI", stream_id=stream_id)

    try:
        # Update progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=75,
            status="processing",
            event_type=ProgressEvent.PROGRESS_UPDATE,
            details={"task": "detecting_highlights_with_ai"},
        )

        # AI-powered highlight detection (simulated)
        # TODO: Implement AI detection
        highlights = [
            {
                "timestamp": 30,
                "duration": 15,
                "confidence": 0.95,
                "type": "action",
                "description": "Epic gaming moment",
            },
            {
                "timestamp": 120,
                "duration": 10,
                "confidence": 0.88,
                "type": "funny",
                "description": "Funny fail",
            },
            {
                "timestamp": 200,
                "duration": 20,
                "confidence": 0.92,
                "type": "skill",
                "description": "Impressive skill display",
            },
        ]

        # Store highlights in database
        with get_db_session() as db:
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")

            for highlight_data in highlights:
                highlight = Highlight(
                    stream_id=stream_id,
                    title=f"Highlight at {highlight_data.get('timestamp', 0)}s",
                    description=highlight_data.get("description", ""),
                    video_url=f"https://example.com/clips/{stream_id}_{highlight_data.get('timestamp', 0)}.mp4",
                    thumbnail_url=f"https://example.com/thumbnails/{stream_id}_{highlight_data.get('timestamp', 0)}.jpg",
                    duration=highlight_data.get("duration", 10),
                    timestamp=highlight_data.get("timestamp", 0),
                    confidence_score=highlight_data.get("confidence", 0.5),
                    tags=[highlight_data.get("type", "general")],
                    extra_metadata=highlight_data.get("metadata", {}),
                )
                db.add(highlight)

            db.commit()

        # Update progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=85,
            status="processing",
            event_type=ProgressEvent.PROGRESS_UPDATE,
            details={
                "task": "highlights_detected",
                "highlights_count": len(highlights),
                "average_confidence": sum(h.get("confidence", 0) for h in highlights)
                / len(highlights)
                if highlights
                else 0,
            },
        )

        # Send highlights detected webhook
        asyncio.create_task(
            self.webhook_dispatcher.dispatch_webhook(
                stream_id=stream_id,
                event=WebhookEvent.HIGHLIGHTS_DETECTED,
                data={
                    "highlights_count": len(highlights),
                    "highlights": highlights[:5],  # Send first 5 highlights
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

        return {
            "highlights": highlights,
            "total_detected": len(highlights),
            "average_confidence": sum(h.get("confidence", 0) for h in highlights)
            / len(highlights)
            if highlights
            else 0,
            "stream_id": stream_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.error("Failed to detect highlights", stream_id=stream_id, error=str(exc))
        raise exc

    def _run_ai_detection(self, processed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI models for highlight detection."""
        # Simulated AI highlight detection
        highlights = [
            {
                "start_time": 12.5,
                "end_time": 18.7,
                "confidence": 0.92,
                "type": "exciting_moment",
                "description": "High energy reaction to game play",
                "metadata": {
                    "visual_score": 0.89,
                    "audio_score": 0.95,
                    "chat_score": 0.88,
                },
            },
            {
                "start_time": 28.3,
                "end_time": 35.1,
                "confidence": 0.87,
                "type": "funny_moment",
                "description": "Humorous interaction with audience",
                "metadata": {
                    "visual_score": 0.76,
                    "audio_score": 0.91,
                    "chat_score": 0.94,
                },
            },
            {
                "start_time": 41.8,
                "end_time": 48.2,
                "confidence": 0.94,
                "type": "skill_showcase",
                "description": "Impressive gameplay technique demonstrated",
                "metadata": {
                    "visual_score": 0.96,
                    "audio_score": 0.87,
                    "chat_score": 0.89,
                },
            },
        ]

        return {
            "highlights": highlights,
            "total_detected": len(highlights),
            "average_confidence": sum(h["confidence"] for h in highlights)
            / len(highlights),
            "detection_metadata": {
                "model_version": "v2.1.0",
                "processing_time": 23.4,
                "gpu_utilization": 0.78,
            },
        }


@celery_app.task(bind=True, base=BaseStreamTask, name="finalize_highlights")
def finalize_highlights(
    self, stream_id: int, detection_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Finalize highlights with post-processing and storage preparation.

    This task performs final processing on detected highlights, including
    video generation, thumbnail creation, and storage preparation.

    Args:
        stream_id: ID of the stream being processed
        detection_results: Results from highlight detection

    Returns:
        Dict[str, Any]: Finalized highlights ready for delivery
    """
    logger.info("Finalizing highlights", stream_id=stream_id)

    try:
        # Update progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=95,
            status="processing",
            event_type=ProgressEvent.PROGRESS_UPDATE,
            details={"task": "finalizing_highlights"},
        )

        # Generate video clips and thumbnails
        finalized_highlights = []
        for highlight in detection_results["highlights"]:
            # TODO: Implement video clip generation and thumbnail creation
            timestamp = highlight.get("timestamp", highlight.get("start_time", 0))
            finalized_highlight = {
                **highlight,
                "clip_url": f"https://example.com/clips/{stream_id}_{timestamp}.mp4",
                "thumbnail_url": f"https://example.com/thumbnails/{stream_id}_{timestamp}.jpg",
                "metadata": {"fps": 30, "resolution": "1920x1080", "codec": "h264"},
            }
            finalized_highlights.append(finalized_highlight)

        # Update stream status to completed
        with get_db_session() as db:
            stream = db.query(Stream).filter(Stream.id == stream_id).first()
            if stream:
                stream.status = StreamStatus.COMPLETED
                stream.completed_at = datetime.now(timezone.utc)
                db.commit()

        finalization_result = {
            "stream_id": stream_id,
            "finalized_highlights": finalized_highlights,
            "total_highlights": len(finalized_highlights),
            "processing_summary": {
                "start_time": "2024-01-01T00:00:00Z",  # Would be actual time
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_duration": 180.5,  # Simulated processing duration
                "success": True,
            },
        }

        # Update final progress
        self.progress_tracker.update_progress(
            stream_id=stream_id,
            progress_percentage=100,
            status="completed",
            event_type=ProgressEvent.COMPLETED,
            details={
                "task": "highlights_finalized",
                "total_highlights": len(finalized_highlights),
            },
        )

        return finalization_result

    except Exception as exc:
        logger.error(
            "Failed to finalize highlights", stream_id=stream_id, error=str(exc)
        )
        raise exc

    def _finalize_single_highlight(
        self, stream_id: int, highlight: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize a single highlight with video generation and storage."""
        # Simulated highlight finalization
        return {
            "highlight_id": f"highlight_{stream_id}_{int(highlight['start_time'])}",
            "start_time": highlight["start_time"],
            "end_time": highlight["end_time"],
            "duration": highlight["end_time"] - highlight["start_time"],
            "confidence": highlight["confidence"],
            "type": highlight["type"],
            "description": highlight["description"],
            "video_url": f"https://highlights.example.com/{stream_id}/highlight_{int(highlight['start_time'])}.mp4",
            "thumbnail_url": f"https://thumbnails.example.com/{stream_id}/thumb_{int(highlight['start_time'])}.jpg",
            "metadata": {
                **highlight.get("metadata", {}),
                "file_size": 1024 * 1024 * 5,  # 5MB simulated
                "resolution": "1920x1080",
                "format": "mp4",
            },
        }


@celery_app.task(bind=True, base=BaseStreamTask, name="notify_completion")
def notify_completion(
    self, stream_id: int, finalization_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send completion notifications via webhooks and other channels.

    This task handles all notification delivery after successful stream processing,
    including webhook dispatch and any other notification channels.

    Args:
        stream_id: ID of the completed stream
        finalization_results: Results from highlight finalization

    Returns:
        Dict[str, Any]: Notification delivery results
    """
    logger.info("Sending completion notifications", stream_id=stream_id)

    try:
        # Send completion webhook
        webhook_result = asyncio.run(
            self.webhook_dispatcher.dispatch_webhook(
                stream_id=stream_id,
                event=WebhookEvent.PROCESSING_COMPLETE,
                data={
                    "stream_id": stream_id,
                    "status": "completed",
                    "highlights_count": finalization_results["total_highlights"],
                    "highlights": finalization_results["finalized_highlights"],
                    "processing_summary": finalization_results["processing_summary"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

        # Additional notification channels could be added here
        # (email, SMS, Slack, etc.)

        notification_result = {
            "stream_id": stream_id,
            "notifications_sent": 1,
            "webhook_result": webhook_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Completion notifications sent successfully", **notification_result)
        return notification_result

    except Exception as exc:
        logger.error(
            "Failed to send completion notifications",
            stream_id=stream_id,
            error=str(exc),
        )
        raise exc


@celery_app.task(bind=True, base=BaseStreamTask, name="cleanup_job_resources")
def cleanup_job_resources(self, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clean up expired job resources and temporary files.

    This maintenance task removes old temporary files, expired job data,
    and other resources that are no longer needed.

    Args:
        max_age_hours: Maximum age in hours for resources to keep

    Returns:
        Dict[str, Any]: Cleanup results and statistics
    """
    logger.info("Starting job resource cleanup", max_age_hours=max_age_hours)

    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        # Clean up old progress tracking data
        progress_cleaned = self.progress_tracker.cleanup_old_progress(cutoff_time)

        # Clean up old webhook attempts
        webhook_cleaned = self.webhook_dispatcher.cleanup_old_attempts(cutoff_time)

        # Clean up temporary files (simulated)
        # TODO: Implement temp file cleanup
        temp_files_cleaned = 0

        # Clean up completed streams older than cutoff
        with get_db_session() as db:
            old_streams = (
                db.query(Stream)
                .filter(
                    Stream.status == StreamStatus.COMPLETED,
                    Stream.completed_at < cutoff_time,
                )
                .count()
            )

            # In real implementation, might archive rather than delete
            db.query(Stream).filter(
                Stream.status == StreamStatus.COMPLETED,
                Stream.completed_at < cutoff_time,
            ).delete()
            db.commit()

        cleanup_result = {
            "progress_records_cleaned": progress_cleaned,
            "webhook_attempts_cleaned": webhook_cleaned,
            "temp_files_cleaned": temp_files_cleaned,
            "old_streams_cleaned": old_streams,
            "cutoff_time": cutoff_time.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Job resource cleanup completed", **cleanup_result)
        return cleanup_result

    except Exception as exc:
        logger.error("Failed to cleanup job resources", error=str(exc))
        raise exc

    def _cleanup_temp_files(self, cutoff_time: datetime) -> int:
        """Clean up temporary files older than cutoff time."""
        # Simulated temp file cleanup
        return 42  # Number of files cleaned


@celery_app.task(bind=True, name="health_check_task")
def health_check_task(self) -> Dict[str, Any]:
    """
    Perform health check on the async processing system.

    This task runs periodic health checks to ensure all components
    of the async processing pipeline are functioning correctly.

    Returns:
        Dict[str, Any]: Health check results
    """
    logger.info("Running health check")

    try:
        # Check database connectivity
        db_healthy = True
        try:
            with get_db_session() as db:
                db.execute("SELECT 1")
        except Exception as e:
            db_healthy = False
            logger.error("Database health check failed", error=str(e))

        # Check Redis connectivity
        redis_healthy = True
        try:
            from src.core.cache import get_redis_client

            redis_client = get_redis_client()
            redis_client.ping()
        except Exception as e:
            redis_healthy = False
            logger.error("Redis health check failed", error=str(e))

        # Check task queue status
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        health_result = {
            "status": "healthy" if db_healthy and redis_healthy else "unhealthy",
            "database": "healthy" if db_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy",
            "active_tasks": len(active_tasks) if active_tasks else 0,
            "scheduled_tasks": len(scheduled_tasks) if scheduled_tasks else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Health check completed", **health_result)
        return health_result

    except Exception as exc:
        logger.error("Health check failed", error=str(exc))
        raise exc


@celery_app.task(bind=True, name="process_dead_letter_queue")
def process_dead_letter_queue(self) -> Dict[str, Any]:
    """
    Process messages from dead letter queue for failed webhook deliveries.

    This maintenance task attempts to reprocess failed webhook deliveries
    and other messages that ended up in the dead letter queue.

    Returns:
        Dict[str, Any]: Processing results
    """
    logger.info("Processing dead letter queue")

    try:
        # This would process actual dead letter queue messages
        # For now, simulate the process

        processed_count = 0
        failed_count = 0

        # Simulated processing of dead letter messages
        dead_letter_messages = []  # Would fetch from actual DLQ

        for message in dead_letter_messages:
            try:
                # Attempt to reprocess the message
                # This would contain the actual reprocessing logic
                processed_count += 1
            except Exception as e:
                logger.error("Failed to reprocess dead letter message", error=str(e))
                failed_count += 1

        result = {
            "processed_count": processed_count,
            "failed_count": failed_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Dead letter queue processing completed", **result)
        return result

    except Exception as exc:
        logger.error("Failed to process dead letter queue", error=str(exc))
        raise exc
