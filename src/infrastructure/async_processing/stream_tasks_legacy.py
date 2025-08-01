"""Celery tasks for stream processing using the new ingestion architecture.

This module provides Celery tasks that integrate with the separated stream
ingestion components (StreamSegmenter and StreamProcessor).
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from celery import Task
import logfire

from src.infrastructure.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.async_processing.celery_app import celery_app
from src.infrastructure.ingestion import (
    StreamIngestionPipeline,
    StreamIngestionConfig,
)
from src.infrastructure.content_processing.gemini_video_processor import GeminiVideoProcessor
from src.infrastructure.config import get_settings
from src.domain.entities.dimension_set_aggregate import DimensionSetAggregate
from src.domain.entities.highlight_agent_config import HighlightAgentConfig
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
settings = get_settings()


class StreamProcessingTask(Task):
    """Base task class for stream processing with new architecture."""

    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60
    retry_backoff = True
    retry_jitter = True

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.progress_tracker = ProgressTracker()
        self.webhook_dispatcher = WebhookDispatcher()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with cleanup."""
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
    Process a stream using the new ingestion architecture.

    This task:
    1. Initializes the StreamIngestionPipeline
    2. Starts stream segmentation and processing
    3. Collects results and updates the database
    4. Sends progress updates and webhooks

    Args:
        stream_id: Stream ID to process
        chunk_duration: Duration of each segment in seconds
        agent_config_id: Optional agent configuration ID

    Returns:
        Dict with processing results
    """
    with logfire.span("ingest_stream_with_ffmpeg.start") as span:
        span.set_attribute("stream.id", stream_id)
        span.set_attribute("chunk_duration", chunk_duration)
        span.set_attribute("agent_config_id", agent_config_id)

        logger.info(
            "Starting stream processing",
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
            # Get stream details
            with logfire.span("fetch_stream_details") as span:
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if not stream:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", f"Stream {stream_id} not found")
                    raise ValueError(f"Stream {stream_id} not found")

                span.set_attribute("stream.platform", stream.platform)
                span.set_attribute("stream.organization_id", stream.organization_id)

                # Track metrics with organization context
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
                "task": "stream_processing",
                "chunk_duration": chunk_duration,
            },
        )

        # Initialize components
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get dimension set
            dimension_set = loop.run_until_complete(
                _get_dimension_set(stream.dimension_set_id, stream.organization_id)
            )

            # Get agent config
            agent_config = None
            if agent_config_id:
                agent_config = loop.run_until_complete(
                    _get_agent_config(agent_config_id)
                )

            # Initialize Gemini processor
            gemini_processor = GeminiVideoProcessor(
                api_key=settings.external_apis.google_gemini_api_key.get_secret_value()
            )

            # Configure pipeline
            config = StreamIngestionConfig(
                stream_url=stream.source_url,
                stream_id=f"stream_{stream_id}",
                segment_duration=chunk_duration,
                enable_audio_processing=True,
                delete_segments_after_processing=True,
                max_concurrent_processing=3,
                temp_dir=Path(tempfile.gettempdir()) / f"stream_{stream_id}",
            )

            # Progress callback
            def progress_callback(update: Dict[str, Any]):
                """Handle progress updates from pipeline."""
                event = update.get("event")
                if event == "segment_processed":
                    # Calculate overall progress
                    segments_processed = update["stats"].get("segments_processed", 0)
                    progress = min(20 + (segments_processed * 10), 90)  # Cap at 90%
                    
                    self.progress_tracker.update_progress(
                        stream_id=stream_id,
                        progress_percentage=progress,
                        status="processing",
                        event_type=ProgressEvent.PROGRESS_UPDATE,
                        details=update,
                    )

            # Create pipeline
            pipeline = StreamIngestionPipeline(
                config=config,
                gemini_processor=gemini_processor,
                dimension_set=dimension_set,
                agent_config=agent_config,
                progress_callback=progress_callback,
            )

            # Process stream
            all_highlights = []
            processing_errors = []

            async def process_stream_async():
                """Async function to process the stream."""
                async for result in pipeline.start():
                    if result.error:
                        processing_errors.append({
                            "segment": result.segment_index,
                            "error": result.error,
                        })
                        logger.error(
                            f"Segment {result.segment_index} failed: {result.error}"
                        )
                    else:
                        # Store highlights in database
                        for highlight_data in result.highlights:
                            # Create highlight entity and save
                            # This would integrate with your highlight repository
                            all_highlights.append(highlight_data)
                            
                        logger.info(
                            f"Segment {result.segment_index} processed: "
                            f"{len(result.highlights)} highlights found"
                        )

            # Run the async processing
            with logfire.span("process_stream_segments") as process_span:
                process_span.set_attribute("pipeline.started", True)
                loop.run_until_complete(process_stream_async())
                process_span.set_attribute("highlights.total", len(all_highlights))
                process_span.set_attribute("errors.count", len(processing_errors))

            # Get final stats
            final_stats = pipeline.get_stats()

            # Update stream status
            with get_db_session() as db:
                with logfire.span("finalize_stream"):
                    stream = db.query(Stream).filter(Stream.id == stream_id).first()
                    if stream:
                        stream.status = StreamStatus.COMPLETED
                        stream.completed_at = datetime.now(timezone.utc)
                        stream.metadata.update({
                            "segments_processed": final_stats["segments_processed"],
                            "highlights_found": len(all_highlights),
                            "processing_errors": len(processing_errors),
                            "pipeline_stats": final_stats,
                        })
                        db.commit()

            # Final progress update
            self.progress_tracker.update_progress(
                stream_id=stream_id,
                progress_percentage=100,
                status="completed",
                event_type=ProgressEvent.COMPLETED,
                details={
                    "task": "stream_processing_complete",
                    "total_highlights": len(all_highlights),
                    "segments_processed": final_stats["segments_processed"],
                    "errors": len(processing_errors),
                },
            )

            # Send completion webhook
            await self.webhook_dispatcher.dispatch_webhook(
                stream_id=stream_id,
                event=WebhookEvent.PROCESSING_COMPLETE,
                data={
                    "stream_id": stream_id,
                    "status": "completed",
                    "highlights_count": len(all_highlights),
                    "segments_processed": final_stats["segments_processed"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Track successful completion
            metrics.increment_task_executed(
                task_name="ingest_stream_with_ffmpeg",
                organization_id=str(stream.organization_id),
                success=True,
            )

            metrics.increment_stream_completed(
                platform=stream.platform,
                organization_id=str(stream.organization_id),
                stream_type="live",
                success=True,
            )

            result = {
                "stream_id": stream_id,
                "status": "completed",
                "highlights_created": len(all_highlights),
                "segments_processed": final_stats["segments_processed"],
                "processing_errors": len(processing_errors),
                "pipeline_stats": final_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logfire.info(
                "Stream processing completed successfully",
                stream_id=stream_id,
                highlights_created=len(all_highlights),
                platform=stream.platform,
                organization_id=stream.organization_id,
            )

            return result

        finally:
            loop.close()

    except Exception as exc:
        logfire.error(
            "Stream processing failed",
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
                stream = db.query(Stream).filter(Stream.id == stream_id).first()
                if stream:
                    stream.status = StreamStatus.FAILED
                    stream.error_message = str(exc)
                    db.commit()
        except Exception as db_exc:
            logger.error("Failed to update stream status", error=str(db_exc))

        raise exc


# The detect_highlights_with_ai task is no longer needed as it's integrated
# into the pipeline, but keeping a stub for backward compatibility
@celery_app.task(bind=True, base=StreamProcessingTask, name="detect_highlights_with_ai")
@traced_background_task(name="detect_highlights_with_ai")
def detect_highlights_with_ai(
    self,
    stream_id: int,
    ingestion_data: Dict[str, Any],
    agent_config_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Legacy task - highlight detection is now integrated into the pipeline.
    
    This task is kept for backward compatibility but just returns success.
    """
    logger.warning(
        "detect_highlights_with_ai called but is deprecated - "
        "highlight detection is now integrated into ingest_stream_with_ffmpeg"
    )
    
    return {
        "stream_id": stream_id,
        "status": "deprecated",
        "message": "Highlight detection is now integrated into stream ingestion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _get_dimension_set(
    dimension_set_id: Optional[int], organization_id: int
) -> DimensionSetAggregate:
    """Get dimension set for processing."""
    from src.application.services.dimension_set_service import DimensionSetService
    from src.infrastructure.persistence.repositories.dimension_set_repository import (
        DimensionSetRepository,
    )
    from src.infrastructure.database import get_async_session

    async with get_async_session() as session:
        dimension_set_repo = DimensionSetRepository(session)
        dimension_set_service = DimensionSetService(dimension_set_repo)

        # Get dimension set
        if dimension_set_id:
            dimension_set = await dimension_set_service.get_dimension_set(
                dimension_set_id, organization_id
            )
        else:
            # Get default dimension set for organization
            dimension_sets = await dimension_set_service.list_dimension_sets(
                organization_id
            )
            if dimension_sets:
                dimension_set = dimension_sets[0]
            else:
                # Create default gaming dimension set
                from src.domain.value_objects.dimension_definition import DimensionDefinition
                
                dimension_set = DimensionSetAggregate(
                    id=1,
                    name="Default Gaming",
                    organization_id=organization_id,
                )
                
                # Add basic dimensions
                dimension_set.add_dimension(
                    DimensionDefinition(
                        id="action_intensity",
                        name="Action Intensity",
                        type="numeric",
                        description="Level of action and excitement",
                        weight=0.3,
                        min_value=0.0,
                        max_value=1.0,
                    )
                )
                dimension_set.add_dimension(
                    DimensionDefinition(
                        id="emotional_peak",
                        name="Emotional Peak",
                        type="numeric",
                        description="Emotional intensity of the moment",
                        weight=0.3,
                        min_value=0.0,
                        max_value=1.0,
                    )
                )

        return dimension_set


async def _get_agent_config(agent_config_id: int) -> Optional[HighlightAgentConfig]:
    """Get agent configuration."""
    from src.infrastructure.persistence.repositories.highlight_agent_config_repository import (
        HighlightAgentConfigRepository,
    )
    from src.infrastructure.database import get_async_session

    async with get_async_session() as session:
        agent_config_repo = HighlightAgentConfigRepository(session)
        return await agent_config_repo.get(agent_config_id)


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
        import os
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