"""Stream processing tasks."""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID
from celery import Task, states
from celery.exceptions import Retry
from structlog import get_logger

from worker.app import celery_app
from worker.services.ffmpeg_processor import FFmpegProcessor
from worker.services.segment_buffer import SegmentBuffer
from shared.infrastructure.storage.repositories import StreamRepository, HighlightRepository
from shared.infrastructure.database.database import Database
from shared.infrastructure.config import get_settings
from shared.domain.models.stream import StreamStatus

logger = get_logger()


class StreamProcessingTask(Task):
    """Base task with callbacks for stream processing."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Update stream status on success."""
        stream_id = kwargs.get("stream_id") or args[0]
        asyncio.run(self._update_stream_status(stream_id, StreamStatus.COMPLETED))
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Update stream status on failure."""
        stream_id = kwargs.get("stream_id") or args[0]
        asyncio.run(self._update_stream_status(stream_id, StreamStatus.FAILED))
        logger.error(
            "Stream processing failed",
            stream_id=stream_id,
            error=str(exc),
            traceback=str(einfo),
        )
    
    async def _update_stream_status(self, stream_id: str, status: StreamStatus):
        """Update stream status in database."""
        settings = get_settings()
        database = Database(settings.database_url)
        await database.connect()
        
        async with database.session() as session:
            stream_repo = StreamRepository(session)
            stream = await stream_repo.get_by_id(UUID(stream_id))
            if stream:
                stream.status = status
                await stream_repo.update(stream)
        
        await database.disconnect()


@celery_app.task(
    bind=True,
    base=StreamProcessingTask,
    name="process_stream",
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
)
def process_stream_task(
    self,
    stream_id: str,
    processing_options: Dict,
) -> Dict:
    """
    Process a livestream or video file.
    
    Args:
        stream_id: UUID of the stream to process
        processing_options: Processing configuration including:
            - dimension_set_id: ID of dimension set to use
            - type_registry_id: ID of type registry to use
            - fusion_strategy: How to combine modality scores
            - enabled_modalities: Which modalities to process
            - confidence_threshold: Minimum confidence for highlights
            
    Returns:
        Dictionary with processing results
    """
    try:
        # Run async processing in sync context
        result = asyncio.run(
            _process_stream_async(stream_id, processing_options, self.request.id)
        )
        return result
        
    except Exception as exc:
        logger.error(
            "Stream processing error",
            stream_id=stream_id,
            error=str(exc),
            exc_info=True,
        )
        
        # Retry with exponential backoff
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


async def _process_stream_async(
    stream_id: str,
    processing_options: Dict,
    task_id: str,
) -> Dict:
    """Async implementation of stream processing."""
    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect()
    
    try:
        async with database.session() as session:
            stream_repo = StreamRepository(session)
            stream = await stream_repo.get_by_id(UUID(stream_id))
            
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Update stream status
            stream.status = StreamStatus.PROCESSING
            stream.celery_task_id = task_id
            await stream_repo.update(stream)
            
            logger.info(
                "Starting stream processing",
                stream_id=stream_id,
                url=stream.url,
                task_id=task_id,
            )
        
        # Initialize processors
        ffmpeg_processor = FFmpegProcessor()
        segment_buffer = SegmentBuffer(max_size=10)
        
        # Process stream segments
        processed_count = 0
        async for segment in ffmpeg_processor.process_stream(
            stream.url,
            segment_duration=processing_options.get("segment_duration", 30),
        ):
            # Add to buffer for context
            segment_buffer.add(segment)
            
            # Queue highlight detection for this segment
            detect_highlights_task.delay(
                stream_id=stream_id,
                segment_data={
                    "id": segment.id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "video_path": segment.video_path,
                    "audio_path": segment.audio_path,
                },
                processing_options=processing_options,
                context_segments=[s.to_dict() for s in segment_buffer.get_context()],
            )
            
            processed_count += 1
            
            # Update progress periodically
            if processed_count % 10 == 0:
                celery_app.send_task(
                    "worker.tasks.webhook_delivery.send_progress_update",
                    args=[stream_id, processed_count],
                )
        
        logger.info(
            "Stream processing completed",
            stream_id=stream_id,
            segments_processed=processed_count,
        )
        
        return {
            "stream_id": stream_id,
            "segments_processed": processed_count,
            "status": "completed",
        }
        
    finally:
        await database.disconnect()


@celery_app.task(
    bind=True,
    name="detect_highlights",
    max_retries=3,
    default_retry_delay=60,
)
def detect_highlights_task(
    self,
    stream_id: str,
    segment_data: Dict,
    processing_options: Dict,
    context_segments: List[Dict],
) -> Dict:
    """
    Detect highlights in a stream segment.
    
    Args:
        stream_id: UUID of the stream
        segment_data: Segment information
        processing_options: Processing configuration
        context_segments: Previous segments for context
        
    Returns:
        Dictionary with detected highlights
    """
    from worker.tasks.highlight_detection import process_segment_for_highlights
    
    try:
        # Delegate to highlight detection module
        highlights = asyncio.run(
            process_segment_for_highlights(
                stream_id,
                segment_data,
                processing_options,
                context_segments,
            )
        )
        
        logger.info(
            "Highlight detection completed",
            stream_id=stream_id,
            segment_id=segment_data["id"],
            highlights_found=len(highlights),
        )
        
        return {
            "stream_id": stream_id,
            "segment_id": segment_data["id"],
            "highlights": highlights,
        }
        
    except Exception as exc:
        logger.error(
            "Highlight detection error",
            stream_id=stream_id,
            segment_id=segment_data["id"],
            error=str(exc),
            exc_info=True,
        )
        
        # Retry with exponential backoff
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)