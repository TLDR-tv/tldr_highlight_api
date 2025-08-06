"""Stream processing tasks."""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID
from celery import Task, states
from celery.exceptions import Retry
from structlog import get_logger

from worker.app import celery_app
from worker.services.ffmpeg_processor import FFmpegProcessor
from worker.services.segment_buffer import SegmentRingBuffer
from worker.services.persistent_segment import PersistentSegmentManager
from worker.tasks.wake_word_detection import detect_wake_words_task
from shared.infrastructure.storage.repositories import StreamRepository, HighlightRepository
from shared.infrastructure.database.database import Database
from shared.infrastructure.config.config import get_settings
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
        
        async with database.session() as session:
            stream_repo = StreamRepository(session)
            stream = await stream_repo.get(UUID(stream_id))
            if stream:
                stream.status = status
                await stream_repo.update(stream)


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
    """Process a livestream or video file.
    
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
    
    # Get stream info and update status
    async with database.session() as session:
        stream_repo = StreamRepository(session)
        stream = await stream_repo.get(UUID(stream_id))
        
        if not stream:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Update stream status
        stream.status = StreamStatus.PROCESSING
        stream.celery_task_id = task_id
        await stream_repo.update(stream)
        
        # Store needed values
        stream_url = stream.url
        organization_id = stream.organization_id
        
        logger.info(
            "Starting stream processing",
            stream_id=stream_id,
            url=stream_url,
            task_id=task_id,
        )
    
    # Create temporary directory for processing
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from worker.services.ffmpeg_processor import FFmpegConfig
    
    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Configure FFmpeg processor
        config = FFmpegConfig(
            segment_duration=processing_options.get("video_segment_duration", 120),
            audio_segment_duration=processing_options.get("audio_segment_duration", 30),
            audio_overlap=processing_options.get("audio_overlap", 5),
            use_readrate=processing_options.get("use_readrate", False),
        )
        
        # Initialize processors with context manager
        ffmpeg_processor = FFmpegProcessor(
            stream_url=stream_url,
            output_dir=output_dir,
            config=config,
        )
        # Use ring buffer for livestream processing (prevents memory overflow)
        segment_buffer = SegmentRingBuffer(max_size=10)
        
        # Process stream segments
        processed_count = 0
        async with ffmpeg_processor:
            # Simple segment handler
            class SegmentHandler:
                async def handle_segment(self, segment):
                    pass
            
            handler = SegmentHandler()
            
            async for segment in ffmpeg_processor.process_stream(handler):
                # Add to buffer for context
                await segment_buffer.add_segment(segment)
                
                # Use context manager to safely process segment before ring buffer cleanup
                async with PersistentSegmentManager(segment, output_dir) as persistent_segment:
                    # Process highlights synchronously within the safe context
                    from worker.tasks.highlight_detection import process_segment_for_highlights
                    
                    try:
                        # Get context segments from buffer
                        context_segments = [
                            {
                                "segment_id": str(s.segment_id),
                                "start_time": s.start_time,
                                "duration": s.duration,
                                "segment_number": s.segment_number,
                            }
                            for s in await segment_buffer.peek(5)
                        ]
                        
                        highlights = await process_segment_for_highlights(
                            stream_id,
                            persistent_segment.to_dict(),
                            processing_options,
                            context_segments,
                        )
                        
                        logger.info(
                            "Highlight detection completed",
                            stream_id=stream_id,
                            segment_id=str(persistent_segment.id),
                            highlights_found=len(highlights),
                        )
                        
                    except Exception as e:
                        logger.error(
                            "Highlight detection failed",
                            stream_id=stream_id,
                            segment_id=str(persistent_segment.id),
                            error=str(e),
                            exc_info=True,
                        )
                
                # Queue wake word detection for the full video segment (async is fine)
                # This uses the same 2-minute segments as Gemini processing for easier clip alignment
                try:
                    detect_wake_words_task.delay(
                        stream_id=stream_id,
                        video_segment={
                            "id": str(segment.segment_id),
                            "video_path": str(segment.path),  # Use the original path since this is queued immediately
                            "start_time": segment.start_time,
                            "end_time": segment.start_time + segment.duration,
                            "segment_number": segment.segment_number,
                        },
                        organization_id=str(organization_id),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to queue wake word detection task",
                        stream_id=stream_id,
                        segment_id=str(segment.segment_id),
                        error=str(e),
                        exc_info=True,
                    )
                
                processed_count += 1
                
                # Update progress periodically
                if processed_count % 10 == 0:
                    try:
                        celery_app.send_task(
                            "worker.tasks.webhook_delivery.send_progress_update",
                            args=[stream_id, processed_count],
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to send progress update",
                            stream_id=stream_id,
                            processed_count=processed_count,
                            error=str(e),
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
    """Detect highlights in a stream segment.
    
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