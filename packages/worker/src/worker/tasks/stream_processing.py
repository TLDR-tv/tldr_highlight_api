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
            )
            
            # Initialize processors with context manager
            ffmpeg_processor = FFmpegProcessor(
                stream_url=stream.url,
                output_dir=output_dir,
                config=config,
            )
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
                    
                    # Queue highlight detection for video segment
                    detect_highlights_task.delay(
                        stream_id=stream_id,
                        segment_data={
                            "id": str(segment.segment_id),
                            "start_time": segment.start_time,
                            "end_time": segment.start_time + segment.duration,
                            "video_path": str(segment.path),
                            "audio_chunks": [
                                {
                                    "id": str(chunk.chunk_id),
                                    "path": str(chunk.path),
                                    "start_time": chunk.start_time,
                                    "end_time": chunk.end_time,
                                }
                                for chunk in segment.audio_chunks
                            ],
                        },
                        processing_options=processing_options,
                        context_segments=[
                            {
                                "segment_id": str(s.segment_id),
                                "start_time": s.start_time,
                                "duration": s.duration,
                                "segment_number": s.segment_number,
                            }
                            for s in await segment_buffer.peek(5)
                        ],
                    )
                    
                    # Also queue wake word detection for audio chunks
                    for chunk in segment.audio_chunks:
                        detect_wake_words_task.delay(
                            stream_id=stream_id,
                            audio_chunk={
                                "id": str(chunk.chunk_id),
                                "path": str(chunk.path),
                                "start_time": chunk.start_time,
                                "end_time": chunk.end_time,
                                "video_segment_number": segment.segment_number,
                            },
                            organization_id=str(stream.organization_id),
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