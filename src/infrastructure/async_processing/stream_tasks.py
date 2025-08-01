"""Refactored Celery tasks - pure orchestration following DDD.

This module contains simplified tasks that only orchestrate between
infrastructure and domain layers, with no business logic.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from celery import Task

from src.infrastructure.database import get_async_session
from src.infrastructure.persistence.repositories.stream_repository import StreamRepository
from src.infrastructure.persistence.repositories.dimension_set_repository import DimensionSetRepository
from src.infrastructure.async_processing.celery_app import celery_app
from src.infrastructure.ingestion import StreamIngestionPipeline, StreamIngestionConfig
from src.infrastructure.content_processing.gemini_video_processor import GeminiVideoProcessor
from src.infrastructure.config import get_settings
from src.domain.entities.stream import StreamStatus
from src.domain.services.highlight_analyzer import HighlightAnalyzer

logger = structlog.get_logger(__name__)
settings = get_settings()


class StreamTask(Task):
    """Base task with simple retry configuration."""
    
    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60


@celery_app.task(bind=True, base=StreamTask, name="process_stream")
async def process_stream(
    self,
    stream_id: int,
    chunk_duration: int = 30,
) -> Dict[str, Any]:
    """Process a stream - pure orchestration, no business logic.
    
    This task:
    1. Loads entities from repositories
    2. Delegates to domain services
    3. Persists results
    4. Returns status
    """
    logger.info(f"Processing stream {stream_id}")
    
    async with get_async_session() as session:
        # Load from repositories
        stream_repo = StreamRepository(session)
        dimension_repo = DimensionSetRepository(session)
        
        stream = await stream_repo.get(stream_id)
        if not stream:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Get dimension set (using stream's configuration)
        dimension_set = await dimension_repo.get(
            stream.dimension_set_id or 1  # Default if not set
        )
        
        # Let domain handle state transition
        stream.start_processing()
        await stream_repo.save(stream)
        
        try:
            # Initialize infrastructure components
            gemini = GeminiVideoProcessor(
                api_key=settings.external_apis.google_gemini_api_key.get_secret_value(),
                model_name="gemini-2.0-flash-exp"
            )
            
            # Create domain service
            analyzer = HighlightAnalyzer(ai_analyzer=gemini)
            
            # Configure pipeline (infrastructure concern)
            config = StreamIngestionConfig(
                stream_url=stream.url.value,
                stream_id=f"stream_{stream_id}",
                segment_duration=chunk_duration,
                enable_audio_processing=True,
                delete_segments_after_processing=True,
                max_concurrent_processing=3,
                temp_dir=Path(tempfile.gettempdir()) / f"stream_{stream_id}",
            )
            
            # Process segments as they arrive
            pipeline = StreamIngestionPipeline(
                config=config,
                gemini_processor=gemini,
                dimension_set=dimension_set,
                agent_config=None,  # Could load from repo if needed
            )
            
            total_highlights = 0
            async for result in pipeline.start():
                if not result.error:
                    # Convert infrastructure results to domain candidates
                    for highlight_data in result.highlights:
                        # Let the stream create its own highlights
                        stream.create_highlight_from_candidate(highlight_data)
                        total_highlights += 1
            
            # Complete processing
            stream.complete_processing()
            await stream_repo.save(stream)
            
            # Clean up
            await gemini.cleanup()
            
            return {
                "stream_id": stream_id,
                "status": "completed",
                "highlights_created": total_highlights,
            }
            
        except Exception as e:
            # Let domain handle failure
            stream.fail_processing(str(e))
            await stream_repo.save(stream)
            raise


@celery_app.task(name="cleanup_old_streams")
async def cleanup_old_streams(days: int = 30) -> Dict[str, Any]:
    """Clean up old completed streams - simple maintenance task."""
    from datetime import datetime, timedelta, timezone
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    async with get_async_session() as session:
        stream_repo = StreamRepository(session)
        
        # Simple cleanup - let repository handle the query
        cleaned = await stream_repo.cleanup_old(older_than=cutoff)
        
        return {
            "cleaned_count": cleaned,
            "cutoff_date": cutoff.isoformat(),
        }


# Simplified task registration
__all__ = ["process_stream", "cleanup_old_streams"]