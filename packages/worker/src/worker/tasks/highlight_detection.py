"""Highlight detection tasks."""

from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from structlog import get_logger

from worker.services.ffmpeg_processor import FFmpegProcessor
from shared.infrastructure.storage.repositories import (
    StreamRepository,
    HighlightRepository,
    OrganizationRepository,
)
from shared.infrastructure.database.database import Database
from shared.infrastructure.config import get_settings
from shared.domain.models.highlight import Highlight, HighlightStatus
from shared.application.services.highlight_detector import HighlightDetector
from shared.infrastructure.gemini.scoring import GeminiScoringStrategy

logger = get_logger()


async def process_segment_for_highlights(
    stream_id: str,
    segment_data: Dict,
    processing_options: Dict,
    context_segments: List[Dict],
) -> List[Dict]:
    """Process a segment to detect highlights using Gemini File API.
    
    Analyzes a video segment using multi-dimensional scoring to identify
    potential highlights. Uses context from previous segments to improve
    accuracy and reduce false positives.
    
    Args:
        stream_id: UUID of the stream being processed.
        segment_data: Current segment information including video_path, 
            start_time, end_time, and audio_path.
        processing_options: Processing configuration including dimension_set_id,
            type_registry_id, fusion_strategy, and confidence thresholds.
        context_segments: Previous segments for temporal context.
        
    Returns:
        List of dictionaries containing detected highlight information
        including scores, timing, and confidence values.

    """
    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect()
    
    try:
        async with database.session() as session:
            # Get stream and organization info
            stream_repo = StreamRepository(session)
            org_repo = OrganizationRepository(session)
            highlight_repo = HighlightRepository(session)
            
            stream = await stream_repo.get_by_id(UUID(stream_id))
            if not stream:
                raise ValueError(f"Stream {stream_id} not found")
            
            organization = await org_repo.get_by_id(stream.organization_id)
            if not organization:
                raise ValueError(f"Organization not found for stream {stream_id}")
        
        # Initialize scoring strategy and detector
        scoring_strategy = GeminiScoringStrategy(
            api_key=settings.gemini_api_key,
            dimension_set_id=processing_options["dimension_set_id"],
        )
        
        highlight_detector = HighlightDetector(
            scoring_strategy=scoring_strategy,
            type_registry_id=processing_options["type_registry_id"],
            confidence_threshold=processing_options.get("confidence_threshold", 0.7),
        )
        
        # Process segment through detector
        detected_highlights = await highlight_detector.process_segment(
            video_path=segment_data["video_path"],
            start_time=segment_data["start_time"],
            end_time=segment_data["end_time"],
            context=context_segments,
        )
        
        # Create highlight records if any detected
        created_highlights = []
        if detected_highlights:
            ffmpeg_processor = FFmpegProcessor()
            
            async with database.session() as session:
                highlight_repo = HighlightRepository(session)
                
                for highlight_data in detected_highlights:
                    # Generate clip and thumbnail
                    clip_path, thumbnail_path = await ffmpeg_processor.create_highlight_clip(
                        source_path=segment_data["video_path"],
                        start_offset=highlight_data["start_offset"],
                        duration=highlight_data["duration"],
                        output_dir=f"/tmp/highlights/{stream_id}",
                    )
                    
                    # Upload to S3
                    clip_url = await _upload_to_s3(clip_path, f"highlights/{stream_id}/clips")
                    thumbnail_url = await _upload_to_s3(
                        thumbnail_path, f"highlights/{stream_id}/thumbnails"
                    )
                    
                    # Create highlight record
                    highlight = Highlight(
                        stream_id=stream.id,
                        organization_id=stream.organization_id,
                        start_time=segment_data["start_time"] + highlight_data["start_offset"],
                        end_time=(
                            segment_data["start_time"]
                            + highlight_data["start_offset"]
                            + highlight_data["duration"]
                        ),
                        duration=highlight_data["duration"],
                        type=highlight_data["type"],
                        confidence=highlight_data["confidence"],
                        title=highlight_data.get("title", ""),
                        description=highlight_data.get("description", ""),
                        clip_url=clip_url,
                        thumbnail_url=thumbnail_url,
                        status=HighlightStatus.READY,
                        metadata={
                            "dimension_scores": highlight_data["dimension_scores"],
                            "detected_at": datetime.utcnow().isoformat(),
                            "segment_id": segment_data["id"],
                        },
                    )
                    
                    await highlight_repo.create(highlight)
                    created_highlights.append(highlight.to_dict())
                    
                    # Send webhook notification
                    from worker.tasks.webhook_delivery import send_highlight_webhook
                    send_highlight_webhook.delay(
                        organization_id=str(stream.organization_id),
                        highlight_data=highlight.to_dict(),
                    )
        
        logger.info(
            "Segment processing completed",
            stream_id=stream_id,
            segment_id=segment_data["id"],
            highlights_found=len(created_highlights),
        )
        
        return created_highlights
        
    finally:
        await database.disconnect()


async def _upload_to_s3(file_path: str, s3_prefix: str) -> str:
    """Upload file to S3 and return URL."""
    import boto3
    from pathlib import Path
    
    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    
    file_name = Path(file_path).name
    key = f"{s3_prefix}/{file_name}"
    
    s3_client.upload_file(
        file_path,
        settings.s3_bucket_name,
        key,
        ExtraArgs={
            "ContentType": "video/mp4" if file_path.endswith(".mp4") else "image/jpeg"
        },
    )
    
    return f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{key}"