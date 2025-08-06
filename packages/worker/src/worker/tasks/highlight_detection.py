"""Highlight detection tasks."""

from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import asdict
from structlog import get_logger

from worker.services.ffmpeg_processor import FFmpegProcessor
from shared.infrastructure.storage.repositories import (
    StreamRepository,
    HighlightRepository,
    OrganizationRepository,
)
from shared.infrastructure.database.database import Database
from shared.infrastructure.config.config import get_settings
from shared.domain.models.highlight import Highlight
from worker.services.highlight_detector import HighlightDetector
from worker.services.gemini_scorer import GeminiVideoScorer
from worker.services.rubric_registry import RubricRegistry

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
    
    async with database.session() as session:
        # Get stream and organization info
        stream_repo = StreamRepository(session)
        org_repo = OrganizationRepository(session)
        highlight_repo = HighlightRepository(session)
        
        stream = await stream_repo.get(UUID(stream_id))
        if not stream:
            raise ValueError(f"Stream {stream_id} not found")
        
        organization = await org_repo.get(stream.organization_id)
        if not organization:
            raise ValueError(f"Organization not found for stream {stream_id}")
    
    # Debug: Log organization details
    logger.info(
        f"Organization details for stream {stream_id}",
        organization_id=str(organization.id),
        organization_name=organization.name,
        organization_rubric_name=organization.rubric_name,
    )
    
    # Get the rubric for this organization
    logger.debug(f"Looking up rubric: '{organization.rubric_name}' for organization {organization.name}")
    rubric = RubricRegistry.get_rubric(organization.rubric_name)
    if not rubric:
        logger.warning(
            f"Rubric '{organization.rubric_name}' not found, using general rubric",
            organization_id=str(organization.id),
            rubric_name=organization.rubric_name,
        )
        rubric = RubricRegistry.get_rubric("general")
        if not rubric:
            raise ValueError("General rubric not found in registry")
    
    logger.info(
        "Using rubric for highlight detection",
        rubric_name=rubric.name,
        organization_id=str(organization.id),
        stream_id=stream_id,
    )
    
    # Initialize scoring strategy and detector
    scoring_strategy = GeminiVideoScorer(
        api_key=settings.gemini_api_key,
    )
    
    highlight_detector = HighlightDetector(
        scoring_strategy=scoring_strategy,
        min_highlight_duration=processing_options.get("min_highlight_duration", 10.0),
        max_highlight_duration=processing_options.get("max_highlight_duration", 120.0),
    )
    
    # Convert segment data to VideoSegment for the detector
    from worker.services.highlight_detector import VideoSegment
    from pathlib import Path
    
    video_segment = VideoSegment(
        file_path=Path(segment_data["video_path"]),
        start_time=segment_data["start_time"],
        duration=segment_data["end_time"] - segment_data["start_time"],
        segment_number=segment_data.get("segment_number", 0),
    )
    
    # Detect highlights in the segment
    detected_highlights = await highlight_detector.detect_highlights(
        stream=stream,
        segments=[video_segment],
        rubric=rubric,
    )
    
    # Create highlight records if any detected
    created_highlights = []
    if detected_highlights:
        async with database.session() as session:
            highlight_repo = HighlightRepository(session)
            
            for highlight_candidate in detected_highlights:
                # Use the candidate's method to get precise offset and duration
                segment_start = segment_data["start_time"] 
                clip_offset, clip_duration = highlight_candidate.get_clip_offset_and_duration(segment_start)
                
                # Log clipping approach for monitoring
                boundary_type = "precise" if highlight_candidate.has_precise_boundaries else "segment-based"
                logger.info(
                    f"Creating {boundary_type} clip: offset={clip_offset:.1f}s, "
                    f"duration={clip_duration:.1f}s, confidence={highlight_candidate.boundary_confidence:.2f}"
                )
                
                # Generate clip and thumbnail using enhanced method
                try:
                    clip_path, thumbnail_path = await FFmpegProcessor.create_highlight_clip(
                        source_path=segment_data["video_path"],
                        start_offset=clip_offset,
                        duration=clip_duration,
                        output_dir=f"/tmp/highlights/{stream_id}",
                        min_duration=processing_options.get("min_highlight_duration", 15.0),
                        max_duration=processing_options.get("max_highlight_duration", 90.0),
                    )
                    
                    logger.info(f"Successfully created clip: {clip_path}")
                    
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Failed to create clip for highlight candidate: {e}")
                    continue  # Skip this candidate and continue with others
                
                # Upload to S3
                clip_url = await _upload_to_s3(clip_path, f"highlights/{stream_id}/clips")
                thumbnail_url = await _upload_to_s3(
                    thumbnail_path, f"highlights/{stream_id}/thumbnails"
                )
                
                # Convert to highlight using the candidate's method
                highlight = highlight_candidate.to_highlight(
                    organization_id=stream.organization_id,
                    clip_url=clip_url,
                    thumbnail_url=thumbnail_url,
                )
                
                await highlight_repo.create(highlight)
                created_highlights.append(asdict(highlight))
                
                # Send webhook notification
                from worker.tasks.webhook_delivery import send_highlight_webhook
                send_highlight_webhook.delay(
                    organization_id=str(stream.organization_id),
                    highlight_data=asdict(highlight),
                )
    
    logger.info(
        "Segment processing completed",
        stream_id=stream_id,
        segment_id=segment_data["id"],
        highlights_found=len(created_highlights),
    )
    
    return created_highlights


async def _upload_to_s3(file_path: str, s3_prefix: str) -> str:
    """Upload file to S3 and return URL."""
    import boto3
    from pathlib import Path
    
    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.s3_region,
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
    
    return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{key}"