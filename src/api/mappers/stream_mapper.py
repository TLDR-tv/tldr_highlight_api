"""Mapper for stream DTOs to domain entities."""

from typing import Dict, Any

from src.api.schemas.streams import StreamCreate, StreamUpdate, StreamResponse, StreamOptions
from src.domain.entities.stream import Stream, StreamPlatform, StreamStatus
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import ProcessingOptions


class StreamMapper:
    """Maps between stream API DTOs and domain entities."""

    @staticmethod
    def stream_create_to_domain(dto: StreamCreate, user_id: int) -> Stream:
        """Convert StreamCreate DTO to Stream domain entity.
        
        Args:
            dto: StreamCreate request data
            user_id: ID of the user creating the stream
            
        Returns:
            Stream domain entity ready for persistence
        """
        # Convert DTO processing options to domain value object
        processing_options = StreamMapper._stream_options_to_processing_options(dto.options)
        
        return Stream(
            id=None,  # Will be set by repository
            url=Url(str(dto.source_url)),
            platform=StreamPlatform(dto.platform.value),
            status=StreamStatus.PENDING,
            user_id=user_id,
            processing_options=processing_options,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )

    @staticmethod
    def stream_update_to_processing_options(dto: StreamUpdate) -> ProcessingOptions:
        """Convert StreamUpdate DTO to ProcessingOptions value object.
        
        Args:
            dto: StreamUpdate request data
            
        Returns:
            ProcessingOptions value object
        """
        if dto.options is None:
            raise ValueError("Stream update must include options")
        
        return StreamMapper._stream_options_to_processing_options(dto.options)

    @staticmethod
    def stream_to_response(entity: Stream, highlight_count: int = 0) -> StreamResponse:
        """Convert Stream domain entity to StreamResponse DTO.
        
        Args:
            entity: Stream domain entity
            highlight_count: Number of highlights for this stream
            
        Returns:
            StreamResponse DTO for API response
        """
        # Convert processing options to dict for API response
        options_dict = entity.processing_options.to_dict()
        
        # Map legacy API field names to domain processing options
        api_options = {
            "highlight_threshold": options_dict["confidence_threshold"],
            "max_highlights": 10,  # Default, could be derived from processing options
            "min_duration": int(options_dict["min_highlight_duration"]),
            "max_duration": int(options_dict["max_highlight_duration"]),
            "enable_audio_analysis": options_dict["analyze_audio"],
            "enable_chat_analysis": options_dict["analyze_chat"],
            "enable_scene_detection": options_dict["analyze_video"],
            "output_format": "mp4",  # Default
            "output_quality": options_dict["video_quality"],
            "generate_thumbnails": True,  # Default
            "custom_tags": [],  # Default
            "webhook_events": []  # Default
        }
        
        # Calculate processing duration if available
        processing_duration = None
        if entity.processing_duration:
            processing_duration = entity.processing_duration.total_seconds
        
        return StreamResponse(
            id=entity.id,
            source_url=str(entity.url),
            platform=entity.platform.value,
            status=entity.status.value,
            options=api_options,
            user_id=entity.user_id,
            created_at=entity.created_at.to_datetime(),
            updated_at=entity.updated_at.to_datetime(),
            completed_at=entity.completed_at.to_datetime() if entity.completed_at else None,
            is_active=entity.status in [StreamStatus.PENDING, StreamStatus.PROCESSING],
            processing_duration=processing_duration,
            highlight_count=highlight_count
        )

    @staticmethod
    def _stream_options_to_processing_options(options: StreamOptions) -> ProcessingOptions:
        """Convert StreamOptions DTO to ProcessingOptions value object.
        
        Args:
            options: StreamOptions from API request
            
        Returns:
            ProcessingOptions domain value object
        """
        # Map video quality string to domain quality
        quality_mapping = {
            "720p": "medium",
            "1080p": "high", 
            "480p": "low",
            "4k": "high"
        }
        video_quality = quality_mapping.get(options.output_quality, "medium")
        
        return ProcessingOptions(
            confidence_threshold=options.highlight_threshold,
            min_highlight_duration=float(options.min_duration),
            max_highlight_duration=float(options.max_duration),
            analyze_video=options.enable_scene_detection,
            analyze_audio=options.enable_audio_analysis,
            analyze_chat=options.enable_chat_analysis,
            analyze_metadata=True,  # Always enabled
            include_chat_sentiment=options.enable_chat_analysis,
            include_viewer_metrics=True,  # Default enabled
            video_quality=video_quality,
            audio_quality="medium",  # Default
            custom_filters={},  # Could be derived from custom_tags
            excluded_categories=[]  # Default
        )