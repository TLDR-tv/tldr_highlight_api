"""Mapper for stream DTOs to domain entities."""


from src.api.schemas.streams import (
    StreamCreate,
    StreamUpdate,
    StreamResponse,
    StreamOptions,
)
from src.domain.entities.stream import Stream, StreamPlatform, StreamStatus
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import (
    ProcessingOptions,
    DetectionStrategy,
    FusionStrategy,
)


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
        processing_options = StreamMapper._stream_options_to_processing_options(
            dto.options
        )

        return Stream(
            id=None,  # Will be set by repository
            url=Url(str(dto.source_url)),
            platform=StreamPlatform(dto.platform.value),
            status=StreamStatus.PENDING,
            user_id=user_id,
            processing_options=processing_options,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
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

        # Map to API field names
        api_options = {
            "dimension_set_id": options_dict.get("dimension_set_id"),
            "type_registry_id": options_dict.get("type_registry_id"),
            "detection_strategy": options_dict.get("detection_strategy"),
            "fusion_strategy": options_dict.get("fusion_strategy"),
            "enabled_modalities": list(options_dict.get("enabled_modalities", [])),
            "modality_weights": options_dict.get("modality_weights", {}),
            "min_confidence_threshold": options_dict.get(
                "min_confidence_threshold", 0.65
            ),
            "target_confidence_threshold": options_dict.get(
                "target_confidence_threshold", 0.75
            ),
            "exceptional_threshold": options_dict.get("exceptional_threshold", 0.85),
            "min_duration": int(options_dict.get("min_highlight_duration", 10)),
            "max_duration": int(options_dict.get("max_highlight_duration", 300)),
            "output_format": "mp4",  # Default
            "generate_thumbnails": True,  # Default
            "custom_tags": [],  # Default
            "webhook_events": [],  # Default
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
            completed_at=entity.completed_at.to_datetime()
            if entity.completed_at
            else None,
            is_active=entity.status in [StreamStatus.PENDING, StreamStatus.PROCESSING],
            processing_duration=processing_duration,
            highlight_count=highlight_count,
        )

    @staticmethod
    def _stream_options_to_processing_options(
        options: StreamOptions,
    ) -> ProcessingOptions:
        """Convert StreamOptions DTO to ProcessingOptions value object.

        Args:
            options: StreamOptions from API request

        Returns:
            ProcessingOptions domain value object
        """
        # Create enabled modalities set
        enabled_modalities = set()
        if options.enable_scene_detection:
            enabled_modalities.add("video")
        if options.enable_audio_analysis:
            enabled_modalities.add("audio")
        if options.enable_chat_analysis:
            enabled_modalities.add("text")

        # Default modality weights
        modality_weights = {"video": 0.4, "audio": 0.3, "text": 0.3}

        return ProcessingOptions(
            dimension_set_id=getattr(options, "dimension_set_id", None),
            type_registry_id=getattr(options, "type_registry_id", None),
            detection_strategy=DetectionStrategy.AI_ONLY,  # Default
            fusion_strategy=FusionStrategy.WEIGHTED,  # Default
            enabled_modalities=enabled_modalities,
            modality_weights=modality_weights,
            min_confidence_threshold=options.highlight_threshold,
            target_confidence_threshold=options.highlight_threshold,
            exceptional_threshold=min(0.9, options.highlight_threshold + 0.15),
            min_highlight_duration=float(options.min_duration),
            max_highlight_duration=float(options.max_duration),
            processing_priority="balanced",
        )
