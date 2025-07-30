"""Mapper for batch DTOs to domain entities."""

from typing import Dict, Any, List

from src.api.schemas.batches import BatchCreate, BatchUpdate, BatchResponse, VideoInput, BatchOptions
from src.domain.entities.batch import Batch, BatchItem, BatchStatus
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import ProcessingOptions, DetectionStrategy, FusionStrategy


class BatchMapper:
    """Maps between batch API DTOs and domain entities."""

    @staticmethod
    def batch_create_to_domain(dto: BatchCreate, user_id: int, batch_name: str) -> Batch:
        """Convert BatchCreate DTO to Batch domain entity.
        
        Args:
            dto: BatchCreate request data
            user_id: ID of the user creating the batch
            batch_name: Name for the batch job
            
        Returns:
            Batch domain entity ready for persistence
        """
        # Convert DTO processing options to domain value object
        processing_options = BatchMapper._batch_options_to_processing_options(dto.options)
        
        # Convert video inputs to batch items
        items = []
        for i, video in enumerate(dto.videos):
            item = BatchItem(
                url=Url(str(video.url)),
                item_id=f"video_{i}_{video.title or 'untitled'}",
                status=BatchStatus.PENDING,
                metadata=video.metadata or {}
            )
            items.append(item)
        
        return Batch(
            id=None,  # Will be set by repository
            name=batch_name,
            user_id=user_id,
            status=BatchStatus.PENDING,
            processing_options=processing_options,
            items=items,
            total_items=len(items),
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )

    @staticmethod
    def batch_update_to_processing_options(dto: BatchUpdate) -> ProcessingOptions:
        """Convert BatchUpdate DTO to ProcessingOptions value object.
        
        Args:
            dto: BatchUpdate request data
            
        Returns:
            ProcessingOptions value object
        """
        if dto.options is None:
            raise ValueError("Batch update must include options")
        
        return BatchMapper._batch_options_to_processing_options(dto.options)

    @staticmethod
    def batch_to_response(entity: Batch, highlight_count: int = 0) -> BatchResponse:
        """Convert Batch domain entity to BatchResponse DTO.
        
        Args:
            entity: Batch domain entity
            highlight_count: Total number of highlights for this batch
            
        Returns:
            BatchResponse DTO for API response
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
            "min_confidence_threshold": options_dict.get("min_confidence_threshold", 0.65),
            "target_confidence_threshold": options_dict.get("target_confidence_threshold", 0.75),
            "exceptional_threshold": options_dict.get("exceptional_threshold", 0.85),
            "min_duration": int(options_dict.get("min_highlight_duration", 10)),
            "max_duration": int(options_dict.get("max_highlight_duration", 300)),
            "max_highlights_per_video": 10,  # Default
            "parallel_processing": True,  # Default
            "output_format": "mp4",  # Default
            "generate_thumbnails": True,  # Default
            "custom_tags": [],  # Default
            "webhook_events": [],  # Default
            "priority": "normal"  # Default
        }
        
        # Calculate estimated completion time (simplified)
        estimated_completion = None
        if entity.status == BatchStatus.PROCESSING and entity.started_at:
            # Simple estimate: if we're 50% done, estimate remaining time
            if entity.progress_percentage > 0:
                elapsed_minutes = entity.started_at.value.timestamp()
                from datetime import datetime, timezone, timedelta
                current_time = datetime.now(timezone.utc)
                elapsed = (current_time - entity.started_at.value).total_seconds() / 60
                
                if entity.progress_percentage > 10:  # Avoid division by very small numbers
                    total_estimated_minutes = elapsed / (entity.progress_percentage / 100)
                    remaining_minutes = total_estimated_minutes - elapsed
                    estimated_completion = current_time + timedelta(minutes=remaining_minutes)
        
        return BatchResponse(
            id=entity.id,
            status=BatchStatus(entity.status.value),
            options=api_options,
            user_id=entity.user_id,
            video_count=entity.total_items,
            created_at=entity.created_at.value,
            updated_at=entity.updated_at.value,
            is_active=entity.status in [BatchStatus.PENDING, BatchStatus.PROCESSING],
            processed_count=entity.processed_items,
            progress_percentage=entity.progress_percentage,
            highlight_count=highlight_count,
            estimated_completion=estimated_completion
        )

    @staticmethod
    def video_input_to_batch_item(video: VideoInput, item_id: str) -> BatchItem:
        """Convert VideoInput DTO to BatchItem domain object.
        
        Args:
            video: VideoInput from API request
            item_id: Unique identifier for this item
            
        Returns:
            BatchItem domain object
        """
        return BatchItem(
            url=Url(str(video.url)),
            item_id=item_id,
            status=BatchStatus.PENDING,
            metadata=video.metadata or {}
        )

    @staticmethod
    def _batch_options_to_processing_options(options: BatchOptions) -> ProcessingOptions:
        """Convert BatchOptions DTO to ProcessingOptions value object.
        
        Args:
            options: BatchOptions from API request
            
        Returns:
            ProcessingOptions domain value object
        """
        # Create enabled modalities set
        enabled_modalities = set()
        if options.enable_scene_detection:
            enabled_modalities.add("video")
        if options.enable_audio_analysis:
            enabled_modalities.add("audio")
        enabled_modalities.add("metadata")  # Always enabled
        
        # Default modality weights
        modality_weights = {
            "video": 0.5,
            "audio": 0.3,
            "metadata": 0.2
        }
        
        return ProcessingOptions(
            dimension_set_id=getattr(options, 'dimension_set_id', None),
            type_registry_id=getattr(options, 'type_registry_id', None),
            detection_strategy=DetectionStrategy.AI_ONLY,  # Default for batch
            fusion_strategy=FusionStrategy.WEIGHTED,  # Default
            enabled_modalities=enabled_modalities,
            modality_weights=modality_weights,
            min_confidence_threshold=options.highlight_threshold,
            target_confidence_threshold=options.highlight_threshold,
            exceptional_threshold=min(0.9, options.highlight_threshold + 0.15),
            min_highlight_duration=float(options.min_duration),
            max_highlight_duration=float(options.max_duration),
            processing_priority="balanced"
        )