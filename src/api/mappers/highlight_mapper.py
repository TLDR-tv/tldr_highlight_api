"""Highlight mapper for converting between API DTOs and domain objects."""

from typing import List, Optional, Dict

from src.api.schemas.highlights import (
    HighlightResponse,
    HighlightListResponse,
    HighlightFilters,
    HighlightUpdate,
)
from src.application.use_cases.highlight_management import (
    GetHighlightRequest,
    ListHighlightsRequest,
    UpdateHighlightRequest,
    DeleteHighlightRequest,
    ExportHighlightRequest,
)
from src.domain.entities.highlight import Highlight


class HighlightMapper:
    """Maps between highlight API DTOs and domain entities."""

    @staticmethod
    def to_get_highlight_request(
        highlight_id: int, user_id: int
    ) -> GetHighlightRequest:
        """Convert parameters to GetHighlightRequest."""
        return GetHighlightRequest(user_id=user_id, highlight_id=highlight_id)

    @staticmethod
    def to_list_highlights_request(
        user_id: int,
        filters: Optional[HighlightFilters] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> ListHighlightsRequest:
        """Convert filter parameters to ListHighlightsRequest."""
        # Extract individual filter parameters
        stream_id = None
        min_confidence = None
        
        if filters:
            stream_id = filters.stream_id
            min_confidence = filters.min_confidence

        return ListHighlightsRequest(
            user_id=user_id,
            stream_id=stream_id,
            min_confidence=min_confidence,
            page=page,
            per_page=per_page
        )

    @staticmethod
    def to_update_highlight_request(
        highlight_id: int, user_id: int, update_dto: HighlightUpdate
    ) -> UpdateHighlightRequest:
        """Convert update DTO to domain request."""
        return UpdateHighlightRequest(
            user_id=user_id,
            highlight_id=highlight_id,
            title=update_dto.title,
            description=update_dto.description,
            tags=update_dto.tags,
        )

    @staticmethod
    def to_delete_highlight_request(
        highlight_id: int, user_id: int
    ) -> DeleteHighlightRequest:
        """Convert parameters to DeleteHighlightRequest."""
        return DeleteHighlightRequest(user_id=user_id, highlight_id=highlight_id)

    @staticmethod
    def to_export_highlight_request(
        highlight_id: int, user_id: int
    ) -> ExportHighlightRequest:
        """Convert parameters to ExportHighlightRequest."""
        return ExportHighlightRequest(user_id=user_id, highlight_id=highlight_id)

    @staticmethod
    def to_highlight_response(
        highlight: Highlight, download_url: Optional[str] = None
    ) -> HighlightResponse:
        """Convert Highlight domain entity to response DTO."""
        # Determine source type
        source_type = "stream" if highlight.stream_id else "batch"

        # Extract metadata
        extra_metadata = highlight.metadata or {}

        return HighlightResponse(
            id=highlight.id,
            stream_id=highlight.stream_id,
            batch_id=None,  # Batch ID not in domain entity yet
            title=highlight.title,
            description=highlight.description or "",
            video_url=highlight.video_url.value,
            thumbnail_url=highlight.thumbnail_url.value
            if highlight.thumbnail_url
            else None,
            duration=highlight.duration,
            timestamp=highlight.timestamp,
            confidence_score=highlight.confidence_score,
            tags=highlight.tags,
            extra_metadata=extra_metadata,
            created_at=highlight.created_at.value,
            source_type=source_type,
            is_high_confidence=highlight.confidence_score > 0.8,
            download_url=download_url,
        )

    @staticmethod
    def to_highlight_list_response(
        highlights: List[Highlight], total: int, page: int, per_page: int
    ) -> HighlightListResponse:
        """Convert list of highlights to response DTO."""
        # Convert highlights
        items = [
            HighlightMapper.to_highlight_response(highlight) for highlight in highlights
        ]

        # Calculate aggregations
        if highlights:
            total_duration = sum(h.duration for h in highlights)
            avg_confidence = sum(h.confidence_score for h in highlights) / len(
                highlights
            )

            # Count tags
            tag_counts: Dict[str, int] = {}
            for highlight in highlights:
                for tag in highlight.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        else:
            total_duration = 0.0
            avg_confidence = 0.0
            tag_counts = {}

        # Calculate pagination info
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1

        return HighlightListResponse(
            page=page,
            per_page=per_page,
            total=total,
            pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            items=items,
            total_duration=total_duration,
            avg_confidence=avg_confidence,
            tag_counts=tag_counts,
        )
