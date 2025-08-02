"""Highlight retrieval and management service."""

from typing import Optional
from uuid import UUID
import structlog

from ...domain.models.highlight import Highlight
from ...infrastructure.storage.repositories import HighlightRepository

logger = structlog.get_logger()


class HighlightService:
    """Service for highlight retrieval and management."""

    def __init__(self, highlight_repository: HighlightRepository):
        """Initialize with dependencies."""
        self.highlight_repository = highlight_repository

    async def get_highlight(
        self, highlight_id: UUID, organization_id: UUID
    ) -> Optional[Highlight]:
        """Get a specific highlight by ID.

        Args:
            highlight_id: Highlight ID
            organization_id: Organization ID (for access control)

        Returns:
            Highlight if found and belongs to organization, None otherwise
        """
        highlight = await self.highlight_repository.get(highlight_id)

        # Ensure highlight belongs to the organization
        if highlight and highlight.organization_id != organization_id:
            logger.warning(
                "Access denied to highlight",
                highlight_id=str(highlight_id),
                requested_org=str(organization_id),
                actual_org=str(highlight.organization_id),
            )
            return None

        return highlight

    async def list_highlights(
        self,
        organization_id: UUID,
        stream_id: Optional[UUID] = None,
        wake_word_triggered: Optional[bool] = None,
        min_score: Optional[float] = None,
        order_by: str = "created_at",
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """List highlights for an organization with filters.

        Args:
            organization_id: Organization ID
            stream_id: Filter by stream ID
            wake_word_triggered: Filter by wake word trigger status
            min_score: Minimum overall score filter
            order_by: Sort order (created_at or score)
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Dictionary with highlights and metadata
        """
        # Build filters
        filters = {
            "organization_id": organization_id,
            "order_by": order_by,
            "limit": limit,
            "offset": offset,
        }

        if stream_id is not None:
            filters["stream_id"] = stream_id
        if wake_word_triggered is not None:
            filters["wake_word_triggered"] = wake_word_triggered
        if min_score is not None:
            filters["min_score"] = min_score

        # Get highlights
        highlights = await self.highlight_repository.list(**filters)

        # Count total (simplified - in production would use a count query)
        total = len(highlights)
        if total == limit:
            # Might be more
            total = offset + limit + 1

        logger.info(
            "Listed highlights",
            organization_id=str(organization_id),
            count=len(highlights),
            filters=filters,
        )

        return {
            "highlights": highlights,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": len(highlights) == limit,
        }

    async def get_stream_highlights(
        self, stream_id: UUID, organization_id: UUID
    ) -> list[Highlight]:
        """Get all highlights for a specific stream.

        Args:
            stream_id: Stream ID
            organization_id: Organization ID (for access control)

        Returns:
            List of highlights for the stream
        """
        # In production, would verify stream belongs to organization
        highlights = await self.highlight_repository.list_by_stream(stream_id)

        # Filter to organization's highlights only
        org_highlights = [h for h in highlights if h.organization_id == organization_id]

        logger.info(
            "Retrieved stream highlights",
            stream_id=str(stream_id),
            organization_id=str(organization_id),
            count=len(org_highlights),
        )

        return org_highlights
