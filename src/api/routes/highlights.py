"""Highlight management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from uuid import UUID
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_current_organization, require_scope, get_session
from ...domain.models.api_key import APIScopes
from ...domain.models.organization import Organization
from ...infrastructure.storage.repositories import HighlightRepository
from ...application.services.highlight_service import HighlightService

router = APIRouter()


@router.get("/streams/{stream_id}/highlights")
async def get_stream_highlights(
    stream_id: UUID,
    organization: Organization = Depends(get_current_organization),
    _=Depends(require_scope(APIScopes.HIGHLIGHTS_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Get all highlights for a specific stream.
    
    Returns highlights in chronological order (by start time).
    """
    highlight_repo = HighlightRepository(session)
    highlight_service = HighlightService(highlight_repo)
    
    highlights = await highlight_service.get_stream_highlights(
        stream_id=stream_id,
        organization_id=organization.id,
    )
    
    return {
        "stream_id": str(stream_id),
        "highlights": [
            {
                "id": str(h.id),
                "start_time": h.start_time,
                "end_time": h.end_time,
                "duration": h.duration,
                "title": h.title,
                "overall_score": h.overall_score,
                "wake_word_triggered": h.wake_word_triggered,
                "created_at": h.created_at.isoformat(),
            }
            for h in highlights
        ],
        "total": len(highlights),
    }


@router.get("/")
async def list_highlights(
    stream_id: Optional[UUID] = Query(None, description="Filter by stream ID"),
    wake_word_triggered: Optional[bool] = Query(None, description="Filter by wake word trigger status"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum overall score"),
    order_by: str = Query("created_at", pattern="^(created_at|score)$", description="Sort order"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    organization: Organization = Depends(get_current_organization),
    _=Depends(require_scope(APIScopes.HIGHLIGHTS_READ)),
    session: AsyncSession = Depends(get_session),
):
    """List highlights for the organization with optional filters.
    
    Query parameters:
    - stream_id: Filter by specific stream
    - wake_word_triggered: Filter by wake word trigger status
    - min_score: Minimum overall score (0.0-1.0)
    - order_by: Sort by 'created_at' or 'score'
    - limit: Maximum results (1-1000)
    - offset: Pagination offset
    """
    highlight_repo = HighlightRepository(session)
    highlight_service = HighlightService(highlight_repo)
    
    result = await highlight_service.list_highlights(
        organization_id=organization.id,
        stream_id=stream_id,
        wake_word_triggered=wake_word_triggered,
        min_score=min_score,
        order_by=order_by,
        limit=limit,
        offset=offset,
    )
    
    # Convert to API response format
    return {
        "highlights": [
            {
                "id": str(h.id),
                "stream_id": str(h.stream_id),
                "start_time": h.start_time,
                "end_time": h.end_time,
                "duration": h.duration,
                "title": h.title,
                "description": h.description,
                "tags": h.tags,
                "overall_score": h.overall_score,
                "dimension_scores": [
                    {
                        "name": ds.name,
                        "score": ds.score,
                        "confidence": ds.confidence,
                    }
                    for ds in h.dimension_scores
                ],
                "clip_url": h.clip_path,
                "thumbnail_url": h.thumbnail_path,
                "transcript": h.transcript,
                "wake_word_triggered": h.wake_word_triggered,
                "wake_word_detected": h.wake_word_detected,
                "created_at": h.created_at.isoformat(),
            }
            for h in result["highlights"]
        ],
        "total": result["total"],
        "limit": result["limit"],
        "offset": result["offset"],
        "has_more": result["has_more"],
    }


@router.get("/{highlight_id}")
async def get_highlight(
    highlight_id: UUID,
    organization: Organization = Depends(get_current_organization),
    _=Depends(require_scope(APIScopes.HIGHLIGHTS_READ)),
    session: AsyncSession = Depends(get_session),
):
    """Get detailed information about a specific highlight.
    
    Returns full highlight data including:
    - Timing information
    - Content metadata
    - Dimension scores
    - Media URLs
    - Transcript (if available)
    """
    highlight_repo = HighlightRepository(session)
    highlight_service = HighlightService(highlight_repo)
    
    highlight = await highlight_service.get_highlight(
        highlight_id=highlight_id,
        organization_id=organization.id,
    )
    
    if not highlight:
        raise HTTPException(status_code=404, detail="Highlight not found")
    
    return {
        "id": str(highlight.id),
        "stream_id": str(highlight.stream_id),
        "organization_id": str(highlight.organization_id),
        "start_time": highlight.start_time,
        "end_time": highlight.end_time,
        "duration": highlight.duration,
        "title": highlight.title,
        "description": highlight.description,
        "tags": highlight.tags,
        "overall_score": highlight.overall_score,
        "dimension_scores": [
            {
                "name": ds.name,
                "score": ds.score,
                "confidence": ds.confidence,
            }
            for ds in highlight.dimension_scores
        ],
        "clip_url": highlight.clip_path,
        "thumbnail_url": highlight.thumbnail_path,
        "transcript": highlight.transcript,
        "wake_word_triggered": highlight.wake_word_triggered,
        "wake_word_detected": highlight.wake_word_detected,
        "created_at": highlight.created_at.isoformat(),
        "updated_at": highlight.updated_at.isoformat(),
    }
