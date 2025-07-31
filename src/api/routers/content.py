"""Content delivery router for the TL;DR Highlight API.

This module provides endpoints for secure content access via signed URLs,
particularly for external streamers who are not managed users in the system.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from fastapi.responses import RedirectResponse, JSONResponse

from src.api.dependencies.use_cases import get_highlight_management_use_case
from src.application.use_cases.highlight_management import HighlightManagementUseCase
from src.application.use_cases.base import ResultStatus
from src.infrastructure.security.url_signer import url_signer
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.api.dependencies.repositories import get_highlight_repository, get_stream_repository

router = APIRouter()


@router.get(
    "/{highlight_id}",
    summary="Access highlight content via signed URL",
    description="Access highlight content using a signed URL token without authentication",
)
async def access_content(
    highlight_id: int,
    token: str,
    request: Request,
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
):
    """Access highlight content via signed URL.

    This endpoint allows access to highlight content using a signed URL token
    without requiring authentication. The token contains the necessary information
    to validate access rights.

    Args:
        highlight_id: ID of the highlight to access
        token: JWT token for validation
        request: FastAPI request object
        highlight_repo: Repository for highlight operations
        stream_repo: Repository for stream operations

    Returns:
        Redirect to the actual content or error response
    """
    # Verify token
    is_valid, payload, error = url_signer.verify_token(
        token, required_claims={"highlight_id": highlight_id}
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {error}",
        )

    # Get highlight
    highlight = await highlight_repo.get(highlight_id)
    if not highlight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Highlight not found"
        )

    # Verify stream association
    stream_id = payload.get("stream_id")
    if stream_id and stream_id != highlight.stream_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is not valid for this highlight",
        )

    # Get video URL
    video_url = str(highlight.video_url)
    if not video_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video URL not found"
        )

    # Redirect to the actual content
    return RedirectResponse(url=video_url)


@router.get(
    "/stream/{stream_id}",
    summary="List highlights for a stream via signed URL",
    description="List all highlights for a stream using a signed URL token without authentication",
)
async def list_stream_highlights(
    stream_id: int,
    token: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    highlight_repo: HighlightRepository = Depends(get_highlight_repository),
    stream_repo: StreamRepository = Depends(get_stream_repository),
):
    """List highlights for a stream via signed URL.

    This endpoint allows listing all highlights for a stream using a signed URL token
    without requiring authentication.

    Args:
        stream_id: ID of the stream
        token: JWT token for validation
        page: Page number for pagination
        per_page: Items per page for pagination
        highlight_repo: Repository for highlight operations
        stream_repo: Repository for stream operations

    Returns:
        List of highlights for the stream
    """
    # Verify token
    is_valid, payload, error = url_signer.verify_token(
        token, required_claims={"stream_id": stream_id}
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {error}",
        )

    # Verify stream exists
    stream = await stream_repo.get(stream_id)
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found"
        )

    # Get highlights for stream with pagination
    offset = (page - 1) * per_page
    result = await highlight_repo.get_by_stream_with_pagination(
        stream_id=stream_id, limit=per_page, offset=offset
    )
    
    # Extract highlights and pagination info
    highlights = result["highlights"]
    pagination = result["pagination"]
    total_count = pagination["total_count"]

    # Format response
    result = {
        "items": [
            {
                "id": h.id,
                "title": h.title,
                "description": h.description,
                "thumbnail_url": str(h.thumbnail_url) if h.thumbnail_url else None,
                "duration": h.duration.seconds,
                "confidence_score": h.confidence_score.value,
                "created_at": h.created_at.isoformat() if h.created_at else None,
                # Generate content access URL for each highlight
                "content_url": url_signer.generate_signed_url(
                    base_url=str(request.base_url).rstrip('/'),
                    highlight_id=h.id,
                    stream_id=stream_id,
                    org_id=payload.get("org_id"),
                    expiry_hours=24,  # Same expiry as the stream token
                ),
            }
            for h in highlights
        ],
        "page": page,
        "per_page": per_page,
        "total": total_count,
        "pages": (total_count + per_page - 1) // per_page,
    }

    return JSONResponse(content=result)


@router.get(
    "/health",
    summary="Content delivery health check",
    description="Check if content delivery service is operational",
)
async def content_health():
    """Get content delivery service health status.

    Returns:
        Health status
    """
    from datetime import datetime

    return {
        "status": "Content delivery service operational",
        "timestamp": datetime.utcnow().isoformat(),
    }