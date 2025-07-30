"""Highlights router for the TL;DR Highlight API.

This module provides endpoints for highlight access and management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import get_highlight_management_use_case
from src.api.mappers.highlight_mapper import HighlightMapper
from src.api.schemas.common import COMMON_RESPONSES, StatusResponse
from src.api.schemas.highlights import (
    HighlightResponse,
    HighlightListResponse,
    HighlightFilters,
)
from src.application.use_cases.highlight_management import HighlightManagementUseCase
from src.application.use_cases.base import ResultStatus
from src.domain.entities.user import User

router = APIRouter()

mapper = HighlightMapper()


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Highlights service status",
    description="Check if highlights service is operational",
    responses=COMMON_RESPONSES,
)
async def highlights_status() -> StatusResponse:
    """Get highlights service status.

    Returns:
        StatusResponse: Highlights service status
    """
    from datetime import datetime

    return StatusResponse(
        status="Highlights service operational", timestamp=datetime.utcnow()
    )


@router.get(
    "/",
    response_model=HighlightListResponse,
    summary="List highlights",
    description="List highlights for the authenticated user with optional filters",
    responses=COMMON_RESPONSES,
)
async def list_highlights(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    stream_id: Optional[int] = Query(None, description="Filter by stream ID"),
    min_confidence: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Minimum confidence"
    ),
    current_user: User = Depends(get_current_user),
    use_case: HighlightManagementUseCase = Depends(get_highlight_management_use_case),
):
    """List highlights for the authenticated user.

    Returns paginated list of highlights with optional filtering.
    """
    # Build filters
    filters = HighlightFilters(stream_id=stream_id, min_confidence=min_confidence)

    request = mapper.to_list_highlights_request(
        user_id=current_user.id, filters=filters, page=page, per_page=per_page
    )

    result = await use_case.list_highlights(request)

    if not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to list highlights",
        )

    return mapper.to_highlight_list_response(
        highlights=result.highlights, total=result.total, page=page, per_page=per_page
    )


@router.get(
    "/{highlight_id}",
    response_model=HighlightResponse,
    summary="Get highlight details",
    description="Get details of a specific highlight",
    responses=COMMON_RESPONSES,
)
async def get_highlight(
    highlight_id: int,
    current_user: User = Depends(get_current_user),
    use_case: HighlightManagementUseCase = Depends(get_highlight_management_use_case),
):
    """Get highlight details.

    Returns detailed information about a specific highlight.
    """
    request = mapper.to_get_highlight_request(highlight_id, current_user.id)
    result = await use_case.get_highlight(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Highlight not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this highlight",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to get highlight",
        )

    return mapper.to_highlight_response(result.highlight)


@router.get(
    "/{highlight_id}/download",
    summary="Download highlight",
    description="Get a presigned URL to download the highlight file",
    responses=COMMON_RESPONSES,
)
async def download_highlight(
    highlight_id: int,
    current_user: User = Depends(get_current_user),
    use_case: HighlightManagementUseCase = Depends(get_highlight_management_use_case),
):
    """Download highlight file.

    Returns a presigned URL for downloading the highlight video.
    """
    request = mapper.to_export_highlight_request(highlight_id, current_user.id)
    result = await use_case.export_highlight(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Highlight not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this highlight",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to export highlight",
        )

    # Redirect to the presigned download URL
    return RedirectResponse(
        url=result.download_url, status_code=status.HTTP_303_SEE_OTHER
    )


@router.delete(
    "/{highlight_id}",
    summary="Delete highlight",
    description="Delete a highlight",
    responses=COMMON_RESPONSES,
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_highlight(
    highlight_id: int,
    current_user: User = Depends(get_current_user),
    use_case: HighlightManagementUseCase = Depends(get_highlight_management_use_case),
) -> None:
    """Delete a highlight.

    Permanently deletes the highlight and its associated files.
    """
    request = mapper.to_delete_highlight_request(highlight_id, current_user.id)
    result = await use_case.delete_highlight(request)

    if result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Highlight not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this highlight",
        )
    elif not result.is_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to delete highlight",
        )
