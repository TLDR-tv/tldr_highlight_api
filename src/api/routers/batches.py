"""Batch processing router for the TL;DR Highlight API.

This module provides endpoints for batch video processing operations.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from src.api.schemas.common import StatusResponse
from src.api.schemas.batches import (
    BatchCreate,
    BatchResponse,
    BatchListResponse,
)
from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import get_batch_processing_use_case
from src.application.use_cases.batch_processing import (
    BatchProcessingUseCase,
    BatchCreateRequest,
    BatchStatusRequest,
    BatchCancelRequest,
)
from src.application.use_cases.base import ResultStatus
from src.domain.entities.user import User

router = APIRouter(prefix="/batches", tags=["batches"])


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Batch service status",
    description="Check if batch processing service is operational",
)
async def batch_status() -> StatusResponse:
    """Get batch processing service status.

    Returns:
        StatusResponse: Batch service status
    """
    from datetime import datetime

    return StatusResponse(
        status="Batch processing service operational", timestamp=datetime.utcnow()
    )


@router.post(
    "/",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create batch job",
    description="Create a new batch processing job for multiple videos",
)
async def create_batch(
    batch_data: BatchCreate,
    current_user: User = Depends(get_current_user),
    use_case: BatchProcessingUseCase = Depends(get_batch_processing_use_case),
) -> BatchResponse:
    """Create a new batch processing job.

    Creates a batch job that will process multiple videos in parallel
    and extract highlights from each using AI-powered detection.

    Args:
        batch_data: Batch configuration including videos and processing options
        current_user: Authenticated user creating the batch
        use_case: Batch processing use case

    Returns:
        BatchResponse: Created batch details

    Raises:
        HTTPException: If batch creation fails
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    # Convert videos to the format expected by use case
    items = []
    for video in batch_data.videos:
        items.append(
            {
                "url": str(video.url),
                "title": video.title or "Untitled",
                "metadata": video.metadata or {},
            }
        )

    # Convert API request to use case request
    request = BatchCreateRequest(
        user_id=current_user.id,
        name=f"Batch job - {len(batch_data.videos)} videos",
        items=items,
        processing_options={
            "confidence_threshold": batch_data.options.highlight_threshold,
            "min_highlight_duration": float(batch_data.options.min_duration),
            "max_highlight_duration": float(batch_data.options.max_duration),
            "parallel_processing": batch_data.options.parallel_processing,
        },
    )

    # Execute use case
    result = await use_case.create_batch(request)

    # Handle result
    if result.status == ResultStatus.SUCCESS:
        # Validate result has required fields
        if result.batch_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from batch processing service",
            )

        # Create response using mapper (simplified for now)
        from datetime import datetime
        from src.infrastructure.persistence.models.batch import BatchStatus

        return BatchResponse(
            id=result.batch_id,
            status=BatchStatus.PENDING,
            options=batch_data.options.model_dump(),
            user_id=current_user.id,
            video_count=result.total_items or len(batch_data.videos),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True,
            processed_count=0,
            progress_percentage=0.0,
            highlight_count=0,
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.errors[0] if result.errors else "User not found",
        )
    elif result.status == ResultStatus.QUOTA_EXCEEDED:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=result.errors[0] if result.errors else "Quota exceeded",
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.errors[0] if result.errors else "Validation error",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to create batch",
        )


@router.get(
    "/",
    response_model=BatchListResponse,
    summary="List batch jobs",
    description="List batch processing jobs for the authenticated user",
)
async def list_batches(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by batch status"),
    current_user: User = Depends(get_current_user),
    use_case: BatchProcessingUseCase = Depends(get_batch_processing_use_case),
) -> BatchListResponse:
    """List batch jobs for the authenticated user.

    Returns a paginated list of batch jobs owned by the authenticated user,
    with optional filtering by status.

    Args:
        page: Page number (1-based)
        per_page: Number of items per page
        status_filter: Optional status filter
        current_user: Authenticated user
        use_case: Batch processing use case

    Returns:
        BatchListResponse: Paginated list of batches
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    # For now, return empty list as this requires repository list methods
    # TODO: Implement batch listing in use case and repository
    return BatchListResponse(
        page=page,
        per_page=per_page,
        total=0,
        pages=0,
        has_next=False,
        has_prev=False,
        items=[],
    )


@router.get(
    "/{batch_id}",
    response_model=BatchResponse,
    summary="Get batch job details",
    description="Get detailed information about a specific batch job",
)
async def get_batch(
    batch_id: int,
    current_user: User = Depends(get_current_user),
    use_case: BatchProcessingUseCase = Depends(get_batch_processing_use_case),
) -> BatchResponse:
    """Get batch job details.

    Retrieves detailed information about a specific batch job including
    current status, processing progress, and item details.

    Args:
        batch_id: Batch ID to retrieve
        current_user: Authenticated user
        use_case: Batch processing use case

    Returns:
        BatchResponse: Batch details

    Raises:
        HTTPException: If batch not found or access denied
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    # Get batch status
    request = BatchStatusRequest(
        user_id=current_user.id, batch_id=batch_id, include_items=False
    )

    result = await use_case.get_batch_status(request)

    if result.status == ResultStatus.SUCCESS:
        # Validate result has required fields
        if result.batch_id is None or result.batch_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from batch processing service",
            )

        # TODO: Get actual batch entity from repository to create proper response
        # For now, create minimal response from status result
        from datetime import datetime
        from src.infrastructure.persistence.models.batch import BatchStatus

        return BatchResponse(
            id=result.batch_id,
            status=BatchStatus(result.batch_status),
            options={},  # Would come from batch entity
            user_id=current_user.id,
            video_count=1,  # Would come from batch entity
            created_at=datetime.utcnow(),  # Would come from batch entity
            updated_at=datetime.utcnow(),  # Would come from batch entity
            is_active=result.batch_status in ["pending", "processing"],
            processed_count=result.successful_items or 0,
            progress_percentage=result.progress_percentage or 0.0,
            highlight_count=0,  # Would need to be calculated
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to get batch",
        )


@router.delete(
    "/{batch_id}",
    response_model=StatusResponse,
    summary="Cancel batch job",
    description="Cancel a batch processing job and stop processing any remaining videos",
)
async def cancel_batch(
    batch_id: int,
    reason: str = Query(
        "User requested cancellation", description="Reason for cancellation"
    ),
    current_user: User = Depends(get_current_user),
    use_case: BatchProcessingUseCase = Depends(get_batch_processing_use_case),
) -> StatusResponse:
    """Cancel a batch processing job.

    Cancels the batch processing job and stops processing any videos
    that haven't been processed yet.

    Args:
        batch_id: Batch ID to cancel
        reason: Reason for cancellation
        current_user: Authenticated user
        use_case: Batch processing use case

    Returns:
        StatusResponse: Operation result

    Raises:
        HTTPException: If operation fails
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    request = BatchCancelRequest(
        user_id=current_user.id, batch_id=batch_id, reason=reason
    )

    result = await use_case.cancel_batch(request)

    if result.status == ResultStatus.SUCCESS:
        from datetime import datetime

        return StatusResponse(
            status=f"Batch {batch_id} cancelled successfully. {result.cancelled_items or 0} items cancelled.",
            timestamp=datetime.utcnow(),
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.errors[0] if result.errors else "Invalid operation",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to cancel batch",
        )
