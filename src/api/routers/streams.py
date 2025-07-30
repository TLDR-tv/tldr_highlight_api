"""Stream management router for the TL;DR Highlight API.

This module provides endpoints for livestream processing and management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from src.api.schemas.common import StatusResponse
from src.api.schemas.streams import (
    StreamCreate,
    StreamUpdate,
    StreamResponse,
    StreamListResponse,
    PaginationParams
)
from src.infrastructure.persistence.models.stream import StreamPlatform, StreamStatus
from src.api.dependencies.auth import get_current_user
from src.api.dependencies.use_cases import get_stream_processing_use_case
from src.api.mappers.stream_mapper import StreamMapper
from src.application.use_cases.stream_processing import (
    StreamProcessingUseCase,
    StreamStartRequest,
    StreamStopRequest,
    StreamStatusRequest
)
from src.application.use_cases.base import ResultStatus
from src.domain.entities.user import User

router = APIRouter(prefix="/streams", tags=["streams"])


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Stream service status",
    description="Check if stream processing service is operational",
)
async def stream_status() -> StatusResponse:
    """Get stream service status.

    Returns:
        StatusResponse: Stream service status
    """
    from datetime import datetime

    return StatusResponse(
        status="Stream processing service operational", timestamp=datetime.utcnow()
    )


@router.post(
    "/",
    response_model=StreamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start stream processing",
    description="Start processing a livestream with AI-powered highlight detection",
)
async def create_stream(
    stream_data: StreamCreate,
    current_user: User = Depends(get_current_user),
    use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> StreamResponse:
    """Start processing a livestream.
    
    Creates a new stream processing job that will analyze the provided
    livestream URL and extract highlights using AI-powered detection.
    
    Args:
        stream_data: Stream configuration including URL and processing options
        current_user: Authenticated user creating the stream
        use_case: Stream processing use case
        
    Returns:
        StreamResponse: Created stream details
        
    Raises:
        HTTPException: If stream creation fails
    """
    # Validate current user has required fields
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # Convert API request to use case request
    request = StreamStartRequest(
        user_id=current_user.id,
        url=str(stream_data.source_url),
        title=f"Stream from {stream_data.platform.value}",
        platform=stream_data.platform.value,
        processing_options={
            "confidence_threshold": stream_data.options.highlight_threshold,
            "min_highlight_duration": float(stream_data.options.min_duration),
            "max_highlight_duration": float(stream_data.options.max_duration),
            "detect_gameplay": True,  # Default
            "detect_reactions": True,  # Default
            "detect_funny_moments": True,  # Default
            "detect_emotional_moments": True  # Default
        }
    )
    
    # Execute use case
    result = await use_case.start_stream(request)
    
    # Handle result
    if result.status == ResultStatus.SUCCESS:
        # Validate result has required fields
        if result.stream_id is None or result.stream_url is None or result.stream_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from stream processing service"
            )
        
        # Convert domain entity to API response (simplified for now)
        from datetime import datetime
        return StreamResponse(
            id=result.stream_id,
            source_url=result.stream_url,
            platform=stream_data.platform,
            status=StreamStatus(result.stream_status),
            options=stream_data.options.model_dump(),
            user_id=current_user.id,
            created_at=current_user.created_at.value,
            updated_at=current_user.updated_at.value,
            is_active=True,
            highlight_count=0
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.errors[0] if result.errors else "User not found"
        )
    elif result.status == ResultStatus.QUOTA_EXCEEDED:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=result.errors[0] if result.errors else "Quota exceeded"
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.errors[0] if result.errors else "Validation error"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to create stream"
        )


@router.get(
    "/",
    response_model=StreamListResponse,
    summary="List streams",
    description="List active and completed streams for the authenticated user",
)
async def list_streams(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by stream status"),
    current_user: User = Depends(get_current_user),
    use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> StreamListResponse:
    """List streams for the authenticated user.
    
    Returns a paginated list of streams owned by the authenticated user,
    with optional filtering by status.
    
    Args:
        page: Page number (1-based)
        per_page: Number of items per page
        status_filter: Optional status filter
        current_user: Authenticated user
        use_case: Stream processing use case
        
    Returns:
        StreamListResponse: Paginated list of streams
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # For now, return empty list as this requires repository list methods
    # TODO: Implement stream listing in use case and repository
    return StreamListResponse(
        page=page,
        per_page=per_page,
        total=0,
        pages=0,
        has_next=False,
        has_prev=False,
        items=[]
    )


@router.get(
    "/{stream_id}",
    response_model=StreamResponse,
    summary="Get stream details",
    description="Get detailed information about a specific stream",
)
async def get_stream(
    stream_id: int,
    current_user: User = Depends(get_current_user),
    use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> StreamResponse:
    """Get stream details.
    
    Retrieves detailed information about a specific stream including
    current status, processing progress, and highlight count.
    
    Args:
        stream_id: Stream ID to retrieve
        current_user: Authenticated user
        use_case: Stream processing use case
        
    Returns:
        StreamResponse: Stream details
        
    Raises:
        HTTPException: If stream not found or access denied
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    # Get stream status
    request = StreamStatusRequest(
        user_id=current_user.id,
        stream_id=stream_id
    )
    
    result = await use_case.get_stream_status(request)
    
    if result.status == ResultStatus.SUCCESS:
        # Validate result has required fields
        if result.stream_id is None or result.stream_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from stream processing service"
            )
        
        # TODO: Get actual stream entity from repository to create proper response
        # For now, create minimal response from status result
        from datetime import datetime
        return StreamResponse(
            id=result.stream_id,
            source_url="",  # Would come from stream entity
            platform=StreamPlatform.CUSTOM,  # Would come from stream entity
            status=StreamStatus(result.stream_status),
            options={},  # Would come from stream entity
            user_id=current_user.id,
            created_at=current_user.created_at.value,
            updated_at=current_user.updated_at.value,
            is_active=result.stream_status in ["pending", "processing"],
            highlight_count=result.highlights_detected or 0
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stream not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to get stream"
        )


@router.delete(
    "/{stream_id}",
    response_model=StatusResponse,
    summary="Stop stream processing",
    description="Stop processing a stream and finalize any extracted highlights",
)
async def stop_stream(
    stream_id: int,
    force: bool = Query(False, description="Force stop even if processing is incomplete"),
    current_user: User = Depends(get_current_user),
    use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> StatusResponse:
    """Stop processing a stream.
    
    Stops the stream processing pipeline and finalizes any highlights
    that have been extracted so far.
    
    Args:
        stream_id: Stream ID to stop
        force: Whether to force stop even if processing is incomplete
        current_user: Authenticated user
        use_case: Stream processing use case
        
    Returns:
        StatusResponse: Operation result
        
    Raises:
        HTTPException: If operation fails
    """
    # Validate user authentication
    if current_user.id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication"
        )
    
    request = StreamStopRequest(
        user_id=current_user.id,
        stream_id=stream_id,
        force=force
    )
    
    result = await use_case.stop_stream(request)
    
    if result.status == ResultStatus.SUCCESS:
        from datetime import datetime
        return StatusResponse(
            status=f"Stream {stream_id} stopped successfully. {result.highlights_count} highlights extracted.",
            timestamp=datetime.utcnow()
        )
    elif result.status == ResultStatus.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stream not found"
        )
    elif result.status == ResultStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    elif result.status == ResultStatus.VALIDATION_ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.errors[0] if result.errors else "Invalid operation"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.errors[0] if result.errors else "Failed to stop stream"
        )


@router.patch(
    "/{stream_id}",
    response_model=StreamResponse,
    summary="Update stream configuration",
    description="Update processing options for a stream (only allowed while pending)",
)
async def update_stream(
    stream_id: int,
    stream_update: StreamUpdate,
    current_user: User = Depends(get_current_user),
    use_case: StreamProcessingUseCase = Depends(get_stream_processing_use_case)
) -> StreamResponse:
    """Update stream configuration.
    
    Updates the processing options for a stream. This is only allowed
    for streams that are still in pending status.
    
    Args:
        stream_id: Stream ID to update
        stream_update: Updated stream configuration
        current_user: Authenticated user
        use_case: Stream processing use case
        
    Returns:
        StreamResponse: Updated stream details
        
    Raises:
        HTTPException: If update fails or stream is not in valid state
    """
    # TODO: Implement stream update in use case
    # For now, return 501 as this requires additional use case methods
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Stream updates will be implemented in a future phase"
    )
