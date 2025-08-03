"""Stream management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from uuid import UUID
from typing import Optional

from ..dependencies import (
    get_current_organization,
    require_scope,
    get_stream_repository,
    get_organization_repository,
    get_settings_dep,
    get_rate_limiter,
)
from ..middleware.rate_limit import create_endpoint_limiter
from ..schemas.streams import (
    StreamCreateRequest,
    StreamProcessRequest,
    StreamResponse,
    StreamListResponse,
    StreamProcessResponse,
    StreamTaskStatusResponse,
)
from shared.domain.models.api_key import APIScopes
from shared.domain.models.stream import Stream, StreamStatus
from shared.domain.models.organization import Organization
from shared.infrastructure.storage.repositories import StreamRepository
from shared.infrastructure.config.config import Settings
from ..celery_client import celery_app

router = APIRouter()



@router.post("/", response_model=StreamResponse)
async def create_stream(
    request: StreamCreateRequest,
    organization: Organization = Depends(get_current_organization),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    _=Depends(require_scope(APIScopes.STREAMS_WRITE)),
    _rate_limit: None = create_endpoint_limiter("20/minute"),
):
    """Create a new stream for processing.
    
    Args:
        request: Stream creation parameters including URL, name, and type.
        organization: Current organization from API key.
        stream_repository: Stream data repository.
        _: API scope validation dependency.
        
    Returns:
        StreamResponse with the created stream details.

    """
    # Create stream record
    stream = Stream(
        organization_id=organization.id,
        url=request.url,
        name=request.name or f"Stream {request.url[:50]}",
        type=request.type,
        status=StreamStatus.PENDING,
        metadata=request.metadata or {},
    )
    
    await stream_repository.create(stream)
    
    return StreamResponse.model_validate(stream)


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: UUID,
    organization: Organization = Depends(get_current_organization),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    _=Depends(require_scope(APIScopes.STREAMS_READ)),
):
    """Get stream details by ID.
    
    Args:
        stream_id: UUID of the stream to retrieve.
        organization: Current organization from API key.
        stream_repository: Stream data repository.
        _: API scope validation dependency.
        
    Returns:
        StreamResponse with stream details.
        
    Raises:
        HTTPException: 404 if stream not found or access denied.

    """
    stream = await stream_repository.get(stream_id)
    
    if not stream or stream.organization_id != organization.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stream not found",
        )
    
    return StreamResponse.model_validate(stream)


@router.post("/{stream_id}/process", response_model=StreamProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_stream(
    stream_id: UUID,
    request: StreamProcessRequest,
    organization: Organization = Depends(get_current_organization),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    _=Depends(require_scope(APIScopes.STREAMS_WRITE)),
    _rate_limit: None = create_endpoint_limiter("10/minute"),
):
    """Start processing a stream for highlight detection.
    
    Queues the stream for asynchronous processing using Celery.
    The stream status is updated to QUEUED and a task ID is returned.
    
    Args:
        stream_id: UUID of the stream to process.
        request: Processing configuration parameters.
        organization: Current organization from API key.
        stream_repository: Stream data repository.
        _: API scope validation dependency.
        
    Returns:
        StreamProcessResponse with task ID and status.
        
    Raises:
        HTTPException: 404 if stream not found, 409 if already processing.

    """
    stream = await stream_repository.get(stream_id)
    
    if not stream or stream.organization_id != organization.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stream not found",
        )
    
    if stream.status == StreamStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stream is already being processed",
        )
    
    # Queue processing task using send_task
    task = celery_app.send_task(
        "process_stream",
        args=[str(stream_id)],
        kwargs={"processing_options": {}},  # Empty for now, will add incrementally
    )
    
    # Update stream with task ID
    stream.celery_task_id = task.id
    stream.status = StreamStatus.QUEUED
    await stream_repository.update(stream)
    
    return StreamProcessResponse(
        stream_id=stream_id,
        task_id=task.id,
        status="queued",
        message="Stream processing has been queued",
    )
