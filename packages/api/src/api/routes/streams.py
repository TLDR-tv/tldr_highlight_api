"""Stream management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File
from uuid import UUID
from typing import Optional
import boto3
from pathlib import Path
import aiofiles
import tempfile
import os

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


@router.get("/", response_model=StreamListResponse)
async def list_streams(
    page: int = 1,
    page_size: int = 20,
    organization: Organization = Depends(get_current_organization),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    _=Depends(require_scope(APIScopes.STREAMS_READ)),
):
    """List all streams for the organization.
    
    Args:
        page: Page number (default: 1).
        page_size: Number of items per page (default: 20, max: 100).
        organization: Current organization from API key.
        stream_repository: Stream data repository.
        _: API scope validation dependency.
        
    Returns:
        StreamListResponse with paginated stream results.
    """
    # Validate pagination
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Get streams for organization
    streams = await stream_repository.list_by_organization(
        org_id=organization.id,
        limit=page_size,
        offset=offset,
    )
    
    # Get total count
    total = await stream_repository.count_by_organization(organization.id)
    
    # Convert to response models
    stream_responses = [StreamResponse.model_validate(stream) for stream in streams]
    
    return StreamListResponse(
        streams=stream_responses,
        total=total,
        page=page,
        per_page=page_size,
    )


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


@router.post("/upload", response_model=StreamResponse)
async def upload_stream_file(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    organization: Organization = Depends(get_current_organization),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    settings: Settings = Depends(get_settings_dep),
    _=Depends(require_scope(APIScopes.STREAMS_WRITE)),
    _rate_limit: None = create_endpoint_limiter("10/minute"),
):
    """Upload a video file and create a stream for processing.
    
    Args:
        file: Video file to upload (MP4, MOV, AVI, etc.).
        name: Optional name for the stream.
        organization: Current organization from API key.
        stream_repository: Stream data repository.
        settings: Application settings.
        _: API scope validation dependency.
        
    Returns:
        StreamResponse with the created stream details.
        
    Raises:
        HTTPException: 400 if file is invalid.

    """
    # Validate file type
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpg', '.mpeg'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Get file size for metadata
    file.file.seek(0, 2)  # Move to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    # Generate S3 key
    import uuid
    file_id = str(uuid.uuid4())
    s3_key = f"uploads/{organization.id}/{file_id}{file_extension}"
    
    # Upload to S3/MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=settings.aws_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    
    try:
        # Upload file in chunks to avoid memory issues
        s3_client.upload_fileobj(
            file.file,
            settings.s3_bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': file.content_type or 'video/mp4',
                'Metadata': {
                    'organization_id': str(organization.id),
                    'original_filename': file.filename.encode('ascii', 'ignore').decode('ascii'),
                }
            }
        )
        
        # Generate S3 URL
        s3_url = f"s3://{settings.s3_bucket_name}/{s3_key}"
        
        # Create stream record
        stream = Stream(
            organization_id=organization.id,
            url=s3_url,
            name=name or file.filename,
            type="file",
            status=StreamStatus.PENDING,
            metadata={
                "file_size": file_size,
                "content_type": file.content_type,
                "original_filename": file.filename.encode('ascii', 'ignore').decode('ascii'),
                "s3_key": s3_key,
            },
        )
        
        await stream_repository.create(stream)
        
        return StreamResponse.model_validate(stream)
        
    except Exception as e:
        # Clean up S3 file if stream creation fails
        try:
            s3_client.delete_object(Bucket=settings.s3_bucket_name, Key=s3_key)
        except:
            pass
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


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
