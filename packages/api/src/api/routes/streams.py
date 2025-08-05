"""Stream management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from uuid import UUID
from typing import Optional
import boto3
from pathlib import Path
import aiofiles
import tempfile
import os
import subprocess
import asyncio
import shutil
import logging

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
logger = logging.getLogger(__name__)


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
    organization_id: str = Form(...),
    name: Optional[str] = Form(None),
    stream_repository: StreamRepository = Depends(get_stream_repository),
    organization_repository = Depends(get_organization_repository),
    settings: Settings = Depends(get_settings_dep),
):
    """Upload a video file and create HLS stream for processing.
    
    Development mode only - creates M3U8 playlist and segments.
    
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
        HTTPException: 400 if file is invalid, 403 if not in development mode.

    """
    # Only allow in development mode
    if settings.environment != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="File upload is only available in development mode"
        )
    
    # Validate organization_id is provided
    if not organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id is required"
        )
    
    # Get organization
    from uuid import UUID
    try:
        org_uuid = UUID(organization_id)
        organization = await organization_repository.get(org_uuid)
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid organization_id format"
        )
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
    
    # Generate unique identifiers
    import uuid
    stream_id = str(uuid.uuid4())
    
    # S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=settings.aws_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / f"input{file_extension}"
        
        # Save uploaded file to temp location
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Create HLS output directory
        hls_dir = temp_path / "hls"
        hls_dir.mkdir(exist_ok=True)
        
        # Log disk space and file info for debugging
        import shutil as shutil_disk
        disk_usage = shutil_disk.disk_usage(temp_path)
        logger.info(f"Temp directory: {temp_path}")
        logger.info(f"Available disk space: {disk_usage.free / (1024**3):.2f} GB")
        logger.info(f"Input file size: {file_size / (1024**2):.2f} MB")
        
        # FFmpeg command to create HLS segments and playlist
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(input_file),
            "-c:v", "copy",  # Copy video codec (no re-encoding)
            "-c:a", "copy",  # Copy audio codec
            "-f", "hls",
            "-hls_time", "10",  # 10 second segments
            "-hls_list_size", "0",  # Include all segments in playlist
            "-hls_segment_type", "mpegts",  # Use TS segments
            "-hls_segment_filename", str(hls_dir / "segment_%03d.ts"),
            "-hls_playlist_type", "vod",  # VOD playlist
            "-hls_flags", "independent_segments",
            str(hls_dir / "playlist.m3u8")
        ]
        
        try:
            # Log FFmpeg command for debugging
            import shlex
            logger.info(f"Running FFmpeg command: {shlex.join(ffmpeg_cmd)}")
            
            # Run FFmpeg to create HLS segments with timeout
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for FFmpeg with timeout (20 minutes for processing large videos)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=1200.0  # 20 minutes timeout for 35+ minute videos
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                raise Exception("FFmpeg timed out after 20 minutes")
            
            # Log FFmpeg output for debugging
            if stderr:
                stderr_text = stderr.decode()
                logger.info(f"FFmpeg stderr: {stderr_text}")
                
            if process.returncode != 0:
                raise Exception(f"FFmpeg failed with code {process.returncode}: {stderr.decode()}")
            
            # Log what files were generated
            all_files = list(hls_dir.iterdir())
            segment_files = list(hls_dir.glob("*.ts"))
            logger.info(f"FFmpeg completed. Generated {len(all_files)} total files, {len(segment_files)} TS segments")
            logger.info(f"Generated files: {[f.name for f in all_files]}")
            
            # Check playlist content
            playlist_path = hls_dir / "playlist.m3u8"
            if playlist_path.exists():
                with open(playlist_path, 'r') as f:
                    playlist_content = f.read()
                    segment_count = playlist_content.count('.ts')
                    logger.info(f"Playlist contains {segment_count} segment references")
                    logger.info(f"Playlist content preview:\n{playlist_content[:500]}")
            
            # Upload all files to S3 with public access
            s3_prefix = f"streams/{organization.id}/{stream_id}"
            
            # Upload playlist
            playlist_key = f"{s3_prefix}/playlist.m3u8"
            with open(hls_dir / "playlist.m3u8", "rb") as f:
                s3_client.upload_fileobj(
                    f,
                    settings.s3_bucket_name,
                    playlist_key,
                    ExtraArgs={
                        'ContentType': 'application/vnd.apple.mpegurl',
                        'CacheControl': 'no-cache',  # Always fetch latest playlist
                    }
                )
            
            # Upload segments
            segment_files = list(hls_dir.glob("*.ts"))
            logger.info(f"Starting upload of {len(segment_files)} segments to S3")
            
            for i, segment_file in enumerate(segment_files):
                segment_key = f"{s3_prefix}/{segment_file.name}"
                file_size = segment_file.stat().st_size
                logger.info(f"Uploading segment {i+1}/{len(segment_files)}: {segment_file.name} ({file_size} bytes)")
                
                with open(segment_file, "rb") as f:
                    s3_client.upload_fileobj(
                        f,
                        settings.s3_bucket_name,
                        segment_key,
                        ExtraArgs={
                            'ContentType': 'video/MP2T',
                            'CacheControl': 'max-age=3600',  # Cache segments
                        }
                    )
            
            logger.info(f"Successfully uploaded {len(segment_files)} segments")
            
            # Generate HTTP URL for playlist
            # Use internal MinIO URL for worker processing
            if settings.environment == "development":
                # Workers should use internal Docker network URL
                http_url = f"http://minio:9000/{settings.s3_bucket_name}/{playlist_key}"
            else:
                # In production, use the configured endpoint
                http_url = f"{settings.aws_endpoint_url}/{settings.s3_bucket_name}/{playlist_key}"
            
            # Create stream record with HTTP URL
            stream = Stream(
                organization_id=organization.id,
                url=http_url,  # Use HTTP URL instead of S3 URL
                name=name or file.filename,
                type="vod",  # Video on demand type
                status=StreamStatus.PENDING,
                metadata={
                    "file_size": file_size,
                    "content_type": "application/vnd.apple.mpegurl",
                    "original_filename": file.filename.encode('ascii', 'ignore').decode('ascii'),
                    "s3_prefix": s3_prefix,
                    "playlist_key": playlist_key,
                    "segment_count": len(segment_files),
                    "segment_duration": 10,  # seconds
                },
            )
            
            await stream_repository.create(stream)
            
            # Automatically queue processing task after successful upload
            task = celery_app.send_task(
                "process_stream",
                args=[str(stream.id)],
                kwargs={"processing_options": {}},
            )
            
            # Update stream with task ID and set to QUEUED status
            stream.celery_task_id = task.id
            stream.status = StreamStatus.QUEUED
            await stream_repository.update(stream)
            
            logger.info(f"Automatically queued processing for uploaded stream {stream.id} with task {task.id}")
            
            return StreamResponse.model_validate(stream)
            
        except Exception as e:
            # Clean up S3 files on error
            try:
                # Delete all uploaded files
                objects = s3_client.list_objects_v2(
                    Bucket=settings.s3_bucket_name,
                    Prefix=s3_prefix
                )
                if 'Contents' in objects:
                    delete_objects = {'Objects': [{'Key': obj['Key']} for obj in objects['Contents']]}
                    s3_client.delete_objects(Bucket=settings.s3_bucket_name, Delete=delete_objects)
            except:
                pass
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file: {str(e)}"
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
