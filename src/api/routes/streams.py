"""Stream API routes - clean Pythonic implementation.

RESTful endpoints for stream processing in the B2B AI highlighting platform.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.application.workflows import StreamProcessor
from src.domain.value_objects.processing_options import ProcessingOptions
from ..dependencies import get_current_user, get_stream_processor


router = APIRouter(prefix="/streams", tags=["streams"])


# Request/Response models (simple Pydantic schemas)

class StreamCreateRequest(BaseModel):
    """Request to create and process a stream."""
    url: str
    title: str
    segment_duration: Optional[int] = 30
    min_confidence: Optional[float] = 0.7
    metadata: Optional[dict] = None


class StreamResponse(BaseModel):
    """Stream information response."""
    id: int
    url: str
    title: str
    status: str
    platform: str
    created_at: str
    highlights_count: int = 0


class StreamStatusResponse(BaseModel):
    """Stream processing status."""
    id: int
    status: str
    progress: float
    highlights_found: int
    duration: Optional[float] = None
    error: Optional[str] = None


# API Endpoints

@router.post("/", response_model=StreamResponse, status_code=status.HTTP_201_CREATED)
async def create_stream(
    request: StreamCreateRequest,
    user=Depends(get_current_user),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """Start processing a new stream."""
    try:
        # Create processing options
        options = ProcessingOptions(
            segment_duration=request.segment_duration,
            min_confidence_threshold=request.min_confidence,
        )
        
        # Process stream
        stream = await processor.process_stream(
            user_id=user.id,
            url=request.url,
            title=request.title,
            options=options,
            metadata=request.metadata,
        )
        
        return StreamResponse(
            id=stream.id,
            url=stream.url.value,
            title=stream.title,
            status=stream.status.value,
            platform=stream.platform.value,
            created_at=stream.created_at.value.isoformat(),
            highlights_count=0,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: int,
    user=Depends(get_current_user),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """Get stream information."""
    try:
        stream = await processor._get_user_stream(stream_id, user.id)
        
        return StreamResponse(
            id=stream.id,
            url=stream.url.value,
            title=stream.title,
            status=stream.status.value,
            platform=stream.platform.value,
            created_at=stream.created_at.value.isoformat(),
            highlights_count=stream.highlight_count,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/{stream_id}/status", response_model=StreamStatusResponse)
async def get_stream_status(
    stream_id: int,
    user=Depends(get_current_user),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """Get current stream processing status."""
    try:
        status_info = await processor.get_stream_status(stream_id, user.id)
        return StreamStatusResponse(**status_info)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/{stream_id}/stop", response_model=StreamResponse)
async def stop_stream(
    stream_id: int,
    user=Depends(get_current_user),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """Stop stream processing."""
    try:
        stream = await processor.stop_processing(stream_id, user.id)
        
        return StreamResponse(
            id=stream.id,
            url=stream.url.value,
            title=stream.title,
            status=stream.status.value,
            platform=stream.platform.value,
            created_at=stream.created_at.value.isoformat(),
            highlights_count=stream.highlight_count,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/{stream_id}/highlights")
async def get_highlights(
    stream_id: int,
    min_confidence: Optional[float] = None,
    user=Depends(get_current_user),
    processor: StreamProcessor = Depends(get_stream_processor)
):
    """Get highlights for a stream."""
    try:
        highlights = await processor.get_highlights(
            stream_id, user.id, min_confidence
        )
        
        return [
            {
                "id": h.id,
                "start_time": h.start_time.seconds,
                "end_time": h.end_time.seconds,
                "confidence": h.confidence_score.value,
                "title": h.title,
                "description": h.description,
                "types": h.highlight_types,
            }
            for h in highlights
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )