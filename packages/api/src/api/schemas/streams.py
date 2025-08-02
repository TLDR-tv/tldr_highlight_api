"""Stream-related request/response schemas."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict

from shared.domain.models.stream import StreamStatus, StreamType


class StreamCreateRequest(BaseModel):
    """Request to create a new stream for processing."""
    
    url: str = Field(..., description="Stream URL (RTMP, HLS, file path, etc.)")
    name: Optional[str] = Field(None, description="Human-readable stream name")
    type: Optional[StreamType] = Field(StreamType.LIVESTREAM, description="Type of stream")
    metadata: Optional[Dict] = Field(
        default_factory=dict,
        description="Additional metadata for the stream",
    )


class StreamProcessRequest(BaseModel):
    """Request to start processing a stream."""
    
    # We'll add processing options incrementally as we build
    pass


class StreamResponse(BaseModel):
    """Stream response model."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    organization_id: UUID
    url: str
    name: Optional[str]
    type: StreamType
    status: StreamStatus
    celery_task_id: Optional[str]
    metadata: Dict
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    stats: Optional[Dict]


class StreamListResponse(BaseModel):
    """List of streams response."""
    
    streams: List[StreamResponse]
    total: int
    page: int = 1
    per_page: int = 20


class StreamProcessResponse(BaseModel):
    """Response after starting stream processing."""
    
    stream_id: UUID
    task_id: str
    status: str = "queued"
    message: str = "Stream processing has been queued"


class StreamTaskStatusResponse(BaseModel):
    """Stream processing task status."""
    
    task_id: str
    stream_id: UUID
    status: str
    progress: Optional[Dict]
    result: Optional[Dict]
    error: Optional[str]