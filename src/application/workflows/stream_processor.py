"""Stream processing workflow - clean Pythonic DDD implementation.

This module orchestrates stream processing for the B2B AI highlighting agent,
coordinating between domain and infrastructure layers.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import logfire

from src.domain.entities.stream import Stream, StreamStatus
from src.domain.entities.highlight import Highlight
from src.domain.value_objects.url import Url
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.exceptions import EntityNotFoundError, UnauthorizedAccessError


@dataclass
class StreamProcessor:
    """Orchestrates stream processing for AI highlight detection.
    
    This is a thin application service that coordinates between
    domain entities and infrastructure services.
    """
    
    stream_repo: Any  # Duck typing - any object with save/get methods
    user_repo: Any
    org_repo: Any
    usage_repo: Any
    highlight_repo: Any
    agent_config_repo: Any
    platform_detector: Any
    task_service: Any
    
    def __post_init__(self):
        self.logger = logfire.get_logger(__name__)
    
    async def process_stream(
        self,
        user_id: int,
        url: str,
        title: str,
        options: Optional[ProcessingOptions] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Stream:
        """Start processing a stream for highlight detection.
        
        Args:
            user_id: User initiating the processing
            url: Stream URL to process
            title: Human-readable title
            options: Processing configuration
            metadata: Additional stream metadata
            
        Returns:
            Stream entity with processing started
        """
        # Verify user exists
        user = await self.user_repo.get(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")
        
        # Get organization if user has one
        org = await self._get_user_org(user_id)
        
        # Detect platform
        platform = self.platform_detector.detect(Url(url))
        
        # Create stream through factory
        stream = Stream.create(
            url=Url(url),
            platform=platform,
            user_id=user_id,
            processing_options=options or ProcessingOptions.default(),
            title=title,
            organization_id=org.id if org else None,
            dimension_set_id=org.default_dimension_set_id if org else None,
            **(metadata or {})
        )
        
        # Persist stream
        stream = await self.stream_repo.save(stream)
        
        # Start processing
        stream.start_processing()
        await self.stream_repo.save(stream)
        
        # Trigger async task
        task_id = await self.task_service.start_stream_task(stream.id)
        
        # Track usage
        await self._track_usage(user_id, stream.id, org.id if org else None)
        
        self.logger.info(
            f"Started processing stream {stream.id}",
            extra={
                "user_id": user_id,
                "platform": platform.value,
                "task_id": task_id,
            }
        )
        
        return stream
    
    async def get_stream_status(
        self,
        stream_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Get current status of stream processing.
        
        Returns simplified status information.
        """
        stream = await self._get_user_stream(stream_id, user_id)
        
        highlights = []
        if stream.status == StreamStatus.COMPLETED:
            highlights = await self.highlight_repo.get_by_stream(stream_id)
        
        return {
            "id": stream.id,
            "status": stream.status.value,
            "progress": self._calculate_progress(stream),
            "highlights_found": len(highlights),
            "duration": stream.duration.total_seconds() if stream.duration else None,
            "error": stream.error_message,
        }
    
    async def stop_processing(
        self,
        stream_id: int,
        user_id: int
    ) -> Stream:
        """Stop stream processing."""
        stream = await self._get_user_stream(stream_id, user_id)
        
        # Cancel tasks
        await self.task_service.cancel_stream_task(stream_id)
        
        # Update stream state
        stream.cancel()
        return await self.stream_repo.save(stream)
    
    async def get_highlights(
        self,
        stream_id: int,
        user_id: int,
        min_confidence: Optional[float] = None
    ) -> List[Highlight]:
        """Get highlights for a completed stream."""
        stream = await self._get_user_stream(stream_id, user_id)
        
        if stream.status != StreamStatus.COMPLETED:
            return []
        
        highlights = stream.highlights
        
        # Filter by confidence if requested
        if min_confidence:
            highlights = [
                h for h in highlights 
                if h.confidence_score.value >= min_confidence
            ]
        
        return sorted(highlights, key=lambda h: h.confidence_score.value, reverse=True)
    
    # Private helpers
    
    async def _get_user_stream(self, stream_id: int, user_id: int) -> Stream:
        """Get stream and verify user access."""
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Check ownership
        if stream.user_id == user_id:
            return stream
        
        # Check organization membership
        user_org = await self._get_user_org(user_id)
        if user_org and stream.organization_id == user_org.id:
            return stream
        
        raise UnauthorizedAccessError("Access denied to this stream")
    
    async def _get_user_org(self, user_id: int):
        """Get user's organization if any."""
        orgs = await self.org_repo.get_by_owner(user_id)
        if orgs:
            return orgs[0]
        
        member_orgs = await self.org_repo.get_by_member(user_id)
        return member_orgs[0] if member_orgs else None
    
    async def _track_usage(self, user_id: int, stream_id: int, org_id: Optional[int]):
        """Track usage for billing."""
        # Simple usage tracking - details handled by domain
        from src.domain.entities.usage_record import UsageRecord
        
        record = UsageRecord.for_stream_processing(
            user_id=user_id,
            stream_id=stream_id,
            duration_minutes=0,  # Updated when complete
            organization_id=org_id,
        )
        await self.usage_repo.save(record)
    
    def _calculate_progress(self, stream: Stream) -> float:
        """Calculate processing progress percentage."""
        if stream.status == StreamStatus.PENDING:
            return 0.0
        elif stream.status == StreamStatus.PROCESSING:
            # Could calculate based on segments processed
            return 50.0
        elif stream.status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            return 100.0
        else:
            return 0.0