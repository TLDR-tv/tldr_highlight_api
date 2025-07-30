"""Stream processing domain service.

This service orchestrates the complete lifecycle of stream processing,
including state management, B2B agent coordination, and usage tracking.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from src.domain.services.base import BaseDomainService
from src.domain.services.b2b_stream_agent import B2BStreamAgent
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.entities.user import User
from src.domain.entities.organization import Organization
from src.domain.entities.usage_record import UsageRecord
from src.domain.entities.highlight_agent_config import HighlightAgentConfig
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.exceptions import (
    BusinessRuleViolation,
    EntityNotFoundError,
    QuotaExceededError,
    InvalidStateTransition
)
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.highlight_agent_config_repository import HighlightAgentConfigRepository


class StreamProcessingService(BaseDomainService):
    """Domain service for stream processing orchestration.
    
    Handles the business logic for starting, monitoring, and completing
    stream processing operations with B2B agent integration while enforcing
    business rules and quotas.
    """
    
    def __init__(
        self,
        stream_repo: StreamRepository,
        user_repo: UserRepository,
        org_repo: OrganizationRepository,
        usage_repo: UsageRecordRepository,
        highlight_repo: HighlightRepository,
        agent_config_repo: HighlightAgentConfigRepository,
        content_analyzer: Optional[Any] = None
    ):
        """Initialize stream processing service.
        
        Args:
            stream_repo: Repository for stream operations
            user_repo: Repository for user operations
            org_repo: Repository for organization operations
            usage_repo: Repository for usage tracking
            highlight_repo: Repository for highlight operations
            agent_config_repo: Repository for agent configurations
            content_analyzer: Optional content analysis service for B2B agents
        """
        super().__init__()
        self.stream_repo = stream_repo
        self.user_repo = user_repo
        self.org_repo = org_repo
        self.usage_repo = usage_repo
        self.highlight_repo = highlight_repo
        self.agent_config_repo = agent_config_repo
        self.content_analyzer = content_analyzer
        self._active_agents: Dict[int, B2BStreamAgent] = {}
    
    async def start_stream_processing(
        self,
        user_id: int,
        url: str,
        title: str,
        processing_options: Optional[ProcessingOptions] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_config_id: Optional[int] = None
    ) -> Stream:
        """Start processing a new stream with B2B agent.
        
        Args:
            user_id: ID of the user starting the stream
            url: Stream URL
            title: Stream title
            processing_options: Optional processing configuration
            metadata: Optional metadata for the stream
            agent_config_id: Optional agent configuration ID
            
        Returns:
            Created stream entity
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            QuotaExceededError: If user has exceeded their stream quota
            BusinessRuleViolation: If business rules are violated
        """
        # Get user and validate
        user = await self.user_repo.get(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")
        
        # Get user's organization for quota checking
        org = await self._get_user_organization(user_id)
        
        # Check concurrent stream limit
        await self._check_concurrent_stream_limit(user_id, org)
        
        # Determine platform from URL
        platform = self._detect_platform(url)
        
        # Create stream entity
        stream = Stream(
            id=None,
            user_id=user_id,
            url=Url(url),
            title=title,
            platform=platform,
            status=StreamStatus.PENDING,
            processing_options=processing_options or ProcessingOptions.default(),
            metadata=metadata or {},
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Save stream
        saved_stream = await self.stream_repo.save(stream)
        
        # Get or create agent configuration
        agent_config = await self._get_or_create_agent_config(
            user_id=user_id,
            organization_id=org.id if org else None,
            agent_config_id=agent_config_id,
            url=url
        )
        
        # Create and start B2B agent
        b2b_agent = B2BStreamAgent(
            stream=saved_stream,
            agent_config=agent_config,
            content_analyzer=self.content_analyzer
        )
        
        # Store active agent
        self._active_agents[saved_stream.id] = b2b_agent
        
        # Start the agent
        await b2b_agent.start()
        
        # Create initial usage record
        usage_record = UsageRecord.for_stream_processing(
            user_id=user_id,
            stream_id=saved_stream.id,
            duration_minutes=0,  # Will be updated as processing continues
            organization_id=org.id if org else None
        )
        await self.usage_repo.save(usage_record)
        
        self.logger.info(f"Started stream processing with B2B agent: {saved_stream.id} for user {user_id}")
        
        return saved_stream
    
    async def update_stream_status(
        self,
        stream_id: int,
        new_status: StreamStatus,
        error_message: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Stream:
        """Update stream processing status.
        
        Args:
            stream_id: Stream ID
            new_status: New status to set
            error_message: Optional error message for failed status
            metadata_updates: Optional metadata updates
            
        Returns:
            Updated stream entity
            
        Raises:
            EntityNotFoundError: If stream doesn't exist
            InvalidStateTransition: If status transition is invalid
        """
        # Get stream
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Validate state transition
        if not self._is_valid_transition(stream.status, new_status):
            raise InvalidStateTransition(
                f"Cannot transition from {stream.status.value} to {new_status.value}"
            )
        
        # Update stream based on new status
        if new_status == StreamStatus.PROCESSING:
            updated_stream = stream.start_processing()
        elif new_status == StreamStatus.COMPLETED:
            updated_stream = stream.complete()
        elif new_status == StreamStatus.FAILED:
            updated_stream = stream.fail(error_message or "Processing failed")
        elif new_status == StreamStatus.CANCELLED:
            updated_stream = stream.cancel()
        else:
            # For other status updates
            updated_stream = Stream(
                id=stream.id,
                user_id=stream.user_id,
                url=stream.url,
                title=stream.title,
                platform=stream.platform,
                status=new_status,
                processing_options=stream.processing_options,
                started_at=stream.started_at,
                completed_at=stream.completed_at,
                duration_seconds=stream.duration_seconds,
                error_message=error_message if new_status == StreamStatus.FAILED else stream.error_message,
                metadata=self._merge_metadata(stream.metadata, metadata_updates),
                highlight_ids=stream.highlight_ids,
                created_at=stream.created_at,
                updated_at=Timestamp.now()
            )
        
        # Save updated stream
        saved_stream = await self.stream_repo.save(updated_stream)
        
        # Stop B2B agent if stream is ending
        if new_status in [StreamStatus.COMPLETED, StreamStatus.FAILED, StreamStatus.CANCELLED]:
            await self._stop_stream_agent(stream_id)
        
        # Update usage record if completed
        if new_status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            await self._finalize_stream_usage(saved_stream)
        
        self.logger.info(f"Updated stream {stream_id} status to {new_status.value}")
        
        return saved_stream
    
    async def get_user_active_streams(self, user_id: int) -> List[Stream]:
        """Get all active streams for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active streams
        """
        return await self.stream_repo.get_by_user(
            user_id,
            status=StreamStatus.PROCESSING
        )
    
    async def get_stream_with_highlights(self, stream_id: int, user_id: int) -> Optional[Stream]:
        """Get stream with highlights, verifying ownership.
        
        Args:
            stream_id: Stream ID
            user_id: User ID for ownership verification
            
        Returns:
            Stream with highlights if found and owned by user
            
        Raises:
            EntityNotFoundError: If stream not found
            UnauthorizedAccessError: If user doesn't own the stream
        """
        stream = await self.stream_repo.get_with_highlights(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        if stream.user_id != user_id:
            # Check if user is in same organization
            user_org = await self._get_user_organization(user_id)
            stream_owner_org = await self._get_user_organization(stream.user_id)
            
            if not (user_org and stream_owner_org and user_org.id == stream_owner_org.id):
                from src.domain.exceptions import UnauthorizedAccessError
                raise UnauthorizedAccessError("You don't have access to this stream")
        
        return stream
    
    async def calculate_stream_cost(self, stream_id: int) -> Dict[str, Any]:
        """Calculate the cost of processing a stream.
        
        Args:
            stream_id: Stream ID
            
        Returns:
            Dictionary with cost breakdown
        """
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")
        
        # Get usage records
        usage_records = await self.usage_repo.get_by_resource(stream_id, "stream")
        
        # Calculate costs
        total_minutes = sum(record.quantity for record in usage_records)
        total_cost = sum(record.total_cost or 0 for record in usage_records)
        
        # Get highlight count
        highlights = await self.highlight_repo.get_by_stream(stream_id)
        
        return {
            "stream_id": stream_id,
            "duration_minutes": total_minutes,
            "total_cost": total_cost,
            "highlight_count": len(highlights),
            "cost_per_highlight": total_cost / len(highlights) if highlights else 0,
            "status": stream.status.value
        }
    
    async def cleanup_old_streams(self, days_old: int = 90) -> int:
        """Clean up old completed/failed streams.
        
        Args:
            days_old: Number of days after which to clean up
            
        Returns:
            Number of streams cleaned up
        """
        cutoff_date = Timestamp.now().subtract_days(days_old)
        return await self.stream_repo.cleanup_old_streams(cutoff_date)
    
    async def process_content_segment(
        self,
        stream_id: int,
        segment_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a content segment using the B2B agent.
        
        Args:
            stream_id: Stream ID
            segment_data: Content segment data
            
        Returns:
            List of highlight candidates
        """
        agent = self._active_agents.get(stream_id)
        if not agent:
            self.logger.warning(f"No active agent found for stream {stream_id}")
            return []
        
        try:
            # Analyze content segment
            candidates = await agent.analyze_content_segment(segment_data)
            
            # Process candidates to create highlights
            created_highlights = []
            for candidate in candidates:
                if await agent.should_create_highlight(candidate):
                    highlight = await agent.create_highlight(candidate)
                    if highlight:
                        # Save highlight to repository
                        saved_highlight = await self.highlight_repo.save(highlight)
                        created_highlights.append({
                            "id": saved_highlight.id,
                            "start_time": candidate.start_time,
                            "end_time": candidate.end_time,
                            "score": candidate.final_score,
                            "description": candidate.description
                        })
            
            return created_highlights
            
        except Exception as e:
            self.logger.error(f"Error processing segment for stream {stream_id}: {str(e)}")
            return []
    
    async def get_agent_metrics(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a stream's B2B agent.
        
        Args:
            stream_id: Stream ID
            
        Returns:
            Agent metrics dictionary or None if agent not found
        """
        agent = self._active_agents.get(stream_id)
        if not agent:
            return None
        
        return agent.get_performance_metrics()
    
    async def stop_stream_processing(
        self,
        stream_id: int,
        force: bool = False
    ) -> Stream:
        """Stop stream processing and clean up agent.
        
        Args:
            stream_id: Stream ID
            force: Whether to force stop
            
        Returns:
            Updated stream entity
        """
        # Update stream status
        updated_stream = await self.update_stream_status(
            stream_id=stream_id,
            new_status=StreamStatus.COMPLETED if not force else StreamStatus.CANCELLED
        )
        
        return updated_stream
    
    # Private helper methods
    
    async def _get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get the organization for a user."""
        orgs = await self.org_repo.get_by_owner(user_id)
        return orgs[0] if orgs else None
    
    async def _check_concurrent_stream_limit(self, user_id: int, org: Optional[Organization]):
        """Check if user can start another concurrent stream."""
        active_streams = await self.stream_repo.get_active_streams()
        user_active_count = sum(1 for s in active_streams if s.user_id == user_id)
        
        # Get limit from organization plan or use default
        limit = 1  # Default for free users
        if org:
            limit = org.plan_limits.concurrent_streams
        
        if user_active_count >= limit:
            raise QuotaExceededError(
                f"Concurrent stream limit ({limit}) exceeded. "
                f"Please wait for current streams to complete."
            )
    
    def _detect_platform(self, url: str) -> StreamPlatform:
        """Detect streaming platform from URL."""
        url_lower = url.lower()
        
        if "twitch.tv" in url_lower:
            return StreamPlatform.TWITCH
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return StreamPlatform.YOUTUBE
        elif "rtmp://" in url_lower:
            return StreamPlatform.RTMP
        else:
            return StreamPlatform.OTHER
    
    def _is_valid_transition(self, from_status: StreamStatus, to_status: StreamStatus) -> bool:
        """Check if status transition is valid."""
        valid_transitions = {
            StreamStatus.PENDING: [StreamStatus.PROCESSING, StreamStatus.FAILED, StreamStatus.CANCELLED],
            StreamStatus.PROCESSING: [StreamStatus.COMPLETED, StreamStatus.FAILED, StreamStatus.CANCELLED],
            StreamStatus.COMPLETED: [],  # Terminal state
            StreamStatus.FAILED: [],     # Terminal state  
            StreamStatus.CANCELLED: [],  # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    def _merge_metadata(self, existing: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata updates with existing metadata."""
        if not updates:
            return existing
        
        merged = existing.copy()
        merged.update(updates)
        return merged
    
    async def _finalize_stream_usage(self, stream: Stream):
        """Finalize usage records when stream completes."""
        if not stream.duration_seconds:
            return
        
        # Get usage records for this stream
        usage_records = await self.usage_repo.get_by_resource(stream.id, "stream")
        
        # Update the main usage record with final duration
        for record in usage_records:
            if not record.is_complete:
                duration_minutes = stream.duration_seconds / 60.0
                completed_record = record.complete(quantity=duration_minutes)
                await self.usage_repo.save(completed_record)
    
    async def _get_or_create_agent_config(
        self,
        user_id: int,
        organization_id: Optional[int],
        agent_config_id: Optional[int],
        url: str
    ) -> HighlightAgentConfig:
        """Get or create an agent configuration for stream processing."""
        # If specific config requested, try to get it
        if agent_config_id:
            config = await self.agent_config_repo.get(agent_config_id)
            if config and (config.user_id == user_id or config.organization_id == organization_id):
                return config
        
        # Try to get organization's default config
        if organization_id:
            org_configs = await self.agent_config_repo.get_active_for_organization(organization_id)
            if org_configs:
                return org_configs[0]  # Use first active config
        
        # Detect content type from URL and create appropriate default
        content_type = self._detect_content_type(url)
        
        if content_type == "valorant":
            return HighlightAgentConfig.create_valorant_config(
                organization_id=organization_id or 0,
                user_id=user_id
            )
        else:
            return HighlightAgentConfig.create_default_gaming_config(
                organization_id=organization_id or 0,
                user_id=user_id
            )
    
    async def _stop_stream_agent(self, stream_id: int) -> None:
        """Stop and clean up a B2B stream agent."""
        agent = self._active_agents.pop(stream_id, None)
        if agent:
            try:
                await agent.stop()
                self.logger.info(f"Stopped B2B agent for stream {stream_id}")
            except Exception as e:
                self.logger.error(f"Error stopping agent for stream {stream_id}: {str(e)}")
    
    def _detect_content_type(self, url: str) -> str:
        """Detect content type from stream URL."""
        url_lower = url.lower()
        
        # Gaming platform detection
        if "twitch.tv" in url_lower:
            # Could parse Twitch categories or game names from URL/metadata
            return "gaming"
        elif "youtube.com" in url_lower:
            # Could analyze YouTube title/description for game detection
            return "gaming"
        
        # Default to general gaming
        return "gaming"