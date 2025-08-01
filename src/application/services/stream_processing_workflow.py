"""Stream processing workflow application service.

This application service orchestrates the complete lifecycle of stream processing,
coordinating between domain entities, repositories, and infrastructure services.
"""

from typing import Optional, List, Dict, Any
import logfire

from src.domain.entities.stream import Stream, StreamStatus
from src.domain.entities.organization import Organization
from src.domain.entities.usage_record import UsageRecord
from src.domain.entities.highlight_agent_config import HighlightAgentConfig
from src.domain.value_objects.url import Url
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.value_objects.duration import Duration
from src.domain.exceptions import (
    EntityNotFoundError,
    InvalidStateTransition,
    BusinessRuleViolation,
)
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.highlight_agent_config_repository import (
    HighlightAgentConfigRepository,
)
from src.domain.services.platform_detection_service import PlatformDetectionService
from src.domain.services.stream_task_service import StreamTaskService


class StreamProcessingWorkflow:
    """Application service for stream processing orchestration.

    Handles the business workflow for starting, monitoring, and completing
    stream processing operations while coordinating between domain and infrastructure.
    """

    def __init__(
        self,
        stream_repo: StreamRepository,
        user_repo: UserRepository,
        org_repo: OrganizationRepository,
        usage_repo: UsageRecordRepository,
        highlight_repo: HighlightRepository,
        agent_config_repo: HighlightAgentConfigRepository,
        platform_service: PlatformDetectionService,
        task_service: StreamTaskService,
    ):
        """Initialize stream processing workflow.

        Args:
            stream_repo: Repository for stream operations
            user_repo: Repository for user operations
            org_repo: Repository for organization operations
            usage_repo: Repository for usage tracking
            highlight_repo: Repository for highlight operations
            agent_config_repo: Repository for agent configurations
        """
        self.stream_repo = stream_repo
        self.user_repo = user_repo
        self.org_repo = org_repo
        self.usage_repo = usage_repo
        self.highlight_repo = highlight_repo
        self.agent_config_repo = agent_config_repo
        self.platform_service = platform_service
        self.task_service = task_service
        self.logger = logfire.get_logger(__name__)

    async def start_stream_processing(
        self,
        user_id: int,
        url: str,
        title: str,
        processing_options: Optional[ProcessingOptions] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_config_id: Optional[int] = None,
    ) -> Stream:
        """Start processing a new stream.

        This orchestrates the complete workflow of:
        1. Validating user and organization
        2. Creating stream entity
        3. Setting up agent configuration
        4. Triggering async processing
        5. Creating usage records

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
            BusinessRuleViolation: If business rules are violated
        """
        # Get user and validate
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")

        # Get user's organization
        org = await self._get_user_organization(user_id)

        # Note: No quota checking for first client - unlimited processing
        if org:
            self.logger.info(
                "Stream processing started",
                user_id=user_id,
                organization_id=org.id,
                note="No limits enforced - unlimited for first client",
            )

        # Determine platform from URL using domain service
        platform = self.platform_service.detect_platform(Url(url))

        # Create stream using factory method
        stream = Stream.create(
            url=Url(url),
            platform=platform,
            user_id=user_id,
            processing_options=processing_options or ProcessingOptions.default(),
            title=title,
            channel_name=metadata.get("channel_name") if metadata else None,
            game_category=metadata.get("game_category") if metadata else None,
            language=metadata.get("language") if metadata else None,
            viewer_count=metadata.get("viewer_count") if metadata else None,
            platform_data=metadata or {},
        )

        # Save stream
        saved_stream = await self.stream_repo.save(stream)

        # Get or create agent configuration
        agent_config = await self._get_or_create_agent_config(
            user_id=user_id,
            organization_id=org.id if org else None,
            agent_config_id=agent_config_id,
            url=url,
        )

        # Trigger async processing via task service
        task_result = await self.task_service.start_stream_ingestion(
            stream_id=saved_stream.id,
            chunk_duration=30,  # 30 second chunks
            agent_config_id=agent_config.id if hasattr(agent_config, "id") else None,
        )

        # Update stream with task ID
        saved_stream.metadata["ingestion_task_id"] = task_result.task_id
        await self.stream_repo.save(saved_stream)

        # Create initial usage record
        usage_record = UsageRecord.for_stream_processing(
            user_id=user_id,
            stream_id=saved_stream.id,
            duration_minutes=0,  # Will be updated as processing continues
            organization_id=org.id if org else None,
        )
        await self.usage_repo.save(usage_record)

        self.logger.info(
            "Stream processing started",
            stream_id=saved_stream.id,
            user_id=user_id,
            platform=saved_stream.platform.value,
            task_id=task_result.task_id,
        )

        return saved_stream

    async def update_stream_status(
        self,
        stream_id: int,
        new_status: StreamStatus,
        error_message: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
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

        # Update stream using aggregate methods
        try:
            if new_status == StreamStatus.PROCESSING:
                stream.start_processing()
            elif new_status == StreamStatus.COMPLETED:
                # Calculate duration if needed
                duration = None
                if metadata_updates and "duration_seconds" in metadata_updates:
                    duration = Duration(metadata_updates["duration_seconds"])
                stream.complete_processing(duration)
            elif new_status == StreamStatus.FAILED:
                stream.fail_processing(error_message or "Processing failed")
            elif new_status == StreamStatus.CANCELLED:
                stream.cancel()
            else:
                raise InvalidStateTransition(
                    f"Unknown status transition to {new_status.value}"
                )
        except (InvalidStateTransition, BusinessRuleViolation) as e:
            self.logger.error(
                "Failed to update stream status",
                stream_id=stream_id,
                current_status=stream.status.value,
                new_status=new_status.value,
                error=str(e),
            )
            raise

        # Apply metadata updates if any
        if metadata_updates:
            stream.metadata.update(metadata_updates)

        # Save updated stream
        saved_stream = await self.stream_repo.save(stream)

        # Cancel Celery tasks if stream is being cancelled
        if new_status == StreamStatus.CANCELLED:
            await self._cancel_stream_tasks(stream_id)

        # Finalize usage if completed or failed
        if new_status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            await self._finalize_stream_usage(saved_stream)

        self.logger.info(
            "Stream status updated",
            stream_id=stream_id,
            old_status=stream.status.value,
            new_status=new_status.value,
        )

        return saved_stream

    async def get_user_active_streams(self, user_id: int) -> List[Stream]:
        """Get all active streams for a user."""
        return await self.stream_repo.get_by_user(
            user_id, status=StreamStatus.PROCESSING
        )

    async def get_stream_with_highlights(
        self, stream_id: int, user_id: int
    ) -> Optional[Stream]:
        """Get stream with highlights, verifying ownership."""
        stream = await self.stream_repo.get_with_highlights(stream_id)
        if not stream:
            raise EntityNotFoundError(f"Stream {stream_id} not found")

        # Verify ownership
        if stream.user_id != user_id:
            # Check if user is in same organization
            user_org = await self._get_user_organization(user_id)
            stream_owner_org = await self._get_user_organization(stream.user_id)

            if not (
                user_org and stream_owner_org and user_org.id == stream_owner_org.id
            ):
                from src.domain.exceptions import UnauthorizedAccessError

                raise UnauthorizedAccessError("You don't have access to this stream")

        return stream

    async def calculate_stream_cost(self, stream_id: int) -> Dict[str, Any]:
        """Calculate the cost of processing a stream."""
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
            "status": stream.status.value,
        }

    async def stop_stream_processing(
        self, stream_id: int, force: bool = False
    ) -> Stream:
        """Stop stream processing and cancel tasks."""
        # Cancel any running tasks first
        await self._cancel_stream_tasks(stream_id)

        # Update stream status
        return await self.update_stream_status(
            stream_id=stream_id,
            new_status=StreamStatus.COMPLETED if not force else StreamStatus.CANCELLED,
        )

    # Private helper methods

    async def _get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get the organization for a user."""
        orgs = await self.org_repo.get_by_owner(user_id)
        if orgs:
            return orgs[0]

        # Check if user is a member of any organization
        member_orgs = await self.org_repo.get_by_member(user_id)
        if member_orgs:
            return member_orgs[0]

        return None

    async def _finalize_stream_usage(self, stream: Stream):
        """Finalize usage records when stream completes."""
        if not stream.duration:
            return

        # Get usage records for this stream
        usage_records = await self.usage_repo.get_by_resource(stream.id, "stream")

        # Update the main usage record with final duration
        for record in usage_records:
            if not record.is_complete:
                duration_minutes = stream.duration.total_seconds() / 60.0
                completed_record = record.complete(quantity=duration_minutes)
                await self.usage_repo.save(completed_record)

    async def _cancel_stream_tasks(self, stream_id: int) -> None:
        """Cancel running background tasks for a stream."""
        try:
            success = await self.task_service.cancel_stream_tasks(stream_id)
            if success:
                self.logger.info(f"Cancelled tasks for stream {stream_id}")
            else:
                self.logger.warning(f"Failed to cancel tasks for stream {stream_id}")
        except Exception as e:
            self.logger.error(f"Error cancelling tasks for stream {stream_id}: {e}")

    async def _get_or_create_agent_config(
        self,
        user_id: int,
        organization_id: Optional[int],
        agent_config_id: Optional[int],
        url: str,
    ) -> HighlightAgentConfig:
        """Get or create an agent configuration for stream processing."""
        # If specific config requested, try to get it
        if agent_config_id:
            config = await self.agent_config_repo.get(agent_config_id)
            if config and (
                config.user_id == user_id or config.organization_id == organization_id
            ):
                return config

        # Try to get organization's default config
        if organization_id:
            org_configs = await self.agent_config_repo.get_active_for_organization(
                organization_id
            )
            if org_configs:
                return org_configs[0]  # Use first active config

        # Detect content type from URL and create appropriate default
        content_type = self._detect_content_type(url)

        if content_type == "valorant":
            return HighlightAgentConfig.create_valorant_config(
                organization_id=organization_id or 0, user_id=user_id
            )
        else:
            return HighlightAgentConfig.create_default_gaming_config(
                organization_id=organization_id or 0, user_id=user_id
            )

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from stream URL."""
        url_lower = url.lower()

        # Gaming platform detection
        if "twitch.tv" in url_lower:
            return "gaming"
        elif "youtube.com" in url_lower:
            return "gaming"

        # Default to general gaming
        return "gaming"
