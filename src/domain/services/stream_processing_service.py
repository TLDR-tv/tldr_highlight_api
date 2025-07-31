"""Stream processing domain service.

This service orchestrates the complete lifecycle of stream processing,
including state management, B2B agent coordination, and usage tracking.
"""

from typing import Optional, List, Dict, Any

from src.domain.services.base import BaseDomainService
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.entities.organization import Organization
from src.domain.entities.usage_record import UsageRecord
from src.domain.entities.highlight_agent_config import HighlightAgentConfig
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.exceptions import (
    EntityNotFoundError,
    InvalidStateTransition,
)
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.highlight_agent_config_repository import (
    HighlightAgentConfigRepository,
)
from src.infrastructure.observability import (
    traced_service_method,
    metrics,
    with_span,
)
import logfire


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
    ):
        """Initialize stream processing service.

        Args:
            stream_repo: Repository for stream operations
            user_repo: Repository for user operations
            org_repo: Repository for organization operations
            usage_repo: Repository for usage tracking
            highlight_repo: Repository for highlight operations
            agent_config_repo: Repository for agent configurations
        """
        super().__init__()
        self.stream_repo = stream_repo
        self.user_repo = user_repo
        self.org_repo = org_repo
        self.usage_repo = usage_repo
        self.highlight_repo = highlight_repo
        self.agent_config_repo = agent_config_repo

    @traced_service_method(name="start_stream_processing")
    async def start_stream_processing(
        self,
        user_id: int,
        url: str,
        title: str,
        processing_options: Optional[ProcessingOptions] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_config_id: Optional[int] = None,
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
        with logfire.span("validate_user") as span:
            span.set_attribute("user.id", user_id)
            user = await self.user_repo.get(user_id)
            if not user:
                span.set_attribute("error", True)
                span.set_attribute("error.message", f"User {user_id} not found")
                raise EntityNotFoundError(f"User {user_id} not found")
            span.set_attribute("user.found", True)

        # Get user's organization for quota checking
        with logfire.span("get_organization") as span:
            org = await self._get_user_organization(user_id)
            if org:
                span.set_attribute("organization.id", org.id)
                span.set_attribute("organization.name", org.name)
                logfire.set_attribute("organization.id", org.id)

        # Check concurrent stream limit
        with logfire.span("check_quota") as span:
            await self._check_concurrent_stream_limit(user_id, org)
            span.set_attribute("quota.check_passed", True)

        # Determine platform from URL
        with logfire.span("detect_platform") as span:
            platform = self._detect_platform(url)
            span.set_attribute("stream.platform", platform.value)
            span.set_attribute("stream.url", url[:50] + "..." if len(url) > 50 else url)

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
            updated_at=Timestamp.now(),
        )

        # Save stream
        with logfire.span("save_stream") as span:
            saved_stream = await self.stream_repo.save(stream)
            span.set_attribute("stream.id", saved_stream.id)
            span.set_attribute("stream.platform", saved_stream.platform.value)
            metrics.increment_stream_started(
                platform=saved_stream.platform.value,
                organization_id=str(org.id) if org else "none",
                stream_type="live",
            )

        # Get or create agent configuration
        with logfire.span("get_agent_config") as span:
            agent_config = await self._get_or_create_agent_config(
                user_id=user_id,
                organization_id=org.id if org else None,
                agent_config_id=agent_config_id,
                url=url,
            )
            span.set_attribute(
                "agent_config.id",
                agent_config.id if hasattr(agent_config, "id") else None,
            )
            span.set_attribute(
                "agent_config.type",
                agent_config.config_type
                if hasattr(agent_config, "config_type")
                else None,
            )

        # Trigger FFmpeg stream ingestion task (which will chain to AI detection)
        with logfire.span("trigger_async_processing") as span:
            from src.infrastructure.async_processing.stream_tasks import (
                ingest_stream_with_ffmpeg,
            )

            task_result = ingest_stream_with_ffmpeg.delay(
                stream_id=saved_stream.id,
                chunk_duration=30,  # 30 second chunks
                agent_config_id=agent_config_id,
            )
            span.set_attribute("task.id", task_result.id)
            span.set_attribute("task.name", "ingest_stream_with_ffmpeg")

        # Store task ID for monitoring
        saved_stream.metadata = {
            **saved_stream.metadata,
            "ingestion_task_id": task_result.id,
        }
        await self.stream_repo.save(saved_stream)

        # Create initial usage record
        with logfire.span("create_usage_record") as span:
            usage_record = UsageRecord.for_stream_processing(
                user_id=user_id,
                stream_id=saved_stream.id,
                duration_minutes=0,  # Will be updated as processing continues
                organization_id=org.id if org else None,
            )
            await self.usage_repo.save(usage_record)
            span.set_attribute("usage_record.created", True)

        self.logger.info(
            f"Started stream processing with B2B agent: {saved_stream.id} for user {user_id}"
        )

        # Log stream start event
        logfire.info(
            "stream.processing.started",
            stream_id=saved_stream.id,
            user_id=user_id,
            platform=saved_stream.platform.value,
            organization_id=org.id if org else None,
            agent_config_id=agent_config_id,
        )

        return saved_stream

    @traced_service_method(name="update_stream_status")
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
        with logfire.span("get_stream_for_update") as span:
            span.set_attribute("stream.id", stream_id)
            span.set_attribute("new_status", new_status.value)
            stream = await self.stream_repo.get(stream_id)
            if not stream:
                span.set_attribute("error", True)
                raise EntityNotFoundError(f"Stream {stream_id} not found")
            span.set_attribute("current_status", stream.status.value)

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
                error_message=error_message
                if new_status == StreamStatus.FAILED
                else stream.error_message,
                metadata=self._merge_metadata(stream.metadata, metadata_updates),
                highlight_ids=stream.highlight_ids,
                created_at=stream.created_at,
                updated_at=Timestamp.now(),
            )

        # Save updated stream
        with logfire.span("save_updated_stream") as span:
            saved_stream = await self.stream_repo.save(updated_stream)
            span.set_attribute("stream.id", saved_stream.id)
            span.set_attribute("stream.status", saved_stream.status.value)

        # Cancel Celery tasks if stream is being cancelled
        if new_status == StreamStatus.CANCELLED:
            await self._cancel_stream_tasks(stream_id)

        # Update usage record if completed
        if new_status in [StreamStatus.COMPLETED, StreamStatus.FAILED]:
            with logfire.span("finalize_usage") as span:
                await self._finalize_stream_usage(saved_stream)
                span.set_attribute("stream.final_status", new_status.value)

                # Track stream completion metrics
                if hasattr(saved_stream, "organization_id"):
                    metrics.increment_stream_completed(
                        platform=saved_stream.platform.value,
                        organization_id=str(saved_stream.organization_id)
                        if saved_stream.organization_id
                        else "none",
                        stream_type="live",
                        success=(new_status == StreamStatus.COMPLETED),
                    )

        self.logger.info(f"Updated stream {stream_id} status to {new_status.value}")

        # Log status update event
        logfire.info(
            "stream.status.updated",
            stream_id=stream_id,
            old_status=stream.status.value,
            new_status=new_status.value,
            error_message=error_message,
        )

        return saved_stream

    @traced_service_method(name="get_user_active_streams")
    async def get_user_active_streams(self, user_id: int) -> List[Stream]:
        """Get all active streams for a user.

        Args:
            user_id: User ID

        Returns:
            List of active streams
        """
        return await self.stream_repo.get_by_user(
            user_id, status=StreamStatus.PROCESSING
        )

    @traced_service_method(name="get_stream_with_highlights")
    async def get_stream_with_highlights(
        self, stream_id: int, user_id: int
    ) -> Optional[Stream]:
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

            if not (
                user_org and stream_owner_org and user_org.id == stream_owner_org.id
            ):
                from src.domain.exceptions import UnauthorizedAccessError

                raise UnauthorizedAccessError("You don't have access to this stream")

        return stream

    @traced_service_method(name="calculate_stream_cost")
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
        with logfire.span("calculate_costs") as span:
            total_minutes = sum(record.quantity for record in usage_records)
            total_cost = sum(record.total_cost or 0 for record in usage_records)

            # Get highlight count
            highlights = await self.highlight_repo.get_by_stream(stream_id)

            span.set_attribute("cost.total_minutes", total_minutes)
            span.set_attribute("cost.total_amount", total_cost)
            span.set_attribute("cost.highlight_count", len(highlights))

            # Record cost metrics
            metrics.record_stream_cost(
                cost=total_cost,
                duration_minutes=total_minutes,
                platform=stream.platform.value,
            )

        return {
            "stream_id": stream_id,
            "duration_minutes": total_minutes,
            "total_cost": total_cost,
            "highlight_count": len(highlights),
            "cost_per_highlight": total_cost / len(highlights) if highlights else 0,
            "status": stream.status.value,
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

    async def get_stream_task_status(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """Get Celery task status for a stream.

        Args:
            stream_id: Stream ID

        Returns:
            Task status information or None if not found
        """
        try:
            stream = await self.stream_repo.get(stream_id)
            if not stream or not stream.metadata.get("ingestion_task_id"):
                return None

            from src.infrastructure.async_processing.celery_app import celery_app

            task_id = stream.metadata["ingestion_task_id"]
            task_result = celery_app.AsyncResult(task_id)

            return {
                "task_id": task_id,
                "status": task_result.status,
                "info": task_result.info if task_result.info else {},
                "ready": task_result.ready(),
                "successful": task_result.successful() if task_result.ready() else None,
                "failed": task_result.failed() if task_result.ready() else None,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting task status for stream {stream_id}: {str(e)}"
            )
            return None

    @traced_service_method(name="stop_stream_processing")
    async def stop_stream_processing(
        self, stream_id: int, force: bool = False
    ) -> Stream:
        """Stop stream processing and cancel tasks.

        Args:
            stream_id: Stream ID
            force: Whether to force stop

        Returns:
            Updated stream entity
        """
        # Cancel any running tasks first
        await self._cancel_stream_tasks(stream_id)

        # Update stream status
        updated_stream = await self.update_stream_status(
            stream_id=stream_id,
            new_status=StreamStatus.COMPLETED if not force else StreamStatus.CANCELLED,
        )

        return updated_stream

    # Private helper methods

    async def _get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get the organization for a user."""
        orgs = await self.org_repo.get_by_owner(user_id)
        return orgs[0] if orgs else None

    @with_span("check_concurrent_stream_limit")
    async def _check_concurrent_stream_limit(
        self, user_id: int, org: Optional[Organization]
    ):
        """Check if user can start another concurrent stream."""
        active_streams = await self.stream_repo.get_active_streams()
        user_active_count = sum(1 for s in active_streams if s.user_id == user_id)

        # No concurrent stream limits for first client - unlimited processing
        # Log usage for statistics but don't enforce limits
        logfire.info(
            "stream.concurrent_usage",
            user_id=user_id,
            organization_id=org.id if org else None,
            current_count=user_active_count,
            note="No limits enforced - unlimited for first client",
        )

    def _detect_platform(self, url: str) -> StreamPlatform:
        """Detect streaming platform/protocol from URL.

        Supports any format that FFmpeg can ingest.
        """
        url_lower = url.lower()

        # Check URL scheme/protocol
        if url_lower.startswith("rtmp://"):
            return StreamPlatform.RTMP
        elif url_lower.startswith("rtmps://"):
            return StreamPlatform.RTMPS
        elif url_lower.startswith("rtsp://"):
            return StreamPlatform.RTSP
        elif url_lower.startswith("rtp://"):
            return StreamPlatform.RTP
        elif url_lower.startswith("udp://"):
            return StreamPlatform.UDP
        elif url_lower.startswith("srt://"):
            return StreamPlatform.SRT
        elif (
            url_lower.startswith("file://")
            or url_lower.startswith("/")
            or (len(url_lower) > 1 and url_lower[1] == ":")
        ):
            # Local file paths
            return StreamPlatform.FILE
        elif url_lower.endswith(".m3u8") or "#ext-x-" in url_lower:
            # HLS playlist
            return StreamPlatform.HLS
        elif url_lower.endswith(".mpd") or "dash" in url_lower:
            # DASH manifest
            return StreamPlatform.DASH
        elif url_lower.startswith("http://") or url_lower.startswith("https://"):
            # Generic HTTP stream (could be HLS, DASH, or direct stream)
            return StreamPlatform.HTTP
        else:
            # Anything else FFmpeg might support
            return StreamPlatform.CUSTOM

    def _is_valid_transition(
        self, from_status: StreamStatus, to_status: StreamStatus
    ) -> bool:
        """Check if status transition is valid."""
        valid_transitions = {
            StreamStatus.PENDING: [
                StreamStatus.PROCESSING,
                StreamStatus.FAILED,
                StreamStatus.CANCELLED,
            ],
            StreamStatus.PROCESSING: [
                StreamStatus.COMPLETED,
                StreamStatus.FAILED,
                StreamStatus.CANCELLED,
            ],
            StreamStatus.COMPLETED: [],  # Terminal state
            StreamStatus.FAILED: [],  # Terminal state
            StreamStatus.CANCELLED: [],  # Terminal state
        }

        return to_status in valid_transitions.get(from_status, [])

    def _merge_metadata(
        self, existing: Dict[str, Any], updates: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge metadata updates with existing metadata."""
        if not updates:
            return existing

        merged = existing.copy()
        merged.update(updates)
        return merged

    @with_span("finalize_stream_usage")
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

    @with_span("cancel_stream_tasks")
    async def _cancel_stream_tasks(self, stream_id: int) -> None:
        """Cancel running Celery tasks for a stream."""
        try:
            stream = await self.stream_repo.get(stream_id)
            if not stream or not stream.metadata.get("ingestion_task_id"):
                return

            from src.infrastructure.async_processing.celery_app import celery_app

            task_id = stream.metadata["ingestion_task_id"]
            celery_app.control.revoke(task_id, terminate=True)

            self.logger.info(f"Cancelled Celery tasks for stream {stream_id}")

        except Exception as e:
            self.logger.error(
                f"Error cancelling tasks for stream {stream_id}: {str(e)}"
            )

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
            # Could parse Twitch categories or game names from URL/metadata
            return "gaming"
        elif "youtube.com" in url_lower:
            # Could analyze YouTube title/description for game detection
            return "gaming"

        # Default to general gaming
        return "gaming"
