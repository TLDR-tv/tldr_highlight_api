"""Stream processing use cases."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.stream import StreamStatus
from src.domain.entities.webhook import WebhookEvent
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.highlight_agent_config_repository import (
    HighlightAgentConfigRepository,
)
from src.domain.services.stream_processing_service import StreamProcessingService
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.services.highlight_detection_service import (
    HighlightDetectionService,
    DetectionResult,
)
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.domain.services.usage_tracking_service import UsageTrackingService
from src.domain.exceptions import (
    BusinessRuleViolation,
    InvalidResourceStateError,
)
from src.infrastructure.observability import traced_use_case, metrics
import logfire


@dataclass
class StreamStartRequest:
    """Request to start stream processing."""

    user_id: int
    url: str
    title: str
    platform: Optional[str] = None
    processing_options: Optional[Dict[str, Any]] = None
    agent_config_id: Optional[int] = None


@dataclass
class StreamStartResult(UseCaseResult):
    """Result of starting stream processing."""

    stream_id: Optional[int] = None
    stream_url: Optional[str] = None
    stream_status: Optional[str] = None
    stream_platform: Optional[str] = None


@dataclass
class StreamStopRequest:
    """Request to stop stream processing."""

    user_id: int
    stream_id: int
    force: bool = False


@dataclass
class StreamStopResult(UseCaseResult):
    """Result of stopping stream processing."""

    stream_id: Optional[int] = None
    final_status: Optional[str] = None
    highlights_count: Optional[int] = None
    processing_duration_minutes: Optional[float] = None


@dataclass
class StreamStatusRequest:
    """Request to get stream status."""

    user_id: int
    stream_id: int


@dataclass
class StreamStatusResult(UseCaseResult):
    """Result of stream status check."""

    stream_id: Optional[int] = None
    stream_status: Optional[str] = None
    progress_percentage: Optional[float] = None
    highlights_detected: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class ProcessHighlightsRequest:
    """Request to process highlights for a stream segment."""

    stream_id: int
    video_results: List[DetectionResult]
    audio_results: List[DetectionResult]
    chat_results: List[DetectionResult]


@dataclass
class ProcessHighlightsResult(UseCaseResult):
    """Result of highlight processing."""

    highlights_created: Optional[int] = None
    highlight_ids: Optional[List[int]] = None


class StreamProcessingUseCase(UseCase[StreamStartRequest, StreamStartResult]):
    """Use case for stream processing operations."""

    def __init__(
        self,
        user_repo: UserRepository,
        stream_repo: StreamRepository,
        highlight_repo: HighlightRepository,
        agent_config_repo: HighlightAgentConfigRepository,
        stream_service: StreamProcessingService,
        highlight_service: HighlightDetectionService,
        webhook_service: WebhookDeliveryService,
        usage_service: UsageTrackingService,
    ):
        """Initialize stream processing use case.

        Args:
            user_repo: Repository for user operations
            stream_repo: Repository for stream operations
            highlight_repo: Repository for highlight operations
            agent_config_repo: Repository for agent configurations
            stream_service: Service for stream processing
            highlight_service: Service for highlight detection
            webhook_service: Service for webhook delivery
            usage_service: Service for usage tracking
        """
        self.user_repo = user_repo
        self.stream_repo = stream_repo
        self.highlight_repo = highlight_repo
        self.agent_config_repo = agent_config_repo
        self.stream_service = stream_service
        self.highlight_service = highlight_service
        self.webhook_service = webhook_service
        self.usage_service = usage_service

    @traced_use_case(name="start_stream")
    async def start_stream(self, request: StreamStartRequest) -> StreamStartResult:
        """Start processing a stream.

        Args:
            request: Stream start request

        Returns:
            Stream start result
        """
        try:
            with logfire.span("start_stream.transaction") as span:
                span.set_attribute("user.id", request.user_id)
                span.set_attribute(
                    "stream.url",
                    request.url[:50] + "..." if len(request.url) > 50 else request.url,
                )
                span.set_attribute("stream.title", request.title)

                # Validate user exists
                user = await self.user_repo.get(request.user_id)
                if not user:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", "user_not_found")
                    return StreamStartResult(
                        status=ResultStatus.NOT_FOUND, errors=["User not found"]
                    )

            # Parse processing options
            processing_options = None
            if request.processing_options:
                processing_options = ProcessingOptions(
                    dimension_set_id=request.processing_options.get("dimension_set_id"),
                    min_highlight_duration=request.processing_options.get(
                        "min_highlight_duration", 10.0
                    ),
                    max_highlight_duration=request.processing_options.get(
                        "max_highlight_duration", 300.0
                    ),
                    typical_highlight_duration=request.processing_options.get(
                        "typical_highlight_duration", 60.0
                    ),
                    min_confidence_threshold=request.processing_options.get(
                        "min_confidence_threshold", 0.5
                    ),
                    target_confidence_threshold=request.processing_options.get(
                        "target_confidence_threshold", 0.7
                    ),
                    exceptional_threshold=request.processing_options.get(
                        "exceptional_threshold", 0.85
                    ),
                    enable_scene_detection=request.processing_options.get(
                        "enable_scene_detection", True
                    ),
                    enable_silence_detection=request.processing_options.get(
                        "enable_silence_detection", True
                    ),
                    enable_motion_detection=request.processing_options.get(
                        "enable_motion_detection", True
                    ),
                    generate_thumbnails=request.processing_options.get(
                        "generate_thumbnails", True
                    ),
                    generate_previews=request.processing_options.get(
                        "generate_previews", True
                    ),
                )

            # Start stream processing with B2B agent
            with logfire.span("start_stream_processing") as processing_span:
                stream = await self.stream_service.start_stream_processing(
                    user_id=request.user_id,
                    url=request.url,
                    title=request.title,
                    processing_options=processing_options,
                    agent_config_id=request.agent_config_id,
                )
                processing_span.set_attribute("stream.id", stream.id)
                processing_span.set_attribute("stream.platform", stream.platform.value)

            # Track API usage
            with logfire.span("track_api_usage"):
                await self.usage_service.track_api_call(
                    user_id=request.user_id,
                    api_key_id=1,  # Would come from auth context
                    endpoint="/streams",
                    method="POST",
                    response_time_ms=100,  # Would be measured
                    status_code=201,
                )

                # Track use case metrics
                metrics.increment_api_calls(
                    endpoint="/streams",
                    method="POST",
                    status_code=201,
                    organization_id=str(stream.organization_id)
                    if hasattr(stream, "organization_id")
                    else "none",
                )

            # Trigger webhook
            with logfire.span("trigger_webhook") as webhook_span:
                await self.webhook_service.trigger_event(
                    event=WebhookEvent.STREAM_STARTED,
                    user_id=request.user_id,
                    resource_id=stream.id,
                    metadata={"platform": stream.platform.value, "title": stream.title},
                )
                webhook_span.set_attribute(
                    "webhook.event", WebhookEvent.STREAM_STARTED.value
                )
                webhook_span.set_attribute("webhook.resource_id", stream.id)

            span.set_attribute("success", True)
            span.set_attribute("stream.id", stream.id)

            # Log successful stream start
            logfire.info(
                "use_case.stream.started",
                user_id=request.user_id,
                stream_id=stream.id,
                platform=stream.platform.value,
                agent_config_id=request.agent_config_id,
            )

            return StreamStartResult(
                status=ResultStatus.SUCCESS,
                stream_id=stream.id,
                stream_url=stream.url.value,
                stream_status=stream.status.value,
                stream_platform=stream.platform.value,
                message="Stream processing started successfully",
            )

        # No quota limits for first client - this exception is no longer raised
        except BusinessRuleViolation as e:
            logfire.warning(
                "use_case.stream.validation_error",
                user_id=request.user_id,
                error=str(e),
            )
            return StreamStartResult(
                status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
            )
        except Exception as e:
            logfire.error(
                "use_case.stream.start_failed",
                user_id=request.user_id,
                error=str(e),
                exc_info=True,
            )
            return StreamStartResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to start stream: {str(e)}"],
            )

    @traced_use_case(name="stop_stream")
    async def stop_stream(self, request: StreamStopRequest) -> StreamStopResult:
        """Stop processing a stream.

        Args:
            request: Stream stop request

        Returns:
            Stream stop result
        """
        try:
            with logfire.span("stop_stream.transaction") as span:
                span.set_attribute("user.id", request.user_id)
                span.set_attribute("stream.id", request.stream_id)
                span.set_attribute("force", request.force)

                # Get stream
                stream = await self.stream_repo.get(request.stream_id)
                if not stream:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", "stream_not_found")
                    return StreamStopResult(
                        status=ResultStatus.NOT_FOUND, errors=["Stream not found"]
                    )

            # Verify ownership
            if stream.user_id != request.user_id:
                return StreamStopResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to stop this stream"],
                )

            # Stop stream processing and agent
            stopped_stream = await self.stream_service.stop_stream_processing(
                stream_id=request.stream_id, force=request.force
            )

            # Get highlights count
            highlights = await self.highlight_repo.get_by_stream(request.stream_id)

            # Calculate processing duration
            duration_minutes = None
            with logfire.span("calculate_usage") as usage_span:
                if stopped_stream.started_at and stopped_stream.completed_at:
                    duration_seconds = (
                        stopped_stream.completed_at.value
                        - stopped_stream.started_at.value
                    ).total_seconds()
                    duration_minutes = duration_seconds / 60

                    usage_span.set_attribute("duration_minutes", duration_minutes)
                    usage_span.set_attribute("highlights_count", len(highlights))

                    # Track stream processing usage
                    await self.usage_service.track_stream_processing(
                        user_id=request.user_id,
                        stream_id=request.stream_id,
                        duration_minutes=duration_minutes,
                        highlights_generated=len(highlights),
                    )

                    # Track completion metrics
                    metrics.record_stream_duration(
                        duration_minutes=duration_minutes,
                        platform=stopped_stream.platform.value,
                        highlights_count=len(highlights),
                    )

            # Trigger webhook
            await self.webhook_service.trigger_event(
                event=WebhookEvent.STREAM_COMPLETED,
                user_id=request.user_id,
                resource_id=request.stream_id,
                metadata={
                    "duration_minutes": duration_minutes,
                    "highlights_count": len(highlights),
                    "forced_stop": request.force,
                },
            )

            span.set_attribute("success", True)
            span.set_attribute("final_status", stopped_stream.status.value)
            span.set_attribute("highlights_count", len(highlights))

            # Log successful stream stop
            logfire.info(
                "use_case.stream.stopped",
                user_id=request.user_id,
                stream_id=request.stream_id,
                final_status=stopped_stream.status.value,
                highlights_count=len(highlights),
                duration_minutes=duration_minutes,
                forced=request.force,
            )

            return StreamStopResult(
                status=ResultStatus.SUCCESS,
                stream_id=stopped_stream.id,
                final_status=stopped_stream.status.value,
                highlights_count=len(highlights),
                processing_duration_minutes=duration_minutes,
                message="Stream processing stopped successfully",
            )

        except InvalidResourceStateError as e:
            return StreamStopResult(
                status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
            )
        except Exception as e:
            return StreamStopResult(
                status=ResultStatus.FAILURE, errors=[f"Failed to stop stream: {str(e)}"]
            )

    @traced_use_case(name="get_stream_status")
    async def get_stream_status(
        self, request: StreamStatusRequest
    ) -> StreamStatusResult:
        """Get current status of a stream.

        Args:
            request: Stream status request

        Returns:
            Stream status result
        """
        try:
            # Get stream
            stream = await self.stream_repo.get(request.stream_id)
            if not stream:
                return StreamStatusResult(
                    status=ResultStatus.NOT_FOUND, errors=["Stream not found"]
                )

            # Verify ownership
            if stream.user_id != request.user_id:
                return StreamStatusResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to view this stream"],
                )

            # Get highlights count
            highlights = await self.highlight_repo.get_by_stream(request.stream_id)

            # Calculate progress (simplified - would be based on actual processing)
            progress = 0.0
            if stream.status == StreamStatus.COMPLETED:
                progress = 100.0
            elif stream.status == StreamStatus.PROCESSING:
                # In real implementation, this would check processing queue
                progress = 50.0
            elif stream.status == StreamStatus.FAILED:
                progress = 0.0

            return StreamStatusResult(
                status=ResultStatus.SUCCESS,
                stream_id=stream.id,
                stream_status=stream.status.value,
                progress_percentage=progress,
                highlights_detected=len(highlights),
                error_message=stream.error_message,
                message="Stream status retrieved successfully",
            )

        except Exception as e:
            return StreamStatusResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get stream status: {str(e)}"],
            )

    @traced_use_case(name="process_highlights")
    async def process_highlights(
        self, request: ProcessHighlightsRequest
    ) -> ProcessHighlightsResult:
        """Process detection results and create highlights.

        Args:
            request: Highlight processing request

        Returns:
            Processing result
        """
        try:
            with logfire.span("process_highlights.transaction") as span:
                span.set_attribute("stream.id", request.stream_id)
                span.set_attribute("video_results.count", len(request.video_results))
                span.set_attribute("audio_results.count", len(request.audio_results))
                span.set_attribute("chat_results.count", len(request.chat_results))

                # Process detection results
                highlights = await self.highlight_service.process_detection_results(
                    stream_id=request.stream_id,
                    video_results=request.video_results,
                    audio_results=request.audio_results,
                    chat_results=request.chat_results,
                )

                span.set_attribute("highlights.created", len(highlights))

            # Get stream for user ID
            stream = await self.stream_repo.get(request.stream_id)
            if not stream:
                return ProcessHighlightsResult(
                    status=ResultStatus.NOT_FOUND, errors=["Stream not found"]
                )

            # Trigger webhooks for each highlight
            with logfire.span("trigger_highlight_webhooks") as webhook_span:
                webhook_span.set_attribute("webhooks.count", len(highlights))

                for i, highlight in enumerate(highlights):
                    await self.webhook_service.trigger_event(
                        event=WebhookEvent.HIGHLIGHT_DETECTED,
                        user_id=stream.user_id,
                        resource_id=highlight.id,
                        metadata={
                            "types": highlight.highlight_types,
                            "confidence": highlight.confidence_score.value,
                            "duration_seconds": highlight.duration.value,
                        },
                    )

                    # Track highlight metrics
                    metrics.record_highlight_confidence(
                        confidence=highlight.confidence_score.value,
                        detection_method="highlight_service",
                        platform=stream.platform.value
                        if hasattr(stream, "platform")
                        else "unknown",
                    )

            span.set_attribute("success", True)

            # Log highlight processing completion
            logfire.info(
                "use_case.highlights.processed",
                stream_id=request.stream_id,
                highlights_created=len(highlights),
                total_results=len(request.video_results)
                + len(request.audio_results)
                + len(request.chat_results),
            )

            return ProcessHighlightsResult(
                status=ResultStatus.SUCCESS,
                highlights_created=len(highlights),
                highlight_ids=[h.id for h in highlights],
                message=f"Created {len(highlights)} highlights",
            )

        except Exception as e:
            return ProcessHighlightsResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to process highlights: {str(e)}"],
            )

    async def get_stream_task_status(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """Get Celery task status for a stream.

        Args:
            stream_id: Stream ID

        Returns:
            Task status information or None
        """
        return await self.stream_service.get_stream_task_status(stream_id)

    async def execute(self, request: StreamStartRequest) -> StreamStartResult:
        """Execute stream start (default use case method).

        Args:
            request: Stream start request

        Returns:
            Stream start result
        """
        return await self.start_stream(request)
