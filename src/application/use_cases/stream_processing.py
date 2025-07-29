"""Stream processing use cases."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.entities.highlight import Highlight
from src.domain.entities.webhook import WebhookEvent
from src.domain.entities.usage_record import UsageType
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.services.stream_processing_service import StreamProcessingService, ProcessingOptions
from src.domain.services.highlight_detection_service import HighlightDetectionService, DetectionResult
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.domain.services.usage_tracking_service import UsageTrackingService
from src.domain.exceptions import (
    EntityNotFoundError,
    QuotaExceededError,
    BusinessRuleViolation,
    InvalidResourceStateError
)


@dataclass
class StreamStartRequest:
    """Request to start stream processing."""
    user_id: int
    url: str
    title: str
    platform: Optional[str] = None
    processing_options: Optional[Dict[str, Any]] = None


@dataclass
class StreamStartResult(UseCaseResult):
    """Result of starting stream processing."""
    stream_id: Optional[int] = None
    stream_url: Optional[str] = None
    status: Optional[str] = None


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
    status: Optional[str] = None
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
        stream_service: StreamProcessingService,
        highlight_service: HighlightDetectionService,
        webhook_service: WebhookDeliveryService,
        usage_service: UsageTrackingService
    ):
        """Initialize stream processing use case.
        
        Args:
            user_repo: Repository for user operations
            stream_repo: Repository for stream operations
            highlight_repo: Repository for highlight operations
            stream_service: Service for stream processing
            highlight_service: Service for highlight detection
            webhook_service: Service for webhook delivery
            usage_service: Service for usage tracking
        """
        self.user_repo = user_repo
        self.stream_repo = stream_repo
        self.highlight_repo = highlight_repo
        self.stream_service = stream_service
        self.highlight_service = highlight_service
        self.webhook_service = webhook_service
        self.usage_service = usage_service
    
    async def start_stream(self, request: StreamStartRequest) -> StreamStartResult:
        """Start processing a stream.
        
        Args:
            request: Stream start request
            
        Returns:
            Stream start result
        """
        try:
            # Validate user exists
            user = await self.user_repo.get(request.user_id)
            if not user:
                return StreamStartResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["User not found"]
                )
            
            # Parse processing options
            processing_options = None
            if request.processing_options:
                processing_options = ProcessingOptions(
                    detect_gameplay=request.processing_options.get("detect_gameplay", True),
                    detect_reactions=request.processing_options.get("detect_reactions", True),
                    detect_funny_moments=request.processing_options.get("detect_funny_moments", True),
                    detect_emotional_moments=request.processing_options.get("detect_emotional_moments", True),
                    min_highlight_duration=request.processing_options.get("min_highlight_duration", 5.0),
                    max_highlight_duration=request.processing_options.get("max_highlight_duration", 120.0),
                    confidence_threshold=request.processing_options.get("confidence_threshold", 0.7)
                )
            
            # Start stream processing
            stream = await self.stream_service.start_stream_processing(
                user_id=request.user_id,
                url=request.url,
                title=request.title,
                processing_options=processing_options
            )
            
            # Track API usage
            await self.usage_service.track_api_call(
                user_id=request.user_id,
                api_key_id=1,  # Would come from auth context
                endpoint="/streams",
                method="POST",
                response_time_ms=100,  # Would be measured
                status_code=201
            )
            
            # Trigger webhook
            await self.webhook_service.trigger_event(
                event=WebhookEvent.STREAM_STARTED,
                user_id=request.user_id,
                resource_id=stream.id,
                metadata={
                    "platform": stream.platform.value,
                    "title": stream.title
                }
            )
            
            return StreamStartResult(
                status=ResultStatus.SUCCESS,
                stream_id=stream.id,
                stream_url=stream.url.value,
                status=stream.status.value,
                message="Stream processing started successfully"
            )
            
        except QuotaExceededError as e:
            return StreamStartResult(
                status=ResultStatus.QUOTA_EXCEEDED,
                errors=[str(e)]
            )
        except BusinessRuleViolation as e:
            return StreamStartResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return StreamStartResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to start stream: {str(e)}"]
            )
    
    async def stop_stream(self, request: StreamStopRequest) -> StreamStopResult:
        """Stop processing a stream.
        
        Args:
            request: Stream stop request
            
        Returns:
            Stream stop result
        """
        try:
            # Get stream
            stream = await self.stream_repo.get(request.stream_id)
            if not stream:
                return StreamStopResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["Stream not found"]
                )
            
            # Verify ownership
            if stream.user_id != request.user_id:
                return StreamStopResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to stop this stream"]
                )
            
            # Stop stream processing
            stopped_stream = await self.stream_service.stop_stream_processing(
                stream_id=request.stream_id,
                force=request.force
            )
            
            # Get highlights count
            highlights = await self.highlight_repo.get_by_stream(request.stream_id)
            
            # Calculate processing duration
            duration_minutes = None
            if stopped_stream.started_at and stopped_stream.completed_at:
                duration_seconds = (
                    stopped_stream.completed_at.value - stopped_stream.started_at.value
                ).total_seconds()
                duration_minutes = duration_seconds / 60
                
                # Track stream processing usage
                await self.usage_service.track_stream_processing(
                    user_id=request.user_id,
                    stream_id=request.stream_id,
                    duration_minutes=duration_minutes,
                    highlights_generated=len(highlights)
                )
            
            # Trigger webhook
            await self.webhook_service.trigger_event(
                event=WebhookEvent.STREAM_COMPLETED,
                user_id=request.user_id,
                resource_id=request.stream_id,
                metadata={
                    "duration_minutes": duration_minutes,
                    "highlights_count": len(highlights),
                    "forced_stop": request.force
                }
            )
            
            return StreamStopResult(
                status=ResultStatus.SUCCESS,
                stream_id=stopped_stream.id,
                final_status=stopped_stream.status.value,
                highlights_count=len(highlights),
                processing_duration_minutes=duration_minutes,
                message="Stream processing stopped successfully"
            )
            
        except InvalidResourceStateError as e:
            return StreamStopResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return StreamStopResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to stop stream: {str(e)}"]
            )
    
    async def get_stream_status(self, request: StreamStatusRequest) -> StreamStatusResult:
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
                    status=ResultStatus.NOT_FOUND,
                    errors=["Stream not found"]
                )
            
            # Verify ownership
            if stream.user_id != request.user_id:
                return StreamStatusResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to view this stream"]
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
                status=stream.status.value,
                progress_percentage=progress,
                highlights_detected=len(highlights),
                error_message=stream.error_message,
                message="Stream status retrieved successfully"
            )
            
        except Exception as e:
            return StreamStatusResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get stream status: {str(e)}"]
            )
    
    async def process_highlights(self, request: ProcessHighlightsRequest) -> ProcessHighlightsResult:
        """Process detection results and create highlights.
        
        Args:
            request: Highlight processing request
            
        Returns:
            Processing result
        """
        try:
            # Process detection results
            highlights = await self.highlight_service.process_detection_results(
                stream_id=request.stream_id,
                video_results=request.video_results,
                audio_results=request.audio_results,
                chat_results=request.chat_results
            )
            
            # Get stream for user ID
            stream = await self.stream_repo.get(request.stream_id)
            if not stream:
                return ProcessHighlightsResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["Stream not found"]
                )
            
            # Trigger webhooks for each highlight
            for highlight in highlights:
                await self.webhook_service.trigger_event(
                    event=WebhookEvent.HIGHLIGHT_DETECTED,
                    user_id=stream.user_id,
                    resource_id=highlight.id,
                    metadata={
                        "type": highlight.highlight_type.value,
                        "confidence": highlight.confidence_score.value,
                        "duration_seconds": highlight.duration.value
                    }
                )
            
            return ProcessHighlightsResult(
                status=ResultStatus.SUCCESS,
                highlights_created=len(highlights),
                highlight_ids=[h.id for h in highlights],
                message=f"Created {len(highlights)} highlights"
            )
            
        except Exception as e:
            return ProcessHighlightsResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to process highlights: {str(e)}"]
            )
    
    async def execute(self, request: StreamStartRequest) -> StreamStartResult:
        """Execute stream start (default use case method).
        
        Args:
            request: Stream start request
            
        Returns:
            Stream start result
        """
        return await self.start_stream(request)