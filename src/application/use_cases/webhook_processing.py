"""Webhook processing use case."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.application.use_cases.stream_processing import (
        StreamProcessingUseCase,
        StreamStartResult,
    )

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.webhook_event import (
    WebhookEvent,
    WebhookEventStatus,
    WebhookEventType,
)
from src.domain.repositories.webhook_event_repository import WebhookEventRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.infrastructure.security.webhook_validator import (
    WebhookValidatorFactory,
)
from src.api.schemas.webhook_models import (
    StreamStartedWebhookEvent,
)
from src.application.use_cases.stream_processing import StreamStartRequest

logger = logging.getLogger(__name__)


@dataclass
class ProcessWebhookRequest:
    """Request to process a webhook."""

    platform: str
    payload: Dict[str, Any]
    raw_payload: bytes
    headers: Dict[str, str]
    api_key: Optional[str] = None
    user_id: Optional[int] = None


@dataclass
class ProcessWebhookResult(UseCaseResult):
    """Result of webhook processing."""

    event_id: Optional[str] = None
    stream_id: Optional[int] = None
    duplicate: bool = False


class WebhookProcessingUseCase(UseCase[ProcessWebhookRequest, ProcessWebhookResult]):
    """Use case for processing incoming webhooks."""

    def __init__(
        self,
        webhook_event_repo: WebhookEventRepository,
        user_repo: UserRepository,
        api_key_repo: APIKeyRepository,
        stream_processing_use_case: "StreamProcessingUseCase",
        validator_factory: WebhookValidatorFactory,
    ):
        """Initialize webhook processing use case.

        Args:
            webhook_event_repo: Repository for webhook events
            user_repo: Repository for users
            api_key_repo: Repository for API keys
            stream_processing_use_case: Use case for stream processing
            validator_factory: Factory for webhook validators
        """
        self.webhook_event_repo = webhook_event_repo
        self.user_repo = user_repo
        self.api_key_repo = api_key_repo
        self.stream_processing_use_case = stream_processing_use_case
        self.validator_factory = validator_factory

    async def execute(self, request: ProcessWebhookRequest) -> ProcessWebhookResult:
        """Process a webhook event.

        Args:
            request: Webhook processing request

        Returns:
            Processing result
        """
        try:
            # Validate webhook signature
            if not await self._validate_webhook_signature(request):
                return ProcessWebhookResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Invalid webhook signature"],
                )

            # Parse webhook payload
            stream_event = await self._parse_webhook_payload(request)
            if not stream_event:
                return ProcessWebhookResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=["Unsupported webhook event type"],
                )

            # Check for duplicate event
            if await self._is_duplicate_event(stream_event.event_id, request.platform):
                logger.info(f"Duplicate webhook event: {stream_event.event_id}")
                return ProcessWebhookResult(
                    status=ResultStatus.SUCCESS,
                    event_id=stream_event.event_id,
                    duplicate=True,
                    message="Duplicate event ignored",
                )

            # Resolve user
            user_id = await self._resolve_user_id(request, stream_event)
            if not user_id:
                return ProcessWebhookResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Unable to authenticate webhook"],
                )

            # Create webhook event record
            webhook_event = await self._create_webhook_event(
                stream_event, request, user_id
            )

            # Process stream started event
            if stream_event.event_type == WebhookEventType.STREAM_STARTED.value:
                result = await self._process_stream_started(
                    webhook_event, stream_event, user_id
                )

                if result.is_success:
                    # Mark webhook as processed
                    webhook_event.mark_processed(result.stream_id)
                    await self.webhook_event_repo.save(webhook_event)

                    return ProcessWebhookResult(
                        status=ResultStatus.SUCCESS,
                        event_id=webhook_event.event_id,
                        stream_id=result.stream_id,
                        message="Stream processing started",
                    )
                else:
                    # Mark webhook as failed
                    webhook_event.mark_failed(
                        result.errors[0]
                        if result.errors
                        else "Stream processing failed"
                    )
                    await self.webhook_event_repo.save(webhook_event)

                    return ProcessWebhookResult(
                        status=result.status,
                        event_id=webhook_event.event_id,
                        errors=result.errors,
                    )

            return ProcessWebhookResult(
                status=ResultStatus.SUCCESS,
                event_id=webhook_event.event_id,
                message="Webhook processed",
            )

        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return ProcessWebhookResult(
                status=ResultStatus.FAILURE,
                errors=[f"Webhook processing failed: {str(e)}"],
            )

    async def _validate_webhook_signature(self, request: ProcessWebhookRequest) -> bool:
        """Validate webhook signature."""
        validator = self.validator_factory.get_validator(request.platform)
        if not validator:
            # No validator configured, allow if API key is provided
            return bool(request.api_key or request.user_id)

        # Extract signature and timestamp from headers (generic approach)
        signature = (
            request.headers.get("x-webhook-signature")
            or request.headers.get("x-hub-signature-256")
            or request.headers.get("signature")
        )

        timestamp = request.headers.get("x-webhook-timestamp") or request.headers.get(
            "timestamp"
        )

        if not signature:
            return False

        try:
            return validator.validate_signature(
                request.raw_payload, signature, timestamp
            )
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False

    async def _parse_webhook_payload(
        self, request: ProcessWebhookRequest
    ) -> Optional[StreamStartedWebhookEvent]:
        """Parse generic webhook payload."""
        try:
            # Attempt to parse as generic StreamStartedWebhookEvent
            return StreamStartedWebhookEvent(**request.payload)
        except Exception as e:
            logger.error(f"Error parsing webhook payload: {e}")
            return None

    async def _is_duplicate_event(self, event_id: str, platform: str) -> bool:
        """Check if event is a duplicate."""
        return await self.webhook_event_repo.exists_by_external_id(event_id, platform)

    async def _resolve_user_id(
        self, request: ProcessWebhookRequest, stream_event: StreamStartedWebhookEvent
    ) -> Optional[int]:
        """Resolve user ID from webhook data."""
        # Priority order:
        # 1. Explicit user_id in request
        if request.user_id:
            return request.user_id

        # 2. User ID in stream event
        if stream_event.user_id:
            return stream_event.user_id

        # 3. API key in request
        if request.api_key:
            api_key = await self.api_key_repo.get_by_key(request.api_key)
            if api_key and api_key.is_valid:
                return api_key.user_id

        # 4. API key in stream event
        if stream_event.api_key:
            api_key = await self.api_key_repo.get_by_key(stream_event.api_key)
            if api_key and api_key.is_valid:
                return api_key.user_id

        return None

    async def _create_webhook_event(
        self,
        stream_event: StreamStartedWebhookEvent,
        request: ProcessWebhookRequest,
        user_id: int,
    ) -> WebhookEvent:
        """Create and save webhook event record."""
        webhook_event = WebhookEvent(
            id=None,
            event_id=stream_event.event_id,
            event_type=WebhookEventType(stream_event.event_type),
            platform=request.platform,
            status=WebhookEventStatus.RECEIVED,
            payload=request.payload,
            user_id=user_id,
        )

        return await self.webhook_event_repo.save(webhook_event)

    async def _process_stream_started(
        self,
        webhook_event: WebhookEvent,
        stream_event: StreamStartedWebhookEvent,
        user_id: int,
    ) -> "StreamStartResult":
        """Process stream started event."""
        # Mark webhook as processing
        webhook_event.mark_processing()
        await self.webhook_event_repo.save(webhook_event)

        # Create stream processing request with generic platform handling
        stream_request = StreamStartRequest(
            user_id=user_id,
            url=stream_event.stream_url,
            title=stream_event.metadata.title or "Stream from webhook",
            platform="custom",  # All webhook streams are treated as custom
            processing_options={
                "webhook_event_id": webhook_event.id,
                "platform_metadata": stream_event.metadata.dict(),
                "original_platform": webhook_event.platform,  # Keep original platform for reference
            },
        )

        # Start stream processing
        return await self.stream_processing_use_case.start_stream(stream_request)
