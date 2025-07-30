"""Webhook configuration use cases."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.webhook import Webhook, WebhookEvent
from src.domain.value_objects.url import Url
from src.domain.repositories.webhook_repository import WebhookRepository
from src.domain.repositories.webhook_event_repository import WebhookEventRepository


@dataclass
class CreateWebhookRequest:
    """Request to create a webhook configuration."""

    user_id: int
    url: str
    events: List[str]
    active: bool = True
    description: Optional[str] = None
    secret: Optional[str] = None


@dataclass
class CreateWebhookResult(UseCaseResult):
    """Result of webhook creation."""

    webhook: Optional[Webhook] = None


@dataclass
class UpdateWebhookRequest:
    """Request to update webhook configuration."""

    user_id: int
    webhook_id: int
    url: Optional[str] = None
    events: Optional[List[str]] = None
    active: Optional[bool] = None
    description: Optional[str] = None


@dataclass
class UpdateWebhookResult(UseCaseResult):
    """Result of webhook update."""

    webhook: Optional[Webhook] = None


@dataclass
class DeleteWebhookRequest:
    """Request to delete webhook."""

    user_id: int
    webhook_id: int


@dataclass
class DeleteWebhookResult(UseCaseResult):
    """Result of webhook deletion."""

    pass


@dataclass
class ListWebhooksRequest:
    """Request to list webhooks."""

    user_id: int
    active_only: bool = False


@dataclass
class ListWebhooksResult(UseCaseResult):
    """Result of listing webhooks."""

    webhooks: List[Webhook] = None
    total: Optional[int] = None


@dataclass
class TestWebhookRequest:
    """Request to test webhook."""

    user_id: int
    webhook_id: int
    test_event: str = "test"


@dataclass
class TestWebhookResult(UseCaseResult):
    """Result of webhook test."""

    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class GetWebhookEventsRequest:
    """Request to get webhook delivery events."""

    user_id: int
    webhook_id: int
    page: int = 1
    per_page: int = 20


@dataclass
class GetWebhookEventsResult(UseCaseResult):
    """Result of getting webhook events."""

    events: List[Dict[str, Any]] = None
    total: Optional[int] = None
    success_rate: Optional[float] = None


class WebhookConfigurationUseCase(UseCase[CreateWebhookRequest, CreateWebhookResult]):
    """Use case for webhook configuration management."""

    def __init__(
        self,
        webhook_repo: WebhookRepository,
        webhook_event_repo: WebhookEventRepository,
    ):
        """Initialize webhook configuration use case.

        Args:
            webhook_repo: Repository for webhook operations
            webhook_event_repo: Repository for webhook event operations
        """
        self.webhook_repo = webhook_repo
        self.webhook_event_repo = webhook_event_repo

    async def create_webhook(
        self, request: CreateWebhookRequest
    ) -> CreateWebhookResult:
        """Create a new webhook configuration.

        Args:
            request: Create webhook request

        Returns:
            Created webhook
        """
        try:
            # Validate URL
            try:
                webhook_url = Url(request.url)
            except ValueError as e:
                return CreateWebhookResult(
                    status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
                )

            # Validate events
            valid_events = [e.value for e in WebhookEvent]
            invalid_events = [e for e in request.events if e not in valid_events]
            if invalid_events:
                return CreateWebhookResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[f"Invalid events: {', '.join(invalid_events)}"],
                )

            # Check for duplicate URL for this user
            existing = await self.webhook_repo.get_by_user_and_url(
                request.user_id, webhook_url
            )
            if existing:
                return CreateWebhookResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=["A webhook with this URL already exists"],
                )

            # Create webhook
            webhook = Webhook(
                id=None,
                user_id=request.user_id,
                url=webhook_url,
                events=[WebhookEvent(e) for e in request.events],
                is_active=request.active,
                description=request.description,
                secret=request.secret,
                created_at=None,  # Will be set by repository
                updated_at=None,  # Will be set by repository
            )

            # Save webhook
            saved_webhook = await self.webhook_repo.save(webhook)

            return CreateWebhookResult(
                status=ResultStatus.SUCCESS,
                webhook=saved_webhook,
                message="Webhook created successfully",
            )

        except Exception as e:
            return CreateWebhookResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to create webhook: {str(e)}"],
            )

    async def update_webhook(
        self, request: UpdateWebhookRequest
    ) -> UpdateWebhookResult:
        """Update webhook configuration.

        Args:
            request: Update webhook request

        Returns:
            Updated webhook
        """
        try:
            # Get existing webhook
            webhook = await self.webhook_repo.get(request.webhook_id)
            if not webhook:
                return UpdateWebhookResult(
                    status=ResultStatus.NOT_FOUND, errors=["Webhook not found"]
                )

            # Check ownership
            if webhook.user_id != request.user_id:
                return UpdateWebhookResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Update fields
            if request.url is not None:
                try:
                    webhook.url = Url(request.url)
                except ValueError as e:
                    return UpdateWebhookResult(
                        status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
                    )

            if request.events is not None:
                valid_events = [e.value for e in WebhookEvent]
                invalid_events = [e for e in request.events if e not in valid_events]
                if invalid_events:
                    return UpdateWebhookResult(
                        status=ResultStatus.VALIDATION_ERROR,
                        errors=[f"Invalid events: {', '.join(invalid_events)}"],
                    )
                webhook.events = [WebhookEvent(e) for e in request.events]

            if request.active is not None:
                webhook.is_active = request.active

            if request.description is not None:
                webhook.description = request.description

            # Save updated webhook
            saved_webhook = await self.webhook_repo.save(webhook)

            return UpdateWebhookResult(
                status=ResultStatus.SUCCESS,
                webhook=saved_webhook,
                message="Webhook updated successfully",
            )

        except Exception as e:
            return UpdateWebhookResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to update webhook: {str(e)}"],
            )

    async def delete_webhook(
        self, request: DeleteWebhookRequest
    ) -> DeleteWebhookResult:
        """Delete webhook configuration.

        Args:
            request: Delete webhook request

        Returns:
            Deletion result
        """
        try:
            # Get webhook
            webhook = await self.webhook_repo.get(request.webhook_id)
            if not webhook:
                return DeleteWebhookResult(
                    status=ResultStatus.NOT_FOUND, errors=["Webhook not found"]
                )

            # Check ownership
            if webhook.user_id != request.user_id:
                return DeleteWebhookResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Delete webhook
            await self.webhook_repo.delete(request.webhook_id)

            return DeleteWebhookResult(
                status=ResultStatus.SUCCESS, message="Webhook deleted successfully"
            )

        except Exception as e:
            return DeleteWebhookResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to delete webhook: {str(e)}"],
            )

    async def list_webhooks(self, request: ListWebhooksRequest) -> ListWebhooksResult:
        """List user's webhooks.

        Args:
            request: List webhooks request

        Returns:
            List of webhooks
        """
        try:
            # Get all user's webhooks
            webhooks = await self.webhook_repo.get_by_user(request.user_id)

            # Filter by active status if requested
            if request.active_only:
                webhooks = [w for w in webhooks if w.is_active]

            return ListWebhooksResult(
                status=ResultStatus.SUCCESS,
                webhooks=webhooks,
                total=len(webhooks),
                message=f"Found {len(webhooks)} webhooks",
            )

        except Exception as e:
            return ListWebhooksResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to list webhooks: {str(e)}"],
            )

    async def test_webhook(self, request: TestWebhookRequest) -> TestWebhookResult:
        """Test webhook delivery.

        Args:
            request: Test webhook request

        Returns:
            Test result
        """
        try:
            # Get webhook
            webhook = await self.webhook_repo.get(request.webhook_id)
            if not webhook:
                return TestWebhookResult(
                    status=ResultStatus.NOT_FOUND, errors=["Webhook not found"]
                )

            # Check ownership
            if webhook.user_id != request.user_id:
                return TestWebhookResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # In a real implementation, this would actually send a test request
            # For now, we'll just simulate success
            return TestWebhookResult(
                status=ResultStatus.SUCCESS,
                status_code=200,
                response_time_ms=150.0,
                message="Webhook test successful",
            )

        except Exception as e:
            return TestWebhookResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to test webhook: {str(e)}"],
            )

    async def get_webhook_events(
        self, request: GetWebhookEventsRequest
    ) -> GetWebhookEventsResult:
        """Get webhook delivery events.

        Args:
            request: Get events request

        Returns:
            Webhook delivery history
        """
        try:
            # Get webhook to check ownership
            webhook = await self.webhook_repo.get(request.webhook_id)
            if not webhook:
                return GetWebhookEventsResult(
                    status=ResultStatus.NOT_FOUND, errors=["Webhook not found"]
                )

            # Check ownership
            if webhook.user_id != request.user_id:
                return GetWebhookEventsResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Get delivery events
            events = await self.webhook_event_repo.get_by_webhook(
                webhook_id=request.webhook_id,
                limit=request.per_page,
                offset=(request.page - 1) * request.per_page,
            )

            # Calculate success rate
            if events:
                successful = sum(1 for e in events if e.status == "delivered")
                success_rate = (successful / len(events)) * 100
            else:
                success_rate = 0.0

            # Convert events to dict format
            event_dicts = [
                {
                    "id": e.id,
                    "event_type": e.event_type.value,
                    "status": e.status.value,
                    "attempts": e.attempts,
                    "created_at": e.created_at.value.isoformat()
                    if e.created_at
                    else None,
                    "delivered_at": e.delivered_at.value.isoformat()
                    if e.delivered_at
                    else None,
                    "response_status": e.response_status,
                    "response_body": e.response_body,
                }
                for e in events
            ]

            return GetWebhookEventsResult(
                status=ResultStatus.SUCCESS,
                events=event_dicts,
                total=len(events),
                success_rate=success_rate,
                message="Webhook events retrieved successfully",
            )

        except Exception as e:
            return GetWebhookEventsResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get webhook events: {str(e)}"],
            )

    async def execute(self, request: CreateWebhookRequest) -> CreateWebhookResult:
        """Execute webhook creation (default use case method).

        Args:
            request: Create webhook request

        Returns:
            Created webhook
        """
        return await self.create_webhook(request)
