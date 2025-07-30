"""Webhook mapper for converting between API DTOs and domain objects."""

from typing import List, Optional
from datetime import datetime, timezone

from src.api.schemas.webhooks import (
    WebhookCreate,
    WebhookUpdate,
    WebhookResponse,
    WebhookTestResponse,
    WebhookListResponse,
)
from src.application.use_cases.webhook_configuration import (
    CreateWebhookRequest,
    UpdateWebhookRequest,
    DeleteWebhookRequest,
    ListWebhooksRequest,
    TestWebhookRequest,
    GetWebhookEventsRequest,
)
from src.domain.entities.webhook import Webhook


class WebhookMapper:
    """Maps between webhook API DTOs and domain entities."""

    @staticmethod
    def to_create_webhook_request(
        user_id: int, create_dto: WebhookCreate
    ) -> CreateWebhookRequest:
        """Convert WebhookCreate DTO to domain request."""
        return CreateWebhookRequest(
            user_id=user_id,
            url=str(create_dto.url),
            events=create_dto.events,
            active=create_dto.active,
            description=None,  # Not in DTO yet
            secret=create_dto.secret,
        )

    @staticmethod
    def to_update_webhook_request(
        user_id: int, webhook_id: int, update_dto: WebhookUpdate
    ) -> UpdateWebhookRequest:
        """Convert WebhookUpdate DTO to domain request."""
        return UpdateWebhookRequest(
            user_id=user_id,
            webhook_id=webhook_id,
            url=str(update_dto.url) if update_dto.url else None,
            events=update_dto.events,
            active=update_dto.active,
            description=None,  # Not in DTO yet
        )

    @staticmethod
    def to_delete_webhook_request(
        user_id: int, webhook_id: int
    ) -> DeleteWebhookRequest:
        """Convert parameters to DeleteWebhookRequest."""
        return DeleteWebhookRequest(user_id=user_id, webhook_id=webhook_id)

    @staticmethod
    def to_list_webhooks_request(
        user_id: int, active_only: bool = False
    ) -> ListWebhooksRequest:
        """Convert parameters to ListWebhooksRequest."""
        return ListWebhooksRequest(user_id=user_id, active_only=active_only)

    @staticmethod
    def to_test_webhook_request(
        user_id: int, webhook_id: int, test_event: str = "test"
    ) -> TestWebhookRequest:
        """Convert parameters to TestWebhookRequest."""
        return TestWebhookRequest(
            user_id=user_id, webhook_id=webhook_id, test_event=test_event
        )

    @staticmethod
    def to_get_webhook_events_request(
        user_id: int, webhook_id: int, page: int = 1, per_page: int = 20
    ) -> GetWebhookEventsRequest:
        """Convert parameters to GetWebhookEventsRequest."""
        return GetWebhookEventsRequest(
            user_id=user_id, webhook_id=webhook_id, page=page, per_page=per_page
        )

    @staticmethod
    def to_webhook_response(webhook: Webhook) -> WebhookResponse:
        """Convert Webhook domain entity to response DTO."""
        # Extract delivery stats (would be computed from events in real implementation)
        delivery_stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "last_delivery_at": None,
            "last_delivery_status": None,
        }

        return WebhookResponse(
            id=webhook.id,
            url=webhook.url.value,
            events=[e.value for e in webhook.events],
            active=webhook.is_active,
            created_at=webhook.created_at.value,
            updated_at=webhook.updated_at.value
            if webhook.updated_at
            else webhook.created_at.value,
            secret_configured=bool(webhook.secret),
            delivery_stats=delivery_stats,
        )

    @staticmethod
    def to_webhook_test_response(
        status_code: int, response_time_ms: float, error_message: Optional[str] = None
    ) -> WebhookTestResponse:
        """Convert test result to response DTO."""
        success = 200 <= status_code < 300

        return WebhookTestResponse(
            success=success,
            status_code=status_code,
            response_time_ms=response_time_ms,
            message="Webhook test successful" if success else "Webhook test failed",
            error=error_message,
            tested_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def to_webhook_list_response(
        webhooks: List[Webhook], page: int, per_page: int
    ) -> WebhookListResponse:
        """Convert list of webhooks to response DTO."""
        items = [WebhookMapper.to_webhook_response(webhook) for webhook in webhooks]

        total = len(webhooks)
        total_pages = (total + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1

        # Calculate stats
        total_active = sum(1 for w in webhooks if w.is_active)
        total_inactive = total - total_active

        return WebhookListResponse(
            page=page,
            per_page=per_page,
            total=total,
            pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            items=items,
            total_active=total_active,
            total_inactive=total_inactive,
        )
