"""Simplified usage tracking domain service.

This service handles tracking of resource usage for analytics and future billing,
without complex billing calculations. Metrics are sent to Logfire for monitoring.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from src.domain.services.base import BaseDomainService
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.exceptions import EntityNotFoundError


class UsageTrackingService(BaseDomainService):
    """Simplified domain service for usage tracking and analytics.

    Tracks API calls, processing time, storage usage, and other
    resources. Provides analytics summaries and sends metrics to Logfire.
    """

    def __init__(
        self,
        usage_repo: UsageRecordRepository,
        org_repo: OrganizationRepository,
        user_repo: UserRepository,
    ):
        """Initialize usage tracking service.

        Args:
            usage_repo: Repository for usage record operations
            org_repo: Repository for organization operations
            user_repo: Repository for user operations
        """
        super().__init__()
        self.usage_repo = usage_repo
        self.org_repo = org_repo
        self.user_repo = user_repo

    async def track_api_call(
        self,
        user_id: int,
        api_key_id: int,
        endpoint: str,
        method: str,
        response_time_ms: int,
        status_code: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> UsageRecord:
        """Track an API call for usage and analytics.

        Args:
            user_id: User making the call
            api_key_id: API key used
            endpoint: API endpoint called
            method: HTTP method
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created usage record
        """
        # Get user and organization for proper tracking
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")

        # Determine organization ID (assume first organization for simplicity)
        organization_id = user.organization_ids[0] if user.organization_ids else None

        usage_record = UsageRecord(
            id=None,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
            user_id=user_id,
            organization_id=organization_id,
            usage_type=UsageType.API_CALL,
            resource_type="api_endpoint",
            quantity=1.0,
            unit="request",
            period_start=Timestamp.now(),
            period_end=Timestamp.now(),
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Save usage record
        saved_record = await self.usage_repo.create(usage_record)

        # Log to metrics (this will be sent to Logfire)
        self._log_usage_metric(
            "api_call",
            {
                "user_id": user_id,
                "organization_id": organization_id,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
            },
        )

        return saved_record

    async def track_stream_processing(
        self,
        user_id: int,
        stream_id: int,
        processing_minutes: float,
        api_key_id: Optional[int] = None,
    ) -> UsageRecord:
        """Track stream processing usage.

        Args:
            user_id: User who owns the stream
            stream_id: Stream being processed
            processing_minutes: Minutes of processing time
            api_key_id: API key used

        Returns:
            Created usage record
        """
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")

        organization_id = user.organization_ids[0] if user.organization_ids else None

        usage_record = UsageRecord(
            id=None,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
            user_id=user_id,
            organization_id=organization_id,
            usage_type=UsageType.STREAM_PROCESSING,
            resource_id=stream_id,
            resource_type="stream",
            quantity=processing_minutes,
            unit="minutes",
            period_start=Timestamp.now(),
            api_key_id=api_key_id,
        )

        saved_record = await self.usage_repo.create(usage_record)

        # Log to metrics
        self._log_usage_metric(
            "stream_processing",
            {
                "user_id": user_id,
                "organization_id": organization_id,
                "stream_id": stream_id,
                "processing_minutes": processing_minutes,
            },
        )

        return saved_record

    async def track_webhook_delivery(
        self,
        user_id: int,
        webhook_id: int,
        api_key_id: Optional[int] = None,
    ) -> UsageRecord:
        """Track webhook delivery usage.

        Args:
            user_id: User who owns the webhook
            webhook_id: Webhook being delivered
            api_key_id: API key used

        Returns:
            Created usage record
        """
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise EntityNotFoundError(f"User {user_id} not found")

        organization_id = user.organization_ids[0] if user.organization_ids else None

        usage_record = UsageRecord(
            id=None,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
            user_id=user_id,
            organization_id=organization_id,
            usage_type=UsageType.WEBHOOK_DELIVERY,
            resource_id=webhook_id,
            resource_type="webhook",
            quantity=1.0,
            unit="delivery",
            period_start=Timestamp.now(),
            period_end=Timestamp.now(),
            api_key_id=api_key_id,
        )

        saved_record = await self.usage_repo.create(usage_record)

        # Log to metrics
        self._log_usage_metric(
            "webhook_delivery",
            {
                "user_id": user_id,
                "organization_id": organization_id,
                "webhook_id": webhook_id,
            },
        )

        return saved_record

    async def get_usage_summary(
        self,
        organization_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get usage summary for an organization.

        Args:
            organization_id: Organization to get summary for
            start_date: Start of period
            end_date: End of period

        Returns:
            Usage summary with metrics
        """
        # Get usage records in the period
        records = await self.usage_repo.get_by_organization_and_period(
            organization_id, start_date, end_date
        )

        # Aggregate by usage type
        usage_by_type = {}
        for record in records:
            usage_type = record.usage_type.value
            if usage_type not in usage_by_type:
                usage_by_type[usage_type] = {
                    "quantity": 0.0,
                    "unit": record.unit,
                    "count": 0,
                }

            usage_by_type[usage_type]["quantity"] += record.quantity
            usage_by_type[usage_type]["count"] += 1

        return {
            "organization_id": organization_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_records": len(records),
            "usage_by_type": usage_by_type,
        }

    def _log_usage_metric(self, metric_name: str, attributes: Dict[str, Any]) -> None:
        """Log usage metric (to be sent to Logfire).

        Args:
            metric_name: Name of the metric
            attributes: Metric attributes
        """
        # Import metrics collector here to avoid circular imports
        from src.infrastructure.observability.logfire_metrics import MetricsCollector

        # Get global metrics collector instance
        metrics_collector = MetricsCollector()

        # Send appropriate metric based on type
        if metric_name == "api_call":
            metrics_collector.track_api_call_usage(
                user_id=str(attributes["user_id"]),
                organization_id=str(attributes["organization_id"]),
                endpoint=attributes["endpoint"],
                method=attributes["method"],
                status_code=attributes["status_code"],
                response_time_ms=attributes["response_time_ms"],
            )
        elif metric_name == "stream_processing":
            metrics_collector.track_stream_processing_usage(
                user_id=str(attributes["user_id"]),
                organization_id=str(attributes["organization_id"]),
                stream_id=str(attributes["stream_id"]),
                processing_minutes=attributes["processing_minutes"],
            )
        elif metric_name == "webhook_delivery":
            metrics_collector.track_webhook_delivery_usage(
                user_id=str(attributes["user_id"]),
                organization_id=str(attributes["organization_id"]),
                webhook_id=str(attributes["webhook_id"]),
                success=True,  # Default to success, can be enhanced later
            )

        # Also log locally for debugging
        self.logger.info(
            f"Usage metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "attributes": attributes,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
