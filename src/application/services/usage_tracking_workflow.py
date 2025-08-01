"""Usage tracking workflow application service.

This application service handles tracking of resource usage for analytics
and future billing, coordinating with metrics infrastructure.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logfire

from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import EntityNotFoundError
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository


class UsageTrackingWorkflow:
    """Application service for usage tracking and analytics.
    
    Tracks API calls, processing time, storage usage, and other
    resources. Provides analytics summaries and coordinates with
    metrics infrastructure.
    """
    
    def __init__(
        self,
        usage_repo: UsageRecordRepository,
        org_repo: OrganizationRepository,
        user_repo: UserRepository,
    ):
        """Initialize usage tracking workflow.
        
        Args:
            usage_repo: Repository for usage record operations
            org_repo: Repository for organization operations
            user_repo: Repository for user operations
        """
        self.usage_repo = usage_repo
        self.org_repo = org_repo
        self.user_repo = user_repo
        self.logger = logfire.get_logger(__name__)
    
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
        
        # Determine organization ID
        organization_id = None
        if user.organization_ids:
            organization_id = user.organization_ids[0]
        
        # Create usage record using factory method
        usage_record = UsageRecord.for_api_call(
            user_id=user_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            ip_address=ip_address,
            organization_id=organization_id,
            rate=None,  # No billing rates for first client
        )
        
        # Add metadata
        usage_record.user_agent = user_agent
        
        # Save usage record
        saved_record = await self.usage_repo.save(usage_record)
        
        # Send metrics to infrastructure
        await self._send_api_metrics(
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
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
        
        # Create usage record using factory method
        usage_record = UsageRecord.for_stream_processing(
            user_id=user_id,
            stream_id=stream_id,
            duration_minutes=processing_minutes,
            organization_id=organization_id,
            rate=None,  # No billing rates for first client
        )
        
        if api_key_id:
            usage_record.api_key_id = api_key_id
        
        saved_record = await self.usage_repo.save(usage_record)
        
        # Send metrics to infrastructure
        await self._send_stream_metrics(
            user_id=user_id,
            organization_id=organization_id,
            stream_id=stream_id,
            processing_minutes=processing_minutes,
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
            billable=True,
            rate=None,  # No billing rates for first client
        )
        
        saved_record = await self.usage_repo.save(usage_record)
        
        # Send metrics to infrastructure
        await self._send_webhook_metrics(
            user_id=user_id,
            organization_id=organization_id,
            webhook_id=webhook_id,
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
    
    async def get_user_usage_breakdown(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get detailed usage breakdown for a user.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Detailed usage breakdown
        """
        end_date = Timestamp.now()
        start_date = end_date.subtract_days(days)
        
        # Get all usage records for user
        records = await self.usage_repo.get_by_user_and_period(
            user_id, start_date.value, end_date.value
        )
        
        # Group by day
        daily_usage = {}
        for record in records:
            day_key = record.period_start.value.date().isoformat()
            if day_key not in daily_usage:
                daily_usage[day_key] = {
                    "api_calls": 0,
                    "stream_minutes": 0.0,
                    "webhook_deliveries": 0,
                }
            
            if record.usage_type == UsageType.API_CALL:
                daily_usage[day_key]["api_calls"] += 1
            elif record.usage_type == UsageType.STREAM_PROCESSING:
                daily_usage[day_key]["stream_minutes"] += record.quantity
            elif record.usage_type == UsageType.WEBHOOK_DELIVERY:
                daily_usage[day_key]["webhook_deliveries"] += 1
        
        # Calculate totals
        totals = {
            "api_calls": sum(d["api_calls"] for d in daily_usage.values()),
            "stream_minutes": sum(d["stream_minutes"] for d in daily_usage.values()),
            "webhook_deliveries": sum(d["webhook_deliveries"] for d in daily_usage.values()),
        }
        
        return {
            "user_id": user_id,
            "period_days": days,
            "daily_usage": daily_usage,
            "totals": totals,
        }
    
    async def complete_usage_period(
        self,
        usage_record_id: int,
        final_quantity: Optional[float] = None
    ) -> UsageRecord:
        """Complete a usage period (e.g., end of stream processing).
        
        Args:
            usage_record_id: Usage record ID
            final_quantity: Final quantity if different from initial
            
        Returns:
            Updated usage record
        """
        record = await self.usage_repo.get(usage_record_id)
        if not record:
            raise EntityNotFoundError(f"Usage record {usage_record_id} not found")
        
        # Complete the record
        completed_record = record.complete(quantity=final_quantity)
        
        # Save updated record
        return await self.usage_repo.save(completed_record)
    
    # Private helper methods
    
    async def _send_api_metrics(
        self,
        user_id: int,
        organization_id: Optional[int],
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
    ) -> None:
        """Send API metrics to infrastructure."""
        # Import metrics collector to avoid circular imports
        from src.infrastructure.observability.logfire_metrics import MetricsCollector
        
        metrics_collector = MetricsCollector()
        
        metrics_collector.track_api_call_usage(
            user_id=str(user_id),
            organization_id=str(organization_id) if organization_id else "none",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
        )
        
        self.logger.info(
            "API usage tracked",
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
        )
    
    async def _send_stream_metrics(
        self,
        user_id: int,
        organization_id: Optional[int],
        stream_id: int,
        processing_minutes: float,
    ) -> None:
        """Send stream processing metrics to infrastructure."""
        from src.infrastructure.observability.logfire_metrics import MetricsCollector
        
        metrics_collector = MetricsCollector()
        
        metrics_collector.track_stream_processing_usage(
            user_id=str(user_id),
            organization_id=str(organization_id) if organization_id else "none",
            stream_id=str(stream_id),
            processing_minutes=processing_minutes,
        )
        
        self.logger.info(
            "Stream usage tracked",
            user_id=user_id,
            organization_id=organization_id,
            stream_id=stream_id,
            processing_minutes=processing_minutes,
        )
    
    async def _send_webhook_metrics(
        self,
        user_id: int,
        organization_id: Optional[int],
        webhook_id: int,
    ) -> None:
        """Send webhook delivery metrics to infrastructure."""
        from src.infrastructure.observability.logfire_metrics import MetricsCollector
        
        metrics_collector = MetricsCollector()
        
        metrics_collector.track_webhook_delivery_usage(
            user_id=str(user_id),
            organization_id=str(organization_id) if organization_id else "none",
            webhook_id=str(webhook_id),
            success=True,  # Tracking attempt, not result
        )
        
        self.logger.info(
            "Webhook usage tracked",
            user_id=user_id,
            organization_id=organization_id,
            webhook_id=webhook_id,
        )