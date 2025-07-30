"""Usage tracking domain service.

This service handles tracking of resource usage for billing,
analytics, and quota enforcement.
"""

from typing import Dict, Any, Optional
from datetime import timedelta
from decimal import Decimal

from src.domain.services.base import BaseDomainService
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.entities.organization import Organization, PlanType
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.usage_record_repository import UsageRecordRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.exceptions import EntityNotFoundError, QuotaExceededError


class UsageTrackingService(BaseDomainService):
    """Domain service for usage tracking and billing calculations.

    Tracks API calls, processing time, storage usage, and other
    billable resources. Provides analytics and billing summaries.
    """

    # Default rates per plan (cost per unit)
    PLAN_RATES = {
        PlanType.STARTER: {
            UsageType.STREAM_PROCESSING: Decimal("0.10"),  # per minute
            UsageType.BATCH_PROCESSING: Decimal("0.08"),  # per minute
            UsageType.API_CALL: Decimal("0.0001"),  # per call
            UsageType.WEBHOOK_DELIVERY: Decimal("0.001"),  # per delivery
            UsageType.STORAGE: Decimal("0.10"),  # per GB per month
            UsageType.BANDWIDTH: Decimal("0.15"),  # per GB
        },
        PlanType.PROFESSIONAL: {
            UsageType.STREAM_PROCESSING: Decimal("0.08"),
            UsageType.BATCH_PROCESSING: Decimal("0.06"),
            UsageType.API_CALL: Decimal("0.00008"),
            UsageType.WEBHOOK_DELIVERY: Decimal("0.0008"),
            UsageType.STORAGE: Decimal("0.08"),
            UsageType.BANDWIDTH: Decimal("0.12"),
        },
        PlanType.ENTERPRISE: {
            UsageType.STREAM_PROCESSING: Decimal("0.05"),
            UsageType.BATCH_PROCESSING: Decimal("0.04"),
            UsageType.API_CALL: Decimal("0.00005"),
            UsageType.WEBHOOK_DELIVERY: Decimal("0.0005"),
            UsageType.STORAGE: Decimal("0.05"),
            UsageType.BANDWIDTH: Decimal("0.08"),
        },
        PlanType.CUSTOM: {
            # Custom rates negotiated per customer
            UsageType.STREAM_PROCESSING: Decimal("0.03"),
            UsageType.BATCH_PROCESSING: Decimal("0.025"),
            UsageType.API_CALL: Decimal("0.00003"),
            UsageType.WEBHOOK_DELIVERY: Decimal("0.0003"),
            UsageType.STORAGE: Decimal("0.03"),
            UsageType.BANDWIDTH: Decimal("0.05"),
        },
    }

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
        # Get user's organization for billing
        org = await self._get_user_organization(user_id)

        # Determine if call is billable (successful calls only)
        billable = 200 <= status_code < 300

        # Get rate for API calls
        rate = None
        if billable and org:
            rate = float(self.PLAN_RATES[org.plan_type][UsageType.API_CALL])

        # Create usage record
        usage_record = UsageRecord(
            id=None,
            user_id=user_id,
            organization_id=org.id if org else None,
            usage_type=UsageType.API_CALL,
            resource_type=f"{method} {endpoint}",
            quantity=1,
            unit="request",
            period_start=Timestamp.now(),
            period_end=Timestamp.now(),
            billable=billable,
            rate=rate,
            total_cost=rate if rate else 0,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={"response_time_ms": response_time_ms, "status_code": status_code},
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

        # Save usage record
        saved_record = await self.usage_repo.save(usage_record)

        # Check API rate limit
        await self._check_api_rate_limit(user_id, org)

        return saved_record

    async def track_stream_processing(
        self,
        user_id: int,
        stream_id: int,
        duration_minutes: float,
        highlights_generated: int = 0,
    ) -> UsageRecord:
        """Track stream processing usage.

        Args:
            user_id: User who owns the stream
            stream_id: Stream ID
            duration_minutes: Processing duration in minutes
            highlights_generated: Number of highlights generated

        Returns:
            Created usage record
        """
        # Get user's organization
        org = await self._get_user_organization(user_id)

        # Check monthly processing quota
        if org:
            await self._check_processing_quota(user_id, org, duration_minutes)

        # Get rate
        rate = None
        if org:
            rate = float(self.PLAN_RATES[org.plan_type][UsageType.STREAM_PROCESSING])

        # Calculate cost
        total_cost = float(duration_minutes * rate) if rate else 0

        # Create usage record
        usage_record = UsageRecord(
            id=None,
            user_id=user_id,
            organization_id=org.id if org else None,
            usage_type=UsageType.STREAM_PROCESSING,
            resource_id=stream_id,
            resource_type="stream",
            quantity=duration_minutes,
            unit="minutes",
            period_start=Timestamp.now().subtract_minutes(int(duration_minutes)),
            period_end=Timestamp.now(),
            billable=True,
            rate=rate,
            total_cost=total_cost,
            metadata={"highlights_generated": highlights_generated},
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

        # Save usage record
        saved_record = await self.usage_repo.save(usage_record)

        self.logger.info(
            f"Tracked {duration_minutes} minutes of stream processing for user {user_id}"
        )

        return saved_record

    async def track_batch_processing(
        self,
        user_id: int,
        batch_id: int,
        total_items: int,
        successful_items: int,
        processing_minutes: float,
    ) -> UsageRecord:
        """Track batch processing usage.

        Args:
            user_id: User who owns the batch
            batch_id: Batch ID
            total_items: Total items in batch
            successful_items: Number of successful items
            processing_minutes: Total processing time in minutes

        Returns:
            Created usage record
        """
        # Get user's organization
        org = await self._get_user_organization(user_id)

        # Get rate
        rate = None
        if org:
            rate = float(self.PLAN_RATES[org.plan_type][UsageType.BATCH_PROCESSING])

        # Calculate cost (based on successful items * minutes)
        billable_minutes = (
            successful_items * (processing_minutes / total_items)
            if total_items > 0
            else 0
        )
        total_cost = float(billable_minutes * rate) if rate else 0

        # Create usage record
        usage_record = UsageRecord(
            id=None,
            user_id=user_id,
            organization_id=org.id if org else None,
            usage_type=UsageType.BATCH_PROCESSING,
            resource_id=batch_id,
            resource_type="batch",
            quantity=billable_minutes,
            unit="minutes",
            period_start=Timestamp.now().subtract_minutes(int(processing_minutes)),
            period_end=Timestamp.now(),
            billable=True,
            rate=rate,
            total_cost=total_cost,
            metadata={
                "total_items": total_items,
                "successful_items": successful_items,
                "failed_items": total_items - successful_items,
            },
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

        # Save usage record
        saved_record = await self.usage_repo.save(usage_record)

        self.logger.info(
            f"Tracked batch processing for user {user_id}: {successful_items}/{total_items} items"
        )

        return saved_record

    async def track_webhook_delivery(
        self,
        user_id: int,
        webhook_id: int,
        event_type: str,
        success: bool,
        response_time_ms: int,
    ) -> UsageRecord:
        """Track webhook delivery attempt.

        Args:
            user_id: User who owns the webhook
            webhook_id: Webhook ID
            event_type: Type of event delivered
            success: Whether delivery was successful
            response_time_ms: Response time in milliseconds

        Returns:
            Created usage record
        """
        # Get user's organization
        org = await self._get_user_organization(user_id)

        # Only successful deliveries are billable
        billable = success

        # Get rate
        rate = None
        if billable and org:
            rate = float(self.PLAN_RATES[org.plan_type][UsageType.WEBHOOK_DELIVERY])

        # Create usage record
        usage_record = UsageRecord(
            id=None,
            user_id=user_id,
            organization_id=org.id if org else None,
            usage_type=UsageType.WEBHOOK_DELIVERY,
            resource_id=webhook_id,
            resource_type="webhook",
            quantity=1,
            unit="delivery",
            period_start=Timestamp.now(),
            period_end=Timestamp.now(),
            billable=billable,
            rate=rate,
            total_cost=rate if rate and billable else 0,
            metadata={
                "event_type": event_type,
                "success": success,
                "response_time_ms": response_time_ms,
            },
            created_at=Timestamp.now(),
            updated_at=Timestamp.now(),
        )

        # Save usage record
        return await self.usage_repo.save(usage_record)

    async def get_usage_summary(
        self, user_id: int, start_date: Timestamp, end_date: Timestamp
    ) -> Dict[str, Any]:
        """Get usage summary for a user over a period.

        Args:
            user_id: User ID
            start_date: Start of period
            end_date: End of period

        Returns:
            Dictionary with usage summary
        """
        # Get user's organization
        org = await self._get_user_organization(user_id)

        # Get usage by type
        usage_by_type = await self.usage_repo.aggregate_by_type(
            user_id, start_date, end_date
        )

        # Calculate totals
        total_cost = sum(data["total_cost"] for data in usage_by_type.values())

        # Get top resources by usage
        all_records = await self.usage_repo.get_by_user(
            user_id, start=start_date, end=end_date
        )

        # Group by resource
        resource_usage = {}
        for record in all_records:
            key = f"{record.resource_type}:{record.resource_id or 'N/A'}"
            if key not in resource_usage:
                resource_usage[key] = {"quantity": 0, "cost": 0, "count": 0}
            resource_usage[key]["quantity"] += record.quantity
            resource_usage[key]["cost"] += record.total_cost or 0
            resource_usage[key]["count"] += 1

        # Sort by cost
        top_resources = sorted(
            resource_usage.items(), key=lambda x: x[1]["cost"], reverse=True
        )[:10]

        return {
            "user_id": user_id,
            "organization_id": org.id if org else None,
            "period": {"start": start_date.iso_string, "end": end_date.iso_string},
            "usage_by_type": {
                usage_type.value: data for usage_type, data in usage_by_type.items()
            },
            "total_cost": total_cost,
            "top_resources": [
                {"resource": resource, **data} for resource, data in top_resources
            ],
            "plan_type": org.plan_type.value if org else "free",
        }

    async def generate_invoice(
        self,
        organization_id: int,
        billing_period_start: Timestamp,
        billing_period_end: Timestamp,
    ) -> Dict[str, Any]:
        """Generate invoice for an organization.

        Args:
            organization_id: Organization ID
            billing_period_start: Start of billing period
            billing_period_end: End of billing period

        Returns:
            Invoice data
        """
        # Get organization
        org = await self.org_repo.get(organization_id)
        if not org:
            raise EntityNotFoundError(f"Organization {organization_id} not found")

        # Get all billable usage for organization members
        all_usage = []
        member_ids = [org.owner_id] + org.member_ids

        for member_id in member_ids:
            member_usage = await self.usage_repo.get_billable_usage(
                member_id, billing_period_start, billing_period_end
            )
            all_usage.extend(member_usage)

        # Group by usage type
        usage_by_type = {}
        for record in all_usage:
            usage_type = record.usage_type.value
            if usage_type not in usage_by_type:
                usage_by_type[usage_type] = {
                    "quantity": 0,
                    "unit": record.unit,
                    "rate": record.rate or 0,
                    "cost": 0,
                    "items": [],
                }

            usage_by_type[usage_type]["quantity"] += record.quantity
            usage_by_type[usage_type]["cost"] += record.total_cost or 0
            usage_by_type[usage_type]["items"].append(
                {
                    "date": record.period_start.iso_string,
                    "quantity": record.quantity,
                    "cost": record.total_cost or 0,
                    "resource": record.resource_type,
                }
            )

        # Calculate totals
        subtotal = sum(data["cost"] for data in usage_by_type.values())

        # Apply any discounts or credits
        discount = 0  # Could be based on plan or promotions
        credits = 0  # Could be from account credits

        total = subtotal - discount - credits

        return {
            "invoice_id": f"INV-{organization_id}-{billing_period_start.value.strftime('%Y%m')}",
            "organization": {
                "id": org.id,
                "name": org.name.value,
                "plan": org.plan_type.value,
            },
            "billing_period": {
                "start": billing_period_start.iso_string,
                "end": billing_period_end.iso_string,
            },
            "usage_details": usage_by_type,
            "financial_summary": {
                "subtotal": subtotal,
                "discount": discount,
                "credits": credits,
                "total": total,
                "currency": "USD",
            },
            "generated_at": Timestamp.now().iso_string,
        }

    async def get_usage_trends(
        self, user_id: int, days: int = 30, granularity: str = "daily"
    ) -> Dict[str, Any]:
        """Get usage trends over time.

        Args:
            user_id: User ID
            days: Number of days to analyze
            granularity: 'daily', 'weekly', or 'monthly'

        Returns:
            Dictionary with usage trends
        """
        end_date = Timestamp.now()
        start_date = end_date.subtract_days(days)

        # Get all usage records
        records = await self.usage_repo.get_by_user(
            user_id, start=start_date, end=end_date
        )

        # Group by time period
        time_buckets = {}

        for record in records:
            # Determine bucket based on granularity
            if granularity == "daily":
                bucket = record.period_start.value.date().isoformat()
            elif granularity == "weekly":
                # Get week start
                week_start = record.period_start.value.date() - timedelta(
                    days=record.period_start.value.weekday()
                )
                bucket = week_start.isoformat()
            else:  # monthly
                bucket = record.period_start.value.strftime("%Y-%m")

            if bucket not in time_buckets:
                time_buckets[bucket] = {
                    "api_calls": 0,
                    "processing_minutes": 0,
                    "webhook_deliveries": 0,
                    "total_cost": 0,
                }

            # Add to appropriate metric
            if record.usage_type == UsageType.API_CALL:
                time_buckets[bucket]["api_calls"] += record.quantity
            elif record.usage_type == UsageType.STREAM_PROCESSING:
                time_buckets[bucket]["processing_minutes"] += record.quantity
            elif record.usage_type == UsageType.WEBHOOK_DELIVERY:
                time_buckets[bucket]["webhook_deliveries"] += record.quantity

            time_buckets[bucket]["total_cost"] += record.total_cost or 0

        # Sort by date
        sorted_buckets = sorted(time_buckets.items())

        return {
            "user_id": user_id,
            "period": {
                "start": start_date.iso_string,
                "end": end_date.iso_string,
                "days": days,
            },
            "granularity": granularity,
            "trends": [
                {"period": period, **metrics} for period, metrics in sorted_buckets
            ],
        }

    async def estimate_monthly_cost(
        self, user_id: int, lookback_days: int = 7
    ) -> Dict[str, Any]:
        """Estimate monthly cost based on recent usage.

        Args:
            user_id: User ID
            lookback_days: Days to look back for estimation

        Returns:
            Dictionary with cost estimate
        """
        # Get recent usage
        end_date = Timestamp.now()
        start_date = end_date.subtract_days(lookback_days)

        summary = await self.get_usage_summary(user_id, start_date, end_date)

        # Calculate daily average
        daily_average = summary["total_cost"] / lookback_days

        # Estimate monthly (30 days)
        estimated_monthly = daily_average * 30

        # Get current month's actual usage
        month_start = Timestamp.now().start_of_month()
        current_month_summary = await self.get_usage_summary(
            user_id, month_start, end_date
        )

        return {
            "user_id": user_id,
            "estimation_basis": {
                "lookback_days": lookback_days,
                "daily_average": daily_average,
            },
            "estimated_monthly_cost": estimated_monthly,
            "current_month_actual": current_month_summary["total_cost"],
            "projected_overrun": max(
                0, estimated_monthly - current_month_summary["total_cost"]
            ),
        }

    # Private helper methods

    async def _get_user_organization(self, user_id: int) -> Optional[Organization]:
        """Get the organization for a user."""
        orgs = await self.org_repo.get_by_owner(user_id)
        if orgs:
            return orgs[0]

        # Check if user is a member
        member_orgs = await self.org_repo.get_by_member(user_id)
        return member_orgs[0] if member_orgs else None

    async def _check_api_rate_limit(self, user_id: int, org: Optional[Organization]):
        """Check if user has exceeded API rate limit."""
        if not org:
            return  # Free users have basic rate limiting elsewhere

        # Get API calls in last minute
        one_minute_ago = Timestamp.now().subtract_minutes(1)

        recent_calls = await self.usage_repo.get_by_user(
            user_id,
            usage_type=UsageType.API_CALL,
            start=one_minute_ago,
            end=Timestamp.now(),
        )

        call_count = len(recent_calls)
        limit = org.plan_limits.api_rate_limit_per_minute

        if call_count >= limit:
            raise QuotaExceededError(
                f"API rate limit exceeded: {call_count}/{limit} calls per minute"
            )

    async def _check_processing_quota(
        self, user_id: int, org: Organization, additional_minutes: float
    ):
        """Check if user has enough processing quota."""
        # Get current month's usage
        month_start = Timestamp.now().start_of_month()
        month_end = Timestamp.now().end_of_month()

        monthly_usage = await self.usage_repo.calculate_usage_totals(
            user_id, UsageType.STREAM_PROCESSING, month_start, month_end
        )

        current_usage = monthly_usage.get("total_quantity", 0)
        limit = org.plan_limits.monthly_processing_minutes

        if current_usage + additional_minutes > limit:
            raise QuotaExceededError(
                f"Monthly processing quota would be exceeded: "
                f"{current_usage + additional_minutes:.1f}/{limit} minutes"
            )
