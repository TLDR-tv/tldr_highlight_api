"""Usage tracking workflow - clean Pythonic implementation.

Tracks API usage, processing time, and other metrics for B2B clients.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.domain.entities.usage_record import UsageRecord
from src.domain.repositories import UserRepository, UsageRecordRepository


@dataclass
class UsageTracker:
    """Tracks usage metrics for billing and analytics.
    
    Simple, focused workflow that creates usage records
    without mixing infrastructure concerns.
    """
    
    user_repo: UserRepository
    usage_repo: UsageRecordRepository
    
    async def track_api_call(
        self,
        user_id: int,
        api_key_id: int,
        endpoint: str,
        response_time_ms: int,
    ) -> UsageRecord:
        """Track an API call."""
        # Get user for organization context
        user = await self.user_repo.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Create usage record using domain factory
        record = UsageRecord.for_api_call(
            user_id=user_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            organization_id=user.organization_id,
        )
        
        return await self.usage_repo.save(record)
    
    async def track_stream_minutes(
        self,
        user_id: int,
        stream_id: int,
        minutes: float,
    ) -> UsageRecord:
        """Track stream processing time."""
        user = await self.user_repo.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        record = UsageRecord.for_stream_processing(
            user_id=user_id,
            stream_id=stream_id,
            duration_minutes=minutes,
            organization_id=user.organization_id,
        )
        
        return await self.usage_repo.save(record)
    
    async def track_webhook(
        self,
        user_id: int,
        webhook_id: int,
    ) -> UsageRecord:
        """Track webhook delivery."""
        user = await self.user_repo.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        record = UsageRecord.for_webhook_delivery(
            user_id=user_id,
            webhook_id=webhook_id,
            organization_id=user.organization_id,
        )
        
        return await self.usage_repo.save(record)
    
    async def get_summary(
        self,
        organization_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get usage summary for organization."""
        records = await self.usage_repo.find_by_organization_and_period(
            organization_id, start_date, end_date
        )
        
        # Let domain entity handle aggregation
        summary = UsageRecord.aggregate_by_type(records)
        
        return {
            "organization_id": organization_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "usage": summary,
            "total_records": len(records),
        }
    
    async def get_user_breakdown(
        self,
        user_id: int,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get usage breakdown for a user."""
        end_date = datetime.utcnow()
        start_date = datetime.utcnow() - timedelta(days=days)
        
        records = await self.usage_repo.find_by_user_and_period(
            user_id, start_date, end_date
        )
        
        # Let domain handle daily aggregation
        daily_breakdown = UsageRecord.aggregate_by_day(records)
        
        return {
            "user_id": user_id,
            "period_days": days,
            "daily": daily_breakdown,
            "totals": UsageRecord.calculate_totals(records),
        }


# Simple helper functions instead of a service class

async def track_api_usage(
    user_repo: UserRepository,
    usage_repo: UsageRecordRepository,
    user_id: int,
    api_key_id: int,
    endpoint: str,
    response_time_ms: int,
) -> None:
    """Convenience function to track API usage."""
    tracker = UsageTracker(user_repo, usage_repo)
    await tracker.track_api_call(
        user_id, api_key_id, endpoint, response_time_ms
    )


async def track_stream_usage(
    user_repo: UserRepository,
    usage_repo: UsageRecordRepository,
    user_id: int,
    stream_id: int,
    minutes: float,
) -> None:
    """Convenience function to track stream usage."""
    tracker = UsageTracker(user_repo, usage_repo)
    await tracker.track_stream_minutes(user_id, stream_id, minutes)