"""Webhook repository implementation."""

from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from src.domain.repositories.webhook_repository import WebhookRepository as IWebhookRepository
from src.domain.entities.webhook import Webhook, WebhookEvent, WebhookStatus
from src.domain.value_objects.url import Url
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import EntityNotFoundError
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.webhook import Webhook as WebhookModel
from src.infrastructure.persistence.mappers.webhook_mapper import WebhookMapper


class WebhookRepository(BaseRepository[Webhook, WebhookModel, int], IWebhookRepository):
    """Concrete implementation of WebhookRepository using SQLAlchemy."""
    
    def __init__(self, session):
        """Initialize WebhookRepository with session."""
        super().__init__(
            session=session,
            model_class=WebhookModel,
            mapper=WebhookMapper()
        )
    
    async def get_by_user(self, user_id: int,
                        status: Optional[WebhookStatus] = None) -> List[Webhook]:
        """Get webhooks for a user, optionally filtered by status.
        
        Args:
            user_id: User ID
            status: Optional status filter
            
        Returns:
            List of webhooks for the user
        """
        stmt = select(WebhookModel).where(
            WebhookModel.user_id == user_id
        )
        
        if status:
            stmt = stmt.where(WebhookModel.status == status.value)
        
        stmt = stmt.order_by(WebhookModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_by_event(self, event: WebhookEvent,
                         active_only: bool = True) -> List[Webhook]:
        """Get all webhooks subscribed to an event.
        
        Args:
            event: Webhook event to filter by
            active_only: If True, only return active webhooks
            
        Returns:
            List of webhooks subscribed to the event
        """
        stmt = select(WebhookModel).where(
            func.json_extract(WebhookModel.events, '$').contains(f'"{event.value}"')
        )
        
        if active_only:
            stmt = stmt.where(WebhookModel.status == WebhookStatus.ACTIVE.value)
        
        stmt = stmt.order_by(WebhookModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_by_url(self, url: Url,
                       user_id: Optional[int] = None) -> List[Webhook]:
        """Get webhooks by URL.
        
        Args:
            url: Webhook URL
            user_id: Optional user ID filter
            
        Returns:
            List of webhooks with the specified URL
        """
        stmt = select(WebhookModel).where(
            WebhookModel.url == url.value
        )
        
        if user_id:
            stmt = stmt.where(WebhookModel.user_id == user_id)
        
        stmt = stmt.order_by(WebhookModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_active_for_event(self, event: WebhookEvent,
                                 user_id: int) -> List[Webhook]:
        """Get active webhooks for a specific event and user.
        
        Args:
            event: Webhook event
            user_id: User ID
            
        Returns:
            List of active webhooks for the event and user
        """
        stmt = select(WebhookModel).where(
            and_(
                WebhookModel.user_id == user_id,
                WebhookModel.status == WebhookStatus.ACTIVE.value,
                func.json_extract(WebhookModel.events, '$').contains(f'"{event.value}"')
            )
        ).order_by(WebhookModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_failed_webhooks(self, failure_threshold: int = 5) -> List[Webhook]:
        """Get webhooks with consecutive failures above threshold.
        
        Args:
            failure_threshold: Minimum number of consecutive failures
            
        Returns:
            List of webhooks with failures above threshold
        """
        stmt = select(WebhookModel).where(
            WebhookModel.consecutive_failures >= failure_threshold
        ).order_by(WebhookModel.consecutive_failures.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def count_by_status(self, user_id: Optional[int] = None) -> Dict[str, int]:
        """Get count of webhooks by status.
        
        Args:
            user_id: Optional user ID filter
            
        Returns:
            Dictionary mapping status to count
        """
        stmt = select(
            WebhookModel.status,
            func.count()
        ).group_by(WebhookModel.status)
        
        if user_id:
            stmt = stmt.where(WebhookModel.user_id == user_id)
        
        result = await self.session.execute(stmt)
        counts = {row[0]: row[1] for row in result}
        
        # Ensure all statuses are represented
        for status in WebhookStatus:
            if status.value not in counts:
                counts[status.value] = 0
        
        return counts
    
    async def get_delivery_stats(self, webhook_id: int) -> Dict[str, Any]:
        """Get delivery statistics for a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Dictionary with delivery statistics
        """
        webhook = await self.get(webhook_id)
        if not webhook:
            raise EntityNotFoundError(f"Webhook {webhook_id} not found")
        
        # Calculate additional stats from the domain entity
        stats = {
            'webhook_id': webhook_id,
            'total_deliveries': webhook.total_deliveries,
            'successful_deliveries': webhook.successful_deliveries,
            'failed_deliveries': webhook.total_deliveries - webhook.successful_deliveries,
            'consecutive_failures': webhook.consecutive_failures,
            'success_rate': webhook.success_rate,
            'last_delivery': {
                'delivered_at': webhook.last_delivery.delivered_at.iso_string if webhook.last_delivery else None,
                'status_code': webhook.last_delivery.status_code if webhook.last_delivery else None,
                'response_time_ms': webhook.last_delivery.response_time_ms if webhook.last_delivery else None,
                'was_successful': webhook.last_delivery.is_successful if webhook.last_delivery else None,
                'error_message': webhook.last_delivery.error_message if webhook.last_delivery else None
            } if webhook.last_delivery else None,
            'status': webhook.status.value,
            'is_active': webhook.is_active,
            'subscribed_events': [event.value for event in webhook.events],
            'created_at': webhook.created_at.iso_string,
            'updated_at': webhook.updated_at.iso_string
        }
        
        return stats
    
    async def cleanup_failed_webhooks(self, failure_threshold: int = 10) -> int:
        """Clean up webhooks with too many failures.
        
        Args:
            failure_threshold: Failure threshold for cleanup
            
        Returns:
            Number of webhooks cleaned up
        """
        # Get webhooks that need cleanup
        failed_webhooks = await self.get_failed_webhooks(failure_threshold)
        
        cleanup_count = 0
        for webhook in failed_webhooks:
            if webhook.status != WebhookStatus.FAILED:
                # Deactivate webhook instead of deleting
                deactivated = webhook.deactivate()
                await self.save(deactivated)
                cleanup_count += 1
        
        return cleanup_count
    
    async def reset_failure_count(self, webhook_id: int) -> Optional[Webhook]:
        """Reset consecutive failure count for a webhook.
        
        Args:
            webhook_id: Webhook ID
            
        Returns:
            Updated webhook if found, None otherwise
        """
        webhook = await self.get(webhook_id)
        if not webhook:
            return None
        
        # Create updated webhook with reset failure count
        reset_webhook = Webhook(
            id=webhook.id,
            url=webhook.url,
            user_id=webhook.user_id,
            events=webhook.events.copy(),
            secret=webhook.secret,
            description=webhook.description,
            status=WebhookStatus.ACTIVE,  # Reactivate when resetting
            last_delivery=webhook.last_delivery,
            consecutive_failures=0,  # Reset to 0
            total_deliveries=webhook.total_deliveries,
            successful_deliveries=webhook.successful_deliveries,
            custom_headers=webhook.custom_headers.copy(),
            created_at=webhook.created_at,
            updated_at=Timestamp.now()
        )
        
        return await self.save(reset_webhook)
    
    async def get_webhooks_by_organization(self, organization_id: int,
                                         status: Optional[WebhookStatus] = None) -> List[Webhook]:
        """Get webhooks for an organization.
        
        Args:
            organization_id: Organization ID
            status: Optional status filter
            
        Returns:
            List of webhooks for the organization
        """
        # This would require joining with users to find organization members
        # For now, implementing a basic version
        # In a real implementation, you'd join with organization membership
        
        stmt = select(WebhookModel)
        
        if status:
            stmt = stmt.where(WebhookModel.status == status.value)
        
        stmt = stmt.order_by(WebhookModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_webhooks_by_success_rate(self, min_success_rate: float = 0.8,
                                         min_deliveries: int = 10) -> List[Webhook]:
        """Get webhooks with high success rate.
        
        Args:
            min_success_rate: Minimum success rate (0.0 to 1.0)
            min_deliveries: Minimum number of deliveries to qualify
            
        Returns:
            List of high-performing webhooks
        """
        # Calculate success rate in SQL
        stmt = select(WebhookModel).where(
            and_(
                WebhookModel.total_deliveries >= min_deliveries,
                (WebhookModel.successful_deliveries / WebhookModel.total_deliveries) >= min_success_rate
            )
        ).order_by(
            (WebhookModel.successful_deliveries / WebhookModel.total_deliveries).desc()
        )
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def bulk_update_status(self, webhook_ids: List[int], 
                               new_status: WebhookStatus) -> int:
        """Bulk update status for multiple webhooks.
        
        Args:
            webhook_ids: List of webhook IDs to update
            new_status: New status to set
            
        Returns:
            Number of webhooks updated
        """
        if not webhook_ids:
            return 0
        
        updated_count = 0
        for webhook_id in webhook_ids:
            webhook = await self.get(webhook_id)
            if webhook and webhook.status != new_status:
                if new_status == WebhookStatus.ACTIVE:
                    updated = webhook.activate()
                elif new_status == WebhookStatus.INACTIVE:
                    updated = webhook.deactivate()
                else:  # FAILED
                    updated = Webhook(
                        id=webhook.id,
                        url=webhook.url,
                        user_id=webhook.user_id,
                        events=webhook.events.copy(),
                        secret=webhook.secret,
                        description=webhook.description,
                        status=WebhookStatus.FAILED,
                        last_delivery=webhook.last_delivery,
                        consecutive_failures=webhook.consecutive_failures,
                        total_deliveries=webhook.total_deliveries,
                        successful_deliveries=webhook.successful_deliveries,
                        custom_headers=webhook.custom_headers.copy(),
                        created_at=webhook.created_at,
                        updated_at=Timestamp.now()
                    )
                
                await self.save(updated)
                updated_count += 1
        
        return updated_count
    
    async def get_webhook_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get webhook usage analytics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with usage analytics
        """
        # Get total webhook count by status
        status_counts = await self.count_by_status()
        
        # Calculate average success rate across all webhooks
        stmt = select(
            func.avg(WebhookModel.successful_deliveries / func.nullif(WebhookModel.total_deliveries, 0)).label('avg_success_rate'),
            func.sum(WebhookModel.total_deliveries).label('total_deliveries'),
            func.sum(WebhookModel.successful_deliveries).label('successful_deliveries'),
            func.count().label('total_webhooks')
        ).where(WebhookModel.total_deliveries > 0)
        
        result = await self.session.execute(stmt)
        row = result.one()
        
        avg_success_rate = row.avg_success_rate or 0.0
        total_deliveries = row.total_deliveries or 0
        successful_deliveries = row.successful_deliveries or 0
        total_webhooks = row.total_webhooks or 0
        
        # Get event subscription counts
        stmt = select(WebhookModel.events).where(
            WebhookModel.status == WebhookStatus.ACTIVE.value
        )
        result = await self.session.execute(stmt)
        
        event_counts = {}
        for row in result:
            if row[0]:  # events field is not null
                events = row[0] if isinstance(row[0], list) else []
                for event in events:
                    event_counts[event] = event_counts.get(event, 0) + 1
        
        return {
            'total_webhooks': sum(status_counts.values()),
            'by_status': status_counts,
            'average_success_rate': float(avg_success_rate),
            'total_deliveries': total_deliveries,
            'successful_deliveries': successful_deliveries,
            'failed_deliveries': total_deliveries - successful_deliveries,
            'event_subscriptions': event_counts,
            'period_days': days
        }