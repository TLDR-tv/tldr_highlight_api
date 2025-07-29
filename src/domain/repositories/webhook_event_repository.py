"""Webhook event repository interface."""

from typing import Optional, List, Protocol
from datetime import datetime

from src.domain.entities.webhook_event import WebhookEvent, WebhookEventStatus, WebhookEventType
from src.domain.value_objects.timestamp import Timestamp


class WebhookEventRepository(Protocol):
    """Repository interface for webhook event operations."""
    
    async def save(self, event: WebhookEvent) -> WebhookEvent:
        """Save a webhook event.
        
        Args:
            event: Webhook event to save
            
        Returns:
            Saved webhook event with ID
        """
        ...
    
    async def get(self, event_id: int) -> Optional[WebhookEvent]:
        """Get a webhook event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            Webhook event if found, None otherwise
        """
        ...
    
    async def get_by_external_id(self, external_event_id: str, platform: str) -> Optional[WebhookEvent]:
        """Get a webhook event by external event ID and platform.
        
        Args:
            external_event_id: External event ID from the platform
            platform: Platform name
            
        Returns:
            Webhook event if found, None otherwise
        """
        ...
    
    async def get_recent_events(
        self,
        limit: int = 100,
        status: Optional[WebhookEventStatus] = None,
        event_type: Optional[WebhookEventType] = None,
        platform: Optional[str] = None
    ) -> List[WebhookEvent]:
        """Get recent webhook events with optional filters.
        
        Args:
            limit: Maximum number of events to return
            status: Filter by status
            event_type: Filter by event type
            platform: Filter by platform
            
        Returns:
            List of webhook events
        """
        ...
    
    async def get_failed_events_for_retry(
        self,
        max_retry_count: int = 3,
        limit: int = 100
    ) -> List[WebhookEvent]:
        """Get failed events that can be retried.
        
        Args:
            max_retry_count: Maximum retry count
            limit: Maximum number of events to return
            
        Returns:
            List of failed webhook events eligible for retry
        """
        ...
    
    async def exists_by_external_id(self, external_event_id: str, platform: str) -> bool:
        """Check if a webhook event exists by external ID and platform.
        
        Args:
            external_event_id: External event ID
            platform: Platform name
            
        Returns:
            True if exists, False otherwise
        """
        ...
    
    async def get_events_by_stream(self, stream_id: int) -> List[WebhookEvent]:
        """Get all webhook events associated with a stream.
        
        Args:
            stream_id: Stream ID
            
        Returns:
            List of webhook events for the stream
        """
        ...
    
    async def cleanup_old_events(self, days: int = 30) -> int:
        """Clean up old processed webhook events.
        
        Args:
            days: Number of days to keep events
            
        Returns:
            Number of events deleted
        """
        ...