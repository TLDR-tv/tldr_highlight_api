"""Stream repository protocol."""

from typing import Protocol, Optional, List

from src.domain.repositories.base import Repository
from src.domain.entities.stream import Stream, StreamStatus, StreamPlatform
from src.domain.value_objects.timestamp import Timestamp


class StreamRepository(Repository[Stream, int], Protocol):
    """Repository protocol for Stream entities.

    Extends the base repository with stream-specific operations.
    """

    async def get_by_user(
        self,
        user_id: int,
        status: Optional[StreamStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Stream]:
        """Get streams for a user, optionally filtered by status."""
        ...

    
    async def get_by_organization(
        self,
        organization_id: int,
        status: Optional[StreamStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Stream]:
        """Get streams for an organization."""
        ...

    
    async def get_active_streams(self) -> List[Stream]:
        """Get all streams currently being processed."""
        ...

    
    async def get_by_platform(
        self, platform: StreamPlatform, limit: int = 100, offset: int = 0
    ) -> List[Stream]:
        """Get streams by platform."""
        ...

    
    async def get_by_date_range(
        self, start: Timestamp, end: Timestamp, user_id: Optional[int] = None
    ) -> List[Stream]:
        """Get streams within a date range."""
        ...

    
    async def count_by_status(
        self, status: StreamStatus, user_id: Optional[int] = None
    ) -> int:
        """Count streams by status."""
        ...

    
    async def get_with_highlights(self, stream_id: int) -> Optional[Stream]:
        """Get stream with its highlights loaded."""
        ...

    
    async def get_processing_stats(self, user_id: int) -> dict:
        """Get processing statistics for a user."""
        ...

    
    async def cleanup_old_streams(self, older_than: Timestamp) -> int:
        """Clean up old completed/failed streams."""
        ...
