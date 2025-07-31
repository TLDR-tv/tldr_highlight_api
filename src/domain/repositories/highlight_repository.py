"""Highlight repository protocol."""

from typing import Protocol, List, Optional, Dict

from src.domain.repositories.base import Repository
from src.domain.entities.highlight import Highlight
from src.domain.value_objects.timestamp import Timestamp


class HighlightRepository(Repository[Highlight, int], Protocol):
    """Repository protocol for Highlight entities.

    Extends the base repository with highlight-specific operations.
    """
    
    async def get_by_stream(
        self,
        stream_id: int,
        min_confidence: Optional[float] = None,
        types: Optional[List[str]] = None,
    ) -> List[Highlight]:
        """Get highlights for a stream with optional filters."""
        ...
        
    async def get_by_user(
        self, user_id: int, limit: int = 100, offset: int = 0
    ) -> List[Highlight]:
        """Get all highlights for a user across all streams."""
        ...
        
    async def get_by_confidence_range(
        self, min_score: float, max_score: float, user_id: Optional[int] = None
    ) -> List[Highlight]:
        """Get highlights within a confidence score range."""
        ...
        
    async def get_by_type(
        self, highlight_type: str, user_id: Optional[int] = None, limit: int = 100
    ) -> List[Highlight]:
        """Get highlights by type."""
        ...
        
    async def get_by_tags(
        self, tags: List[str], match_all: bool = False, user_id: Optional[int] = None
    ) -> List[Highlight]:
        """Get highlights by tags (match any or all)."""
        ...
        
    async def get_trending(
        self, time_window: Timestamp, limit: int = 10
    ) -> List[Highlight]:
        """Get trending highlights based on engagement."""
        ...
        
    async def get_statistics(self, user_id: int) -> Dict[str, any]:
        """Get highlight statistics for a user."""
        ...
        
    async def search(
        self, query: str, user_id: Optional[int] = None, limit: int = 100
    ) -> List[Highlight]:
        """Search highlights by title or description."""
        ...
        
    async def bulk_create(self, highlights: List[Highlight]) -> List[Highlight]:
        """Create multiple highlights at once."""
        ...
        
    async def count_by_stream(self, stream_id: int) -> int:
        """Count the total number of highlights for a stream."""
        ...
        
    async def get_by_stream_with_pagination(
        self,
        stream_id: int,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "confidence",
    ) -> Dict[str, any]:
        """Get highlights for a stream with pagination and metadata."""
        ...
