"""Highlight repository protocol."""

from typing import Protocol, List, Optional, Dict
from abc import abstractmethod

from src.domain.repositories.base import Repository
from src.domain.entities.highlight import Highlight, HighlightType
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.timestamp import Timestamp


class HighlightRepository(Repository[Highlight, int], Protocol):
    """Repository protocol for Highlight entities.
    
    Extends the base repository with highlight-specific operations.
    """
    
    @abstractmethod
    async def get_by_stream(self, stream_id: int,
                          min_confidence: Optional[float] = None,
                          types: Optional[List[HighlightType]] = None) -> List[Highlight]:
        """Get highlights for a stream with optional filters."""
        ...
    
    @abstractmethod
    async def get_by_user(self, user_id: int,
                        limit: int = 100,
                        offset: int = 0) -> List[Highlight]:
        """Get all highlights for a user across all streams."""
        ...
    
    @abstractmethod
    async def get_by_confidence_range(self, min_score: float, max_score: float,
                                    user_id: Optional[int] = None) -> List[Highlight]:
        """Get highlights within a confidence score range."""
        ...
    
    @abstractmethod
    async def get_by_type(self, highlight_type: HighlightType,
                        user_id: Optional[int] = None,
                        limit: int = 100) -> List[Highlight]:
        """Get highlights by type."""
        ...
    
    @abstractmethod
    async def get_by_tags(self, tags: List[str],
                        match_all: bool = False,
                        user_id: Optional[int] = None) -> List[Highlight]:
        """Get highlights by tags (match any or all)."""
        ...
    
    @abstractmethod
    async def get_trending(self, time_window: Timestamp,
                         limit: int = 10) -> List[Highlight]:
        """Get trending highlights based on engagement."""
        ...
    
    @abstractmethod
    async def get_statistics(self, user_id: int) -> Dict[str, any]:
        """Get highlight statistics for a user."""
        ...
    
    @abstractmethod
    async def search(self, query: str,
                   user_id: Optional[int] = None,
                   limit: int = 100) -> List[Highlight]:
        """Search highlights by title or description."""
        ...
    
    @abstractmethod
    async def bulk_create(self, highlights: List[Highlight]) -> List[Highlight]:
        """Create multiple highlights at once."""
        ...