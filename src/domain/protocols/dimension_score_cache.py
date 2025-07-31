"""Domain protocol for dimension score caching.

This protocol defines the interface for caching dimension scores
to improve performance and reduce redundant AI evaluations.
"""

from typing import Protocol, Optional, Dict, List
from datetime import timedelta

from src.domain.value_objects.dimension_score import DimensionScore


class DimensionScoreCache(Protocol):
    """Protocol for caching dimension evaluation results.

    This interface allows the domain layer to remain agnostic
    about the specific caching implementation while providing
    performance benefits through score caching.
    """

    async def get(self, cache_key: str) -> Optional[DimensionScore]:
        """Retrieve a cached dimension score.

        Args:
            cache_key: Unique key for the cached score

        Returns:
            Cached DimensionScore if found, None otherwise
        """
        ...

    async def get_batch(
        self, cache_keys: List[str]
    ) -> Dict[str, Optional[DimensionScore]]:
        """Retrieve multiple cached scores at once.

        Args:
            cache_keys: List of cache keys to retrieve

        Returns:
            Dictionary mapping cache keys to scores (None if not found)
        """
        ...

    async def set(
        self, cache_key: str, score: DimensionScore, ttl: Optional[int] = None
    ) -> None:
        """Cache a dimension score.

        Args:
            cache_key: Unique key for the score
            score: The dimension score to cache
            ttl: Time-to-live in seconds (None for default)
        """
        ...

    async def set_batch(
        self, scores: Dict[str, DimensionScore], ttl: Optional[int] = None
    ) -> None:
        """Cache multiple dimension scores at once.

        Args:
            scores: Dictionary mapping cache keys to scores
            ttl: Time-to-live in seconds (None for default)
        """
        ...

    async def invalidate(self, cache_key: str) -> None:
        """Invalidate a specific cached score.

        Args:
            cache_key: Key of the score to invalidate
        """
        ...

    async def invalidate_by_pattern(self, pattern: str) -> None:
        """Invalidate all cached scores matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "dimension_set:123:*")
        """
        ...

    async def invalidate_by_dimension(self, dimension_id: str) -> None:
        """Invalidate all cached scores for a specific dimension.

        Args:
            dimension_id: ID of the dimension to invalidate
        """
        ...

    async def invalidate_by_dimension_set(self, dimension_set_id: int) -> None:
        """Invalidate all cached scores for a dimension set.

        Args:
            dimension_set_id: ID of the dimension set to invalidate
        """
        ...

    async def exists(self, cache_key: str) -> bool:
        """Check if a score exists in the cache.

        Args:
            cache_key: Key to check

        Returns:
            True if the key exists in cache
        """
        ...

    async def get_ttl(self, cache_key: str) -> Optional[int]:
        """Get remaining TTL for a cached score.

        Args:
            cache_key: Key to check

        Returns:
            Remaining TTL in seconds, None if key doesn't exist
        """
        ...


class CacheKeyGenerator(Protocol):
    """Protocol for generating cache keys for dimension scores."""

    def generate_score_key(
        self,
        dimension_set_id: int,
        dimension_id: str,
        content_hash: str,
        version: Optional[str] = None,
    ) -> str:
        """Generate a cache key for a dimension score.

        Args:
            dimension_set_id: ID of the dimension set
            dimension_id: ID of the specific dimension
            content_hash: Hash of the content being evaluated
            version: Optional version identifier

        Returns:
            Generated cache key
        """
        ...

    def generate_batch_key(
        self, dimension_set_id: int, content_hash: str, version: Optional[str] = None
    ) -> str:
        """Generate a cache key for a batch of dimension scores.

        Args:
            dimension_set_id: ID of the dimension set
            content_hash: Hash of the content being evaluated
            version: Optional version identifier

        Returns:
            Generated cache key for the batch
        """
        ...

    def parse_key(self, cache_key: str) -> Dict[str, str]:
        """Parse a cache key to extract components.

        Args:
            cache_key: The cache key to parse

        Returns:
            Dictionary with key components
        """
        ...


class CacheConfiguration:
    """Configuration for dimension score caching."""

    def __init__(
        self,
        default_ttl: timedelta = timedelta(hours=24),
        max_ttl: timedelta = timedelta(days=7),
        enable_compression: bool = True,
        enable_batch_operations: bool = True,
        max_batch_size: int = 100,
        namespace: str = "dimension_scores",
    ):
        self.default_ttl = default_ttl
        self.max_ttl = max_ttl
        self.enable_compression = enable_compression
        self.enable_batch_operations = enable_batch_operations
        self.max_batch_size = max_batch_size
        self.namespace = namespace

    def get_ttl_seconds(self, custom_ttl: Optional[timedelta] = None) -> int:
        """Get TTL in seconds, respecting max TTL."""
        if custom_ttl is None:
            return int(self.default_ttl.total_seconds())

        ttl_seconds = int(custom_ttl.total_seconds())
        max_seconds = int(self.max_ttl.total_seconds())

        return min(ttl_seconds, max_seconds)


class CacheStats(Protocol):
    """Protocol for cache statistics and monitoring."""

    async def get_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        ...

    async def get_size(self) -> int:
        """Get number of items in cache."""
        ...

    async def get_memory_usage(self) -> int:
        """Get cache memory usage in bytes."""
        ...

    async def reset_stats(self) -> None:
        """Reset cache statistics."""
        ...
