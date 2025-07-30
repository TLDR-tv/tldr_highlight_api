"""Factory for creating stream adapters.

This module provides a factory pattern implementation for creating
the appropriate stream adapter based on the URL or platform.
"""

import logging
from typing import Optional, Type, Dict
from urllib.parse import urlparse

from aiohttp import ClientSession

from .base import StreamAdapter
from .twitch import TwitchStreamAdapter
from .youtube import YouTubeStreamAdapter
from .rtmp import RTMPStreamAdapter

logger = logging.getLogger(__name__)


class StreamAdapterFactory:
    """Factory for creating stream adapters based on URL or platform."""

    # Registry of adapter implementations
    _adapters: Dict[str, Type[StreamAdapter]] = {
        "twitch": TwitchStreamAdapter,
        "youtube": YouTubeStreamAdapter,
        "rtmp": RTMPStreamAdapter,
        "rtmps": RTMPStreamAdapter,
    }

    @classmethod
    def register_adapter(
        cls, platform: str, adapter_class: Type[StreamAdapter]
    ) -> None:
        """Register a new adapter implementation.

        Args:
            platform: Platform identifier
            adapter_class: Adapter class that implements StreamAdapter protocol
        """
        cls._adapters[platform.lower()] = adapter_class
        logger.info(f"Registered adapter for platform: {platform}")

    @classmethod
    def create(
        cls, url: str, session: Optional[ClientSession] = None, **kwargs
    ) -> StreamAdapter:
        """Create a stream adapter for the given URL.

        Args:
            url: Stream URL
            session: Optional aiohttp ClientSession
            **kwargs: Additional adapter-specific configuration

        Returns:
            StreamAdapter: Appropriate adapter instance

        Raises:
            ValueError: If no adapter found for the URL
        """
        platform = cls._detect_platform(url)

        if platform not in cls._adapters:
            raise ValueError(
                f"No adapter found for platform: {platform}. "
                f"Available platforms: {list(cls._adapters.keys())}"
            )

        adapter_class = cls._adapters[platform]
        logger.info(f"Creating {adapter_class.__name__} for URL: {url}")

        return adapter_class(url, session=session, **kwargs)

    @staticmethod
    def _detect_platform(url: str) -> str:
        """Detect the platform from the URL.

        Args:
            url: Stream URL

        Returns:
            str: Platform identifier

        Raises:
            ValueError: If platform cannot be detected
        """
        url_lower = url.lower()

        # Check common platforms
        if "twitch.tv" in url_lower:
            return "twitch"
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"

        # Check URL scheme for protocol-based adapters
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        if scheme in ["rtmp", "rtmps"]:
            return scheme

        # Default detection based on scheme
        if scheme in ["http", "https"]:
            # Could be HLS or other HTTP-based streaming
            # Would need more sophisticated detection
            raise ValueError(
                f"Cannot detect platform from HTTP(S) URL: {url}. "
                "Please specify platform explicitly."
            )

        raise ValueError(f"Cannot detect platform from URL: {url}")

    @classmethod
    def get_supported_platforms(cls) -> list[str]:
        """Get list of supported platforms.

        Returns:
            List of platform identifiers
        """
        return list(cls._adapters.keys())


def get_stream_adapter(
    url: str,
    platform: Optional[str] = None,
    session: Optional[ClientSession] = None,
    **kwargs,
) -> StreamAdapter:
    """Convenience function to get a stream adapter.

    Args:
        url: Stream URL
        platform: Optional platform override
        session: Optional aiohttp ClientSession
        **kwargs: Additional adapter configuration

    Returns:
        StreamAdapter: Appropriate adapter instance
    """
    if platform:
        # Use specified platform
        if platform.lower() not in StreamAdapterFactory._adapters:
            raise ValueError(
                f"Unknown platform: {platform}. "
                f"Available: {StreamAdapterFactory.get_supported_platforms()}"
            )

        adapter_class = StreamAdapterFactory._adapters[platform.lower()]
        return adapter_class(url, session=session, **kwargs)

    # Auto-detect platform
    return StreamAdapterFactory.create(url, session=session, **kwargs)
