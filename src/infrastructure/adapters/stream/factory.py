"""Factory functions for creating stream adapters.

This module provides factory functions for creating
the appropriate stream adapter based on the URL or platform.
"""

import logging
from typing import Optional, Type, Dict
from urllib.parse import urlparse

from aiohttp import ClientSession

from .base import StreamAdapter
from .rtmp import RTMPStreamAdapter
from .ffmpeg import FFmpegStreamAdapter

logger = logging.getLogger(__name__)

# Registry of adapter implementations
_ADAPTERS: Dict[str, Type[StreamAdapter]] = {
    "rtmp": RTMPStreamAdapter,
    "rtmps": RTMPStreamAdapter,
    "ffmpeg": FFmpegStreamAdapter,  # Generic adapter for any format
}


def register_stream_adapter(platform: str, adapter_class: Type[StreamAdapter]) -> None:
    """Register a new adapter implementation.

    Args:
        platform: Platform identifier
        adapter_class: Adapter class that implements StreamAdapter protocol
    """
    _ADAPTERS[platform.lower()] = adapter_class
    logger.info(f"Registered adapter for platform: {platform}")


def create_stream_adapter(
    url: str, session: Optional[ClientSession] = None, **kwargs
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
    platform = _detect_platform(url)

    if platform not in _ADAPTERS:
        raise ValueError(
            f"No adapter found for platform: {platform}. "
            f"Available platforms: {list(_ADAPTERS.keys())}"
        )

    adapter_class = _ADAPTERS[platform]
    logger.info(f"Creating {adapter_class.__name__} for URL: {url}")

    return adapter_class(url, session=session, **kwargs)


def _detect_platform(url: str) -> str:
    """Detect the platform from the URL.

    Args:
        url: Stream URL

    Returns:
        str: Platform identifier

    Raises:
        ValueError: If platform cannot be detected
    """
    # Check URL scheme for protocol-based adapters
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in ["rtmp", "rtmps"]:
        return scheme

    # For any other URL, use the generic FFmpeg adapter
    # which supports all formats that FFmpeg can handle
    return "ffmpeg"


def get_supported_platforms() -> list[str]:
    """Get list of supported platforms.

    Returns:
        List of platform identifiers
    """
    return list(_ADAPTERS.keys())


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
        if platform.lower() not in _ADAPTERS:
            raise ValueError(
                f"Unknown platform: {platform}. Available: {get_supported_platforms()}"
            )

        adapter_class = _ADAPTERS[platform.lower()]
        return adapter_class(url, session=session, **kwargs)

    # Auto-detect platform
    return create_stream_adapter(url, session=session, **kwargs)
