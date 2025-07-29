"""Stream adapter factory for creating platform-specific adapters.

This module provides the StreamAdapterFactory class and utility functions
for creating the appropriate stream adapter based on URL or platform type.
It follows the Factory pattern to encapsulate adapter creation logic.
"""

import logging
from typing import Dict, Type, Optional, Any
from aiohttp import ClientSession

from src.infrastructure.persistence.models.stream import StreamPlatform
from src.utils.stream_validation import detect_platform, validate_stream_url
from .base import BaseStreamAdapter
from .twitch import TwitchAdapter
from .youtube import YouTubeAdapter
from .rtmp import RTMPAdapter


logger = logging.getLogger(__name__)


class StreamAdapterFactory:
    """Factory class for creating stream adapters.

    This factory creates the appropriate stream adapter based on the
    platform type or by detecting the platform from the URL.
    """

    # Mapping of platforms to adapter classes
    _ADAPTER_CLASSES: Dict[StreamPlatform, Type[BaseStreamAdapter]] = {
        StreamPlatform.TWITCH: TwitchAdapter,
        StreamPlatform.YOUTUBE: YouTubeAdapter,
        StreamPlatform.RTMP: RTMPAdapter,
        StreamPlatform.CUSTOM: RTMPAdapter,  # Use RTMP adapter for custom streams
    }

    @classmethod
    def create_adapter(
        cls,
        url: str,
        platform: Optional[StreamPlatform] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ) -> BaseStreamAdapter:
        """Create a stream adapter for the given URL and platform.

        Args:
            url: The stream URL
            platform: Optional platform type (will be detected if not provided)
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options for the adapter

        Returns:
            BaseStreamAdapter: The appropriate stream adapter instance

        Raises:
            ValueError: If the platform is not supported
            ValidationError: If the URL is invalid
        """
        # Detect platform if not provided
        if platform is None:
            platform = detect_platform(url)

        # Validate the URL for the platform
        validate_stream_url(url, platform)

        # Get the adapter class
        adapter_class = cls._ADAPTER_CLASSES.get(platform)
        if not adapter_class:
            raise ValueError(f"Unsupported platform: {platform.value}")

        # Create and return the adapter
        logger.info(f"Creating {adapter_class.__name__} for URL: {url}")

        return adapter_class(url=url, session=session, **kwargs)

    @classmethod
    def create_twitch_adapter(
        cls,
        url: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ) -> TwitchAdapter:
        """Create a Twitch adapter with specific configuration.

        Args:
            url: Twitch stream URL
            client_id: Twitch Client ID
            client_secret: Twitch Client Secret
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options

        Returns:
            TwitchAdapter: Configured Twitch adapter
        """
        return TwitchAdapter(
            url=url,
            client_id=client_id,
            client_secret=client_secret,
            session=session,
            **kwargs,
        )

    @classmethod
    def create_youtube_adapter(
        cls,
        url: str,
        api_key: Optional[str] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ) -> YouTubeAdapter:
        """Create a YouTube adapter with specific configuration.

        Args:
            url: YouTube stream URL
            api_key: YouTube Data API v3 key
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options

        Returns:
            YouTubeAdapter: Configured YouTube adapter
        """
        return YouTubeAdapter(url=url, api_key=api_key, session=session, **kwargs)

    @classmethod
    def create_rtmp_adapter(
        cls,
        url: str,
        buffer_size: Optional[int] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ) -> RTMPAdapter:
        """Create an RTMP adapter with specific configuration.

        Args:
            url: RTMP stream URL
            buffer_size: Buffer size for RTMP data
            session: Optional aiohttp ClientSession (not used for RTMP)
            **kwargs: Additional configuration options

        Returns:
            RTMPAdapter: Configured RTMP adapter
        """
        return RTMPAdapter(url=url, buffer_size=buffer_size, session=session, **kwargs)

    @classmethod
    def get_supported_platforms(cls) -> Dict[str, str]:
        """Get a dictionary of supported platforms and their descriptions.

        Returns:
            Dict[str, str]: Mapping of platform names to descriptions
        """
        return {
            StreamPlatform.TWITCH.value: "Twitch live streams with API integration",
            StreamPlatform.YOUTUBE.value: "YouTube Live streams with Data API v3",
            StreamPlatform.RTMP.value: "Generic RTMP/RTMPS streams",
            StreamPlatform.CUSTOM.value: "Custom streaming formats (HLS, DASH, etc.)",
        }

    @classmethod
    def is_platform_supported(cls, platform: StreamPlatform) -> bool:
        """Check if a platform is supported by the factory.

        Args:
            platform: The platform to check

        Returns:
            bool: True if the platform is supported
        """
        return platform in cls._ADAPTER_CLASSES


# Convenience functions for easier adapter creation


def create_stream_adapter(
    url: str,
    platform: Optional[StreamPlatform] = None,
    session: Optional[ClientSession] = None,
    **kwargs,
) -> BaseStreamAdapter:
    """Create a stream adapter for the given URL.

    This is a convenience function that uses the StreamAdapterFactory
    to create the appropriate adapter.

    Args:
        url: The stream URL
        platform: Optional platform type (will be detected if not provided)
        session: Optional aiohttp ClientSession
        **kwargs: Additional configuration options for the adapter

    Returns:
        BaseStreamAdapter: The appropriate stream adapter instance

    Raises:
        ValueError: If the platform is not supported
        ValidationError: If the URL is invalid
    """
    return StreamAdapterFactory.create_adapter(
        url=url, platform=platform, session=session, **kwargs
    )


def create_adapter_from_stream_model(
    stream_model, session: Optional[ClientSession] = None, **kwargs
) -> BaseStreamAdapter:
    """Create a stream adapter from a Stream model instance.

    Args:
        stream_model: Stream model instance with source_url and platform
        session: Optional aiohttp ClientSession
        **kwargs: Additional configuration options for the adapter

    Returns:
        BaseStreamAdapter: The appropriate stream adapter instance

    Raises:
        ValueError: If the platform is not supported
        ValidationError: If the URL is invalid
    """
    # Convert string platform to enum
    platform = StreamPlatform(stream_model.platform)

    # Extract any platform-specific options from the stream model
    options = stream_model.options or {}

    # Merge options with kwargs (kwargs take precedence)
    adapter_kwargs = {**options, **kwargs}

    return StreamAdapterFactory.create_adapter(
        url=stream_model.source_url,
        platform=platform,
        session=session,
        **adapter_kwargs,
    )


def detect_and_validate_stream(url: str) -> Dict[str, Any]:
    """Detect platform and validate stream URL.

    This is a utility function that combines platform detection
    and URL validation in a single call.

    Args:
        url: The stream URL to analyze

    Returns:
        Dict containing:
        - platform: The detected platform
        - validation_result: URL validation details
        - adapter_class: The adapter class that would be used

    Raises:
        ValidationError: If the URL is invalid
    """
    # Detect platform
    platform = detect_platform(url)

    # Validate URL
    validation_result = validate_stream_url(url, platform)

    # Get adapter class
    adapter_class = StreamAdapterFactory._ADAPTER_CLASSES.get(platform)

    return {
        "platform": platform,
        "validation_result": validation_result,
        "adapter_class": adapter_class,
        "adapter_name": adapter_class.__name__ if adapter_class else None,
        "is_supported": StreamAdapterFactory.is_platform_supported(platform),
    }
