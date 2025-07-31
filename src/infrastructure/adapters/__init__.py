"""Infrastructure adapters for external system integration.

This module provides adapters for integrating with external streaming platforms.
"""

from .stream import (
    StreamAdapter,
    StreamConnection,
    StreamMetadata,
    ConnectionStatus,
    StreamHealth,
    StreamAdapterError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    StreamNotFoundError,
    StreamOfflineError,
    StreamAdapterFactory,
    get_stream_adapter,
    RTMPStreamAdapter,
)

__all__ = [
    # Base protocol and types
    "StreamAdapter",
    "StreamConnection",
    "StreamMetadata",
    "ConnectionStatus",
    "StreamHealth",
    # Exceptions
    "StreamAdapterError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "StreamNotFoundError",
    "StreamOfflineError",
    # Factory
    "StreamAdapterFactory",
    "get_stream_adapter",
    # Implementations
    "RTMPStreamAdapter",
]
