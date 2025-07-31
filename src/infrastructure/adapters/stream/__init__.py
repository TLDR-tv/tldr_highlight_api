"""Stream adapters infrastructure module.

Provides adapters for various streaming platforms as infrastructure components.
"""

from .base import (
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
)
from .factory import StreamAdapterFactory, get_stream_adapter
from .rtmp import RTMPStreamAdapter
from .ffmpeg import FFmpegStreamAdapter

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
    "FFmpegStreamAdapter",
]
