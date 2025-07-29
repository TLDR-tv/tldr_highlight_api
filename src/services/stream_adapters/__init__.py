"""Stream adapters package for platform-specific stream ingestion.

This package provides adapters for various streaming platforms including
Twitch, YouTube Live, and generic RTMP streams. The adapters normalize
different streaming platforms into a unified processing pipeline.

The package follows the Adapter pattern to provide a consistent interface
for different streaming platforms while handling platform-specific
authentication, API calls, and data formats.
"""

from .base import BaseStreamAdapter, StreamMetadata, StreamConnection
from .factory import StreamAdapterFactory, create_stream_adapter
from .twitch import TwitchAdapter
from .youtube import YouTubeAdapter
from .rtmp import RTMPAdapter

__all__ = [
    "BaseStreamAdapter",
    "StreamMetadata",
    "StreamConnection",
    "StreamAdapterFactory",
    "create_stream_adapter",
    "TwitchAdapter",
    "YouTubeAdapter",
    "RTMPAdapter",
]
