"""Infrastructure adapters for external system integration.

This module provides adapters for integrating with external systems
like streaming platforms and chat services as infrastructure concerns.
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
    TwitchStreamAdapter,
    YouTubeStreamAdapter,
    RTMPStreamAdapter,
)
from .chat import (
    ChatAdapter,
    ChatMessage,
    ChatUser,
    ChatMessageType,
    ChatConnection,
    SentimentAnalyzer,
    SentimentResult,
    ChatAdapterError,
    TwitchChatAdapter,
    YouTubeChatAdapter,
)

__all__ = [
    # Stream adapters
    "StreamAdapter",
    "StreamConnection",
    "StreamMetadata",
    "ConnectionStatus",
    "StreamHealth",
    "StreamAdapterError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "StreamNotFoundError",
    "StreamOfflineError",
    "StreamAdapterFactory",
    "get_stream_adapter",
    "TwitchStreamAdapter",
    "YouTubeStreamAdapter",
    "RTMPStreamAdapter",
    # Chat adapters
    "ChatAdapter",
    "ChatMessage",
    "ChatUser",
    "ChatMessageType",
    "ChatConnection",
    "SentimentAnalyzer",
    "SentimentResult",
    "ChatAdapterError",
    "TwitchChatAdapter",
    "YouTubeChatAdapter",
]
