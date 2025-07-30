"""Chat adapters infrastructure module.

Provides adapters for various chat platforms as infrastructure components.
"""

from .base import (
    ChatAdapter,
    ChatMessage,
    ChatUser,
    ChatMessageType,
    ChatConnection,
    ConnectionStatus,
    ChatAdapterError,
)
from .sentiment import SentimentAnalyzer, SentimentResult
from .twitch import TwitchChatAdapter
from .youtube import YouTubeChatAdapter

__all__ = [
    # Base protocol and types
    "ChatAdapter",
    "ChatMessage",
    "ChatUser",
    "ChatMessageType",
    "ChatConnection",
    "ConnectionStatus",
    "ChatAdapterError",
    # Sentiment analysis
    "SentimentAnalyzer",
    "SentimentResult",
    # Implementations
    "TwitchChatAdapter",
    "YouTubeChatAdapter",
]