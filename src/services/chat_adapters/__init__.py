"""Chat adapters for real-time chat integration.

This module provides adapters for various chat platforms to receive and process
real-time chat messages and events. Currently supports:
- Twitch EventSub WebSocket
- YouTube Live Chat API (polling-based)
"""

from .base import BaseChatAdapter, ChatMessage, ChatEvent, ChatEventType
from .twitch_eventsub import TwitchEventSubAdapter
from .youtube import YouTubeChatAdapter

__all__ = [
    "BaseChatAdapter",
    "ChatMessage",
    "ChatEvent",
    "ChatEventType",
    "TwitchEventSubAdapter",
    "YouTubeChatAdapter",
]