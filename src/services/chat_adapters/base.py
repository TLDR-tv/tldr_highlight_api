"""Base chat adapter interface and common functionality.

This module defines the base interface for chat adapters that integrate with
platform-specific chat systems for real-time message and event processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Union

from src.utils.metrics import MetricsContext, counter, gauge, histogram


logger = logging.getLogger(__name__)


class ChatEventType(str, Enum):
    """Types of chat events."""
    
    MESSAGE = "message"
    FOLLOW = "follow"
    SUBSCRIBE = "subscribe"
    RESUBSCRIBE = "resubscribe"
    CHEER = "cheer"
    RAID = "raid"
    HOST = "host"
    HYPE_TRAIN_BEGIN = "hype_train_begin"
    HYPE_TRAIN_PROGRESS = "hype_train_progress"
    HYPE_TRAIN_END = "hype_train_end"
    MODERATOR_ACTION = "moderator_action"
    CHANNEL_UPDATE = "channel_update"
    STREAM_ONLINE = "stream_online"
    STREAM_OFFLINE = "stream_offline"
    USER_UPDATE = "user_update"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    RECONNECT = "reconnect"


@dataclass
class ChatUser:
    """Represents a chat user."""
    
    id: str
    username: str
    display_name: str
    badges: List[str] = field(default_factory=list)
    is_subscriber: bool = False
    is_moderator: bool = False
    is_vip: bool = False
    is_broadcaster: bool = False
    color: Optional[str] = None
    profile_image_url: Optional[str] = None


@dataclass
class ChatMessage:
    """Represents a chat message."""
    
    id: str
    user: ChatUser
    text: str
    timestamp: datetime
    emotes: List[Dict[str, Any]] = field(default_factory=list)
    bits: Optional[int] = None
    reply_to: Optional[str] = None
    is_action: bool = False
    is_highlight: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatEvent:
    """Represents a chat event."""
    
    id: str
    type: ChatEventType
    timestamp: datetime
    user: Optional[ChatUser] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Common event data accessors
    @property
    def message(self) -> Optional[ChatMessage]:
        """Get message if this is a message event."""
        if self.type == ChatEventType.MESSAGE and "message" in self.data:
            return self.data["message"]
        return None
    
    @property
    def amount(self) -> Optional[int]:
        """Get amount for events like cheers, subs."""
        return self.data.get("amount")
    
    @property
    def tier(self) -> Optional[str]:
        """Get tier for subscription events."""
        return self.data.get("tier")
    
    @property
    def months(self) -> Optional[int]:
        """Get months for subscription events."""
        return self.data.get("months")
    
    @property
    def viewers(self) -> Optional[int]:
        """Get viewer count for raid events."""
        return self.data.get("viewers")


class ChatAdapterError(Exception):
    """Base exception for chat adapter errors."""
    pass


class ChatConnectionError(ChatAdapterError):
    """Exception raised for connection-related errors."""
    pass


class ChatAuthenticationError(ChatAdapterError):
    """Exception raised for authentication-related errors."""
    pass


class BaseChatAdapter(ABC):
    """Base class for all chat adapters.
    
    This abstract base class provides the interface and common functionality
    that all platform-specific chat adapters must implement.
    """
    
    def __init__(
        self,
        channel_id: str,
        access_token: Optional[str] = None,
        **kwargs
    ):
        """Initialize the chat adapter.
        
        Args:
            channel_id: The channel/broadcaster ID
            access_token: Optional OAuth access token
            **kwargs: Additional platform-specific configuration
        """
        self.channel_id = channel_id
        self.access_token = access_token
        
        # Connection state
        self.is_connected = False
        self.connection_id: Optional[str] = None
        self.connected_at: Optional[datetime] = None
        self.last_message_at: Optional[datetime] = None
        
        # Event callbacks
        self._event_callbacks: Dict[ChatEventType, List[Callable]] = {}
        self._global_callbacks: List[Callable] = []
        
        # Metrics
        self._init_metrics()
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(f"Initialized {self.__class__.__name__} for channel: {channel_id}")
    
    @property
    def platform_name(self) -> str:
        """Get the platform name."""
        return self.__class__.__name__.replace("ChatAdapter", "").replace("EventSubAdapter", "").lower()
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the chat service.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ChatConnectionError: If connection fails
            ChatAuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the chat service."""
        pass
    
    @abstractmethod
    async def send_message(self, text: str) -> bool:
        """Send a message to the chat.
        
        Args:
            text: The message text to send
            
        Returns:
            bool: True if message was sent successfully
        """
        pass
    
    @abstractmethod
    async def get_events(self) -> AsyncGenerator[ChatEvent, None]:
        """Get chat events as an async generator.
        
        Yields:
            ChatEvent: Chat events as they occur
            
        Raises:
            ChatAdapterError: If event retrieval fails
        """
        pass
    
    @abstractmethod
    async def subscribe_to_events(self, event_types: List[ChatEventType]) -> bool:
        """Subscribe to specific event types.
        
        Args:
            event_types: List of event types to subscribe to
            
        Returns:
            bool: True if subscription was successful
        """
        pass
    
    # Common functionality
    
    def on_event(self, event_type: Union[ChatEventType, str], callback: Callable) -> None:
        """Register a callback for a specific event type.
        
        Args:
            event_type: The event type to listen for
            callback: The callback function to call when event occurs
        """
        if isinstance(event_type, str):
            event_type = ChatEventType(event_type)
        
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        
        self._event_callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for event type: {event_type}")
    
    def on_any_event(self, callback: Callable) -> None:
        """Register a callback for all events.
        
        Args:
            callback: The callback function to call for any event
        """
        self._global_callbacks.append(callback)
        logger.debug("Registered global event callback")
    
    async def _notify_event(self, event: ChatEvent) -> None:
        """Notify all registered callbacks for an event.
        
        Args:
            event: The event to notify about
        """
        # Update metrics
        self._events_received.increment(labels={"event_type": event.type.value})
        
        # Notify specific event callbacks
        if event.type in self._event_callbacks:
            for callback in self._event_callbacks[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    self._callback_errors.increment()
        
        # Notify global callbacks
        for callback in self._global_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in global event callback: {e}")
                self._callback_errors.increment()
    
    async def start(self) -> None:
        """Start the chat adapter.
        
        This method connects to the chat service and begins processing events.
        """
        logger.info(f"Starting {self.platform_name} chat adapter")
        
        async with MetricsContext("chat_adapter_start", {"platform": self.platform_name}):
            try:
                # Connect
                if not await self.connect():
                    raise ChatConnectionError("Failed to connect to chat service")
                
                logger.info(f"Started {self.platform_name} chat adapter successfully")
                
            except Exception as e:
                logger.error(f"Failed to start {self.platform_name} chat adapter: {e}")
                await self.stop()
                raise
    
    async def stop(self) -> None:
        """Stop the chat adapter."""
        logger.info(f"Stopping {self.platform_name} chat adapter")
        
        self._shutdown = True
        
        # Disconnect
        await self.disconnect()
        
        logger.info(f"Stopped {self.platform_name} chat adapter")
    
    def _init_metrics(self) -> None:
        """Initialize metrics for this adapter."""
        platform = self.platform_name
        
        # Connection metrics
        self._connection_attempts = counter(
            "chat_adapter_connection_attempts_total",
            "Total connection attempts",
            {"platform": platform}
        )
        self._connection_successes = counter(
            "chat_adapter_connection_successes_total",
            "Total successful connections",
            {"platform": platform}
        )
        self._connection_failures = counter(
            "chat_adapter_connection_failures_total",
            "Total connection failures",
            {"platform": platform}
        )
        
        # Event metrics
        self._events_received = counter(
            "chat_adapter_events_received_total",
            "Total events received",
            {"platform": platform}
        )
        self._messages_received = counter(
            "chat_adapter_messages_received_total",
            "Total chat messages received",
            {"platform": platform}
        )
        self._messages_sent = counter(
            "chat_adapter_messages_sent_total",
            "Total messages sent",
            {"platform": platform}
        )
        
        # Error metrics
        self._errors = counter(
            "chat_adapter_errors_total",
            "Total errors encountered",
            {"platform": platform}
        )
        self._callback_errors = counter(
            "chat_adapter_callback_errors_total",
            "Total callback errors",
            {"platform": platform}
        )
        
        # Connection status gauge
        self._connection_status = gauge(
            "chat_adapter_connection_status",
            "Current connection status (1=connected, 0=disconnected)",
            {"platform": platform}
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of adapter metrics."""
        return {
            "platform": self.platform_name,
            "channel_id": self.channel_id,
            "is_connected": self.is_connected,
            "connection_id": self.connection_id,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"{self.__class__.__name__}(channel_id='{self.channel_id}', "
            f"connected={self.is_connected})"
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()