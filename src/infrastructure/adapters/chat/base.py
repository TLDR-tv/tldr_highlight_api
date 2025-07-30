"""Base chat adapter protocol and common functionality.

This module defines the chat adapter protocol and common functionality
using Pythonic patterns with Protocol instead of ABC.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Protocol, runtime_checkable, Union
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ChatMessageType(str, Enum):
    """Types of chat messages."""
    MESSAGE = "message"
    SYSTEM = "system"
    EMOTE = "emote"
    ANNOUNCEMENT = "announcement"
    
    
class ConnectionStatus(str, Enum):
    """Chat connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


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
    type: ChatMessageType = ChatMessageType.MESSAGE
    emotes: List[Dict[str, Any]] = field(default_factory=list)
    bits: Optional[int] = None
    reply_to: Optional[str] = None
    is_highlight: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatConnection:
    """Information about a chat connection."""
    channel_id: str
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    messages_received: int = 0
    messages_sent: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None


class ChatAdapterError(Exception):
    """Base exception for chat adapter errors."""
    pass


@runtime_checkable
class ChatAdapter(Protocol):
    """Protocol for chat adapters.
    
    This protocol defines the interface that all chat adapters must implement.
    Uses Python Protocol for structural subtyping instead of ABC.
    """
    
    @property
    def channel_id(self) -> str:
        """Channel/broadcaster ID."""
        ...
    
    @property
    def connection(self) -> ChatConnection:
        """Connection information."""
        ...
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        ...
    
    @property
    def platform_name(self) -> str:
        """Get the platform name."""
        ...
    
    async def connect(self) -> bool:
        """Connect to the chat service.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ChatAdapterError: If connection fails
        """
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from the chat service."""
        ...
    
    async def send_message(self, text: str) -> bool:
        """Send a message to the chat.
        
        Args:
            text: The message text to send
            
        Returns:
            bool: True if message was sent successfully
        """
        ...
    
    async def get_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Get chat messages as an async generator.
        
        Yields:
            ChatMessage: Chat messages as they arrive
            
        Raises:
            ChatAdapterError: If message retrieval fails
        """
        ...
    
    async def start(self) -> None:
        """Start the chat adapter."""
        ...
    
    async def stop(self) -> None:
        """Stop the chat adapter."""
        ...


class BaseChatAdapter:
    """Base implementation with common functionality for chat adapters.
    
    This class provides common functionality that can be shared by concrete
    adapter implementations. It's not abstract - subclasses just extend it.
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
        self.connection = ChatConnection(channel_id=channel_id)
        
        # Message callbacks
        self._message_callbacks: List[Callable] = []
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(f"Initialized {self.__class__.__name__} for channel: {channel_id}")
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self.connection.status == ConnectionStatus.CONNECTED
    
    @property
    def platform_name(self) -> str:
        """Get the platform name."""
        return self.__class__.__name__.replace("ChatAdapter", "").lower()
    
    def on_message(self, callback: Callable) -> None:
        """Register a callback for message events.
        
        Args:
            callback: The callback function to call when message arrives
        """
        self._message_callbacks.append(callback)
        logger.debug("Registered message callback")
    
    async def _notify_message(self, message: ChatMessage) -> None:
        """Notify all registered callbacks for a message.
        
        Args:
            message: The message to notify about
        """
        for callback in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
    
    async def start(self) -> None:
        """Start the chat adapter.
        
        This method connects to the chat service.
        """
        logger.info(f"Starting {self.platform_name} chat adapter")
        
        try:
            # Connect
            if not await self.connect():
                raise ChatAdapterError("Failed to connect to chat service")
            
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
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of adapter metrics."""
        return {
            "platform": self.platform_name,
            "channel_id": self.channel_id,
            "is_connected": self.is_connected,
            "status": self.connection.status.value,
            "connected_at": self.connection.connected_at.isoformat() 
                if self.connection.connected_at else None,
            "last_message_at": self.connection.last_message_at.isoformat() 
                if self.connection.last_message_at else None,
            "messages_received": self.connection.messages_received,
            "messages_sent": self.connection.messages_sent,
            "error_count": self.connection.error_count,
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


@asynccontextmanager
async def chat_adapter_context(adapter: ChatAdapter):
    """Context manager for chat adapters.
    
    This provides a convenient way to ensure proper cleanup of chat adapters.
    
    Args:
        adapter: The chat adapter to manage
        
    Yields:
        The started chat adapter
    """
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()