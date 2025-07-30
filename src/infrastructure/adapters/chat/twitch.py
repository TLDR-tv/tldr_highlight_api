"""Twitch chat adapter infrastructure implementation.

This module provides the Twitch chat integration using
Pythonic patterns and async/await throughout.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, List, Dict, Any
from contextlib import asynccontextmanager

from .base import (
    BaseChatAdapter,
    ChatMessage,
    ChatUser,
    ChatMessageType,
    ConnectionStatus,
    ChatAdapterError,
)

logger = logging.getLogger(__name__)


class TwitchChatAdapter(BaseChatAdapter):
    """Twitch chat adapter for EventSub or IRC integration.
    
    Provides basic Twitch chat connectivity. Full implementation
    would require WebSocket or IRC protocol handling.
    """
    
    def __init__(
        self,
        channel_id: str,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Twitch chat adapter.
        
        Args:
            channel_id: Twitch channel ID or username
            access_token: OAuth access token for Twitch API
            client_id: Twitch application client ID
            **kwargs: Additional configuration
        """
        super().__init__(channel_id, access_token, **kwargs)
        
        self.client_id = client_id
        self.channel_name = channel_id.lower()  # Twitch channels are lowercase
        
        # Connection state
        self._websocket = None
        self._irc_connection = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Background task
        self._receive_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized Twitch chat adapter for channel: {self.channel_name}")
    
    async def connect(self) -> bool:
        """Connect to Twitch chat service.
        
        Returns:
            bool: True if connection was successful
        """
        logger.info(f"Connecting to Twitch chat for channel: {self.channel_name}")
        
        try:
            self.connection.status = ConnectionStatus.CONNECTING
            
            # In a full implementation, we would:
            # 1. Connect to Twitch EventSub WebSocket
            # 2. Or connect to Twitch IRC
            # 3. Subscribe to chat events
            
            logger.warning(
                "Full Twitch chat protocol not implemented. "
                "This would require EventSub WebSocket or IRC implementation."
            )
            
            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.now(timezone.utc)
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            logger.info(f"Connected to Twitch chat for channel: {self.channel_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Twitch chat: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            self.connection.error_count += 1
            raise ChatAdapterError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Twitch chat service."""
        logger.info(f"Disconnecting from Twitch chat for channel: {self.channel_name}")
        
        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        # Close connections
        if self._websocket:
            # Close WebSocket
            self._websocket = None
        
        if self._irc_connection:
            # Close IRC
            self._irc_connection = None
        
        self.connection.status = ConnectionStatus.DISCONNECTED
        
        logger.info(f"Disconnected from Twitch chat for channel: {self.channel_name}")
    
    async def send_message(self, text: str) -> bool:
        """Send a message to Twitch chat.
        
        Args:
            text: The message text to send
            
        Returns:
            bool: True if message was sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send message: not connected")
            return False
        
        try:
            # In a full implementation, we would send via IRC or API
            logger.info(f"Would send message to Twitch chat: {text}")
            
            self.connection.messages_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.connection.error_count += 1
            return False
    
    async def get_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Get chat messages as an async generator.
        
        Yields:
            ChatMessage: Chat messages as they arrive
        """
        while not self._shutdown:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                self.connection.messages_received += 1
                self.connection.last_message_at = datetime.now(timezone.utc)
                
                # Notify callbacks
                await self._notify_message(message)
                
                yield message
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error getting message: {e}")
                self.connection.error_count += 1
    
    async def _receive_messages(self) -> None:
        """Background task to receive messages."""
        while not self._shutdown and self.is_connected:
            try:
                # Simulate receiving messages
                # In a full implementation, this would receive from WebSocket/IRC
                await asyncio.sleep(5)
                
                # Create a sample message
                sample_message = ChatMessage(
                    id=f"msg_{datetime.now().timestamp()}",
                    user=ChatUser(
                        id="123456",
                        username="sampleuser",
                        display_name="SampleUser",
                        badges=["subscriber"],
                        is_subscriber=True,
                    ),
                    text="This is a sample Twitch chat message!",
                    timestamp=datetime.now(timezone.utc),
                    type=ChatMessageType.MESSAGE,
                )
                
                await self._message_queue.put(sample_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receive task: {e}")
                await asyncio.sleep(1)
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"TwitchChatAdapter(channel='{self.channel_name}', connected={self.is_connected})"


@asynccontextmanager
async def twitch_chat(channel_id: str, **kwargs):
    """Context manager for Twitch chat.
    
    Args:
        channel_id: Twitch channel ID or username
        **kwargs: Additional adapter configuration
        
    Yields:
        Connected TwitchChatAdapter
    """
    adapter = TwitchChatAdapter(channel_id, **kwargs)
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()