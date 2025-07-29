"""Twitch EventSub WebSocket client implementation.

This module provides a production-ready Twitch EventSub WebSocket client that handles:
- WebSocket connection management
- OAuth2 authentication
- Automatic reconnection
- Subscription management
- Message parsing and event dispatching
- Timeline synchronization with stream adapters
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Any, Set
from urllib.parse import urlencode
import uuid

import aiohttp
from aiohttp import ClientSession, WSMsgType
import backoff

from src.core.config import get_settings
from src.utils.metrics import MetricsContext, counter, gauge, histogram
from src.utils.circuit_breaker import get_circuit_breaker, CircuitBreakerError

from .base import (
    BaseChatAdapter,
    ChatEvent,
    ChatEventType,
    ChatMessage,
    ChatUser,
    ChatConnectionError,
    ChatAuthenticationError,
)
from .models import (
    EventSubSubscription,
    EventSubCondition,
    EventSubTransport,
    WebSocketMessage,
    MessageType,
    WelcomePayload,
    NotificationPayload,
    ReconnectPayload,
    ChannelChatMessageEvent,
    ChannelFollowEvent,
    ChannelSubscribeEvent,
    ChannelSubscriptionMessageEvent,
    ChannelCheerEvent,
    ChannelRaidEvent,
    ChannelHypeTrainBeginEvent,
    ChannelHypeTrainProgressEvent,
    ChannelHypeTrainEndEvent,
)


logger = logging.getLogger(__name__)
settings = get_settings()


# Mapping of EventSub subscription types to our ChatEventType
EVENTSUB_TYPE_MAPPING = {
    "channel.chat.message": ChatEventType.MESSAGE,
    "channel.follow": ChatEventType.FOLLOW,
    "channel.subscribe": ChatEventType.SUBSCRIBE,
    "channel.subscription.message": ChatEventType.RESUBSCRIBE,
    "channel.cheer": ChatEventType.CHEER,
    "channel.raid": ChatEventType.RAID,
    "channel.hype_train.begin": ChatEventType.HYPE_TRAIN_BEGIN,
    "channel.hype_train.progress": ChatEventType.HYPE_TRAIN_PROGRESS,
    "channel.hype_train.end": ChatEventType.HYPE_TRAIN_END,
}


class TwitchEventSubAdapter(BaseChatAdapter):
    """Twitch EventSub WebSocket client adapter.
    
    This adapter implements the Twitch EventSub WebSocket protocol for receiving
    real-time events from Twitch channels.
    """
    
    def __init__(
        self,
        channel_id: str,
        access_token: Optional[str] = None,
        client_id: Optional[str] = None,
        session: Optional[ClientSession] = None,
        websocket_url: str = "wss://eventsub.wss.twitch.tv/ws",
        reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
        keepalive_timeout: int = 10,
        **kwargs
    ):
        """Initialize the Twitch EventSub adapter.
        
        Args:
            channel_id: Twitch broadcaster user ID
            access_token: OAuth2 access token (user or app token)
            client_id: Twitch client ID
            session: Optional aiohttp session
            websocket_url: EventSub WebSocket URL
            reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Base delay between reconnections (exponential backoff)
            keepalive_timeout: Keepalive timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(channel_id, access_token, **kwargs)
        
        # Twitch configuration
        self.client_id = client_id or settings.twitch_client_id
        self.websocket_url = websocket_url
        self.api_base_url = settings.twitch_api_base_url
        
        # Session management
        self._session = session
        self._session_owned = session is None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # EventSub session
        self.session_id: Optional[str] = None
        self.reconnect_url: Optional[str] = None
        self.keepalive_timeout_seconds: int = keepalive_timeout
        
        # Connection configuration
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._reconnect_count = 0
        
        # Subscription management
        self._subscriptions: Dict[str, EventSubSubscription] = {}
        self._pending_subscriptions: Set[str] = set()
        self._subscription_types: Set[str] = set()
        
        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._last_keepalive: Optional[datetime] = None
        
        # Message queue for processing
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
        # Circuit breaker for API calls
        self._circuit_breaker_name = f"twitch_eventsub_{channel_id}"
        
        logger.info(f"Initialized Twitch EventSub adapter for channel: {channel_id}")
    
    @property
    def session(self) -> ClientSession:
        """Get the HTTP session, creating one if necessary."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = ClientSession(timeout=timeout)
            self._session_owned = True
        return self._session
    
    async def connect(self) -> bool:
        """Connect to Twitch EventSub WebSocket.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ChatConnectionError: If connection fails
            ChatAuthenticationError: If authentication fails
        """
        if not self.access_token:
            raise ChatAuthenticationError("Access token is required for EventSub")
        
        if not self.client_id:
            raise ChatAuthenticationError("Client ID is required for EventSub")
        
        logger.info(f"Connecting to Twitch EventSub for channel {self.channel_id}")
        
        try:
            self._connection_attempts.increment()
            
            # Connect to WebSocket
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Client-Id": self.client_id,
            }
            
            self._websocket = await self.session.ws_connect(
                self.websocket_url,
                headers=headers,
                heartbeat=30,
            )
            
            # Wait for welcome message
            welcome_msg = await self._wait_for_welcome()
            if not welcome_msg:
                raise ChatConnectionError("Did not receive welcome message")
            
            # Extract session info
            if hasattr(welcome_msg.payload, 'session'):
                session_data = welcome_msg.payload.session
            else:
                # Handle dict payload
                session_data = welcome_msg.payload['session']
                from .models import SessionPayload
                session_data = SessionPayload(**session_data)
            
            self.session_id = session_data.id
            self.keepalive_timeout_seconds = session_data.keepalive_timeout_seconds
            self.reconnect_url = session_data.reconnect_url
            
            # Update connection state
            self.is_connected = True
            self.connection_id = self.session_id
            self.connected_at = datetime.now(timezone.utc)
            self._reconnect_count = 0
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            
            # Subscribe to configured events
            if self._subscription_types:
                await self._create_subscriptions()
            
            self._connection_successes.increment()
            self._connection_status.set(1)
            
            # Notify connection established
            await self._notify_event(ChatEvent(
                id=str(uuid.uuid4()),
                type=ChatEventType.CONNECTION_ESTABLISHED,
                timestamp=datetime.now(timezone.utc),
                data={"session_id": self.session_id}
            ))
            
            logger.info(f"Successfully connected to EventSub with session: {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to EventSub: {e}")
            self._connection_failures.increment()
            self._errors.increment(labels={"error_type": type(e).__name__})
            await self.disconnect()
            raise ChatConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from EventSub WebSocket."""
        logger.info("Disconnecting from Twitch EventSub")
        
        # Cancel background tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        
        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        # Clear session info
        self.session_id = None
        self.reconnect_url = None
        self.is_connected = False
        self.connection_id = None
        self._connection_status.set(0)
        
        # Delete subscriptions if we're shutting down
        if self._shutdown and self._subscriptions:
            await self._delete_subscriptions()
        
        # Notify disconnection
        if not self._shutdown:
            await self._notify_event(ChatEvent(
                id=str(uuid.uuid4()),
                type=ChatEventType.CONNECTION_LOST,
                timestamp=datetime.now(timezone.utc),
                data={}
            ))
        
        logger.info("Disconnected from Twitch EventSub")
    
    async def send_message(self, text: str) -> bool:
        """Send a message to the chat.
        
        Note: EventSub is receive-only. Sending messages requires Twitch IRC.
        
        Args:
            text: The message text to send
            
        Returns:
            bool: Always False as EventSub doesn't support sending
        """
        logger.warning("EventSub WebSocket is receive-only. Use Twitch IRC to send messages.")
        return False
    
    async def get_events(self) -> AsyncGenerator[ChatEvent, None]:
        """Get chat events as an async generator.
        
        Yields:
            ChatEvent: Chat events as they occur
        """
        while not self._shutdown:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # Process message into ChatEvent
                event = await self._process_message(message)
                if event:
                    yield event
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self._errors.increment(labels={"error_type": "processing_error"})
    
    async def subscribe_to_events(self, event_types: List[ChatEventType]) -> bool:
        """Subscribe to specific event types.
        
        Args:
            event_types: List of event types to subscribe to
            
        Returns:
            bool: True if subscription was successful
        """
        # Map ChatEventType to EventSub subscription types
        sub_types = set()
        for event_type in event_types:
            for eventsub_type, chat_type in EVENTSUB_TYPE_MAPPING.items():
                if chat_type == event_type:
                    sub_types.add(eventsub_type)
        
        self._subscription_types.update(sub_types)
        
        # If connected, create subscriptions immediately
        if self.is_connected and self.session_id:
            return await self._create_subscriptions()
        
        # Otherwise, subscriptions will be created on connect
        return True
    
    async def _wait_for_welcome(self) -> Optional[WebSocketMessage]:
        """Wait for and process the welcome message.
        
        Returns:
            WebSocketMessage or None if welcome not received
        """
        try:
            # Wait for first message (should be welcome)
            msg = await asyncio.wait_for(
                self._websocket.receive(),
                timeout=10.0
            )
            
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                message = WebSocketMessage(**data)
                
                if message.metadata.message_type == MessageType.SESSION_WELCOME:
                    logger.info("Received EventSub welcome message")
                    return message
                else:
                    logger.error(f"First message was not welcome: {message.metadata.message_type}")
            else:
                logger.error(f"Unexpected WebSocket message type: {msg.type}")
                
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for welcome message")
        except Exception as e:
            logger.error(f"Error processing welcome message: {e}")
        
        return None
    
    async def _receive_loop(self) -> None:
        """Background task to receive WebSocket messages."""
        logger.info("Started EventSub receive loop")
        
        while not self._shutdown and self._websocket and not self._websocket.closed:
            try:
                msg = await self._websocket.receive()
                
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                    
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    self._errors.increment(labels={"error_type": "websocket_error"})
                    break
                    
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                    logger.info("WebSocket closed by server")
                    break
                    
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self._errors.increment(labels={"error_type": "receive_error"})
                break
        
        # Connection lost, attempt reconnection
        if not self._shutdown:
            logger.info("Receive loop ended, attempting reconnection")
            await self._reconnect()
    
    async def _keepalive_loop(self) -> None:
        """Background task to monitor keepalive messages."""
        logger.info(f"Started keepalive monitor (timeout: {self.keepalive_timeout_seconds}s)")
        
        while not self._shutdown and self.is_connected:
            try:
                # Calculate time since last keepalive
                if self._last_keepalive:
                    elapsed = (datetime.now(timezone.utc) - self._last_keepalive).total_seconds()
                    
                    # Check if keepalive timeout exceeded
                    if elapsed > self.keepalive_timeout_seconds + 5:  # 5s grace period
                        logger.warning(f"Keepalive timeout exceeded ({elapsed:.1f}s)")
                        await self._reconnect()
                        break
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in keepalive loop: {e}")
                self._errors.increment(labels={"error_type": "keepalive_error"})
    
    async def _handle_message(self, data: str) -> None:
        """Handle a WebSocket message.
        
        Args:
            data: Raw message data
        """
        try:
            message_data = json.loads(data)
            message = WebSocketMessage(**message_data)
            
            # Update metrics
            self._events_received.increment(labels={"event_type": message.metadata.message_type.value})
            
            # Handle message based on type
            if message.metadata.message_type == MessageType.SESSION_KEEPALIVE:
                self._last_keepalive = datetime.now(timezone.utc)
                logger.debug("Received keepalive")
                
            elif message.metadata.message_type == MessageType.SESSION_RECONNECT:
                logger.info("Received reconnect message")
                await self._handle_reconnect(message)
                
            elif message.metadata.message_type == MessageType.NOTIFICATION:
                # Queue for processing
                await self._message_queue.put(message)
                
            elif message.metadata.message_type == MessageType.REVOCATION:
                logger.warning(f"Subscription revoked: {message.payload}")
                await self._handle_revocation(message)
                
            else:
                logger.warning(f"Unknown message type: {message.metadata.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._errors.increment(labels={"error_type": "message_handling_error"})
    
    async def _handle_reconnect(self, message: WebSocketMessage) -> None:
        """Handle a reconnect message.
        
        Args:
            message: The reconnect message
        """
        if message.payload and hasattr(message.payload, "session"):
            self.reconnect_url = message.payload.session.reconnect_url
            logger.info(f"Updated reconnect URL: {self.reconnect_url}")
        
        # Reconnect to new URL
        await self._reconnect()
    
    async def _handle_revocation(self, message: WebSocketMessage) -> None:
        """Handle a subscription revocation.
        
        Args:
            message: The revocation message
        """
        if message.payload and hasattr(message.payload, "subscription"):
            sub = message.payload.subscription
            if sub.id in self._subscriptions:
                del self._subscriptions[sub.id]
                logger.warning(f"Removed revoked subscription: {sub.type} (reason: {sub.status})")
    
    @backoff.on_exception(
        backoff.expo,
        (ChatConnectionError, aiohttp.ClientError),
        max_tries=5,
        max_time=300
    )
    async def _reconnect(self) -> None:
        """Reconnect to EventSub WebSocket."""
        if self._shutdown:
            return
        
        self._reconnect_count += 1
        logger.info(f"Attempting reconnection #{self._reconnect_count}")
        
        # Disconnect current connection
        await self.disconnect()
        
        # Use reconnect URL if available
        if self.reconnect_url:
            self.websocket_url = self.reconnect_url
            self.reconnect_url = None
        
        # Notify reconnection attempt
        await self._notify_event(ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.RECONNECT,
            timestamp=datetime.now(timezone.utc),
            data={"attempt": self._reconnect_count}
        ))
        
        # Reconnect
        await self.connect()
    
    async def _create_subscriptions(self) -> bool:
        """Create EventSub subscriptions.
        
        Returns:
            bool: True if all subscriptions were created successfully
        """
        if not self.session_id:
            logger.error("Cannot create subscriptions without session ID")
            return False
        
        circuit_breaker = await get_circuit_breaker(
            self._circuit_breaker_name,
            failure_threshold=5,
            timeout_seconds=30,
            recovery_timeout_seconds=60
        )
        
        success = True
        
        for sub_type in self._subscription_types:
            if sub_type in self._pending_subscriptions:
                continue
            
            try:
                self._pending_subscriptions.add(sub_type)
                
                # Determine version based on subscription type
                version = "1"
                if sub_type == "channel.chat.message":
                    version = "1"
                
                # Create subscription
                subscription = await circuit_breaker.call(
                    self._create_single_subscription,
                    sub_type,
                    version
                )
                
                if subscription:
                    self._subscriptions[subscription.id] = subscription
                    logger.info(f"Created subscription: {sub_type}")
                else:
                    success = False
                    
            except CircuitBreakerError as e:
                logger.error(f"Circuit breaker open for EventSub API: {e}")
                success = False
            except Exception as e:
                logger.error(f"Failed to create subscription {sub_type}: {e}")
                success = False
            finally:
                self._pending_subscriptions.discard(sub_type)
        
        return success
    
    async def _create_single_subscription(self, sub_type: str, version: str) -> Optional[EventSubSubscription]:
        """Create a single EventSub subscription.
        
        Args:
            sub_type: Subscription type
            version: Subscription version
            
        Returns:
            EventSubSubscription or None if creation failed
        """
        url = f"{self.api_base_url}/eventsub/subscriptions"
        
        # Build condition based on subscription type
        condition = {"broadcaster_user_id": self.channel_id}
        
        # Some subscriptions need additional conditions
        if sub_type in ["channel.chat.message"]:
            condition["user_id"] = self.channel_id  # Bot user ID
        
        payload = {
            "type": sub_type,
            "version": version,
            "condition": condition,
            "transport": {
                "method": "websocket",
                "session_id": self.session_id
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Client-Id": self.client_id,
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 202:
                    data = await response.json()
                    return EventSubSubscription(**data["data"][0])
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to create subscription: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            return None
    
    async def _delete_subscriptions(self) -> None:
        """Delete all active subscriptions."""
        url = f"{self.api_base_url}/eventsub/subscriptions"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Client-Id": self.client_id
        }
        
        for sub_id, subscription in list(self._subscriptions.items()):
            try:
                async with self.session.delete(
                    f"{url}?id={sub_id}",
                    headers=headers
                ) as response:
                    if response.status == 204:
                        logger.info(f"Deleted subscription: {subscription.type}")
                    else:
                        logger.error(f"Failed to delete subscription: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error deleting subscription: {e}")
        
        self._subscriptions.clear()
    
    async def _process_message(self, message: WebSocketMessage) -> Optional[ChatEvent]:
        """Process a notification message into a ChatEvent.
        
        Args:
            message: The notification message
            
        Returns:
            ChatEvent or None if processing failed
        """
        try:
            if message.metadata.message_type != MessageType.NOTIFICATION:
                return None
            
            sub_type = message.metadata.subscription_type
            event_type = EVENTSUB_TYPE_MAPPING.get(sub_type)
            
            if not event_type:
                logger.warning(f"Unknown subscription type: {sub_type}")
                return None
            
            # Extract event data
            if hasattr(message.payload, "event"):
                event_data = message.payload.event
            elif isinstance(message.payload, dict) and "event" in message.payload:
                event_data = message.payload["event"]
            else:
                event_data = {}
            
            # Process based on event type
            if event_type == ChatEventType.MESSAGE:
                return await self._process_chat_message(event_data)
            elif event_type == ChatEventType.FOLLOW:
                return await self._process_follow(event_data)
            elif event_type == ChatEventType.SUBSCRIBE:
                return await self._process_subscribe(event_data)
            elif event_type == ChatEventType.RESUBSCRIBE:
                return await self._process_resubscribe(event_data)
            elif event_type == ChatEventType.CHEER:
                return await self._process_cheer(event_data)
            elif event_type == ChatEventType.RAID:
                return await self._process_raid(event_data)
            elif event_type == ChatEventType.HYPE_TRAIN_BEGIN:
                return await self._process_hype_train_begin(event_data)
            elif event_type == ChatEventType.HYPE_TRAIN_PROGRESS:
                return await self._process_hype_train_progress(event_data)
            elif event_type == ChatEventType.HYPE_TRAIN_END:
                return await self._process_hype_train_end(event_data)
            else:
                logger.warning(f"Unhandled event type: {event_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._errors.increment(labels={"error_type": "message_processing_error"})
            return None
    
    async def _process_chat_message(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a chat message event."""
        event_data = ChannelChatMessageEvent(**data)
        
        # Create ChatUser
        user = ChatUser(
            id=event_data.chatter_user_id,
            username=event_data.chatter_user_login,
            display_name=event_data.chatter_user_name,
            badges=[f"{b.set_id}:{b.id}" for b in event_data.badges],
            color=event_data.color
        )
        
        # Check badge types
        badge_sets = {b.set_id for b in event_data.badges}
        user.is_broadcaster = "broadcaster" in badge_sets
        user.is_moderator = "moderator" in badge_sets
        user.is_subscriber = "subscriber" in badge_sets
        user.is_vip = "vip" in badge_sets
        
        # Create ChatMessage
        message = ChatMessage(
            id=event_data.message_id,
            user=user,
            text=event_data.message.text,
            timestamp=datetime.now(timezone.utc),
            bits=event_data.cheer.get("bits") if event_data.cheer else None,
            reply_to=event_data.reply.parent_message_id if event_data.reply else None,
            is_action=event_data.message_type == "action",
            metadata={
                "fragments": event_data.message.fragments,
                "channel_points_reward_id": event_data.channel_points_custom_reward_id,
            }
        )
        
        # Update last message time
        self.last_message_at = datetime.now(timezone.utc)
        self._messages_received.increment()
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.MESSAGE,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={"message": message}
        )
    
    async def _process_follow(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a follow event."""
        event_data = ChannelFollowEvent(**data)
        
        user = ChatUser(
            id=event_data.user_id,
            username=event_data.user_login,
            display_name=event_data.user_name
        )
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.FOLLOW,
            timestamp=event_data.followed_at,
            user=user,
            data={}
        )
    
    async def _process_subscribe(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a subscribe event."""
        event_data = ChannelSubscribeEvent(**data)
        
        user = ChatUser(
            id=event_data.user_id,
            username=event_data.user_login,
            display_name=event_data.user_name
        )
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.SUBSCRIBE,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={
                "tier": event_data.tier,
                "is_gift": event_data.is_gift
            }
        )
    
    async def _process_resubscribe(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a resubscribe event."""
        event_data = ChannelSubscriptionMessageEvent(**data)
        
        user = ChatUser(
            id=event_data.user_id,
            username=event_data.user_login,
            display_name=event_data.user_name,
            is_subscriber=True
        )
        
        # Create message if present
        message = None
        if event_data.message:
            message = ChatMessage(
                id=str(uuid.uuid4()),
                user=user,
                text=event_data.message.text,
                timestamp=datetime.now(timezone.utc),
                metadata={"fragments": event_data.message.fragments}
            )
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.RESUBSCRIBE,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={
                "tier": event_data.tier,
                "months": event_data.cumulative_months,
                "streak_months": event_data.streak_months,
                "message": message
            }
        )
    
    async def _process_cheer(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a cheer event."""
        event_data = ChannelCheerEvent(**data)
        
        user = None
        if not event_data.is_anonymous:
            user = ChatUser(
                id=event_data.user_id,
                username=event_data.user_login,
                display_name=event_data.user_name
            )
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.CHEER,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={
                "amount": event_data.bits,
                "message": event_data.message,
                "is_anonymous": event_data.is_anonymous
            }
        )
    
    async def _process_raid(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a raid event."""
        event_data = ChannelRaidEvent(**data)
        
        user = ChatUser(
            id=event_data.from_broadcaster_user_id,
            username=event_data.from_broadcaster_user_login,
            display_name=event_data.from_broadcaster_user_name,
            is_broadcaster=True
        )
        
        return ChatEvent(
            id=str(uuid.uuid4()),
            type=ChatEventType.RAID,
            timestamp=datetime.now(timezone.utc),
            user=user,
            data={
                "viewers": event_data.viewers,
                "from_broadcaster": {
                    "id": event_data.from_broadcaster_user_id,
                    "login": event_data.from_broadcaster_user_login,
                    "name": event_data.from_broadcaster_user_name
                }
            }
        )
    
    async def _process_hype_train_begin(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a hype train begin event."""
        event_data = ChannelHypeTrainBeginEvent(**data)
        
        return ChatEvent(
            id=event_data.id,
            type=ChatEventType.HYPE_TRAIN_BEGIN,
            timestamp=event_data.started_at,
            data={
                "level": event_data.level,
                "total": event_data.total,
                "progress": event_data.progress,
                "goal": event_data.goal,
                "expires_at": event_data.expires_at.isoformat()
            }
        )
    
    async def _process_hype_train_progress(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a hype train progress event."""
        event_data = ChannelHypeTrainProgressEvent(**data)
        
        return ChatEvent(
            id=event_data.id,
            type=ChatEventType.HYPE_TRAIN_PROGRESS,
            timestamp=datetime.now(timezone.utc),
            data={
                "level": event_data.level,
                "total": event_data.total,
                "progress": event_data.progress,
                "goal": event_data.goal,
                "expires_at": event_data.expires_at.isoformat()
            }
        )
    
    async def _process_hype_train_end(self, data: Dict[str, Any]) -> ChatEvent:
        """Process a hype train end event."""
        event_data = ChannelHypeTrainEndEvent(**data)
        
        return ChatEvent(
            id=event_data.id,
            type=ChatEventType.HYPE_TRAIN_END,
            timestamp=event_data.ended_at,
            data={
                "level": event_data.level,
                "total": event_data.total,
                "started_at": event_data.started_at.isoformat(),
                "ended_at": event_data.ended_at.isoformat(),
                "cooldown_ends_at": event_data.cooldown_ends_at.isoformat()
            }
        )
    
    async def stop(self) -> None:
        """Stop the chat adapter and clean up resources."""
        await super().stop()
        
        # Close session if we own it
        if self._session_owned and self._session:
            await self._session.close()
            self._session = None
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"TwitchEventSubAdapter(channel_id='{self.channel_id}', "
            f"connected={self.is_connected}, session_id='{self.session_id}')"
        )