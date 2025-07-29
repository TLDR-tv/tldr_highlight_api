"""YouTube Live Chat API integration adapter.

This module provides a YouTube Live Chat adapter that uses the YouTube Data API v3
to poll for live chat messages and events. Since YouTube doesn't support WebSocket
for chat, this implementation uses efficient polling with pageToken pagination.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Any, Set, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
from dataclasses import dataclass, field
from enum import Enum
import re

import aiohttp
from aiohttp import ClientSession
import backoff

from src.core.config import get_settings
from src.utils.metrics import MetricsContext, counter, gauge, histogram
from src.utils.circuit_breaker import CircuitBreakerError

from .base import (
    BaseChatAdapter,
    ChatEvent,
    ChatEventType,
    ChatMessage,
    ChatUser,
    ChatConnectionError,
    ChatAuthenticationError,
    ChatAdapterError,
)


logger = logging.getLogger(__name__)
settings = get_settings()


class YouTubeMessageType(str, Enum):
    """YouTube live chat message types."""
    
    TEXT_MESSAGE = "textMessageEvent"
    SUPER_CHAT = "superChatEvent"
    SUPER_STICKER = "superStickerEvent"
    NEW_SPONSOR = "newSponsorEvent"  # Member join
    MEMBER_MILESTONE_CHAT = "memberMilestoneChatEvent"
    MESSAGE_DELETED = "messageDeletedEvent"
    USER_BANNED = "userBannedEvent"
    SPONSOR_ONLY_MODE_STARTED = "sponsorOnlyModeStartedEvent"
    SPONSOR_ONLY_MODE_ENDED = "sponsorOnlyModeEndedEvent"


# Mapping of YouTube message types to our ChatEventType
YOUTUBE_TYPE_MAPPING = {
    YouTubeMessageType.TEXT_MESSAGE: ChatEventType.MESSAGE,
    YouTubeMessageType.SUPER_CHAT: ChatEventType.CHEER,  # Map donations to cheer
    YouTubeMessageType.SUPER_STICKER: ChatEventType.CHEER,
    YouTubeMessageType.NEW_SPONSOR: ChatEventType.SUBSCRIBE,
    YouTubeMessageType.MESSAGE_DELETED: ChatEventType.MODERATOR_ACTION,
    YouTubeMessageType.USER_BANNED: ChatEventType.MODERATOR_ACTION,
}


@dataclass
class YouTubeChatMessage:
    """Represents a YouTube live chat message."""
    
    id: str
    author_channel_id: str
    author_display_name: str
    author_profile_image_url: str
    author_is_chat_owner: bool
    author_is_chat_moderator: bool
    author_is_chat_sponsor: bool
    message_text: str
    display_message: str
    published_at: datetime
    message_type: YouTubeMessageType
    
    # Super Chat/Sticker specific
    amount_micros: Optional[int] = None
    currency: Optional[str] = None
    tier: Optional[int] = None
    
    # Metadata
    has_display_content: bool = True
    live_chat_id: Optional[str] = None


@dataclass
class PollState:
    """Tracks the state of chat polling."""
    
    next_page_token: Optional[str] = None
    polling_interval_ms: int = 5000  # Default 5 seconds
    last_poll_time: Optional[datetime] = None
    messages_since_last_poll: int = 0
    consecutive_empty_polls: int = 0
    is_replay: bool = False


class YouTubeChatAdapter(BaseChatAdapter):
    """YouTube Live Chat adapter using polling-based API.
    
    This adapter implements efficient polling of YouTube's Live Chat API,
    handling rate limits, pagination, and converting YouTube events to
    the unified ChatEvent model.
    """
    
    def __init__(
        self,
        channel_id: str,  # This should be the live_chat_id
        access_token: Optional[str] = None,
        api_key: Optional[str] = None,
        session: Optional[ClientSession] = None,
        video_id: Optional[str] = None,
        oauth2_credentials: Optional[Dict[str, Any]] = None,
        initial_polling_interval_ms: int = 5000,
        min_polling_interval_ms: int = 1000,
        max_polling_interval_ms: int = 30000,
        max_results: int = 200,
        **kwargs
    ):
        """Initialize the YouTube chat adapter.
        
        Args:
            channel_id: The live_chat_id for the stream
            access_token: OAuth2 access token (for authenticated features)
            api_key: YouTube Data API key (for public features)
            session: Optional aiohttp session
            video_id: The YouTube video ID (optional, for metadata)
            oauth2_credentials: Full OAuth2 credentials dict
            initial_polling_interval_ms: Initial polling interval in milliseconds
            min_polling_interval_ms: Minimum polling interval
            max_polling_interval_ms: Maximum polling interval
            max_results: Maximum results per poll (max 2000)
            **kwargs: Additional configuration
        """
        super().__init__(channel_id, access_token, **kwargs)
        
        # YouTube configuration
        self.api_key = api_key or settings.youtube_api_key
        self.video_id = video_id
        self.live_chat_id = channel_id  # In YouTube, we need the live_chat_id
        self.oauth2_credentials = oauth2_credentials
        
        # API endpoints
        self.api_base_url = settings.youtube_api_base_url
        self.live_chat_messages_url = f"{self.api_base_url}/liveChat/messages"
        
        # Session management
        self._session = session
        self._session_owned = session is None
        
        # Polling configuration
        self.initial_polling_interval_ms = initial_polling_interval_ms
        self.min_polling_interval_ms = min_polling_interval_ms
        self.max_polling_interval_ms = max_polling_interval_ms
        self.max_results = min(max_results, 2000)  # YouTube max is 2000
        
        # Polling state
        self.poll_state = PollState(polling_interval_ms=initial_polling_interval_ms)
        self._polling_task: Optional[asyncio.Task] = None
        
        # Message deduplication
        self._seen_message_ids: Set[str] = set()
        self._message_cache_size = 10000
        self._last_cleanup_time = datetime.now(timezone.utc)
        
        # Rate limiting
        self.quota_used = 0
        self.quota_limit = settings.youtube_rate_limit_per_day
        self._request_times: List[datetime] = []
        
        # Circuit breaker for API calls
        self._circuit_breaker_name = f"youtube_chat_{channel_id}"
        
        # Event queue for async processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
        # Subscription tracking
        self._subscription_types: Set[ChatEventType] = set()
        
        logger.info(f"Initialized YouTube chat adapter for live_chat_id: {channel_id}")
    
    @property
    def session(self) -> ClientSession:
        """Get the HTTP session, creating one if necessary."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = ClientSession(timeout=timeout)
            self._session_owned = True
        return self._session
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        return headers
    
    def _get_api_params(self, **kwargs) -> Dict[str, Any]:
        """Get API parameters including authentication."""
        params = kwargs.copy()
        
        if self.api_key and "key" not in params:
            params["key"] = self.api_key
        
        return params
    
    async def connect(self) -> bool:
        """Connect to YouTube Live Chat service.
        
        Since YouTube uses polling, this validates the live_chat_id
        and starts the polling task.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ChatConnectionError: If connection fails
            ChatAuthenticationError: If authentication fails
        """
        if not self.api_key and not self.access_token:
            raise ChatAuthenticationError("Either API key or OAuth2 access token required")
        
        logger.info(f"Connecting to YouTube Live Chat: {self.live_chat_id}")
        
        try:
            self._connection_attempts.increment()
            
            # Validate the live chat by making a test request
            params = self._get_api_params(
                liveChatId=self.live_chat_id,
                part="snippet",
                maxResults=1
            )
            
            headers = self._get_auth_headers()
            
            async with self.session.get(
                self.live_chat_messages_url,
                params=params,
                headers=headers
            ) as response:
                if response.status == 401:
                    raise ChatAuthenticationError("Invalid API key or access token")
                elif response.status == 403:
                    raise ChatAuthenticationError("Insufficient permissions or quota exceeded")
                elif response.status == 404:
                    raise ChatConnectionError(f"Live chat not found: {self.live_chat_id}")
                elif response.status != 200:
                    raise ChatConnectionError(f"Failed to connect: {response.status}")
                
                data = await response.json()
                
                # Extract initial polling state
                if "nextPageToken" in data:
                    self.poll_state.next_page_token = data["nextPageToken"]
                
                if "pollingIntervalMillis" in data:
                    self.poll_state.polling_interval_ms = data["pollingIntervalMillis"]
                
                # Check if this is a replay
                if "offlineAt" in data.get("pageInfo", {}):
                    self.poll_state.is_replay = True
                    logger.info("Connected to YouTube chat replay")
            
            # Update connection state
            self.is_connected = True
            self.connected_at = datetime.now(timezone.utc)
            self._connection_successes.increment()
            self._connection_status.set(1)
            
            # Start polling task
            self._polling_task = asyncio.create_task(self._polling_loop())
            
            # Notify connection established
            await self._event_queue.put(ChatEvent(
                id=f"youtube_connection_{self.live_chat_id}",
                type=ChatEventType.CONNECTION_ESTABLISHED,
                timestamp=datetime.now(timezone.utc),
                data={"live_chat_id": self.live_chat_id, "is_replay": self.poll_state.is_replay}
            ))
            
            logger.info("Successfully connected to YouTube Live Chat")
            return True
            
        except Exception as e:
            self._connection_failures.increment()
            self._connection_status.set(0)
            logger.error(f"Failed to connect to YouTube chat: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from YouTube Live Chat service."""
        logger.info("Disconnecting from YouTube Live Chat")
        
        self._shutdown = True
        
        # Cancel polling task
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        
        # Update connection state
        self.is_connected = False
        self._connection_status.set(0)
        
        # Notify disconnection
        await self._event_queue.put(ChatEvent(
            id=f"youtube_disconnection_{self.live_chat_id}",
            type=ChatEventType.CONNECTION_LOST,
            timestamp=datetime.now(timezone.utc),
            data={"live_chat_id": self.live_chat_id}
        ))
        
        # Close session if we own it
        if self._session_owned and self._session:
            await self._session.close()
            self._session = None
        
        logger.info("Disconnected from YouTube Live Chat")
    
    async def send_message(self, text: str) -> bool:
        """Send a message to YouTube live chat.
        
        Requires OAuth2 authentication with appropriate scopes.
        
        Args:
            text: The message text to send
            
        Returns:
            bool: True if message was sent successfully
        """
        if not self.access_token:
            logger.error("OAuth2 access token required to send messages")
            return False
        
        try:
            url = f"{self.api_base_url}/liveChat/messages"
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            params = self._get_api_params(part="snippet")
            
            body = {
                "snippet": {
                    "liveChatId": self.live_chat_id,
                    "type": "textMessageEvent",
                    "textMessageDetails": {
                        "messageText": text
                    }
                }
            }
            
            async with self.session.post(
                url,
                params=params,
                headers=headers,
                json=body
            ) as response:
                if response.status == 200:
                    self._messages_sent.increment()
                    logger.info(f"Sent message to YouTube chat: {text[:50]}...")
                    return True
                else:
                    error_data = await response.json()
                    logger.error(f"Failed to send message: {error_data}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self._errors.increment()
            return False
    
    async def get_events(self) -> AsyncGenerator[ChatEvent, None]:
        """Get chat events as an async generator.
        
        Yields:
            ChatEvent: Chat events as they are polled from the API
        """
        while not self._shutdown:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                # Notify listeners
                await self._notify_event(event)
                
                yield event
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self._errors.increment()
    
    async def subscribe_to_events(self, event_types: List[ChatEventType]) -> bool:
        """Subscribe to specific event types.
        
        YouTube doesn't have granular subscriptions, so this just
        tracks what events the client is interested in.
        
        Args:
            event_types: List of event types to subscribe to
            
        Returns:
            bool: Always returns True for YouTube
        """
        # Store subscription preferences for filtering
        self._subscription_types = set(event_types)
        logger.info(f"Subscribed to event types: {event_types}")
        return True
    
    async def _polling_loop(self) -> None:
        """Main polling loop for retrieving chat messages."""
        logger.info("Starting YouTube chat polling loop")
        
        while not self._shutdown and self.is_connected:
            try:
                # Respect polling interval
                if self.poll_state.last_poll_time:
                    elapsed = (datetime.now(timezone.utc) - self.poll_state.last_poll_time).total_seconds() * 1000
                    wait_time = max(0, self.poll_state.polling_interval_ms - elapsed)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time / 1000)
                
                # Poll for messages
                await self._poll_messages()
                
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                self._errors.increment()
                
                # Exponential backoff on errors
                await asyncio.sleep(min(
                    self.poll_state.polling_interval_ms * 2 / 1000,
                    self.max_polling_interval_ms / 1000
                ))
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _poll_messages(self) -> None:
        """Poll for new chat messages."""
        # Skip circuit breaker for now - it has async issues in tests
        # TODO: Fix circuit breaker to work properly with async context managers
        
        try:
            params = self._get_api_params(
                liveChatId=self.live_chat_id,
                part="id,snippet,authorDetails",
                maxResults=self.max_results
            )
            
            if self.poll_state.next_page_token:
                params["pageToken"] = self.poll_state.next_page_token
            
            headers = self._get_auth_headers()
            
            async with self.session.get(
                self.live_chat_messages_url,
                params=params,
                headers=headers
            ) as response:
                if response.status == 403:
                    # Check if it's quota exceeded
                    error_data = await response.json()
                    if "quotaExceeded" in str(error_data):
                        raise RateLimitError("YouTube API quota exceeded")
                
                response.raise_for_status()
                data = await response.json()
                
                # Update polling state
                self.poll_state.last_poll_time = datetime.now(timezone.utc)
                self.poll_state.next_page_token = data.get("nextPageToken")
                
                if "pollingIntervalMillis" in data:
                    self.poll_state.polling_interval_ms = data["pollingIntervalMillis"]
                
                # Process messages
                items = data.get("items", [])
                self.poll_state.messages_since_last_poll = len(items)
                
                if items:
                    self.poll_state.consecutive_empty_polls = 0
                    await self._process_messages(items)
                else:
                    self.poll_state.consecutive_empty_polls += 1
                    
                    # Adaptive polling - slow down if no activity
                    if self.poll_state.consecutive_empty_polls > 5:
                        self.poll_state.polling_interval_ms = min(
                            self.poll_state.polling_interval_ms * 1.5,
                            self.max_polling_interval_ms
                        )
                
                # Update quota tracking
                self.quota_used += 1  # Each request costs 1 quota unit
                
                # Cleanup old message IDs periodically
                if (datetime.now(timezone.utc) - self._last_cleanup_time).total_seconds() > 3600:
                    await self._cleanup_message_cache()
                    
        except CircuitBreakerError:
            logger.warning("Circuit breaker open for YouTube API")
            await asyncio.sleep(30)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error polling messages: {e}")
            raise
    
    async def _process_messages(self, items: List[Dict[str, Any]]) -> None:
        """Process a batch of chat messages."""
        for item in items:
            try:
                # Deduplicate messages
                message_id = item.get("id")
                if message_id in self._seen_message_ids:
                    continue
                
                self._seen_message_ids.add(message_id)
                
                # Parse message
                youtube_message = self._parse_message(item)
                if not youtube_message:
                    continue
                
                # Convert to ChatEvent
                event = self._convert_to_chat_event(youtube_message)
                if event:
                    await self._event_queue.put(event)
                    
                    # Update last message time
                    self.last_message_at = datetime.now(timezone.utc)
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self._errors.increment()
    
    def _parse_message(self, item: Dict[str, Any]) -> Optional[YouTubeChatMessage]:
        """Parse a YouTube API message response."""
        try:
            snippet = item.get("snippet", {})
            author = item.get("authorDetails", {})
            
            # Determine message type
            message_type = YouTubeMessageType(snippet.get("type", "textMessageEvent"))
            
            # Extract message text based on type
            message_text = ""
            display_message = ""
            
            if message_type == YouTubeMessageType.TEXT_MESSAGE:
                details = snippet.get("textMessageDetails", {})
                message_text = details.get("messageText", "")
                display_message = snippet.get("displayMessage", message_text)
                
            elif message_type == YouTubeMessageType.SUPER_CHAT:
                details = snippet.get("superChatDetails", {})
                message_text = details.get("userComment", "")
                display_message = snippet.get("displayMessage", message_text)
                
            elif message_type == YouTubeMessageType.SUPER_STICKER:
                details = snippet.get("superStickerDetails", {})
                display_message = f"[Super Sticker: {details.get('superStickerMetadata', {}).get('altText', '')}]"
            
            # Parse published time
            published_at = datetime.fromisoformat(
                snippet.get("publishedAt", "").replace("Z", "+00:00")
            )
            
            # Create message object
            message = YouTubeChatMessage(
                id=item.get("id"),
                author_channel_id=author.get("channelId", ""),
                author_display_name=author.get("displayName", ""),
                author_profile_image_url=author.get("profileImageUrl", ""),
                author_is_chat_owner=author.get("isChatOwner", False),
                author_is_chat_moderator=author.get("isChatModerator", False),
                author_is_chat_sponsor=author.get("isChatSponsor", False),
                message_text=message_text,
                display_message=display_message,
                published_at=published_at,
                message_type=message_type,
                has_display_content=snippet.get("hasDisplayContent", True),
                live_chat_id=snippet.get("liveChatId")
            )
            
            # Add Super Chat/Sticker specific data
            if message_type == YouTubeMessageType.SUPER_CHAT:
                details = snippet.get("superChatDetails", {})
                message.amount_micros = details.get("amountMicros")
                message.currency = details.get("currency")
                message.tier = details.get("tier")
                
            elif message_type == YouTubeMessageType.SUPER_STICKER:
                details = snippet.get("superStickerDetails", {})
                message.amount_micros = details.get("amountMicros")
                message.currency = details.get("currency")
                message.tier = details.get("tier")
            
            return message
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None
    
    def _convert_to_chat_event(self, youtube_message: YouTubeChatMessage) -> Optional[ChatEvent]:
        """Convert a YouTube message to a ChatEvent."""
        try:
            # Map message type to event type
            event_type = YOUTUBE_TYPE_MAPPING.get(
                youtube_message.message_type,
                ChatEventType.MESSAGE
            )
            
            # Filter by subscription if set
            if self._subscription_types and event_type not in self._subscription_types:
                return None
            
            # Create ChatUser
            badges = []
            if youtube_message.author_is_chat_owner:
                badges.append("broadcaster")
            if youtube_message.author_is_chat_moderator:
                badges.append("moderator")
            if youtube_message.author_is_chat_sponsor:
                badges.append("member")
            
            user = ChatUser(
                id=youtube_message.author_channel_id,
                username=youtube_message.author_display_name.lower().replace(" ", "_"),
                display_name=youtube_message.author_display_name,
                badges=badges,
                is_subscriber=youtube_message.author_is_chat_sponsor,
                is_moderator=youtube_message.author_is_chat_moderator,
                is_broadcaster=youtube_message.author_is_chat_owner,
                profile_image_url=youtube_message.author_profile_image_url
            )
            
            # Create event data
            event_data = {
                "platform": "youtube",
                "raw_type": youtube_message.message_type.value,
            }
            
            # Handle text messages
            if event_type == ChatEventType.MESSAGE:
                # Extract emotes from message
                emotes = self._extract_emotes(youtube_message.display_message)
                
                chat_message = ChatMessage(
                    id=youtube_message.id,
                    user=user,
                    text=youtube_message.message_text,
                    timestamp=youtube_message.published_at,
                    emotes=emotes,
                    metadata={
                        "display_message": youtube_message.display_message,
                        "has_display_content": youtube_message.has_display_content
                    }
                )
                
                event_data["message"] = chat_message
                self._messages_received.increment()
            
            # Handle Super Chat/Sticker (mapped to CHEER)
            elif event_type == ChatEventType.CHEER:
                if youtube_message.amount_micros:
                    # Convert micros to main currency unit (amount_micros is a string)
                    amount = int(youtube_message.amount_micros) / 1_000_000
                    event_data["amount"] = amount
                    event_data["currency"] = youtube_message.currency
                    event_data["tier"] = youtube_message.tier
                    
                    # Include message if present
                    if youtube_message.message_text:
                        chat_message = ChatMessage(
                            id=youtube_message.id,
                            user=user,
                            text=youtube_message.message_text,
                            timestamp=youtube_message.published_at,
                            metadata={
                                "is_super_chat": youtube_message.message_type == YouTubeMessageType.SUPER_CHAT,
                                "is_super_sticker": youtube_message.message_type == YouTubeMessageType.SUPER_STICKER,
                                "amount": amount,
                                "currency": youtube_message.currency
                            }
                        )
                        event_data["message"] = chat_message
            
            # Handle member join (mapped to SUBSCRIBE)
            elif event_type == ChatEventType.SUBSCRIBE:
                event_data["tier"] = "member"  # YouTube only has one tier
            
            # Handle moderator actions
            elif event_type == ChatEventType.MODERATOR_ACTION:
                if youtube_message.message_type == YouTubeMessageType.MESSAGE_DELETED:
                    event_data["action"] = "delete_message"
                elif youtube_message.message_type == YouTubeMessageType.USER_BANNED:
                    event_data["action"] = "ban_user"
            
            # Create the event
            event = ChatEvent(
                id=youtube_message.id,
                type=event_type,
                timestamp=youtube_message.published_at,
                user=user,
                data=event_data
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error converting message to event: {e}")
            return None
    
    def _extract_emotes(self, text: str) -> List[Dict[str, Any]]:
        """Extract YouTube emojis and custom emotes from text.
        
        YouTube doesn't provide emote positions in the API, so we
        do basic emoji detection.
        """
        emotes = []
        
        # Basic emoji pattern
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"   # Symbols & pictographs
            "\U0001F680-\U0001F6FF"   # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"   # Flags
            "\U00002702-\U000027B0"   # Dingbats
            "\U000024C2-\U0001F251"   # Enclosed characters
            "]+"
        )
        
        for match in emoji_pattern.finditer(text):
            emotes.append({
                "id": match.group(),
                "name": match.group(),
                "positions": [[match.start(), match.end()]]
            })
        
        return emotes
    
    async def _cleanup_message_cache(self) -> None:
        """Clean up old message IDs to prevent memory growth."""
        if len(self._seen_message_ids) > self._message_cache_size:
            # Keep only the most recent half
            to_remove = len(self._seen_message_ids) - (self._message_cache_size // 2)
            for _ in range(to_remove):
                self._seen_message_ids.pop()
        
        self._last_cleanup_time = datetime.now(timezone.utc)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of adapter metrics."""
        summary = super().get_metrics_summary()
        summary.update({
            "live_chat_id": self.live_chat_id,
            "video_id": self.video_id,
            "polling_interval_ms": self.poll_state.polling_interval_ms,
            "messages_since_last_poll": self.poll_state.messages_since_last_poll,
            "consecutive_empty_polls": self.poll_state.consecutive_empty_polls,
            "is_replay": self.poll_state.is_replay,
            "quota_used": self.quota_used,
            "quota_limit": self.quota_limit,
            "cached_message_ids": len(self._seen_message_ids)
        })
        return summary


class RateLimitError(ChatAdapterError):
    """Exception raised when YouTube API rate limit is exceeded."""
    pass