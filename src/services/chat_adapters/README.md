# Chat Adapters

This module provides real-time chat integration for various streaming platforms. Chat events are synchronized with stream timelines to enable accurate highlight detection based on viewer engagement.

## Features

### Twitch EventSub WebSocket Client

The `TwitchEventSubAdapter` provides a production-ready implementation of Twitch's EventSub WebSocket protocol:

- **WebSocket Connection Management**: Automatic connection handling with exponential backoff
- **OAuth2 Authentication**: Support for both user and app access tokens
- **Automatic Reconnection**: Handles disconnects and reconnect messages
- **Subscription Management**: Easy subscription to different event types
- **Message Parsing**: Full support for all EventSub event types
- **Timeline Synchronization**: Integration with stream adapters for accurate timing

### YouTube Live Chat API Client

The `YouTubeChatAdapter` provides comprehensive YouTube Live Chat integration:

- **Polling-Based Architecture**: Efficient polling with adaptive intervals
- **API Key & OAuth Support**: Public access via API key, authenticated features with OAuth
- **Automatic Rate Limiting**: Respects YouTube API quotas with built-in tracking
- **Message Deduplication**: Prevents processing duplicate messages
- **Super Chat/Sticker Support**: Full support for YouTube monetization events
- **Chat Replay**: Process chat history from completed streams
- **Adaptive Polling**: Automatically adjusts polling frequency based on chat activity

### Supported Event Types

#### Twitch
- `channel.chat.message` - Chat messages
- `channel.follow` - New followers
- `channel.subscribe` - New subscriptions
- `channel.subscription.message` - Resubscription messages
- `channel.cheer` - Bit cheers
- `channel.raid` - Incoming raids
- `channel.hype_train.begin/progress/end` - Hype train events

#### YouTube
- `textMessageEvent` - Regular chat messages
- `superChatEvent` - Super Chat donations (mapped to CHEER)
- `superStickerEvent` - Super Sticker donations (mapped to CHEER)
- `newSponsorEvent` - New channel members (mapped to SUBSCRIBE)
- `messageDeletedEvent` - Message deletions (mapped to MODERATOR_ACTION)
- `userBannedEvent` - User bans (mapped to MODERATOR_ACTION)

## Usage

### Twitch Example

```python
from src.services.chat_adapters.twitch_eventsub import TwitchEventSubAdapter
from src.services.chat_adapters.base import ChatEventType

# Create adapter
adapter = TwitchEventSubAdapter(
    channel_id="123456789",  # Twitch broadcaster user ID
    access_token="your_access_token",
    client_id="your_client_id"
)

# Subscribe to events
await adapter.subscribe_to_events([
    ChatEventType.MESSAGE,
    ChatEventType.FOLLOW,
    ChatEventType.SUBSCRIBE
])

# Register event handlers
@adapter.on_event(ChatEventType.MESSAGE)
async def handle_message(event):
    message = event.data["message"]
    print(f"{message.user.display_name}: {message.text}")

# Connect and start receiving events
async with adapter:
    async for event in adapter.get_events():
        # Events are also dispatched to registered handlers
        pass
```

### YouTube Example

```python
from src.services.chat_adapters.youtube import YouTubeChatAdapter
from src.services.chat_adapters.base import ChatEventType

# First, get the live_chat_id from stream metadata
stream_adapter = YouTubeAdapter(url="https://youtube.com/watch?v=VIDEO_ID")
await stream_adapter.connect()
metadata = await stream_adapter.get_stream_metadata()

# Create chat adapter
chat_adapter = YouTubeChatAdapter(
    channel_id=metadata.live_chat_id,  # YouTube live_chat_id
    api_key="your_api_key",
    initial_polling_interval_ms=5000,  # 5 seconds
    max_results=200  # Messages per poll
)

# Register event handlers
@chat_adapter.on_event(ChatEventType.MESSAGE)
async def handle_message(event):
    print(f"{event.user.display_name}: {event.message.text}")

@chat_adapter.on_event(ChatEventType.CHEER)
async def handle_super_chat(event):
    amount = event.data.get("amount", 0)
    currency = event.data.get("currency", "USD")
    print(f"Super Chat: {currency} {amount} from {event.user.display_name}")

# Connect and process events
async with chat_adapter:
    async for event in chat_adapter.get_events():
        # Events are processed by handlers
        pass
```

### Stream Synchronization

For highlight detection, synchronize chat events with the video stream:

```python
from src.services.stream_adapters.twitch import TwitchAdapter
from src.services.chat_adapters.twitch_eventsub import TwitchEventSubAdapter
from src.services.chat_adapters.stream_sync import StreamChatSynchronizer

# Create adapters
stream_adapter = TwitchAdapter(url="https://twitch.tv/channel")
chat_adapter = TwitchEventSubAdapter(
    channel_id="123456789",
    access_token="token"
)

# Create synchronizer
synchronizer = StreamChatSynchronizer(
    stream_adapter=stream_adapter,
    chat_adapter=chat_adapter,
    buffer_seconds=30.0
)

# Handle synchronized events
@synchronizer.on_sync_event
async def handle_sync_event(sync_event):
    print(f"Event at {sync_event.stream_time:.1f}s: {sync_event.chat_event.type}")

# Start everything
await stream_adapter.start()
await chat_adapter.start()
await synchronizer.start()
```

## Architecture

### Base Classes

- `BaseChatAdapter`: Abstract base class for all chat adapters
- `ChatEvent`: Unified event model for all platforms
- `ChatMessage`: Structured chat message data
- `ChatUser`: User information with badges and roles

### Event Flow

1. Platform-specific adapter connects to chat service
2. Raw events are parsed into unified `ChatEvent` objects
3. Events are dispatched to registered callbacks
4. Synchronizer aligns events with stream timeline
5. Synchronized events are used for highlight detection

### Error Handling

The adapters implement robust error handling:

- Automatic reconnection with exponential backoff
- Circuit breaker pattern for API calls
- Graceful degradation on service issues
- Comprehensive logging and metrics

## Configuration

Configure via environment variables or settings:

```python
# Twitch EventSub
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
TWITCH_EVENTSUB_WEBSOCKET_URL=wss://eventsub.wss.twitch.tv/ws
TWITCH_EVENTSUB_RECONNECT_ATTEMPTS=5
TWITCH_EVENTSUB_KEEPALIVE_TIMEOUT=10

# YouTube Live Chat
YOUTUBE_API_KEY=your_api_key
YOUTUBE_API_BASE_URL=https://www.googleapis.com/youtube/v3
YOUTUBE_RATE_LIMIT_PER_DAY=10000
YOUTUBE_RATE_LIMIT_PER_100_SECONDS=3000000
```

## Metrics

The chat adapters provide detailed metrics:

- Connection attempts/successes/failures
- Events received by type
- Message rates
- Error counts
- Callback performance

Access metrics via:

```python
metrics = adapter.get_metrics_summary()
```

## Testing

Run tests with:

```bash
uv run pytest tests/unit/chat_adapters/
```

## Future Enhancements

- Discord integration for gaming streams
- Custom chat protocols via plugins
- ML-based sentiment analysis
- Spam/bot detection
- Chat command handling
- YouTube chat participation (requires OAuth2 with broader scopes)
- Enhanced emoji/emoticon analysis