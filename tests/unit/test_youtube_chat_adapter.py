"""Unit tests for YouTube Live Chat adapter."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from aiohttp import ClientSession

from src.services.chat_adapters import YouTubeChatAdapter, ChatEventType, ChatEvent
from src.services.chat_adapters.youtube import YouTubeMessageType, YouTubeChatMessage, PollState
from src.services.chat_adapters.base import ChatAuthenticationError, ChatConnectionError


@pytest.fixture
def youtube_adapter():
    """Create a YouTube chat adapter instance."""
    return YouTubeChatAdapter(
        channel_id="test_live_chat_id",
        api_key="test_api_key",
        initial_polling_interval_ms=1000,
        max_results=50
    )


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = MagicMock(spec=ClientSession)
    return session


@pytest.fixture
def sample_youtube_message():
    """Sample YouTube live chat message response."""
    return {
        "id": "message_123",
        "snippet": {
            "type": "textMessageEvent",
            "liveChatId": "test_live_chat_id",
            "authorChannelId": "UCtest123",
            "publishedAt": "2024-01-01T12:00:00Z",
            "hasDisplayContent": True,
            "displayMessage": "Hello YouTube! 😀",
            "textMessageDetails": {
                "messageText": "Hello YouTube! 😀"
            }
        },
        "authorDetails": {
            "channelId": "UCtest123",
            "displayName": "Test User",
            "profileImageUrl": "https://example.com/avatar.jpg",
            "isVerified": False,
            "isChatOwner": False,
            "isChatSponsor": False,
            "isChatModerator": False
        }
    }


@pytest.fixture
def sample_super_chat_message():
    """Sample YouTube Super Chat message."""
    return {
        "id": "superchat_456",
        "snippet": {
            "type": "superChatEvent",
            "liveChatId": "test_live_chat_id",
            "authorChannelId": "UCdonor123",
            "publishedAt": "2024-01-01T12:05:00Z",
            "hasDisplayContent": True,
            "displayMessage": "Great stream! Keep it up!",
            "superChatDetails": {
                "amountMicros": "5000000",
                "currency": "USD",
                "amountDisplayString": "$5.00",
                "userComment": "Great stream! Keep it up!",
                "tier": 2
            }
        },
        "authorDetails": {
            "channelId": "UCdonor123",
            "displayName": "Generous Viewer",
            "profileImageUrl": "https://example.com/donor.jpg",
            "isVerified": True,
            "isChatOwner": False,
            "isChatSponsor": True,
            "isChatModerator": False
        }
    }


class TestYouTubeChatAdapter:
    """Test YouTube chat adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, youtube_adapter):
        """Test adapter initialization."""
        assert youtube_adapter.live_chat_id == "test_live_chat_id"
        assert youtube_adapter.api_key == "test_api_key"
        assert youtube_adapter.initial_polling_interval_ms == 1000
        assert youtube_adapter.max_results == 50
        assert not youtube_adapter.is_connected
        assert youtube_adapter.poll_state.polling_interval_ms == 1000
    
    @pytest.mark.asyncio
    async def test_connect_success(self, youtube_adapter, mock_session):
        """Test successful connection to YouTube chat."""
        youtube_adapter._session = mock_session
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "nextPageToken": "token_123",
            "pollingIntervalMillis": 5000,
            "items": []
        }
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Connect
        result = await youtube_adapter.connect()
        
        assert result is True
        assert youtube_adapter.is_connected
        assert youtube_adapter.poll_state.next_page_token == "token_123"
        assert youtube_adapter.poll_state.polling_interval_ms == 5000
        
        # Cleanup
        await youtube_adapter.disconnect()
    
    @pytest.mark.asyncio
    async def test_connect_authentication_error(self, youtube_adapter, mock_session):
        """Test connection failure due to authentication error."""
        youtube_adapter._session = mock_session
        
        # Mock 401 response
        mock_response = AsyncMock()
        mock_response.status = 401
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Should raise authentication error
        with pytest.raises(ChatAuthenticationError, match="Invalid API key"):
            await youtube_adapter.connect()
        
        assert not youtube_adapter.is_connected
    
    @pytest.mark.asyncio
    async def test_connect_quota_exceeded(self, youtube_adapter, mock_session):
        """Test connection failure due to quota exceeded."""
        youtube_adapter._session = mock_session
        
        # Mock 403 response
        mock_response = AsyncMock()
        mock_response.status = 403
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Should raise authentication error
        with pytest.raises(ChatAuthenticationError, match="quota exceeded"):
            await youtube_adapter.connect()
    
    @pytest.mark.asyncio
    async def test_connect_chat_not_found(self, youtube_adapter, mock_session):
        """Test connection failure when chat is not found."""
        youtube_adapter._session = mock_session
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Should raise connection error
        with pytest.raises(ChatConnectionError, match="Live chat not found"):
            await youtube_adapter.connect()
    
    @pytest.mark.asyncio
    async def test_parse_text_message(self, youtube_adapter, sample_youtube_message):
        """Test parsing a regular text message."""
        message = youtube_adapter._parse_message(sample_youtube_message)
        
        assert message is not None
        assert message.id == "message_123"
        assert message.author_channel_id == "UCtest123"
        assert message.author_display_name == "Test User"
        assert message.message_text == "Hello YouTube! 😀"
        assert message.display_message == "Hello YouTube! 😀"
        assert message.message_type == YouTubeMessageType.TEXT_MESSAGE
        assert not message.author_is_chat_moderator
        assert not message.author_is_chat_sponsor
    
    @pytest.mark.asyncio
    async def test_parse_super_chat(self, youtube_adapter, sample_super_chat_message):
        """Test parsing a Super Chat message."""
        message = youtube_adapter._parse_message(sample_super_chat_message)
        
        assert message is not None
        assert message.id == "superchat_456"
        assert message.author_channel_id == "UCdonor123"
        assert message.author_display_name == "Generous Viewer"
        assert message.message_text == "Great stream! Keep it up!"
        assert message.message_type == YouTubeMessageType.SUPER_CHAT
        assert message.amount_micros == "5000000"
        assert message.currency == "USD"
        assert message.tier == 2
        assert message.author_is_chat_sponsor
    
    @pytest.mark.asyncio
    async def test_convert_text_message_to_event(self, youtube_adapter, sample_youtube_message):
        """Test converting a text message to ChatEvent."""
        youtube_message = youtube_adapter._parse_message(sample_youtube_message)
        event = youtube_adapter._convert_to_chat_event(youtube_message)
        
        assert event is not None
        assert event.type == ChatEventType.MESSAGE
        assert event.user.id == "UCtest123"
        assert event.user.display_name == "Test User"
        assert event.user.username == "test_user"  # Normalized
        assert event.message is not None
        assert event.message.text == "Hello YouTube! 😀"
        assert len(event.message.emotes) == 1  # Should detect the emoji
    
    @pytest.mark.asyncio
    async def test_convert_super_chat_to_event(self, youtube_adapter, sample_super_chat_message):
        """Test converting a Super Chat to ChatEvent."""
        youtube_message = youtube_adapter._parse_message(sample_super_chat_message)
        event = youtube_adapter._convert_to_chat_event(youtube_message)
        
        assert event is not None
        assert event.type == ChatEventType.CHEER  # Super Chats map to CHEER
        assert event.user.id == "UCdonor123"
        assert event.user.is_subscriber  # Sponsors are subscribers
        assert event.amount == 5.0  # Converted from micros
        assert event.data["currency"] == "USD"
        assert event.data["tier"] == 2
        assert "message" in event.data
        assert event.data["message"] is not None
        assert event.data["message"].text == "Great stream! Keep it up!"
    
    @pytest.mark.asyncio
    async def test_extract_emotes(self, youtube_adapter):
        """Test emoji extraction from messages."""
        text = "Hello 😀 YouTube! 🎉 Party time! 🎊"
        emotes = youtube_adapter._extract_emotes(text)
        
        assert len(emotes) == 3
        assert emotes[0]["id"] == "😀"
        assert emotes[1]["id"] == "🎉"
        assert emotes[2]["id"] == "🎊"
        assert emotes[0]["positions"] == [[6, 7]]
    
    @pytest.mark.asyncio
    async def test_send_message_without_oauth(self, youtube_adapter):
        """Test sending message without OAuth token."""
        result = await youtube_adapter.send_message("Test message")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_message_with_oauth(self, youtube_adapter, mock_session):
        """Test sending message with OAuth token."""
        youtube_adapter.access_token = "oauth_token"
        youtube_adapter._session = mock_session
        
        # Mock successful send
        mock_response = AsyncMock()
        mock_response.status = 200
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        result = await youtube_adapter.send_message("Test message")
        
        assert result is True
        
        # Verify request
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer oauth_token"
    
    @pytest.mark.asyncio
    async def test_poll_messages_deduplication(self, youtube_adapter, mock_session, sample_youtube_message):
        """Test that duplicate messages are filtered out."""
        youtube_adapter._session = mock_session
        youtube_adapter.is_connected = True
        
        # Mock API response with duplicate message IDs
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "nextPageToken": "token_456",
            "pollingIntervalMillis": 5000,
            "items": [sample_youtube_message, sample_youtube_message]  # Duplicate
        }
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Poll messages
        await youtube_adapter._poll_messages()
        
        # Should only process one message
        assert "message_123" in youtube_adapter._seen_message_ids
        assert len(youtube_adapter._seen_message_ids) == 1
    
    @pytest.mark.asyncio
    async def test_adaptive_polling_interval(self, youtube_adapter, mock_session):
        """Test adaptive polling interval adjustment."""
        youtube_adapter._session = mock_session
        youtube_adapter.is_connected = True
        
        # Mock empty responses
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "nextPageToken": "token_789",
            "items": []
        }
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Initial interval
        assert youtube_adapter.poll_state.polling_interval_ms == 1000
        
        # Poll multiple times with no messages
        for i in range(7):
            await youtube_adapter._poll_messages()
        
        # Interval should increase after consecutive empty polls
        assert youtube_adapter.poll_state.consecutive_empty_polls == 7
        assert youtube_adapter.poll_state.polling_interval_ms > 1000
    
    @pytest.mark.asyncio
    async def test_message_cache_cleanup(self, youtube_adapter):
        """Test message ID cache cleanup."""
        # Add many message IDs
        for i in range(youtube_adapter._message_cache_size + 100):
            youtube_adapter._seen_message_ids.add(f"msg_{i}")
        
        # Force cleanup
        await youtube_adapter._cleanup_message_cache()
        
        # Cache should be reduced
        assert len(youtube_adapter._seen_message_ids) <= youtube_adapter._message_cache_size // 2
    
    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, youtube_adapter):
        """Test metrics summary generation."""
        youtube_adapter.live_chat_id = "test_chat"
        youtube_adapter.video_id = "test_video"
        youtube_adapter.poll_state.polling_interval_ms = 5000
        youtube_adapter.poll_state.messages_since_last_poll = 10
        youtube_adapter.quota_used = 100
        
        summary = youtube_adapter.get_metrics_summary()
        
        assert summary["platform"] == "youtube"
        assert summary["live_chat_id"] == "test_chat"
        assert summary["video_id"] == "test_video"
        assert summary["polling_interval_ms"] == 5000
        assert summary["messages_since_last_poll"] == 10
        assert summary["quota_used"] == 100
        assert summary["quota_limit"] > 0
    
    @pytest.mark.asyncio
    async def test_subscription_filtering(self, youtube_adapter, sample_youtube_message):
        """Test event type subscription filtering."""
        # Subscribe only to FOLLOW events
        await youtube_adapter.subscribe_to_events([ChatEventType.FOLLOW])
        
        # Parse and convert a text message
        youtube_message = youtube_adapter._parse_message(sample_youtube_message)
        event = youtube_adapter._convert_to_chat_event(youtube_message)
        
        # Should be filtered out since we only subscribed to FOLLOW
        assert event is None
    
    @pytest.mark.asyncio
    async def test_member_join_event(self, youtube_adapter):
        """Test parsing member join events."""
        member_join_message = {
            "id": "member_789",
            "snippet": {
                "type": "newSponsorEvent",
                "liveChatId": "test_live_chat_id",
                "authorChannelId": "UCmember123",
                "publishedAt": "2024-01-01T12:10:00Z",
                "hasDisplayContent": True,
                "displayMessage": "Welcome to the channel!"
            },
            "authorDetails": {
                "channelId": "UCmember123",
                "displayName": "New Member",
                "profileImageUrl": "https://example.com/member.jpg",
                "isVerified": False,
                "isChatOwner": False,
                "isChatSponsor": True,
                "isChatModerator": False
            }
        }
        
        message = youtube_adapter._parse_message(member_join_message)
        event = youtube_adapter._convert_to_chat_event(message)
        
        assert event is not None
        assert event.type == ChatEventType.SUBSCRIBE
        assert event.data["tier"] == "member"
        assert event.user.is_subscriber
    
    @pytest.mark.asyncio
    async def test_context_manager(self, youtube_adapter, mock_session):
        """Test using adapter as async context manager."""
        youtube_adapter._session = mock_session
        
        # Mock successful connection
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"items": []}
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        async with youtube_adapter as adapter:
            assert adapter.is_connected
        
        # Should be disconnected after exit
        assert not youtube_adapter.is_connected