"""Tests for Twitch EventSub WebSocket adapter."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
import uuid

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, ClientWebSocketResponse

from src.services.chat_adapters.twitch_eventsub import TwitchEventSubAdapter
from src.services.chat_adapters.base import (
    ChatEventType,
    ChatConnectionError,
    ChatAuthenticationError,
)


@pytest_asyncio.fixture
async def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    session.ws_connect = AsyncMock()
    session.post = AsyncMock()
    session.delete = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest_asyncio.fixture
async def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock(spec=ClientWebSocketResponse)
    ws.closed = False
    ws.close = AsyncMock()
    ws.receive = AsyncMock()
    return ws


@pytest_asyncio.fixture
async def adapter(mock_session):
    """Create a TwitchEventSubAdapter instance."""
    adapter = TwitchEventSubAdapter(
        channel_id="123456789",
        access_token="test_token",
        client_id="test_client_id",
        session=mock_session,
        reconnect_attempts=2,
        reconnect_delay=0.1,
        keepalive_timeout=5,
    )
    yield adapter
    # Cleanup
    adapter._shutdown = True
    if adapter.is_connected:
        await adapter.disconnect()


class TestTwitchEventSubAdapter:
    """Test cases for TwitchEventSubAdapter."""

    @pytest.mark.asyncio
    async def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.channel_id == "123456789"
        assert adapter.access_token == "test_token"
        assert adapter.client_id == "test_client_id"
        assert not adapter.is_connected
        assert adapter.session_id is None
        assert adapter.reconnect_attempts == 2
        assert adapter.reconnect_delay == 0.1
        assert adapter.keepalive_timeout_seconds == 5

    @pytest.mark.asyncio
    async def test_connect_success(self, adapter, mock_session, mock_websocket):
        """Test successful connection."""
        # Setup mock WebSocket
        mock_session.ws_connect.return_value = mock_websocket

        # Mock welcome message
        welcome_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "session_welcome",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "payload": {
                "session": {
                    "id": "test_session_id",
                    "status": "connected",
                    "connected_at": datetime.now(timezone.utc).isoformat(),
                    "keepalive_timeout_seconds": 10,
                    "reconnect_url": "wss://example.com/reconnect",
                }
            },
        }

        mock_msg = Mock()
        mock_msg.type = WSMsgType.TEXT
        mock_msg.data = json.dumps(welcome_data)
        mock_websocket.receive.return_value = mock_msg

        # Connect
        result = await adapter.connect()

        assert result is True
        assert adapter.is_connected
        assert adapter.session_id == "test_session_id"
        assert adapter.keepalive_timeout_seconds == 10
        assert adapter.reconnect_url == "wss://example.com/reconnect"

        # Verify WebSocket connection
        mock_session.ws_connect.assert_called_once()
        call_args = mock_session.ws_connect.call_args
        assert call_args[0][0] == adapter.websocket_url
        assert "Authorization" in call_args[1]["headers"]
        assert "Client-Id" in call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_connect_no_token(self, adapter):
        """Test connection without access token."""
        adapter.access_token = None

        with pytest.raises(ChatAuthenticationError):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_no_client_id(self, adapter):
        """Test connection without client ID."""
        adapter.client_id = None

        with pytest.raises(ChatAuthenticationError):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_welcome_timeout(self, adapter, mock_session, mock_websocket):
        """Test connection with welcome message timeout."""
        mock_session.ws_connect.return_value = mock_websocket

        # Mock timeout on receive
        mock_websocket.receive.side_effect = asyncio.TimeoutError()

        with pytest.raises(ChatConnectionError):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter, mock_session, mock_websocket):
        """Test disconnection."""
        # Setup connected state
        adapter._websocket = mock_websocket
        adapter.is_connected = True
        adapter.session_id = "test_session"
        adapter.connection_id = "test_session"

        # Create mock tasks
        adapter._receive_task = asyncio.create_task(asyncio.sleep(10))
        adapter._keepalive_task = asyncio.create_task(asyncio.sleep(10))

        # Disconnect
        await adapter.disconnect()

        assert not adapter.is_connected
        assert adapter.session_id is None
        assert adapter.connection_id is None
        assert adapter._receive_task is None
        assert adapter._keepalive_task is None

        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, adapter):
        """Test subscribing to event types."""
        event_types = [
            ChatEventType.MESSAGE,
            ChatEventType.FOLLOW,
            ChatEventType.SUBSCRIBE,
        ]

        result = await adapter.subscribe_to_events(event_types)

        assert result is True
        assert "channel.chat.message" in adapter._subscription_types
        assert "channel.follow" in adapter._subscription_types
        assert "channel.subscribe" in adapter._subscription_types

    @pytest.mark.asyncio
    async def test_send_message(self, adapter):
        """Test send message (not supported)."""
        result = await adapter.send_message("Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_keepalive(self, adapter):
        """Test keepalive message handling."""
        keepalive_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "session_keepalive",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "payload": {},
        }

        await adapter._handle_message(json.dumps(keepalive_data))

        assert adapter._last_keepalive is not None

    @pytest.mark.asyncio
    async def test_handle_notification_chat_message(self, adapter):
        """Test handling chat message notification."""
        notification_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "notification",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
                "subscription_type": "channel.chat.message",
                "subscription_version": "1",
            },
            "payload": {
                "subscription": {
                    "id": "sub_123",
                    "status": "enabled",
                    "type": "channel.chat.message",
                    "version": "1",
                    "condition": {"broadcaster_user_id": "123456789"},
                    "transport": {"method": "websocket", "session_id": "test_session"},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "cost": 1,
                },
                "event": {
                    "broadcaster_user_id": "123456789",
                    "broadcaster_user_login": "testchannel",
                    "broadcaster_user_name": "TestChannel",
                    "chatter_user_id": "987654321",
                    "chatter_user_login": "testuser",
                    "chatter_user_name": "TestUser",
                    "message_id": "msg_123",
                    "message": {
                        "text": "Hello world!",
                        "fragments": [{"type": "text", "text": "Hello world!"}],
                    },
                    "color": "#FF0000",
                    "badges": [{"set_id": "subscriber", "id": "12", "info": "12"}],
                    "message_type": "text",
                },
            },
        }

        await adapter._handle_message(json.dumps(notification_data))

        # Check that message was queued
        assert not adapter._message_queue.empty()

        # Process the message
        message = await adapter._message_queue.get()
        event = await adapter._process_message(message)

        assert event is not None
        assert event.type == ChatEventType.MESSAGE
        assert event.user.id == "987654321"
        assert event.user.username == "testuser"
        assert event.data["message"].text == "Hello world!"

    @pytest.mark.asyncio
    async def test_handle_notification_follow(self, adapter):
        """Test handling follow notification."""
        notification_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "notification",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
                "subscription_type": "channel.follow",
                "subscription_version": "2",
            },
            "payload": {
                "subscription": {
                    "id": "sub_124",
                    "status": "enabled",
                    "type": "channel.follow",
                    "version": "2",
                    "condition": {"broadcaster_user_id": "123456789"},
                    "transport": {"method": "websocket", "session_id": "test_session"},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "cost": 1,
                },
                "event": {
                    "user_id": "987654321",
                    "user_login": "testuser",
                    "user_name": "TestUser",
                    "broadcaster_user_id": "123456789",
                    "broadcaster_user_login": "testchannel",
                    "broadcaster_user_name": "TestChannel",
                    "followed_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        }

        await adapter._handle_message(json.dumps(notification_data))

        # Process the message
        message = await adapter._message_queue.get()
        event = await adapter._process_message(message)

        assert event is not None
        assert event.type == ChatEventType.FOLLOW
        assert event.user.id == "987654321"
        assert event.user.username == "testuser"

    @pytest.mark.asyncio
    async def test_handle_reconnect(self, adapter):
        """Test handling reconnect message."""
        reconnect_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "session_reconnect",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "payload": {
                "session": {
                    "id": "new_session_id",
                    "status": "reconnecting",
                    "connected_at": datetime.now(timezone.utc).isoformat(),
                    "keepalive_timeout_seconds": 10,
                    "reconnect_url": "wss://example.com/new_reconnect",
                }
            },
        }

        with patch.object(adapter, "_reconnect", new_callable=AsyncMock):
            await adapter._handle_message(json.dumps(reconnect_data))

            assert adapter.reconnect_url == "wss://example.com/new_reconnect"
            adapter._reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_subscription(self, adapter, mock_session):
        """Test creating EventSub subscription."""
        adapter.session_id = "test_session"

        # Mock successful subscription creation
        mock_response = AsyncMock()
        mock_response.status = 202
        mock_response.json = AsyncMock(
            return_value={
                "data": [
                    {
                        "id": "sub_123",
                        "status": "webhook_callback_verification_pending",
                        "type": "channel.chat.message",
                        "version": "1",
                        "condition": {"broadcaster_user_id": "123456789"},
                        "transport": {
                            "method": "websocket",
                            "session_id": "test_session",
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "cost": 1,
                    }
                ]
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        subscription = await adapter._create_single_subscription(
            "channel.chat.message", "1"
        )

        assert subscription is not None
        assert subscription.id == "sub_123"
        assert subscription.type == "channel.chat.message"

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "eventsub/subscriptions" in call_args[0][0]
        assert call_args[1]["json"]["type"] == "channel.chat.message"
        assert call_args[1]["json"]["transport"]["method"] == "websocket"

    @pytest.mark.asyncio
    async def test_keepalive_timeout(self, adapter):
        """Test keepalive timeout detection."""
        adapter.is_connected = True
        adapter._last_keepalive = datetime.now(timezone.utc) - timedelta(
            seconds=adapter.keepalive_timeout_seconds + 10
        )

        with patch.object(adapter, "_reconnect", new_callable=AsyncMock):
            # Run one iteration of keepalive loop
            adapter._shutdown = False
            keepalive_task = asyncio.create_task(adapter._keepalive_loop())

            # Wait a bit and then cancel
            await asyncio.sleep(0.1)
            adapter._shutdown = True
            keepalive_task.cancel()

            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

            adapter._reconnect.assert_called()

    @pytest.mark.asyncio
    async def test_event_callbacks(self, adapter):
        """Test event callback registration and notification."""
        received_events = []

        async def on_message(event):
            received_events.append(event)

        adapter.on_event(ChatEventType.MESSAGE, on_message)

        # Create a test event
        test_event = Mock()
        test_event.type = ChatEventType.MESSAGE

        await adapter._notify_event(test_event)

        assert len(received_events) == 1
        assert received_events[0] == test_event

    @pytest.mark.asyncio
    async def test_metrics_summary(self, adapter):
        """Test getting metrics summary."""
        adapter.is_connected = True
        adapter.connection_id = "test_session"
        adapter.connected_at = datetime.now(timezone.utc)
        adapter.last_message_at = datetime.now(timezone.utc)

        metrics = adapter.get_metrics_summary()

        assert metrics["platform"] == "twitch"
        assert metrics["channel_id"] == "123456789"
        assert metrics["is_connected"] is True
        assert metrics["connection_id"] == "test_session"
        assert metrics["connected_at"] is not None
        assert metrics["last_message_at"] is not None

    @pytest.mark.asyncio
    async def test_context_manager(self, adapter, mock_session, mock_websocket):
        """Test using adapter as async context manager."""
        mock_session.ws_connect.return_value = mock_websocket

        # Mock welcome message
        welcome_data = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "message_type": "session_welcome",
                "message_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "payload": {
                "session": {
                    "id": "test_session_id",
                    "status": "connected",
                    "connected_at": datetime.now(timezone.utc).isoformat(),
                    "keepalive_timeout_seconds": 10,
                    "reconnect_url": None,
                }
            },
        }

        mock_msg = Mock()
        mock_msg.type = WSMsgType.TEXT
        mock_msg.data = json.dumps(welcome_data)
        mock_websocket.receive.return_value = mock_msg

        async with adapter as a:
            assert a.is_connected
            assert a.session_id == "test_session_id"

        assert not adapter.is_connected
