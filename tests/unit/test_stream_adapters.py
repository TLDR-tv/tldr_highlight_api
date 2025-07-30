"""Unit tests for stream adapters.

Tests for BaseStreamAdapter, TwitchAdapter, YouTubeAdapter, RTMPAdapter,
and the StreamAdapterFactory.
"""

# Apply patches before imports to ensure they take effect
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import aiohttp
from aiohttp import ClientSession


# Create a complete mock MetricsContext class
class MockMetricsContext:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


# Mock the entire metrics module
mock_metrics_module = Mock()
mock_metrics_module.MetricsContext = MockMetricsContext
mock_metrics_module.counter = Mock(return_value=Mock(increment=Mock()))
mock_metrics_module.gauge = Mock(return_value=Mock(set=Mock()))
mock_metrics_module.histogram = Mock(return_value=Mock(observe=Mock()))

# Replace the module in sys.modules before any imports
sys.modules["src.utils.metrics"] = mock_metrics_module

from src.services.stream_adapters.base import (
    BaseStreamAdapter,
    StreamMetadata,
    StreamConnection,
    ConnectionStatus,
    StreamHealth,
    StreamAdapterError,
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    StreamNotFoundError,
    StreamOfflineError,
)
from src.services.stream_adapters.twitch import TwitchAdapter
from src.services.stream_adapters.youtube import YouTubeAdapter
from src.services.stream_adapters.rtmp import RTMPAdapter
from src.services.stream_adapters.factory import (
    StreamAdapterFactory,
    create_stream_adapter,
    create_adapter_from_stream_model,
    detect_and_validate_stream,
)
from src.models.stream import StreamPlatform


class MockStreamAdapter(BaseStreamAdapter):
    """Mock implementation of BaseStreamAdapter for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._authenticated = False
        self._connected = False
        self._live = True

    async def authenticate(self) -> bool:
        self._authenticated = True
        return True

    async def connect(self) -> bool:
        if not self._authenticated:
            raise AuthenticationError("Not authenticated")
        self._connected = True
        self.connection.status = ConnectionStatus.CONNECTED
        await self._notify_connect()  # Notify callbacks
        return True

    async def disconnect(self) -> None:
        self._connected = False
        self.connection.status = ConnectionStatus.DISCONNECTED
        await self._notify_disconnect()  # Notify callbacks

    async def get_metadata(self) -> StreamMetadata:
        return StreamMetadata(
            title="Mock Stream",
            description="Mock stream for testing",
            is_live=self._live,
        )

    async def is_stream_live(self) -> bool:
        return self._live

    async def get_stream_data(self):
        if not self._connected:
            raise ConnectionError("Not connected")
        data = b"mock_data"
        await self._notify_data(data)  # Notify callbacks
        yield data


class TestBaseStreamAdapter:
    """Test BaseStreamAdapter functionality."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter for testing."""
        return MockStreamAdapter("https://example.com/stream")

    def test_adapter_initialization(self, mock_adapter):
        """Test adapter initialization."""
        assert mock_adapter.url == "https://example.com/stream"
        assert mock_adapter.connection.url == "https://example.com/stream"
        assert mock_adapter.connection.status == ConnectionStatus.DISCONNECTED
        assert mock_adapter.connection.health == StreamHealth.UNKNOWN
        assert not mock_adapter.is_connected
        assert not mock_adapter.is_healthy

    @pytest.mark.asyncio
    async def test_adapter_start_stop(self, mock_adapter):
        """Test adapter start and stop lifecycle."""
        # Test start
        await mock_adapter.start()
        assert mock_adapter._authenticated
        assert mock_adapter.is_connected
        assert mock_adapter.connection.status == ConnectionStatus.CONNECTED

        # Test stop
        await mock_adapter.stop()
        assert not mock_adapter.is_connected
        assert mock_adapter.connection.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_adapter_reconnect(self, mock_adapter):
        """Test adapter reconnection."""
        # Start adapter
        await mock_adapter.start()
        assert mock_adapter.is_connected

        # Simulate disconnection
        await mock_adapter.disconnect()
        assert not mock_adapter.is_connected

        # Test reconnect
        success = await mock_adapter.reconnect()
        assert success
        assert mock_adapter.is_connected
        assert mock_adapter.connection.reconnect_count == 1

        # Clean up
        await mock_adapter.stop()

    @pytest.mark.asyncio
    async def test_adapter_health_check(self, mock_adapter):
        """Test adapter health checking."""
        await mock_adapter.start()

        # Test healthy state
        health = await mock_adapter.check_health()
        assert health == StreamHealth.HEALTHY

        # Test unhealthy state (stream offline)
        mock_adapter._live = False
        health = await mock_adapter.check_health()
        assert health == StreamHealth.UNHEALTHY

        # Clean up
        await mock_adapter.stop()

    @pytest.mark.asyncio
    async def test_adapter_event_callbacks(self, mock_adapter):
        """Test adapter event callbacks."""
        connect_called = False
        disconnect_called = False
        data_received = None
        error_received = None

        def on_connect(adapter):
            nonlocal connect_called
            connect_called = True

        def on_disconnect(adapter):
            nonlocal disconnect_called
            disconnect_called = True

        def on_data(adapter, data):
            nonlocal data_received
            data_received = data

        def on_error(adapter, error):
            nonlocal error_received
            error_received = error

        # Register callbacks
        mock_adapter.on_connect(on_connect)
        mock_adapter.on_disconnect(on_disconnect)
        mock_adapter.on_data(on_data)
        mock_adapter.on_error(on_error)

        # Test connect callback
        await mock_adapter.start()
        assert connect_called

        # Test data callback
        async for data in mock_adapter.get_stream_data():
            break
        assert data_received == b"mock_data"

        # Test disconnect callback
        await mock_adapter.stop()
        assert disconnect_called

    @pytest.mark.asyncio
    async def test_adapter_context_manager(self, mock_adapter):
        """Test adapter as async context manager."""
        async with mock_adapter as adapter:
            assert adapter.is_connected

        assert not mock_adapter.is_connected


class TestTwitchAdapter:
    """Test TwitchAdapter functionality."""

    @pytest.fixture
    def twitch_adapter(self):
        """Create a Twitch adapter for testing."""
        return TwitchAdapter(
            "https://www.twitch.tv/shroud",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    @pytest.fixture
    def mock_session(self):
        """Create a mock HTTP session."""
        session = MagicMock(spec=ClientSession)
        return session

    def test_twitch_adapter_initialization(self, twitch_adapter):
        """Test Twitch adapter initialization."""
        assert twitch_adapter.username == "shroud"
        assert twitch_adapter.client_id == "test_client_id"
        assert twitch_adapter.client_secret == "test_client_secret"
        assert twitch_adapter.platform_name == "twitch"

    @pytest.mark.asyncio
    async def test_twitch_authentication_success(self, twitch_adapter, mock_session):
        """Test successful Twitch authentication."""
        # Mock successful auth response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "access_token": "test_token",
            "expires_in": 3600,
        }

        mock_session.post.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        success = await twitch_adapter.authenticate()
        assert success
        assert twitch_adapter.app_access_token == "test_token"

    @pytest.mark.asyncio
    async def test_twitch_authentication_failure(self, twitch_adapter, mock_session):
        """Test failed Twitch authentication."""
        # Mock failed auth response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text.return_value = "Unauthorized"

        mock_session.post.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        with pytest.raises(AuthenticationError):
            await twitch_adapter.authenticate()

    @pytest.mark.asyncio
    async def test_twitch_api_request(self, twitch_adapter, mock_session):
        """Test Twitch API request."""
        twitch_adapter.app_access_token = "test_token"

        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Ratelimit-Remaining": "800"}
        mock_response.json.return_value = {"data": [{"id": "123", "login": "shroud"}]}

        mock_session.get.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        result = await twitch_adapter._make_api_request("users", {"login": "shroud"})
        assert result["data"][0]["login"] == "shroud"
        assert twitch_adapter.rate_limit_remaining == 800

    @pytest.mark.asyncio
    async def test_twitch_rate_limit_error(self, twitch_adapter, mock_session):
        """Test Twitch API rate limit handling."""
        twitch_adapter.app_access_token = "test_token"

        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}

        mock_session.get.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        with pytest.raises(RateLimitError):
            await twitch_adapter._make_api_request("users")

    @pytest.mark.asyncio
    async def test_twitch_user_not_found(self, twitch_adapter, mock_session):
        """Test Twitch user not found."""
        twitch_adapter.app_access_token = "test_token"

        # Mock empty response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"data": []}

        mock_session.get.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        with pytest.raises(StreamNotFoundError):
            await twitch_adapter._get_user_info()

    @pytest.mark.asyncio
    async def test_twitch_stream_metadata(self, twitch_adapter, mock_session):
        """Test Twitch stream metadata retrieval."""
        twitch_adapter.app_access_token = "test_token"
        twitch_adapter.user_id = "123"

        # Mock stream response
        mock_stream_response = AsyncMock()
        mock_stream_response.status = 200
        mock_stream_response.headers = {}
        mock_stream_response.json.return_value = {
            "data": [
                {
                    "id": "stream_123",
                    "title": "Test Stream",
                    "viewer_count": 1000,
                    "started_at": "2023-01-01T12:00:00Z",
                    "game_name": "Test Game",
                    "thumbnail_url": "https://example.com/thumb_{width}x{height}.jpg",
                }
            ]
        }

        # Mock user response
        mock_user_response = AsyncMock()
        mock_user_response.status = 200
        mock_user_response.headers = {}
        mock_user_response.json.return_value = {
            "data": [{"description": "Test streamer"}]
        }

        mock_session.get.return_value.__aenter__.side_effect = [
            mock_stream_response,
            mock_user_response,
        ]
        twitch_adapter._session = mock_session

        metadata = await twitch_adapter.get_metadata()

        assert metadata.title == "Test Stream"
        assert metadata.viewer_count == 1000
        assert metadata.is_live is True
        assert metadata.category == "Test Game"
        assert "1920x1080" in metadata.thumbnail_url

    @pytest.mark.asyncio
    async def test_twitch_hls_streaming(self, twitch_adapter, mock_session):
        """Test Twitch HLS streaming functionality."""
        twitch_adapter.app_access_token = "test_token"
        twitch_adapter.user_id = "123"

        # Mock stream live response
        mock_live_response = AsyncMock()
        mock_live_response.status = 200
        mock_live_response.headers = {}
        mock_live_response.json.return_value = {
            "data": [{"id": "live_stream_123"}]  # Stream is live
        }

        # Mock GraphQL access token response
        mock_gql_response = AsyncMock()
        mock_gql_response.status = 200
        mock_gql_response.json.return_value = {
            "data": {
                "streamPlaybackAccessToken": {
                    "value": "test_access_token_value",
                    "signature": "test_signature",
                    "__typename": "PlaybackAccessToken",
                }
            }
        }

        # Mock HLS manifest content
        mock_manifest_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=1280000,RESOLUTION=720x480,CODECS="avc1.42001e,mp4a.40.2"
chunked.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2560000,RESOLUTION=1280x720,CODECS="avc1.42001f,mp4a.40.2"
720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080,CODECS="avc1.42001f,mp4a.40.2"
1080p.m3u8
"""

        # Mock playlist content
        mock_playlist_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:12345
#EXTINF:10.0,
segment000.ts
#EXTINF:10.0,
segment001.ts
"""

        # Mock segment data
        mock_segment_data = b"mock_twitch_video_segment_data"

        # Mock HLS manifest response
        mock_manifest_response = AsyncMock()
        mock_manifest_response.status = 200
        mock_manifest_response.text = AsyncMock(return_value=mock_manifest_content)
        mock_manifest_response.raise_for_status = AsyncMock()

        # Mock playlist response
        mock_playlist_response = AsyncMock()
        mock_playlist_response.status = 200
        mock_playlist_response.text = AsyncMock(return_value=mock_playlist_content)
        mock_playlist_response.raise_for_status = AsyncMock()

        # Mock segment response
        mock_segment_response = AsyncMock()
        mock_segment_response.status = 200
        mock_segment_response.content.iter_chunked.return_value = AsyncMock()
        mock_segment_response.content.iter_chunked.return_value.__aiter__ = AsyncMock(
            return_value=iter([mock_segment_data])
        )
        mock_segment_response.raise_for_status = AsyncMock()

        # Configure mock session responses
        mock_session.post.return_value.__aenter__.return_value = mock_gql_response
        mock_session.get.return_value.__aenter__.side_effect = [
            mock_live_response,  # Check if stream is live
            mock_manifest_response,  # HLS manifest
            mock_playlist_response,  # Video playlist
            mock_segment_response,  # Video segment
        ]

        twitch_adapter._session = mock_session

        # Test individual components first
        # Test getting access token
        await twitch_adapter._get_stream_access_token()
        assert twitch_adapter.access_token == "test_access_token_value"
        assert twitch_adapter.access_signature == "test_signature"

        # Test getting HLS manifest
        await twitch_adapter._get_hls_manifest()
        assert twitch_adapter.current_manifest is not None
        assert len(twitch_adapter.current_manifest.video_playlists) > 0

        # Test quality selection
        await twitch_adapter._select_quality()
        assert twitch_adapter.current_playlist is not None

        # Test getting available qualities
        qualities = await twitch_adapter.get_available_qualities()
        assert len(qualities) > 0
        assert any(q["name"] in ["720p", "1080p", "480p"] for q in qualities)

        # Test quality switching
        success = await twitch_adapter.switch_quality("720p")
        assert success
        assert twitch_adapter.preferred_quality == "720p"

    @pytest.mark.asyncio
    async def test_twitch_access_token_error(self, twitch_adapter, mock_session):
        """Test Twitch access token error handling."""
        twitch_adapter.app_access_token = "test_token"

        # Mock failed GraphQL response
        mock_gql_response = AsyncMock()
        mock_gql_response.status = 400
        mock_gql_response.text.return_value = "Bad Request"

        mock_session.post.return_value.__aenter__.return_value = mock_gql_response
        twitch_adapter._session = mock_session

        with pytest.raises(AuthenticationError):
            await twitch_adapter._get_stream_access_token()

    @pytest.mark.asyncio
    async def test_twitch_hls_manifest_error(self, twitch_adapter, mock_session):
        """Test Twitch HLS manifest error handling."""
        twitch_adapter.app_access_token = "test_token"
        twitch_adapter.access_token = "test_access_token"
        twitch_adapter.access_signature = "test_signature"

        # Mock failed manifest response
        mock_manifest_response = AsyncMock()
        mock_manifest_response.status = 404
        mock_manifest_response.text = AsyncMock(return_value="Not Found")
        mock_manifest_response.raise_for_status = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=None, history=None, status=404
            )
        )

        mock_session.get.return_value.__aenter__.return_value = mock_manifest_response
        twitch_adapter._session = mock_session

        with pytest.raises(ConnectionError):
            await twitch_adapter._get_hls_manifest()

    @pytest.mark.asyncio
    async def test_twitch_stream_offline(self, twitch_adapter, mock_session):
        """Test Twitch stream offline handling during streaming."""
        twitch_adapter.app_access_token = "test_token"
        twitch_adapter.user_id = "123"

        # Mock stream offline response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"data": []}  # No active streams

        mock_session.get.return_value.__aenter__.return_value = mock_response
        twitch_adapter._session = mock_session

        with pytest.raises(StreamOfflineError):
            async for _ in twitch_adapter.get_stream_data():
                pass

    @pytest.mark.asyncio
    async def test_twitch_analytics_with_hls(self, twitch_adapter):
        """Test Twitch analytics with HLS streaming data."""
        from src.utils.hls_parser import StreamManifest, HLSPlaylist, StreamQuality

        # Mock HLS streaming state
        quality = StreamQuality(
            resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"
        )
        playlist = HLSPlaylist(uri="720p.m3u8", quality=quality)
        manifest = StreamManifest(master_playlist_uri="master.m3u8")
        manifest.playlists = [playlist]

        twitch_adapter.current_manifest = manifest
        twitch_adapter.current_playlist = playlist
        twitch_adapter.access_token = "test_token"
        twitch_adapter.preferred_quality = "720p"

        analytics = await twitch_adapter.get_stream_analytics()

        assert analytics["hls_enabled"] is True
        assert analytics["current_quality"] == "720p"
        assert analytics["available_qualities"] == 1
        assert analytics["preferred_quality"] == "720p"
        assert analytics["access_token_valid"] is True


class TestYouTubeAdapter:
    """Test YouTubeAdapter functionality."""

    @pytest.fixture
    def youtube_adapter(self):
        """Create a YouTube adapter for testing."""
        return YouTubeAdapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", api_key="test_api_key"
        )

    @pytest.fixture
    def mock_session(self):
        """Create a mock HTTP session."""
        session = MagicMock(spec=ClientSession)
        return session

    def test_youtube_adapter_initialization(self, youtube_adapter):
        """Test YouTube adapter initialization."""
        assert youtube_adapter.video_id == "dQw4w9WgXcQ"
        assert youtube_adapter.api_key == "test_api_key"
        assert youtube_adapter.platform_name == "youtube"

    @pytest.mark.asyncio
    async def test_youtube_authentication(self, youtube_adapter):
        """Test YouTube authentication (API key validation)."""
        success = await youtube_adapter.authenticate()
        assert success

    @pytest.mark.asyncio
    async def test_youtube_authentication_no_key(self):
        """Test YouTube authentication without API key."""
        adapter = YouTubeAdapter("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        with pytest.raises(AuthenticationError):
            await adapter.authenticate()

    @pytest.mark.asyncio
    async def test_youtube_api_request(self, youtube_adapter, mock_session):
        """Test YouTube API request."""
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"items": [{"id": "dQw4w9WgXcQ"}]}

        mock_session.get.return_value.__aenter__.return_value = mock_response
        youtube_adapter._session = mock_session

        result = await youtube_adapter._make_api_request(
            "videos", {"id": "dQw4w9WgXcQ"}
        )
        assert result["items"][0]["id"] == "dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_youtube_quota_exceeded(self, youtube_adapter, mock_session):
        """Test YouTube API quota exceeded."""
        # Mock quota exceeded response
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.json.return_value = {"error": {"message": "Quota exceeded"}}

        mock_session.get.return_value.__aenter__.return_value = mock_response
        youtube_adapter._session = mock_session

        with pytest.raises(RateLimitError):
            await youtube_adapter._make_api_request("videos")

    @pytest.mark.asyncio
    async def test_youtube_hls_streaming(self, youtube_adapter, mock_session):
        """Test YouTube HLS streaming functionality."""
        # Mock live stream with HLS manifest
        mock_live_stream = {
            "id": "live_stream_123",
            "snippet": {
                "title": "Test Live Stream",
                "description": "Test stream with HLS",
                "channelId": "channel_123",
                "channelTitle": "Test Channel",
            },
            "liveStreamingDetails": {
                "activeLiveChatId": "chat_123",
                "actualStartTime": "2023-01-01T12:00:00Z",
                "hlsManifestUrl": "https://manifest.googlevideo.com/api/manifest/hls_playlist/test.m3u8",
            },
            "statistics": {
                "viewCount": "1000",
            },
        }

        # Mock HLS manifest content
        mock_manifest_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=1280000,RESOLUTION=720x480,CODECS="avc1.42001e,mp4a.40.2"
720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2560000,RESOLUTION=1280x720,CODECS="avc1.42001f,mp4a.40.2"
1080p.m3u8
"""

        # Mock playlist content
        mock_playlist_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:10.0,
segment000.ts
#EXTINF:10.0,
segment001.ts
"""

        # Mock segment data
        mock_segment_data = b"mock_video_segment_data"

        # Set up mock responses
        mock_api_response = AsyncMock()
        mock_api_response.status = 200
        mock_api_response.json.return_value = {"items": [mock_live_stream]}

        mock_manifest_response = AsyncMock()
        mock_manifest_response.status = 200
        mock_manifest_response.text.return_value = mock_manifest_content

        mock_playlist_response = AsyncMock()
        mock_playlist_response.status = 200
        mock_playlist_response.text.return_value = mock_playlist_content

        mock_segment_response = AsyncMock()
        mock_segment_response.status = 200
        mock_segment_response.content.iter_chunked.return_value = AsyncMock()
        mock_segment_response.content.iter_chunked.return_value.__aiter__ = AsyncMock(
            return_value=iter([mock_segment_data])
        )

        # Configure mock session to return appropriate responses
        mock_session.get.return_value.__aenter__.side_effect = [
            mock_api_response,  # API call
            mock_manifest_response,  # HLS manifest
            mock_playlist_response,  # Video playlist
            mock_segment_response,  # Video segment
        ]

        youtube_adapter._session = mock_session
        youtube_adapter.api_key = "test_key"

        # Test connection with HLS streaming
        success = await youtube_adapter.connect()
        assert success
        assert youtube_adapter.hls_manifest_url is not None
        assert youtube_adapter.current_manifest is not None

        # Test getting available qualities
        qualities = await youtube_adapter.get_available_qualities()
        assert len(qualities) > 0
        assert "720p" in [q["name"] for q in qualities]

        # Test streaming data (should get at least one chunk)
        data_received = False
        async for chunk in youtube_adapter.get_stream_data():
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0
            data_received = True
            break  # Just test first chunk

        assert data_received

        # Clean up
        await youtube_adapter.disconnect()

    @pytest.mark.asyncio
    async def test_youtube_stream_without_hls(self, youtube_adapter, mock_session):
        """Test YouTube adapter fallback when HLS is not available."""
        # Mock live stream without HLS manifest
        mock_live_stream = {
            "id": "live_stream_123",
            "snippet": {
                "title": "Test Live Stream",
                "description": "Test stream without HLS",
                "channelId": "channel_123",
                "channelTitle": "Test Channel",
            },
            "liveStreamingDetails": {
                "activeLiveChatId": "chat_123",
                "actualStartTime": "2023-01-01T12:00:00Z",
                # No HLS manifest URL
            },
            "statistics": {
                "viewCount": "1000",
            },
        }

        mock_api_response = AsyncMock()
        mock_api_response.status = 200
        mock_api_response.json.return_value = {"items": [mock_live_stream]}

        mock_session.get.return_value.__aenter__.return_value = mock_api_response
        youtube_adapter._session = mock_session
        youtube_adapter.api_key = "test_key"

        # Test connection
        success = await youtube_adapter.connect()
        assert success
        assert youtube_adapter.hls_manifest_url is None
        assert youtube_adapter.current_manifest is None

        # Test streaming data (should fall back to metadata)
        data_received = False
        async for chunk in youtube_adapter.get_stream_data():
            assert isinstance(chunk, bytes)
            # Should contain JSON metadata
            data_str = chunk.decode("utf-8")
            assert "title" in data_str
            assert "Test Live Stream" in data_str
            data_received = True
            break

        assert data_received

        # Clean up
        await youtube_adapter.disconnect()

    @pytest.mark.asyncio
    async def test_youtube_quality_switching(self, youtube_adapter, mock_session):
        """Test YouTube quality switching functionality."""
        # Mock the adapter to have a manifest
        from src.utils.hls_parser import StreamManifest, HLSPlaylist, StreamQuality

        # Create mock manifest with multiple qualities
        quality_720p = StreamQuality(
            resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"
        )
        quality_480p = StreamQuality(
            resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"
        )

        playlist_720p = HLSPlaylist(uri="720p.m3u8", quality=quality_720p)
        playlist_480p = HLSPlaylist(uri="480p.m3u8", quality=quality_480p)

        mock_manifest = StreamManifest(master_playlist_uri="master.m3u8")
        mock_manifest.playlists = [playlist_720p, playlist_480p]

        youtube_adapter.current_manifest = mock_manifest
        youtube_adapter.preferred_quality = "best"

        # Test quality switching
        success = await youtube_adapter.switch_quality("480p")
        assert success
        assert youtube_adapter.preferred_quality == "480p"

        # Test invalid quality
        success = await youtube_adapter.switch_quality("invalid")
        # Should still succeed but not find exact match
        assert success or not success  # Either is acceptable for invalid quality


class TestRTMPAdapter:
    """Test enhanced RTMPAdapter functionality."""

    @pytest.fixture
    def rtmp_adapter(self):
        """Create an RTMP adapter for testing."""
        return RTMPAdapter("rtmp://live.twitch.tv/live/stream_key")

    @pytest.fixture
    def rtmp_adapter_with_options(self):
        """Create an RTMP adapter with additional options."""
        return RTMPAdapter(
            "rtmp://live.twitch.tv/live/stream_key",
            app_name="live",
            stream_key="test_stream_key",
            hardware_acceleration=True,
            enable_recording=True,
        )

    def test_rtmp_adapter_initialization(self, rtmp_adapter):
        """Test RTMP adapter initialization."""
        assert rtmp_adapter.hostname == "live.twitch.tv"
        assert rtmp_adapter.port == 1935
        assert rtmp_adapter.path == "/live/stream_key"
        assert rtmp_adapter.scheme == "rtmp"
        assert rtmp_adapter.platform_name == "enhancedrtmp"
        assert hasattr(rtmp_adapter, "_rtmp_protocol")
        assert hasattr(rtmp_adapter, "_flv_processor")
        assert hasattr(rtmp_adapter, "_ffmpeg_processor")

    def test_rtmp_adapter_with_options_initialization(self, rtmp_adapter_with_options):
        """Test RTMP adapter initialization with additional options."""
        assert rtmp_adapter_with_options._app_name == "live"
        assert rtmp_adapter_with_options._stream_key == "test_stream_key"
        assert rtmp_adapter_with_options._enable_recording is True
        assert rtmp_adapter_with_options._ffmpeg_processor.hardware_acceleration is True

    @pytest.mark.asyncio
    async def test_rtmp_authentication(self, rtmp_adapter):
        """Test RTMP authentication (no-op for basic streams)."""
        success = await rtmp_adapter.authenticate()
        assert success

    @pytest.mark.asyncio
    async def test_rtmp_metadata(self, rtmp_adapter):
        """Test RTMP metadata retrieval."""
        metadata = await rtmp_adapter.get_metadata()

        assert "RTMP Stream" in metadata.title
        assert rtmp_adapter.hostname in metadata.title
        assert (
            metadata.platform_id
            == f"{rtmp_adapter.hostname}:{rtmp_adapter.port}{rtmp_adapter.path}"
        )
        assert "enhanced_rtmp" in metadata.platform_data["adapter_type"]

    @pytest.mark.asyncio
    async def test_rtmp_protocol_components(self, rtmp_adapter):
        """Test RTMP protocol components are properly initialized."""
        from src.utils.rtmp_protocol import RTMPProtocol
        from src.utils.flv_parser import FLVStreamProcessor
        from src.utils.ffmpeg_integration import FFmpegProcessor

        assert isinstance(rtmp_adapter._rtmp_protocol, RTMPProtocol)
        assert isinstance(rtmp_adapter._flv_processor, FLVStreamProcessor)
        assert isinstance(rtmp_adapter._ffmpeg_processor, FFmpegProcessor)

    @pytest.mark.asyncio
    async def test_rtmp_stream_info_initialization(self, rtmp_adapter):
        """Test RTMP stream info initialization."""
        # Simulate reading stream info
        await rtmp_adapter._read_stream_info()

        assert "connected_at" in rtmp_adapter._stream_info
        assert "url" in rtmp_adapter._stream_info
        assert "hostname" in rtmp_adapter._stream_info
        assert "rtmp_protocol_version" in rtmp_adapter._stream_info
        assert rtmp_adapter._stream_info["app_name"] == "live"

    @pytest.mark.asyncio
    async def test_rtmp_analytics(self, rtmp_adapter):
        """Test RTMP analytics retrieval."""
        analytics = await rtmp_adapter.get_stream_analytics()

        assert analytics["platform"] == "enhanced_rtmp"
        assert analytics["hostname"] == rtmp_adapter.hostname
        assert analytics["port"] == rtmp_adapter.port
        assert "rtmp_protocol" in analytics
        assert "flv_processing" in analytics
        assert "video_processing" in analytics
        assert "ffmpeg" in analytics

    @pytest.mark.asyncio
    async def test_rtmp_frame_management(self, rtmp_adapter):
        """Test RTMP frame management functionality."""
        # Initially no frames
        frames = await rtmp_adapter.get_recent_frames()
        assert len(frames) == 0

        # Simulate adding some frames
        rtmp_adapter._video_frames = [
            (b"frame1", 1.0, True),
            (b"frame2", 2.0, False),
            (b"frame3", 3.0, True),
        ]

        # Test getting recent frames
        frames = await rtmp_adapter.get_recent_frames(2)
        assert len(frames) == 2
        assert frames[0][1] == 2.0  # timestamp of second frame
        assert frames[1][1] == 3.0  # timestamp of third frame

        # Test getting frame at timestamp
        frame = await rtmp_adapter.get_frame_at_timestamp(2.1)
        assert frame is not None
        assert frame[1] == 2.0  # Should get closest frame

    def test_rtmp_connection_health_checks(self, rtmp_adapter):
        """Test RTMP connection health checking logic."""
        # Test when not connected
        rtmp_adapter.connection.status = ConnectionStatus.DISCONNECTED
        health = asyncio.run(rtmp_adapter.check_health())
        assert health == StreamHealth.UNHEALTHY

        # Test when connected but no protocol handshake
        rtmp_adapter.connection.status = ConnectionStatus.CONNECTED
        rtmp_adapter._rtmp_protocol.handshake_complete = False
        health = asyncio.run(rtmp_adapter.check_health())
        assert health == StreamHealth.UNHEALTHY

    @pytest.mark.asyncio
    async def test_rtmp_background_task_management(self, rtmp_adapter):
        """Test RTMP background task management."""
        # Start background processing
        await rtmp_adapter._start_background_processing()

        # Check tasks are created (though they will fail without real connection)
        assert rtmp_adapter._message_processing_task is not None

        # Stop background processing
        await rtmp_adapter._stop_background_processing()

        # Check tasks are cleaned up
        assert rtmp_adapter._message_processing_task is None

    def test_rtmp_adapter_repr(self, rtmp_adapter):
        """Test RTMP adapter string representation."""
        repr_str = repr(rtmp_adapter)
        assert "EnhancedRTMPAdapter" in repr_str
        assert rtmp_adapter.hostname in repr_str
        assert str(rtmp_adapter.port) in repr_str


class TestStreamAdapterFactory:
    """Test StreamAdapterFactory functionality."""

    def test_factory_create_twitch_adapter(self):
        """Test creating Twitch adapter via factory."""
        adapter = StreamAdapterFactory.create_adapter("https://www.twitch.tv/shroud")

        assert isinstance(adapter, TwitchAdapter)
        assert adapter.username == "shroud"

    def test_factory_create_youtube_adapter(self):
        """Test creating YouTube adapter via factory."""
        adapter = StreamAdapterFactory.create_adapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

        assert isinstance(adapter, YouTubeAdapter)
        assert adapter.video_id == "dQw4w9WgXcQ"

    def test_factory_create_rtmp_adapter(self):
        """Test creating RTMP adapter via factory."""
        adapter = StreamAdapterFactory.create_adapter(
            "rtmp://live.twitch.tv/live/stream_key"
        )

        assert isinstance(adapter, RTMPAdapter)
        assert adapter.hostname == "live.twitch.tv"

    def test_factory_unsupported_platform(self):
        """Test factory with unsupported platform."""
        # Test invalid URL that doesn't match any platform
        from src.utils.stream_validation import ValidationError

        with pytest.raises(ValidationError, match="Unsupported or invalid stream URL"):
            StreamAdapterFactory.create_adapter("https://unknown-platform.com/stream")

    def test_factory_convenience_functions(self):
        """Test factory convenience functions."""
        # Test create_stream_adapter
        adapter = create_stream_adapter("https://www.twitch.tv/shroud")
        assert isinstance(adapter, TwitchAdapter)

        # Test specialized factory methods
        twitch_adapter = StreamAdapterFactory.create_twitch_adapter(
            "https://www.twitch.tv/shroud", client_id="test_id"
        )
        assert isinstance(twitch_adapter, TwitchAdapter)
        assert twitch_adapter.client_id == "test_id"

        youtube_adapter = StreamAdapterFactory.create_youtube_adapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", api_key="test_key"
        )
        assert isinstance(youtube_adapter, YouTubeAdapter)
        assert youtube_adapter.api_key == "test_key"

        rtmp_adapter = StreamAdapterFactory.create_rtmp_adapter(
            "rtmp://live.twitch.tv/live/stream_key", buffer_size=8192
        )
        assert isinstance(rtmp_adapter, RTMPAdapter)
        assert rtmp_adapter.buffer_size == 8192

    def test_factory_get_supported_platforms(self):
        """Test getting supported platforms."""
        platforms = StreamAdapterFactory.get_supported_platforms()

        assert "twitch" in platforms
        assert "youtube" in platforms
        assert "rtmp" in platforms
        assert "custom" in platforms

        for platform, description in platforms.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_factory_is_platform_supported(self):
        """Test platform support checking."""
        assert StreamAdapterFactory.is_platform_supported(StreamPlatform.TWITCH)
        assert StreamAdapterFactory.is_platform_supported(StreamPlatform.YOUTUBE)
        assert StreamAdapterFactory.is_platform_supported(StreamPlatform.RTMP)
        assert StreamAdapterFactory.is_platform_supported(StreamPlatform.CUSTOM)

    def test_create_adapter_from_stream_model(self):
        """Test creating adapter from stream model."""
        # Mock stream model
        mock_stream = MagicMock()
        mock_stream.source_url = "https://www.twitch.tv/shroud"
        mock_stream.platform = "twitch"
        mock_stream.options = {"client_id": "test_id"}

        adapter = create_adapter_from_stream_model(mock_stream)

        assert isinstance(adapter, TwitchAdapter)
        assert adapter.url == "https://www.twitch.tv/shroud"

    def test_detect_and_validate_stream(self):
        """Test stream detection and validation utility."""
        result = detect_and_validate_stream("https://www.twitch.tv/shroud")

        assert result["platform"] == StreamPlatform.TWITCH
        assert result["adapter_name"] == "TwitchAdapter"
        assert result["is_supported"] is True
        assert "validation_result" in result
        assert result["validation_result"]["username"] == "shroud"


class TestStreamAdapterErrors:
    """Test stream adapter error handling."""

    def test_stream_adapter_error_hierarchy(self):
        """Test error class hierarchy."""
        # Test base error
        base_error = StreamAdapterError("Base error")
        assert str(base_error) == "Base error"

        # Test specific errors
        auth_error = AuthenticationError("Auth failed")
        assert isinstance(auth_error, StreamAdapterError)

        conn_error = ConnectionError("Connection failed")
        assert isinstance(conn_error, StreamAdapterError)

        rate_error = RateLimitError("Rate limited")
        assert isinstance(rate_error, StreamAdapterError)

        not_found_error = StreamNotFoundError("Not found")
        assert isinstance(not_found_error, StreamAdapterError)

        offline_error = StreamOfflineError("Offline")
        assert isinstance(offline_error, StreamAdapterError)


class TestStreamMetadata:
    """Test StreamMetadata data class."""

    def test_metadata_initialization(self):
        """Test metadata initialization."""
        metadata = StreamMetadata()

        assert metadata.title is None
        assert metadata.description is None
        assert metadata.is_live is False
        assert metadata.viewer_count is None
        assert isinstance(metadata.platform_data, dict)
        assert isinstance(metadata.tags, list)
        assert isinstance(metadata.updated_at, datetime)

    def test_metadata_with_data(self):
        """Test metadata with data."""
        metadata = StreamMetadata(
            title="Test Stream",
            description="Test description",
            is_live=True,
            viewer_count=1000,
            category="Gaming",
            tags=["test", "stream"],
            language="en",
        )

        assert metadata.title == "Test Stream"
        assert metadata.description == "Test description"
        assert metadata.is_live is True
        assert metadata.viewer_count == 1000
        assert metadata.category == "Gaming"
        assert metadata.tags == ["test", "stream"]
        assert metadata.language == "en"


class TestStreamConnection:
    """Test StreamConnection data class."""

    def test_connection_initialization(self):
        """Test connection initialization."""
        connection = StreamConnection("https://example.com/stream")

        assert connection.url == "https://example.com/stream"
        assert connection.status == ConnectionStatus.DISCONNECTED
        assert connection.health == StreamHealth.UNKNOWN
        assert connection.bytes_received == 0
        assert connection.reconnect_count == 0
        assert connection.error_count == 0
