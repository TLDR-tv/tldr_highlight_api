"""Integration tests for platform connections.

These tests verify that the stream adapters can actually connect to
real streaming platforms and handle real API responses. Some tests
may be skipped if API credentials are not available.
"""

import asyncio
import os

import pytest
import aiohttp

from src.infrastructure.config import get_settings
from src.services.stream_adapters.factory import StreamAdapterFactory
from src.services.stream_adapters.twitch import TwitchAdapter
from src.services.stream_adapters.youtube import YouTubeAdapter
from src.services.stream_adapters.rtmp import RTMPAdapter
from src.services.stream_adapters.base import (
    AuthenticationError,
    ConnectionError,
    StreamNotFoundError,
)


settings = get_settings()


@pytest.fixture
def session():
    """Create an aiohttp session for tests."""
    return aiohttp.ClientSession()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class TestTwitchIntegration:
    """Integration tests for Twitch adapter."""

    @pytest.fixture
    def twitch_credentials(self):
        """Get Twitch credentials from environment or settings."""
        client_id = os.getenv("TWITCH_CLIENT_ID") or settings.twitch_client_id
        client_secret = (
            os.getenv("TWITCH_CLIENT_SECRET") or settings.twitch_client_secret
        )

        if not client_id or not client_secret:
            pytest.skip("Twitch credentials not available")

        return {"client_id": client_id, "client_secret": client_secret}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_authentication_real(self, session, twitch_credentials):
        """Test real Twitch authentication."""
        adapter = TwitchAdapter(
            "https://www.twitch.tv/shroud", session=session, **twitch_credentials
        )

        try:
            success = await adapter.authenticate()
            assert success
            assert adapter.app_access_token is not None
            assert len(adapter.app_access_token) > 0
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_authentication_invalid_credentials(self, session):
        """Test Twitch authentication with invalid credentials."""
        adapter = TwitchAdapter(
            "https://www.twitch.tv/shroud",
            client_id="invalid_id",
            client_secret="invalid_secret",
            session=session,
        )

        try:
            with pytest.raises(AuthenticationError):
                await adapter.authenticate()
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_user_lookup_existing(self, session, twitch_credentials):
        """Test looking up an existing Twitch user."""
        adapter = TwitchAdapter(
            "https://www.twitch.tv/shroud",  # Known popular streamer
            session=session,
            **twitch_credentials,
        )

        try:
            await adapter.authenticate()
            user_info = await adapter._get_user_info()

            assert user_info["login"] == "shroud"
            assert "id" in user_info
            assert adapter.user_id is not None
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_user_lookup_nonexistent(self, session, twitch_credentials):
        """Test looking up a non-existent Twitch user."""
        adapter = TwitchAdapter(
            "https://www.twitch.tv/nonexistent_user_12345678",
            session=session,
            **twitch_credentials,
        )

        try:
            await adapter.authenticate()
            with pytest.raises(StreamNotFoundError):
                await adapter._get_user_info()
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_stream_status_check(self, session, twitch_credentials):
        """Test checking Twitch stream live status."""
        # Use a popular streamer who is likely to be streaming
        adapter = TwitchAdapter(
            "https://www.twitch.tv/shroud", session=session, **twitch_credentials
        )

        try:
            await adapter.authenticate()
            await adapter._get_user_info()

            # Check if stream is live (result may vary)
            is_live = await adapter.is_stream_live()
            assert isinstance(is_live, bool)

            # Get metadata regardless of live status
            metadata = await adapter.get_metadata()
            assert metadata is not None
            assert metadata.platform_data["user_login"] == "shroud"
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_twitch_rate_limiting(self, session, twitch_credentials):
        """Test Twitch API rate limiting behavior."""
        adapter = TwitchAdapter(
            "https://www.twitch.tv/shroud", session=session, **twitch_credentials
        )

        try:
            await adapter.authenticate()

            # Make multiple API requests to test rate limiting
            initial_remaining = adapter.rate_limit_remaining

            for _ in range(5):
                await adapter._make_api_request("users", {"login": "shroud"})

            # Rate limit should have decreased
            assert adapter.rate_limit_remaining <= initial_remaining
        finally:
            await adapter.stop()


class TestYouTubeIntegration:
    """Integration tests for YouTube adapter."""

    @pytest.fixture
    def youtube_credentials(self):
        """Get YouTube credentials from environment or settings."""
        api_key = os.getenv("YOUTUBE_API_KEY") or settings.youtube_api_key

        if not api_key:
            pytest.skip("YouTube API key not available")

        return {"api_key": api_key}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_authentication_real(self, session, youtube_credentials):
        """Test real YouTube authentication."""
        adapter = YouTubeAdapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            session=session,
            **youtube_credentials,
        )

        try:
            success = await adapter.authenticate()
            assert success
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_authentication_invalid_key(self, session):
        """Test YouTube authentication with invalid API key."""
        adapter = YouTubeAdapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            api_key="invalid_key",
            session=session,
        )

        try:
            # Authentication itself succeeds (just checks if key exists)
            success = await adapter.authenticate()
            assert success

            # But API requests should fail
            with pytest.raises((AuthenticationError, ConnectionError)):
                await adapter._make_api_request("videos", {"id": "dQw4w9WgXcQ"})
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_video_lookup_existing(self, session, youtube_credentials):
        """Test looking up an existing YouTube video."""
        # Use Rick Astley's "Never Gonna Give You Up" - famous and stable video
        adapter = YouTubeAdapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            session=session,
            **youtube_credentials,
        )

        try:
            await adapter.authenticate()

            response = await adapter._make_api_request(
                "videos", {"part": "snippet,statistics", "id": "dQw4w9WgXcQ"}
            )

            videos = response.get("items", [])
            assert len(videos) > 0

            video = videos[0]
            assert video["id"] == "dQw4w9WgXcQ"
            assert "snippet" in video
            assert "statistics" in video
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_video_lookup_nonexistent(self, session, youtube_credentials):
        """Test looking up a non-existent YouTube video."""
        adapter = YouTubeAdapter(
            "https://www.youtube.com/watch?v=invalidvideoID",
            session=session,
            **youtube_credentials,
        )

        try:
            await adapter.authenticate()

            response = await adapter._make_api_request(
                "videos", {"part": "snippet", "id": "invalidvideoID"}
            )

            videos = response.get("items", [])
            assert len(videos) == 0
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_channel_resolution(self, session, youtube_credentials):
        """Test resolving channel ID from various URL formats."""
        test_cases = [
            # Use well-known channels that are unlikely to disappear
            (
                "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw",
                "UCuAXFkgsw1L7xaCfnd5JJOw",
            ),  # Real channel ID
        ]

        for url, expected_channel_id in test_cases:
            adapter = YouTubeAdapter(url, session=session, **youtube_credentials)

            try:
                await adapter.authenticate()
                resolved_id = await adapter._resolve_channel_id()
                assert resolved_id == expected_channel_id
            finally:
                await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_youtube_quota_usage(self, session, youtube_credentials):
        """Test YouTube API quota usage tracking."""
        adapter = YouTubeAdapter(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            session=session,
            **youtube_credentials,
        )

        try:
            await adapter.authenticate()

            # Make a few API requests
            for _ in range(3):
                await adapter._make_api_request(
                    "videos", {"part": "snippet", "id": "dQw4w9WgXcQ"}
                )

            # Should complete without quota errors for small number of requests
            # Actual quota tracking would require monitoring daily usage
        finally:
            await adapter.stop()


class TestRTMPIntegration:
    """Integration tests for RTMP adapter."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rtmp_connection_invalid_host(self, session):
        """Test RTMP connection to invalid host."""
        adapter = RTMPAdapter(
            "rtmp://nonexistent.example.com/live/stream",
            session=session,
            connection_timeout=5,  # Short timeout for test
        )

        try:
            await adapter.authenticate()

            with pytest.raises(ConnectionError):
                await adapter.connect()
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rtmp_connection_unreachable_port(self, session):
        """Test RTMP connection to unreachable port."""
        adapter = RTMPAdapter(
            "rtmp://google.com:9999/live/stream",  # Google doesn't run RTMP on port 9999
            session=session,
            connection_timeout=5,
        )

        try:
            await adapter.authenticate()

            with pytest.raises(ConnectionError):
                await adapter.connect()
        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rtmp_metadata_generation(self, session):
        """Test RTMP metadata generation without connection."""
        adapter = RTMPAdapter("rtmp://live.twitch.tv/live/stream_key", session=session)

        try:
            # Test metadata generation without connecting
            metadata = await adapter.get_metadata()

            assert "RTMP Stream" in metadata.title
            assert "live.twitch.tv" in metadata.title
            assert metadata.platform_id == "live.twitch.tv:1935/live/stream_key"
            assert metadata.platform_data["hostname"] == "live.twitch.tv"
            assert metadata.platform_data["port"] == 1935
            assert metadata.platform_data["scheme"] == "rtmp"
        finally:
            await adapter.stop()


class TestFactoryIntegration:
    """Integration tests for adapter factory."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_factory_creates_working_adapters(self, session):
        """Test that factory creates working adapters for different platforms."""
        test_urls = [
            "https://www.twitch.tv/shroud",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "rtmp://live.twitch.tv/live/stream_key",
        ]

        for url in test_urls:
            adapter = StreamAdapterFactory.create_adapter(url, session=session)

            try:
                # Test basic functionality
                assert adapter.url == url
                assert hasattr(adapter, "authenticate")
                assert hasattr(adapter, "connect")
                assert hasattr(adapter, "get_metadata")

                # Test authentication (may fail for missing credentials, but should not crash)
                try:
                    await adapter.authenticate()
                except AuthenticationError:
                    # Expected if credentials are not available
                    pass

            finally:
                await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_adapter_error_handling(self, session):
        """Test adapter error handling in real scenarios."""
        # Test with invalid Twitch user
        adapter = StreamAdapterFactory.create_adapter(
            "https://www.twitch.tv/nonexistent_user_12345678", session=session
        )

        try:
            # This should work even without valid credentials for basic initialization
            assert isinstance(adapter, TwitchAdapter)
            assert adapter.username == "nonexistent_user_12345678"
        finally:
            await adapter.stop()


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_adapter_lifecycle(self, session):
        """Test complete adapter lifecycle from creation to cleanup."""
        # Use a stable YouTube video for testing
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # Skip if no API key available
        if not (os.getenv("YOUTUBE_API_KEY") or settings.youtube_api_key):
            pytest.skip("YouTube API key not available")

        # Create adapter
        adapter = StreamAdapterFactory.create_adapter(url, session=session)

        try:
            # Test full lifecycle
            await adapter.start()

            # Get metadata
            metadata = await adapter.get_metadata()
            assert metadata is not None

            # Check health
            health = await adapter.check_health()
            assert health is not None

            # Get analytics
            analytics = await adapter.get_stream_analytics()
            assert analytics is not None
            assert analytics["platform"] == "youtube"

        finally:
            await adapter.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_adapters_concurrent(self, session):
        """Test running multiple adapters concurrently."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "rtmp://live.twitch.tv/live/stream1",
            "rtmp://live.twitch.tv/live/stream2",
        ]

        adapters = []

        try:
            # Create adapters
            for url in urls:
                adapter = StreamAdapterFactory.create_adapter(url, session=session)
                adapters.append(adapter)

            # Test concurrent authentication
            auth_tasks = [adapter.authenticate() for adapter in adapters]
            auth_results = await asyncio.gather(*auth_tasks, return_exceptions=True)

            # At least some should succeed (YouTube with valid key, RTMP always succeeds)
            successful_auths = [r for r in auth_results if r is True]
            assert len(successful_auths) > 0

            # Test concurrent metadata retrieval
            metadata_tasks = [adapter.get_metadata() for adapter in adapters]
            metadata_results = await asyncio.gather(
                *metadata_tasks, return_exceptions=True
            )

            # All should return metadata objects (even if not connected)
            successful_metadata = [r for r in metadata_results if hasattr(r, "title")]
            assert len(successful_metadata) > 0

        finally:
            # Clean up all adapters
            cleanup_tasks = [adapter.stop() for adapter in adapters]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
