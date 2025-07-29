"""Twitch stream adapter for Twitch API integration.

This module provides the TwitchAdapter class for connecting to and processing
Twitch streams. It handles Twitch API authentication, stream metadata retrieval,
and stream data access.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Any
from urllib.parse import quote, urlencode

import aiohttp
from aiohttp import ClientSession, ClientResponseError

from src.core.config import get_settings
from src.utils.stream_validation import validate_stream_url
from src.utils.circuit_breaker import get_circuit_breaker, CircuitBreakerError
from src.utils.hls_parser import HLSParser, StreamManifest, HLSPlaylist, StreamQuality
from src.infrastructure.persistence.models.stream import StreamPlatform
from .base import (
    BaseStreamAdapter,
    StreamMetadata,
    ConnectionStatus,
    StreamHealth,
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    StreamNotFoundError,
    StreamOfflineError,
)


logger = logging.getLogger(__name__)
settings = get_settings()


class TwitchAdapter(BaseStreamAdapter):
    """Twitch stream adapter for Twitch API integration.

    This adapter handles authentication with Twitch API, retrieves stream
    metadata, and provides access to stream data for processing.
    """

    def __init__(
        self,
        url: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ):
        """Initialize the Twitch adapter.

        Args:
            url: Twitch stream URL
            client_id: Twitch Client ID (optional, will use config if not provided)
            client_secret: Twitch Client Secret (optional, will use config)
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options
        """
        super().__init__(url, session, **kwargs)

        # Validate URL and extract username
        validation_result = validate_stream_url(url, StreamPlatform.TWITCH)
        self.username = validation_result["username"]

        # API configuration
        self.client_id = client_id or settings.twitch_client_id
        self.client_secret = client_secret or settings.twitch_client_secret
        self.api_base_url = settings.twitch_api_base_url
        self.auth_url = settings.twitch_auth_url

        # Authentication
        self.app_access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

        # Rate limiting
        self.rate_limit_remaining = settings.twitch_rate_limit_per_minute
        self.rate_limit_reset_time: Optional[datetime] = None

        # Stream information
        self.user_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        
        # HLS streaming
        self.hls_parser: Optional[HLSParser] = None
        self.current_manifest: Optional[StreamManifest] = None
        self.current_playlist: Optional[HLSPlaylist] = None
        self.preferred_quality: str = "best"  # best, worst, or specific quality like "720p"
        self.access_token: Optional[str] = None
        self.access_signature: Optional[str] = None
        
        # Twitch GraphQL and Usher endpoints
        self.graphql_url = "https://gql.twitch.tv/gql"
        self.usher_base_url = "https://usher.ttvnw.net/api/channel/hls"

        # Circuit breaker for API resilience
        self._circuit_breaker_name = f"twitch_api_{self.username}"

        logger.info(f"Initialized Twitch adapter for username: {self.username}")

    async def authenticate(self) -> bool:
        """Authenticate with Twitch API using Client Credentials flow.

        Returns:
            bool: True if authentication was successful

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.client_id or not self.client_secret:
            raise AuthenticationError(
                "Twitch Client ID and Client Secret are required for authentication"
            )

        logger.info("Authenticating with Twitch API")

        try:
            # Prepare authentication request
            auth_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            }

            # Make authentication request
            async with self.session.post(self.auth_url, data=auth_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(
                        f"Twitch authentication failed: {response.status} - {error_text}"
                    )

                auth_response = await response.json()

                self.app_access_token = auth_response["access_token"]
                expires_in = auth_response.get("expires_in", 3600)
                self.token_expires_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                )

                logger.info("Successfully authenticated with Twitch API")
                return True

        except ClientResponseError as e:
            logger.error(f"HTTP error during Twitch authentication: {e}")
            raise AuthenticationError(f"HTTP error during authentication: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Twitch authentication: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid authentication token."""
        if not self.app_access_token or (
            self.token_expires_at
            and datetime.utcnow().replace(tzinfo=timezone.utc) >= self.token_expires_at
        ):
            await self.authenticate()

    async def _make_api_request(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to the Twitch API with circuit breaker.

        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters

        Returns:
            Dict: API response data

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ConnectionError: If request fails
            CircuitBreakerError: If circuit breaker is open
        """
        await self._ensure_authenticated()

        # Get circuit breaker for this API
        circuit_breaker = await get_circuit_breaker(
            self._circuit_breaker_name,
            failure_threshold=5,
            timeout_seconds=30,
            recovery_timeout_seconds=60,
            expected_exceptions=(aiohttp.ClientError, RateLimitError, ConnectionError),
        )

        async def make_request():
            url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
            headers = {
                "Client-ID": self.client_id,
                "Authorization": f"Bearer {self.app_access_token}",
                "Accept": "application/json",
            }

            logger.debug(f"Making Twitch API request: {endpoint} with params: {params}")

            try:
                async with self.session.get(
                    url, headers=headers, params=params
                ) as response:
                    # Check rate limiting
                    remaining = response.headers.get("Ratelimit-Remaining")
                    if remaining:
                        self.rate_limit_remaining = int(remaining)
                        logger.debug(f"Twitch API rate limit remaining: {remaining}")

                    reset_time = response.headers.get("Ratelimit-Reset")
                    if reset_time:
                        self.rate_limit_reset_time = datetime.fromtimestamp(
                            int(reset_time), tz=timezone.utc
                        )

                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After", "60")
                        error_msg = f"Twitch API rate limit exceeded. Retry after {retry_after} seconds"
                        logger.warning(error_msg)
                        raise RateLimitError(error_msg)

                    # Handle authentication errors
                    if response.status == 401:
                        # Token may have expired, try to re-authenticate
                        logger.info("Twitch token expired, re-authenticating")
                        self.app_access_token = None
                        await self.authenticate()
                        return await self._make_api_request(endpoint, params)

                    # Handle other errors
                    if response.status >= 400:
                        error_text = await response.text()
                        error_msg = f"Twitch API request failed: {response.status} - {error_text}"
                        logger.error(error_msg)
                        raise ConnectionError(error_msg)

                    response_data = await response.json()
                    logger.debug(f"Twitch API response received for {endpoint}")
                    return response_data

            except aiohttp.ClientError as e:
                error_msg = f"HTTP client error during Twitch API request: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)

        try:
            return await circuit_breaker.call(make_request)
        except CircuitBreakerError as e:
            logger.error(f"Twitch API circuit breaker is open: {e}")
            raise ConnectionError(f"Twitch API temporarily unavailable: {e}")

    async def _get_user_info(self) -> Dict[str, Any]:
        """Get user information by username.

        Returns:
            Dict: User information from Twitch API

        Raises:
            StreamNotFoundError: If user is not found
        """
        response = await self._make_api_request("users", {"login": self.username})

        users = response.get("data", [])
        if not users:
            raise StreamNotFoundError(f"Twitch user '{self.username}' not found")

        user_info = users[0]
        self.user_id = user_info["id"]

        return user_info

    async def connect(self) -> bool:
        """Connect to the Twitch stream.

        Returns:
            bool: True if connection was successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to Twitch stream for user: {self.username}")

        try:
            self.connection.status = ConnectionStatus.CONNECTING

            # Get user information
            _user_info = await self._get_user_info()

            # Check if stream is live
            if not await self.is_stream_live():
                raise StreamOfflineError(f"Twitch stream '{self.username}' is offline")

            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.utcnow()
            self.connection.health = StreamHealth.HEALTHY

            # Notify connection
            await self._notify_connect()

            logger.info(f"Successfully connected to Twitch stream: {self.username}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Twitch stream: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.utcnow()
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect to Twitch stream: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the Twitch stream."""
        logger.info(f"Disconnecting from Twitch stream: {self.username}")

        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN
        
        # Clear HLS streaming state
        self.current_manifest = None
        self.current_playlist = None
        self.access_token = None
        self.access_signature = None
        
        # Clear HLS parser cache if it exists
        if self.hls_parser:
            self.hls_parser.clear_cache()

        await self._notify_disconnect()

        logger.info(f"Disconnected from Twitch stream: {self.username}")

    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata from Twitch API.

        Returns:
            StreamMetadata: Current stream metadata

        Raises:
            StreamAdapterError: If metadata cannot be retrieved
        """
        if not self.user_id:
            await self._get_user_info()

        # Get stream information
        stream_response = await self._make_api_request(
            "streams", {"user_id": self.user_id}
        )

        streams = stream_response.get("data", [])
        if not streams:
            # Stream is offline
            self.metadata.is_live = False
            self.metadata.updated_at = datetime.utcnow()
            return self.metadata

        stream_data = streams[0]

        # Get user information for additional metadata
        user_response = await self._make_api_request("users", {"id": self.user_id})
        user_data = user_response.get("data", [{}])[0]

        # Get game/category information
        game_id = stream_data.get("game_id")
        game_name = stream_data.get("game_name", "")

        # Parse started_at timestamp
        started_at = None
        if stream_data.get("started_at"):
            started_at = datetime.fromisoformat(
                stream_data["started_at"].replace("Z", "+00:00")
            )

        # Update metadata
        self.metadata = StreamMetadata(
            title=stream_data.get("title", ""),
            description=user_data.get("description", ""),
            thumbnail_url=stream_data.get("thumbnail_url", "")
            .replace("{width}", "1920")
            .replace("{height}", "1080"),
            is_live=True,
            viewer_count=stream_data.get("viewer_count", 0),
            started_at=started_at,
            platform_id=self.user_id,
            platform_url=self.url,
            platform_data={
                "stream_id": stream_data.get("id"),
                "user_login": stream_data.get("user_login"),
                "user_name": stream_data.get("user_name"),
                "game_id": game_id,
                "game_name": game_name,
                "type": stream_data.get("type", "live"),
                "language": stream_data.get("language", "en"),
                "is_mature": stream_data.get("is_mature", False),
                "profile_image_url": user_data.get("profile_image_url"),
                "broadcaster_type": user_data.get("broadcaster_type", ""),
            },
            category=game_name,
            tags=stream_data.get("tag_ids", []),
            language=stream_data.get("language", "en"),
            updated_at=datetime.utcnow(),
        )

        self.stream_id = stream_data.get("id")

        return self.metadata

    async def is_stream_live(self) -> bool:
        """Check if the Twitch stream is currently live.

        Returns:
            bool: True if stream is live

        Raises:
            StreamAdapterError: If status cannot be determined
        """
        if not self.user_id:
            await self._get_user_info()

        try:
            response = await self._make_api_request(
                "streams", {"user_id": self.user_id}
            )

            streams = response.get("data", [])
            is_live = len(streams) > 0

            logger.debug(f"Twitch stream {self.username} live status: {is_live}")
            return is_live

        except Exception as e:
            logger.error(f"Error checking Twitch stream live status: {e}")
            raise

    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get stream data as an async generator from Twitch HLS streams.

        This implementation:
        1. Gets a stream access token from Twitch GraphQL API
        2. Retrieves the HLS manifest from usher.ttvnw.net
        3. Parses the manifest and selects appropriate quality
        4. Downloads and yields video segments

        Yields:
            bytes: HLS video segment data chunks

        Raises:
            StreamAdapterError: If stream data cannot be retrieved
            StreamOfflineError: If stream is offline
            AuthenticationError: If access token cannot be obtained
        """
        if not self.user_id:
            await self._get_user_info()

        # Check if stream is live
        if not await self.is_stream_live():
            raise StreamOfflineError(f"Twitch stream '{self.username}' is offline")

        try:
            # Get stream access token
            await self._get_stream_access_token()
            
            # Get HLS manifest
            if not self.current_manifest:
                await self._get_hls_manifest()
            
            # Select quality and get playlist
            if not self.current_playlist:
                await self._select_quality()
            
            # Stream video segments
            async for chunk in self._stream_video_segments():
                # Update connection stats
                self.connection.bytes_received += len(chunk)
                self.connection.packets_received += 1
                self.connection.last_data_at = datetime.now(timezone.utc)
                
                # Notify data callbacks
                await self._notify_data(chunk)
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming Twitch video data: {e}")
            self.connection.error_count += 1
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            await self._notify_error(e)
            raise

    async def get_chat_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent chat messages (if available).

        Note: This requires additional Twitch IRC or EventSub integration.
        This is a placeholder method.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List[Dict]: Chat messages
        """
        logger.warning(
            "Chat message access not implemented. "
            "This would require Twitch IRC or EventSub integration."
        )
        return []

    async def get_stream_markers(self) -> List[Dict[str, Any]]:
        """Get stream markers (if available).

        Returns:
            List[Dict]: Stream markers
        """
        if not self.user_id:
            await self._get_user_info()

        try:
            response = await self._make_api_request(
                "streams/markers", {"user_id": self.user_id}
            )

            return response.get("data", [])

        except Exception as e:
            logger.error(f"Error getting Twitch stream markers: {e}")
            return []

    async def get_stream_analytics(self) -> Dict[str, Any]:
        """Get stream analytics and statistics.

        Returns:
            Dict: Stream analytics data
        """
        analytics = {
            "platform": "twitch",
            "username": self.username,
            "user_id": self.user_id,
            "stream_id": self.stream_id,
            "connection_status": self.connection.status.value,
            "health_status": self.connection.health.value,
            "reconnect_count": self.connection.reconnect_count,
            "error_count": self.connection.error_count,
            "rate_limit_remaining": self.rate_limit_remaining,
            "last_data_at": self.connection.last_data_at,
            "metadata": self.metadata.__dict__ if self.metadata else None,
            # HLS streaming analytics
            "hls_enabled": self.current_manifest is not None,
            "current_quality": self.current_playlist.quality.quality_name if self.current_playlist else None,
            "available_qualities": len(self.current_manifest.video_playlists) if self.current_manifest else 0,
            "preferred_quality": self.preferred_quality,
            "access_token_valid": self.access_token is not None,
            "bytes_received": self.connection.bytes_received,
            "packets_received": self.connection.packets_received,
        }

        return analytics

    async def _get_stream_access_token(self) -> None:
        """Get stream access token from Twitch GraphQL API.
        
        This token is required to access the HLS manifest from usher.ttvnw.net
        
        Raises:
            AuthenticationError: If token cannot be obtained
        """
        logger.info(f"Getting stream access token for {self.username}")
        
        # GraphQL query for getting playback access token
        query = f"""
        {{
            streamPlaybackAccessToken(
                channelName: "{self.username}",
                params: {{
                    platform: "web",
                    playerBackend: "mediaplayer",
                    playerType: "site"
                }}
            ) {{
                value
                signature
                __typename
            }}
        }}
        """
        
        payload = {
            "query": query,
            "variables": {}
        }
        
        headers = {
            "Client-ID": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            async with self.session.post(
                self.graphql_url, 
                json=payload, 
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(
                        f"Failed to get Twitch access token: {response.status} - {error_text}"
                    )
                
                data = await response.json()
                
                if "errors" in data:
                    error_msg = data["errors"][0].get("message", "Unknown GraphQL error")
                    raise AuthenticationError(f"GraphQL error getting access token: {error_msg}")
                
                token_data = data.get("data", {}).get("streamPlaybackAccessToken", {})
                if not token_data:
                    raise AuthenticationError("No access token data in response")
                
                self.access_token = token_data.get("value")
                self.access_signature = token_data.get("signature")
                
                if not self.access_token or not self.access_signature:
                    raise AuthenticationError("Invalid access token or signature")
                
                logger.info("Successfully obtained Twitch stream access token")
                
        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            logger.error(f"Error getting Twitch access token: {e}")
            raise AuthenticationError(f"Failed to get access token: {e}")

    async def _get_hls_manifest(self) -> None:
        """Get HLS manifest from Twitch usher service.
        
        Raises:
            ConnectionError: If manifest cannot be retrieved
        """
        if not self.access_token or not self.access_signature:
            await self._get_stream_access_token()
        
        # Initialize HLS parser if not already done
        if not self.hls_parser:
            self.hls_parser = HLSParser(session=self.session)
        
        logger.info(f"Getting HLS manifest for {self.username}")
        
        # Build usher URL with access token
        params = {
            "token": self.access_token,
            "sig": self.access_signature,
            "allow_source": "true",
            "allow_audio_only": "true",
            "allow_spectre": "false",
            "fast_bread": "true",
            "p": str(hash(self.username) % 9999999),  # Random player ID
            "player_backend": "mediaplayer",
            "playlist_include_framerate": "true",
            "reassignments_supported": "true",
            "rtqos": "control",
            "cdm": "wv",
            "player_version": "1.0.0"
        }
        
        manifest_url = f"{self.usher_base_url}/{self.username}.m3u8"
        
        try:
            # Parse HLS manifest
            self.current_manifest = await self.hls_parser.parse_manifest(
                f"{manifest_url}?{urlencode(params)}", 
                is_youtube=False
            )
            
            logger.info(f"Successfully parsed HLS manifest with {len(self.current_manifest.playlists)} quality options")
            
        except Exception as e:
            logger.error(f"Error getting HLS manifest: {e}")
            raise ConnectionError(f"Failed to get HLS manifest: {e}")

    async def _select_quality(self) -> None:
        """Select quality from available options and get the media playlist.
        
        Raises:
            ConnectionError: If quality selection fails
        """
        if not self.current_manifest or not self.current_manifest.video_playlists:
            raise ConnectionError("No video playlists available")
        
        logger.info(f"Selecting quality: {self.preferred_quality}")
        
        # Get available qualities
        video_playlists = self.current_manifest.video_playlists
        
        # Select playlist based on preference
        if self.preferred_quality == "best":
            selected_playlist = max(video_playlists, key=lambda p: p.quality.bandwidth)
        elif self.preferred_quality == "worst":
            selected_playlist = min(video_playlists, key=lambda p: p.quality.bandwidth)
        elif self.preferred_quality.endswith("p"):
            # Try to match resolution (e.g., "720p")
            target_height = int(self.preferred_quality[:-1])
            selected_playlist = min(
                video_playlists, 
                key=lambda p: abs(p.quality.height - target_height)
            )
        else:
            # Default to best quality
            selected_playlist = max(video_playlists, key=lambda p: p.quality.bandwidth)
        
        self.current_playlist = selected_playlist
        
        logger.info(
            f"Selected quality: {selected_playlist.quality.quality_name} "
            f"({selected_playlist.quality.resolution}, {selected_playlist.quality.bandwidth} bps)"
        )

    async def _stream_video_segments(self) -> AsyncGenerator[bytes, None]:
        """Stream video segments from the selected playlist.
        
        Yields:
            bytes: Video segment data chunks
            
        Raises:
            ConnectionError: If segment streaming fails
        """
        if not self.current_playlist:
            raise ConnectionError("No playlist selected")
        
        logger.info("Starting video segment streaming")
        
        segment_count = 0
        last_sequence = -1
        
        try:
            while not self._shutdown:
                # Refresh playlist to get new segments (for live streams)
                try:
                    refreshed_manifest = await self.hls_parser.refresh_manifest(
                        self.current_manifest, is_youtube=False
                    )
                    
                    # Find the matching playlist in refreshed manifest
                    for playlist in refreshed_manifest.video_playlists:
                        if playlist.quality.bandwidth == self.current_playlist.quality.bandwidth:
                            self.current_playlist = playlist
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to refresh manifest, continuing with current: {e}")
                
                # Get segments to download
                segments_to_download = []
                for segment in self.current_playlist.segments:
                    if segment.sequence_number > last_sequence:
                        segments_to_download.append(segment)
                
                if not segments_to_download:
                    # No new segments, wait and refresh
                    await asyncio.sleep(2.0)
                    continue
                
                # Download and yield segment data
                for segment in segments_to_download:
                    try:
                        logger.debug(f"Downloading segment {segment.sequence_number}")
                        
                        async for chunk in self.hls_parser.download_segment(segment, is_youtube=False):
                            yield chunk
                        
                        last_sequence = segment.sequence_number
                        segment_count += 1
                        
                        logger.debug(f"Successfully streamed segment {segment.sequence_number}")
                        
                    except Exception as e:
                        logger.error(f"Error downloading segment {segment.sequence_number}: {e}")
                        # Continue with next segment
                        continue
                
                # For non-live streams, break after all segments
                if not self.current_manifest.is_live:
                    break
                
                # Small delay between segments
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in video segment streaming: {e}")
            raise ConnectionError(f"Segment streaming failed: {e}")
        
        logger.info(f"Finished streaming {segment_count} video segments")

    async def get_available_qualities(self) -> List[Dict[str, Any]]:
        """Get available stream qualities.
        
        Returns:
            List of quality options with details
        """
        if not self.current_manifest:
            try:
                if not self.hls_parser:
                    self.hls_parser = HLSParser(session=self.session)
                await self._get_stream_access_token()
                await self._get_hls_manifest()
            except Exception as e:
                logger.error(f"Error getting qualities: {e}")
                return []
        
        qualities = []
        for playlist in self.current_manifest.video_playlists:
            quality = playlist.quality
            qualities.append({
                "name": quality.quality_name,
                "resolution": quality.resolution,
                "bandwidth": quality.bandwidth,
                "fps": quality.fps,
                "codecs": quality.codecs,
                "audio_only": quality.audio_only
            })
        
        # Sort by bandwidth (highest first)
        qualities.sort(key=lambda q: q["bandwidth"], reverse=True)
        return qualities

    async def switch_quality(self, quality: str) -> bool:
        """Switch to a different quality.
        
        Args:
            quality: Quality preference ("best", "worst", or specific like "720p")
            
        Returns:
            bool: True if quality switch was successful
        """
        logger.info(f"Switching quality from {self.preferred_quality} to {quality}")
        
        old_quality = self.preferred_quality
        self.preferred_quality = quality
        
        try:
            await self._select_quality()
            logger.info(f"Successfully switched to quality: {quality}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch quality: {e}")
            # Revert to old quality
            self.preferred_quality = old_quality
            return False

    def __repr__(self) -> str:
        """String representation of the Twitch adapter."""
        return (
            f"TwitchAdapter(username='{self.username}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )
