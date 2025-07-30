"""Twitch stream adapter infrastructure implementation.

This module provides the Twitch streaming platform adapter using
Pythonic patterns and async/await throughout.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Any
from urllib.parse import urlencode
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientSession, ClientResponseError

from src.core.config import get_settings
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


class TwitchStreamAdapter(BaseStreamAdapter):
    """Twitch stream adapter for API integration.
    
    Handles authentication, metadata retrieval, and HLS stream access
    for Twitch streaming platform.
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
            client_id: Twitch Client ID (optional, uses config if not provided)
            client_secret: Twitch Client Secret (optional, uses config if not provided)
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options
        """
        super().__init__(url, session, **kwargs)
        
        # Extract username from URL
        self.username = self._extract_username(url)
        
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
        self.hls_manifest_url: Optional[str] = None
        self.access_token: Optional[str] = None
        self.access_signature: Optional[str] = None
        self.preferred_quality: str = kwargs.get("quality", "best")
        
        # GraphQL and Usher endpoints
        self.graphql_url = "https://gql.twitch.tv/gql"
        self.usher_base_url = "https://usher.ttvnw.net/api/channel/hls"
        
        logger.info(f"Initialized Twitch adapter for username: {self.username}")
    
    def _extract_username(self, url: str) -> str:
        """Extract username from Twitch URL.
        
        Args:
            url: Twitch stream URL
            
        Returns:
            str: Extracted username
            
        Raises:
            ValueError: If username cannot be extracted
        """
        # Remove protocol and domain
        parts = url.replace("https://", "").replace("http://", "").replace("www.", "")
        parts = parts.replace("twitch.tv/", "").strip("/")
        
        # Get first part (username)
        username = parts.split("/")[0].split("?")[0]
        
        if not username:
            raise ValueError(f"Cannot extract username from URL: {url}")
        
        return username.lower()
    
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
            and datetime.now(timezone.utc) >= self.token_expires_at
        ):
            await self.authenticate()
    
    async def _make_api_request(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to the Twitch API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            Dict: API response data
            
        Raises:
            Various exceptions based on response
        """
        await self._ensure_authenticated()
        
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.app_access_token}",
            "Accept": "application/json",
        }
        
        logger.debug(f"Making Twitch API request: {endpoint}")
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                # Check rate limiting
                remaining = response.headers.get("Ratelimit-Remaining")
                if remaining:
                    self.rate_limit_remaining = int(remaining)
                
                reset_time = response.headers.get("Ratelimit-Reset")
                if reset_time:
                    self.rate_limit_reset_time = datetime.fromtimestamp(
                        int(reset_time), tz=timezone.utc
                    )
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    raise RateLimitError(
                        f"Twitch API rate limit exceeded. Retry after {retry_after} seconds"
                    )
                
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
                    raise ConnectionError(
                        f"Twitch API request failed: {response.status} - {error_text}"
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during Twitch API request: {e}")
            raise ConnectionError(f"HTTP client error: {e}")
    
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
            await self._get_user_info()
            
            # Check if stream is live
            if not await self.is_stream_live():
                raise StreamOfflineError(f"Twitch stream '{self.username}' is offline")
            
            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.now(timezone.utc)
            self.connection.health = StreamHealth.HEALTHY
            
            # Notify connection
            await self._notify_connect()
            
            logger.info(f"Successfully connected to Twitch stream: {self.username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Twitch stream: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the Twitch stream."""
        logger.info(f"Disconnecting from Twitch stream: {self.username}")
        
        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN
        
        # Clear HLS state
        self.hls_manifest_url = None
        self.access_token = None
        self.access_signature = None
        
        await self._notify_disconnect()
        
        logger.info(f"Disconnected from Twitch stream: {self.username}")
    
    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata from Twitch API.
        
        Returns:
            StreamMetadata: Current stream metadata
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
            self.metadata.updated_at = datetime.now(timezone.utc)
            return self.metadata
        
        stream_data = streams[0]
        
        # Get user information for additional metadata
        user_response = await self._make_api_request("users", {"id": self.user_id})
        user_data = user_response.get("data", [{}])[0]
        
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
                "game_id": stream_data.get("game_id"),
                "game_name": stream_data.get("game_name", ""),
                "type": stream_data.get("type", "live"),
                "language": stream_data.get("language", "en"),
                "is_mature": stream_data.get("is_mature", False),
                "profile_image_url": user_data.get("profile_image_url"),
                "broadcaster_type": user_data.get("broadcaster_type", ""),
            },
            category=stream_data.get("game_name", ""),
            tags=stream_data.get("tag_ids", []),
            language=stream_data.get("language", "en"),
            updated_at=datetime.now(timezone.utc),
        )
        
        self.stream_id = stream_data.get("id")
        
        return self.metadata
    
    async def is_stream_live(self) -> bool:
        """Check if the Twitch stream is currently live.
        
        Returns:
            bool: True if stream is live
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
        """Get stream data as an async generator.
        
        Currently returns empty generator - full HLS implementation
        would require additional HLS parsing infrastructure.
        
        Yields:
            bytes: Stream data chunks
        """
        if not await self.is_stream_live():
            raise StreamOfflineError(f"Twitch stream '{self.username}' is offline")
        
        logger.warning(
            "Full HLS streaming not implemented in infrastructure layer. "
            "This would require HLS parser infrastructure component."
        )
        
        # Placeholder implementation
        yield b""
    
    async def _get_stream_access_token(self) -> Dict[str, str]:
        """Get stream access token from Twitch GraphQL API.
        
        Returns:
            Dict with 'value' and 'signature' keys
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
        
        payload = {"query": query, "variables": {}}
        
        headers = {
            "Client-ID": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        try:
            async with self.session.post(
                self.graphql_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(
                        f"Failed to get access token: {response.status} - {error_text}"
                    )
                
                data = await response.json()
                
                if "errors" in data:
                    error_msg = data["errors"][0].get("message", "Unknown error")
                    raise AuthenticationError(f"GraphQL error: {error_msg}")
                
                token_data = data.get("data", {}).get("streamPlaybackAccessToken", {})
                if not token_data:
                    raise AuthenticationError("No access token data in response")
                
                self.access_token = token_data.get("value")
                self.access_signature = token_data.get("signature")
                
                if not self.access_token or not self.access_signature:
                    raise AuthenticationError("Invalid access token or signature")
                
                logger.info("Successfully obtained stream access token")
                return token_data
                
        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            logger.error(f"Error getting access token: {e}")
            raise AuthenticationError(f"Failed to get access token: {e}")
    
    async def get_hls_manifest_url(self) -> str:
        """Get the HLS manifest URL for the stream.
        
        Returns:
            str: HLS manifest URL with access token
        """
        if not self.access_token or not self.access_signature:
            await self._get_stream_access_token()
        
        params = {
            "token": self.access_token,
            "sig": self.access_signature,
            "allow_source": "true",
            "allow_audio_only": "true",
            "allow_spectre": "false",
            "fast_bread": "true",
            "p": str(hash(self.username) % 9999999),
            "player_backend": "mediaplayer",
            "playlist_include_framerate": "true",
            "reassignments_supported": "true",
            "rtqos": "control",
            "cdm": "wv",
            "player_version": "1.0.0"
        }
        
        manifest_url = f"{self.usher_base_url}/{self.username}.m3u8?{urlencode(params)}"
        self.hls_manifest_url = manifest_url
        
        return manifest_url
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"TwitchStreamAdapter(username='{self.username}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )


@asynccontextmanager
async def twitch_stream(url: str, **kwargs):
    """Context manager for Twitch streams.
    
    Args:
        url: Twitch stream URL
        **kwargs: Additional adapter configuration
        
    Yields:
        Connected TwitchStreamAdapter
    """
    adapter = TwitchStreamAdapter(url, **kwargs)
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()