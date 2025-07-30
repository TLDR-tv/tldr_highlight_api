"""YouTube stream adapter infrastructure implementation.

This module provides the YouTube streaming platform adapter using
Pythonic patterns and async/await throughout.
"""

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Optional, Any, List
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientSession

from src.core.config import get_settings
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


class YouTubeStreamAdapter(BaseStreamAdapter):
    """YouTube stream adapter for API integration.
    
    Handles authentication, metadata retrieval, and stream access
    for YouTube streaming platform.
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ):
        """Initialize the YouTube adapter.
        
        Args:
            url: YouTube stream URL
            api_key: YouTube Data API v3 key (optional, uses config if not provided)
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options
        """
        super().__init__(url, session, **kwargs)
        
        # Extract video/channel info from URL
        self.video_id = self._extract_video_id(url)
        self.channel_id = self._extract_channel_id(url)
        self.channel_handle = self._extract_channel_handle(url)
        
        # API configuration
        self.api_key = api_key or settings.youtube_api_key
        self.api_base_url = settings.youtube_api_base_url
        
        # Rate limiting
        self.quota_remaining = settings.youtube_rate_limit_per_day
        self.requests_remaining = settings.youtube_rate_limit_per_100_seconds
        self.rate_limit_reset_time: Optional[datetime] = None
        
        # Stream information
        self.resolved_channel_id: Optional[str] = None
        self.live_stream_id: Optional[str] = None
        self.live_chat_id: Optional[str] = None
        
        # Streaming configuration
        self.preferred_quality: str = kwargs.get("preferred_quality", "best")
        
        logger.info(f"Initialized YouTube adapter for URL: {url}")
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Optional[str]: Video ID if found
        """
        # Simple extraction - full implementation would handle more URL formats
        if "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return None
    
    def _extract_channel_id(self, url: str) -> Optional[str]:
        """Extract channel ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Optional[str]: Channel ID if found
        """
        if "/channel/" in url:
            return url.split("/channel/")[1].split("/")[0].split("?")[0]
        return None
    
    def _extract_channel_handle(self, url: str) -> Optional[str]:
        """Extract channel handle from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Optional[str]: Channel handle if found
        """
        if "/@" in url:
            return url.split("/@")[1].split("/")[0].split("?")[0]
        return None
    
    async def authenticate(self) -> bool:
        """Authenticate with YouTube Data API.
        
        YouTube uses API keys, so this validates the key is available.
        
        Returns:
            bool: True if authentication is valid
            
        Raises:
            AuthenticationError: If API key is not available
        """
        if not self.api_key:
            raise AuthenticationError(
                "YouTube Data API v3 key is required for authentication"
            )
        
        logger.info("YouTube API key validation successful")
        return True
    
    async def _make_api_request(
        self, endpoint: str, params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to the YouTube Data API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            Dict: API response data
            
        Raises:
            Various exceptions based on response
        """
        if not self.api_key:
            raise AuthenticationError("YouTube API key is required")
        
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        
        # Add API key to parameters
        request_params = params or {}
        request_params["key"] = self.api_key
        
        try:
            async with self.session.get(url, params=request_params) as response:
                # Handle rate limiting and quota exceeded
                if response.status == 403:
                    error_data = await response.json()
                    error_message = error_data.get("error", {}).get("message", "")
                    
                    if "quota exceeded" in error_message.lower():
                        raise RateLimitError(
                            f"YouTube API quota exceeded: {error_message}"
                        )
                    elif "api key" in error_message.lower():
                        raise AuthenticationError(
                            f"YouTube API key error: {error_message}"
                        )
                    else:
                        raise ConnectionError(
                            f"YouTube API access denied: {error_message}"
                        )
                
                # Handle other client errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise ConnectionError(
                        f"YouTube API request failed: {response.status} - {error_text}"
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error during YouTube API request: {e}")
            raise ConnectionError(f"HTTP request failed: {e}")
    
    async def _resolve_channel_id(self) -> str:
        """Resolve channel ID from various URL formats.
        
        Returns:
            str: The resolved channel ID
            
        Raises:
            StreamNotFoundError: If channel cannot be found
        """
        if self.resolved_channel_id:
            return self.resolved_channel_id
        
        # If we already have a channel ID
        if self.channel_id:
            self.resolved_channel_id = self.channel_id
            return self.resolved_channel_id
        
        # If we have a video ID, get channel from video
        if self.video_id:
            response = await self._make_api_request(
                "videos", {"part": "snippet", "id": self.video_id}
            )
            
            videos = response.get("items", [])
            if not videos:
                raise StreamNotFoundError(f"YouTube video '{self.video_id}' not found")
            
            self.resolved_channel_id = videos[0]["snippet"]["channelId"]
            return self.resolved_channel_id
        
        # If we have a handle (@username format)
        if self.channel_handle:
            response = await self._make_api_request(
                "channels", {"part": "id", "forHandle": self.channel_handle}
            )
            
            channels = response.get("items", [])
            if channels:
                self.resolved_channel_id = channels[0]["id"]
                return self.resolved_channel_id
        
        raise StreamNotFoundError(f"Could not resolve channel ID from URL: {self.url}")
    
    async def _get_live_stream_info(self) -> Optional[Dict[str, Any]]:
        """Get live stream information for the channel.
        
        Returns:
            Optional[Dict]: Live stream data if available, None if offline
        """
        if self.video_id:
            # Check specific video for live status
            response = await self._make_api_request(
                "videos",
                {
                    "part": "snippet,liveStreamingDetails,statistics",
                    "id": self.video_id,
                },
            )
            
            videos = response.get("items", [])
            if videos:
                video = videos[0]
                live_details = video.get("liveStreamingDetails", {})
                
                # Check if it's a live stream
                if live_details and not live_details.get("actualEndTime"):
                    return video
        else:
            # Search for live streams on the channel
            channel_id = await self._resolve_channel_id()
            
            response = await self._make_api_request(
                "search",
                {
                    "part": "id,snippet",
                    "channelId": channel_id,
                    "type": "video",
                    "eventType": "live",
                    "maxResults": "1",
                },
            )
            
            items = response.get("items", [])
            if items:
                # Get full video details
                video_id = items[0]["id"]["videoId"]
                video_response = await self._make_api_request(
                    "videos",
                    {"part": "snippet,liveStreamingDetails,statistics", "id": video_id},
                )
                
                videos = video_response.get("items", [])
                if videos:
                    return videos[0]
        
        return None
    
    async def connect(self) -> bool:
        """Connect to the YouTube stream.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to YouTube stream: {self.url}")
        
        try:
            self.connection.status = ConnectionStatus.CONNECTING
            
            # Check if stream is live
            live_stream = await self._get_live_stream_info()
            if not live_stream:
                raise StreamOfflineError(f"YouTube stream is offline: {self.url}")
            
            # Store live stream information
            self.live_stream_id = live_stream["id"]
            live_details = live_stream.get("liveStreamingDetails", {})
            self.live_chat_id = live_details.get("activeLiveChatId")
            
            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.now(timezone.utc)
            self.connection.health = StreamHealth.HEALTHY
            
            # Notify connection
            await self._notify_connect()
            
            logger.info(
                f"Successfully connected to YouTube stream: {self.live_stream_id}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to YouTube stream: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the YouTube stream."""
        logger.info(f"Disconnecting from YouTube stream: {self.url}")
        
        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN
        
        await self._notify_disconnect()
        
        logger.info("Disconnected from YouTube stream")
    
    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata from YouTube Data API.
        
        Returns:
            StreamMetadata: Current stream metadata
        """
        live_stream = await self._get_live_stream_info()
        
        if not live_stream:
            # Stream is offline
            self.metadata.is_live = False
            self.metadata.updated_at = datetime.now(timezone.utc)
            return self.metadata
        
        snippet = live_stream.get("snippet", {})
        live_details = live_stream.get("liveStreamingDetails", {})
        statistics = live_stream.get("statistics", {})
        
        # Parse timestamps
        started_at = None
        if live_details.get("actualStartTime"):
            started_at = datetime.fromisoformat(
                live_details["actualStartTime"].replace("Z", "+00:00")
            )
        
        # Get thumbnail URL (highest quality available)
        thumbnails = snippet.get("thumbnails", {})
        thumbnail_url = None
        for quality in ["maxres", "standard", "medium", "default"]:
            if quality in thumbnails:
                thumbnail_url = thumbnails[quality]["url"]
                break
        
        # Update metadata
        self.metadata = StreamMetadata(
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            thumbnail_url=thumbnail_url,
            is_live=True,
            viewer_count=int(live_details.get("concurrentViewers", 0) or statistics.get("viewCount", 0)),
            started_at=started_at,
            platform_id=live_stream["id"],
            platform_url=f"https://www.youtube.com/watch?v={live_stream['id']}",
            platform_data={
                "video_id": live_stream["id"],
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "category_id": snippet.get("categoryId"),
                "default_language": snippet.get("defaultLanguage"),
                "default_audio_language": snippet.get("defaultAudioLanguage"),
                "live_chat_id": live_details.get("activeLiveChatId"),
                "scheduled_start_time": live_details.get("scheduledStartTime"),
                "concurrent_viewers": live_details.get("concurrentViewers"),
                "like_count": statistics.get("likeCount"),
                "comment_count": statistics.get("commentCount"),
            },
            tags=snippet.get("tags", []),
            language=snippet.get("defaultLanguage", "en"),
            updated_at=datetime.now(timezone.utc),
        )
        
        return self.metadata
    
    async def is_stream_live(self) -> bool:
        """Check if the YouTube stream is currently live.
        
        Returns:
            bool: True if stream is live
        """
        try:
            live_stream = await self._get_live_stream_info()
            is_live = live_stream is not None
            
            logger.debug(f"YouTube stream {self.url} live status: {is_live}")
            return is_live
            
        except Exception as e:
            logger.error(f"Error checking YouTube stream live status: {e}")
            raise
    
    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get stream data as an async generator.
        
        Currently returns empty generator - full HLS implementation
        would require additional HLS parsing infrastructure.
        
        Yields:
            bytes: Stream data chunks
        """
        if not self.connection.status == ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to YouTube stream")
        
        logger.warning(
            "Full HLS streaming not implemented in infrastructure layer. "
            "This would require HLS parser infrastructure component."
        )
        
        # Placeholder implementation
        yield b""
    
    async def get_chat_messages(
        self, limit: int = 100, page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get live chat messages from the stream.
        
        Args:
            limit: Maximum number of messages to retrieve
            page_token: Pagination token for continued requests
            
        Returns:
            Dict: Chat messages and pagination info
        """
        if not self.live_chat_id:
            logger.warning("No live chat ID available")
            return {"items": [], "nextPageToken": None}
        
        params = {
            "liveChatId": self.live_chat_id,
            "part": "snippet,authorDetails",
            "maxResults": str(min(limit, 2000)),  # YouTube API limit
        }
        
        if page_token:
            params["pageToken"] = page_token
        
        try:
            response = await self._make_api_request("liveChat/messages", params)
            return response
            
        except Exception as e:
            logger.error(f"Error getting YouTube chat messages: {e}")
            return {"items": [], "nextPageToken": None}
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        identifier = self.video_id or self.channel_id or self.channel_handle or "unknown"
        return (
            f"YouTubeStreamAdapter(identifier='{identifier}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )


@asynccontextmanager
async def youtube_stream(url: str, **kwargs):
    """Context manager for YouTube streams.
    
    Args:
        url: YouTube stream URL
        **kwargs: Additional adapter configuration
        
    Yields:
        Connected YouTubeStreamAdapter
    """
    adapter = YouTubeStreamAdapter(url, **kwargs)
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()