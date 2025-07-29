"""YouTube Live stream adapter for YouTube Data API v3 integration.

This module provides the YouTubeAdapter class for connecting to and processing
YouTube Live streams. It handles YouTube API authentication, stream metadata
retrieval, and stream data access.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Any, List

import aiohttp
from aiohttp import ClientSession

from src.utils.hls_parser import (
    HLSParser,
    StreamManifest,
    StreamQuality,
    YouTubeStreamExtractor,
    select_optimal_quality,
    estimate_bandwidth_requirement,
)

from src.core.config import get_settings
from src.utils.stream_validation import validate_stream_url
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


class YouTubeAdapter(BaseStreamAdapter):
    """YouTube Live stream adapter for YouTube Data API v3 integration.

    This adapter handles YouTube API requests, retrieves stream metadata,
    and provides access to stream data for processing.
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
            api_key: YouTube Data API v3 key (optional, will use config if not provided)
            session: Optional aiohttp ClientSession
            **kwargs: Additional configuration options
        """
        super().__init__(url, session, **kwargs)

        # Validate URL and extract information
        self.validation_result = validate_stream_url(url, StreamPlatform.YOUTUBE)
        self.video_id = self.validation_result.get("video_id")
        self.channel_id = self.validation_result.get("channel_id")
        self.channel_custom_url = self.validation_result.get("channel_custom_url")
        self.handle = self.validation_result.get("handle")
        self.url_type = self.validation_result.get("type", "video")

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

        # HLS streaming components
        self.hls_parser: Optional[HLSParser] = None
        self.current_manifest: Optional[StreamManifest] = None
        self.current_quality: Optional[StreamQuality] = None
        self.hls_manifest_url: Optional[str] = None
        self.dash_manifest_url: Optional[str] = None

        # Streaming configuration
        self.preferred_quality: str = kwargs.get("preferred_quality", "best")
        self.max_bandwidth: Optional[int] = kwargs.get("max_bandwidth")
        self.target_height: Optional[int] = kwargs.get("target_height")
        self.enable_adaptive_streaming: bool = kwargs.get(
            "enable_adaptive_streaming", True
        )
        self.segment_buffer_size: int = kwargs.get("segment_buffer_size", 5)

        # Reconnection settings for live streams
        self.manifest_refresh_interval: float = kwargs.get(
            "manifest_refresh_interval", 30.0
        )
        self.segment_retry_attempts: int = kwargs.get("segment_retry_attempts", 3)
        self.segment_timeout: float = kwargs.get("segment_timeout", 10.0)

        # Circuit breaker for API resilience
        self._circuit_breaker_name = f"youtube_api_{self.video_id or self.channel_id or self.handle or 'unknown'}"

        logger.info(f"Initialized YouTube adapter for URL: {url}")
        logger.debug(f"YouTube adapter details: {self.validation_result}")

    async def authenticate(self) -> bool:
        """Authenticate with YouTube Data API.

        YouTube Data API v3 uses API keys, so this method just validates
        that an API key is available.

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
        """Make an authenticated request to the YouTube Data API v3.

        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters

        Returns:
            Dict: API response data

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit/quota is exceeded
            ConnectionError: If request fails
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
                    logger.error(f"YouTube API error {response.status}: {error_text}")
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

        # If we have a channel custom URL
        if self.channel_custom_url:
            response = await self._make_api_request(
                "channels", {"part": "id", "forUsername": self.channel_custom_url}
            )

            channels = response.get("items", [])
            if channels:
                self.resolved_channel_id = channels[0]["id"]
                return self.resolved_channel_id

        # If we have a handle (@username format)
        if self.handle:
            response = await self._make_api_request(
                "channels", {"part": "id", "forHandle": self.handle}
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
        """Connect to the YouTube Live stream.

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

            # Extract HLS/DASH URLs from live stream metadata
            await self._extract_stream_urls(live_stream)

            # Initialize HLS parser if we have a manifest URL
            if self.hls_manifest_url:
                await self._initialize_hls_streaming()
            elif self.dash_manifest_url:
                logger.warning(
                    "DASH streaming not yet implemented, falling back to metadata-only mode"
                )

            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.utcnow()
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
            self.connection.last_error_at = datetime.utcnow()
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect to YouTube stream: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the YouTube Live stream."""
        logger.info(f"Disconnecting from YouTube stream: {self.url}")

        # Clean up HLS parser
        if self.hls_parser:
            try:
                await self.hls_parser.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing HLS parser: {e}")
            finally:
                self.hls_parser = None

        # Clear streaming state
        self.current_manifest = None
        self.current_quality = None
        self.hls_manifest_url = None
        self.dash_manifest_url = None

        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN

        await self._notify_disconnect()

        logger.info("Disconnected from YouTube stream")

    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata from YouTube Data API.

        Returns:
            StreamMetadata: Current stream metadata

        Raises:
            StreamAdapterError: If metadata cannot be retrieved
        """
        live_stream = await self._get_live_stream_info()

        if not live_stream:
            # Stream is offline
            self.metadata.is_live = False
            self.metadata.updated_at = datetime.utcnow()
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

        # Add stream quality information if available
        resolution = None
        framerate = None
        bitrate = None

        if self.current_quality:
            resolution = self.current_quality.resolution
            framerate = self.current_quality.fps
            bitrate = self.current_quality.bandwidth

        # Update metadata
        self.metadata = StreamMetadata(
            title=snippet.get("title", ""),
            description=snippet.get("description", ""),
            thumbnail_url=thumbnail_url,
            is_live=True,
            viewer_count=int(statistics.get("viewCount", 0)),
            started_at=started_at,
            platform_id=live_stream["id"],
            platform_url=f"https://www.youtube.com/watch?v={live_stream['id']}",
            resolution=resolution,
            framerate=framerate,
            bitrate=bitrate,
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
                "hls_manifest_url": self.hls_manifest_url,
                "dash_manifest_url": self.dash_manifest_url,
                "streaming_available": bool(
                    self.hls_manifest_url or self.dash_manifest_url
                ),
                "available_qualities": len(self.current_manifest.video_playlists)
                if self.current_manifest
                else 0,
            },
            tags=snippet.get("tags", []),
            language=snippet.get("defaultLanguage", "en"),
            updated_at=datetime.utcnow(),
        )

        return self.metadata

    async def is_stream_live(self) -> bool:
        """Check if the YouTube stream is currently live.

        Returns:
            bool: True if stream is live

        Raises:
            StreamAdapterError: If status cannot be determined
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

        This implementation provides real HLS video streaming from YouTube.

        Yields:
            bytes: Video stream data chunks (TS segments)

        Raises:
            StreamAdapterError: If stream data cannot be retrieved
        """
        if not self.connection.status == ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to YouTube stream")

        # If no HLS streaming is available, fall back to metadata
        if not self.hls_parser or not self.current_manifest:
            logger.warning(
                "HLS streaming not available, falling back to metadata-only mode. "
                "This may occur if the stream doesn't provide HLS manifests."
            )
            metadata = await self.get_metadata()
            metadata_json = json.dumps(metadata.__dict__, default=str, indent=2)
            yield metadata_json.encode("utf-8")
            return

        logger.info(
            f"Starting HLS video streaming for YouTube stream: {self.live_stream_id}"
        )

        try:
            # Select the best quality playlist
            playlist = self._select_optimal_playlist()
            if not playlist:
                raise ConnectionError("No suitable video quality found")

            logger.info(
                f"Selected quality: {playlist.quality.quality_name} ({playlist.quality.bandwidth} bps)"
            )

            # Start streaming segments
            async for chunk in self._stream_segments(playlist):
                yield chunk

                # Update connection stats
                self.connection.bytes_received += len(chunk)
                self.connection.packets_received += 1
                self.connection.last_data_at = datetime.utcnow()

                # Notify data callback
                await self._notify_data(chunk)

        except Exception as e:
            logger.error(f"Error streaming YouTube data: {e}")
            self.connection.error_count += 1
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.utcnow()
            await self._notify_error(e)
            raise

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

    async def get_stream_analytics(self) -> Dict[str, Any]:
        """Get stream analytics and statistics.

        Returns:
            Dict: Stream analytics data
        """
        analytics = {
            "platform": "youtube",
            "url": self.url,
            "video_id": self.video_id,
            "channel_id": self.resolved_channel_id,
            "live_stream_id": self.live_stream_id,
            "live_chat_id": self.live_chat_id,
            "connection_status": self.connection.status.value,
            "health_status": self.connection.health.value,
            "reconnect_count": self.connection.reconnect_count,
            "error_count": self.connection.error_count,
            "quota_remaining": self.quota_remaining,
            "requests_remaining": self.requests_remaining,
            "last_data_at": self.connection.last_data_at,
            "metadata": self.metadata.__dict__ if self.metadata else None,
            "streaming": {
                "hls_manifest_url": self.hls_manifest_url,
                "dash_manifest_url": self.dash_manifest_url,
                "streaming_available": bool(
                    self.hls_manifest_url or self.dash_manifest_url
                ),
                "current_quality": self.current_quality.__dict__
                if self.current_quality
                else None,
                "manifest_info": {
                    "is_live": self.current_manifest.is_live
                    if self.current_manifest
                    else None,
                    "playlist_count": len(self.current_manifest.playlists)
                    if self.current_manifest
                    else 0,
                    "video_qualities": len(self.current_manifest.video_playlists)
                    if self.current_manifest
                    else 0,
                }
                if self.current_manifest
                else None,
                "configuration": {
                    "preferred_quality": self.preferred_quality,
                    "max_bandwidth": self.max_bandwidth,
                    "target_height": self.target_height,
                    "enable_adaptive_streaming": self.enable_adaptive_streaming,
                    "segment_buffer_size": self.segment_buffer_size,
                },
            },
        }

        return analytics

    async def _extract_stream_urls(self, live_stream: Dict[str, Any]) -> None:
        """Extract HLS/DASH URLs from live stream metadata."""
        try:
            # Extract URLs using the utility class
            stream_urls = YouTubeStreamExtractor.extract_stream_urls(live_stream)

            self.hls_manifest_url = stream_urls.get("hls_url")
            self.dash_manifest_url = stream_urls.get("dash_url")

            if self.hls_manifest_url:
                logger.info(f"Found HLS manifest URL: {self.hls_manifest_url}")
            if self.dash_manifest_url:
                logger.info(f"Found DASH manifest URL: {self.dash_manifest_url}")

            if not self.hls_manifest_url and not self.dash_manifest_url:
                logger.warning(
                    "No streaming manifest URLs found in live stream metadata"
                )

        except Exception as e:
            logger.error(f"Failed to extract stream URLs: {e}")

    async def _initialize_hls_streaming(self) -> None:
        """Initialize HLS parser and parse the manifest."""
        try:
            if not self.hls_manifest_url:
                raise ValueError("No HLS manifest URL available")

            # Create HLS parser with our session
            self.hls_parser = HLSParser(session=self.session)

            # Parse the manifest
            self.current_manifest = await self.hls_parser.parse_manifest(
                self.hls_manifest_url, is_youtube=True
            )

            logger.info(
                f"Parsed HLS manifest with {len(self.current_manifest.playlists)} playlists"
            )

            # Log available qualities
            for playlist in self.current_manifest.video_playlists:
                quality = playlist.quality
                logger.debug(
                    f"Available quality: {quality.quality_name} ({quality.bandwidth} bps, {quality.resolution})"
                )

        except Exception as e:
            logger.error(f"Failed to initialize HLS streaming: {e}")
            raise

    def _select_optimal_playlist(self):
        """Select the optimal playlist based on configuration."""
        if not self.current_manifest:
            return None

        video_playlists = self.current_manifest.video_playlists
        if not video_playlists:
            logger.error("No video playlists available")
            return None

        # Use the quality selection utility
        qualities = [p.quality for p in video_playlists]
        selected_quality = select_optimal_quality(
            qualities,
            target_height=self.target_height,
            max_bandwidth=self.max_bandwidth,
            prefer_quality=self.preferred_quality,
        )

        if not selected_quality:
            logger.warning("No suitable quality found, using first available")
            return video_playlists[0]

        # Find the playlist with the selected quality
        for playlist in video_playlists:
            if playlist.quality.bandwidth == selected_quality.bandwidth:
                self.current_quality = selected_quality
                return playlist

        # Fallback to first playlist
        return video_playlists[0]

    async def _stream_segments(self, playlist):
        """Stream video segments from the HLS playlist."""
        last_segment_sequence = None
        consecutive_errors = 0
        max_consecutive_errors = 3

        while self.connection.status == ConnectionStatus.CONNECTED:
            try:
                # Refresh manifest for live streams to get new segments
                if self.current_manifest.is_live:
                    self.current_manifest = await self.hls_parser.refresh_manifest(
                        self.current_manifest, is_youtube=True
                    )

                    # Update playlist reference
                    playlist = self._select_optimal_playlist()
                    if not playlist:
                        logger.error("Lost playlist after manifest refresh")
                        break

                # Get segments to download
                segments_to_download = []
                for segment in playlist.segments:
                    if (
                        last_segment_sequence is None
                        or segment.sequence_number > last_segment_sequence
                    ):
                        segments_to_download.append(segment)

                if not segments_to_download:
                    if self.current_manifest.is_live:
                        # Wait before checking for new segments
                        await asyncio.sleep(playlist.target_duration / 2)
                        continue
                    else:
                        # End of VOD stream
                        logger.info("Reached end of video stream")
                        break

                # Download and yield segments
                for segment in segments_to_download:
                    try:
                        async for chunk in self.hls_parser.download_segment(
                            segment, is_youtube=True
                        ):
                            yield chunk

                        last_segment_sequence = segment.sequence_number
                        consecutive_errors = 0  # Reset error counter on success

                    except Exception as segment_error:
                        logger.warning(
                            f"Failed to download segment {segment.sequence_number}: {segment_error}"
                        )
                        consecutive_errors += 1

                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                f"Too many consecutive segment errors ({consecutive_errors})"
                            )
                            raise

                        # Skip this segment and continue
                        continue

                # For live streams, wait a bit before checking for new segments
                if self.current_manifest.is_live and segments_to_download:
                    await asyncio.sleep(max(1.0, playlist.target_duration * 0.1))

            except Exception as e:
                logger.error(f"Error in segment streaming loop: {e}")
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    raise

                # Wait before retrying
                await asyncio.sleep(2.0 * consecutive_errors)

    async def get_available_qualities(self) -> List[Dict[str, Any]]:
        """Get list of available stream qualities."""
        if not self.current_manifest:
            return []

        qualities = []
        for playlist in self.current_manifest.video_playlists:
            quality = playlist.quality
            qualities.append(
                {
                    "name": quality.quality_name,
                    "resolution": quality.resolution,
                    "bandwidth": quality.bandwidth,
                    "fps": quality.fps,
                    "codecs": quality.codecs,
                    "estimated_bandwidth_requirement": estimate_bandwidth_requirement(
                        quality
                    ),
                }
            )

        return sorted(qualities, key=lambda q: q["bandwidth"], reverse=True)

    async def switch_quality(self, target_quality: str) -> bool:
        """Switch to a different quality during streaming."""
        try:
            if not self.current_manifest:
                logger.error("No manifest available for quality switching")
                return False

            # Update preferred quality
            self.preferred_quality = target_quality

            # Select new playlist
            new_playlist = self._select_optimal_playlist()
            if not new_playlist:
                logger.error(f"Cannot switch to quality: {target_quality}")
                return False

            logger.info(f"Switched to quality: {new_playlist.quality.quality_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch quality: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of the YouTube adapter."""
        identifier = self.video_id or self.channel_id or self.handle or "unknown"
        return (
            f"YouTubeAdapter(identifier='{identifier}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )
