"""
HLS/DASH manifest parsing utilities for YouTube and other streaming platforms.

This module provides comprehensive HLS (HTTP Live Streaming) and DASH (Dynamic
Adaptive Streaming over HTTP) parsing capabilities, with specialized support
for YouTube's streaming infrastructure.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Any
from urllib.parse import urljoin

import aiohttp
import m3u8

logger = logging.getLogger(__name__)


@dataclass
class StreamQuality:
    """Stream quality information."""

    resolution: str  # e.g., "1920x1080", "1280x720"
    bandwidth: int  # bits per second
    codecs: str  # e.g., "avc1.64001f,mp4a.40.2"
    fps: Optional[float] = None
    audio_only: bool = False
    video_only: bool = False

    @property
    def height(self) -> int:
        """Get video height from resolution."""
        if "x" in self.resolution:
            return int(self.resolution.split("x")[1])
        return 0

    @property
    def width(self) -> int:
        """Get video width from resolution."""
        if "x" in self.resolution:
            return int(self.resolution.split("x")[0])
        return 0

    @property
    def quality_name(self) -> str:
        """Get quality name (e.g., 1080p, 720p)."""
        height = self.height
        if height >= 2160:
            return "4K"
        elif height >= 1440:
            return "1440p"
        elif height >= 1080:
            return "1080p"
        elif height >= 720:
            return "720p"
        elif height >= 480:
            return "480p"
        elif height >= 360:
            return "360p"
        elif height >= 240:
            return "240p"
        else:
            return f"{height}p"


@dataclass
class StreamSegment:
    """Individual stream segment information."""

    uri: str
    duration: float
    sequence_number: int
    byte_range: Optional[str] = None
    discontinuity: bool = False

    # Timing information
    program_date_time: Optional[datetime] = None

    @property
    def is_encrypted(self) -> bool:
        """Check if segment is encrypted."""
        return "key=" in self.uri.lower() or "enc=" in self.uri.lower()


@dataclass
class HLSPlaylist:
    """HLS playlist information."""

    uri: str
    quality: StreamQuality
    segments: List[StreamSegment] = field(default_factory=list)
    target_duration: float = 6.0
    sequence_number: int = 0
    is_live: bool = True

    # Media metadata
    media_type: str = "video"  # video, audio, subtitles
    language: Optional[str] = None
    name: Optional[str] = None

    # Encryption info
    encryption_method: Optional[str] = None
    encryption_key_uri: Optional[str] = None
    encryption_iv: Optional[str] = None

    @property
    def duration(self) -> float:
        """Total duration of all segments."""
        return sum(segment.duration for segment in self.segments)

    @property
    def latest_segment(self) -> Optional[StreamSegment]:
        """Get the latest segment."""
        return self.segments[-1] if self.segments else None


@dataclass
class StreamManifest:
    """Complete stream manifest with all available qualities."""

    master_playlist_uri: str
    playlists: List[HLSPlaylist] = field(default_factory=list)
    is_live: bool = True
    version: int = 3

    # Content metadata
    title: Optional[str] = None
    description: Optional[str] = None

    # Stream capabilities
    independent_segments: bool = False
    start_time_offset: float = 0.0

    @property
    def video_playlists(self) -> List[HLSPlaylist]:
        """Get video-only playlists."""
        return [
            p
            for p in self.playlists
            if p.media_type == "video" and not p.quality.audio_only
        ]

    @property
    def audio_playlists(self) -> List[HLSPlaylist]:
        """Get audio-only playlists."""
        return [
            p for p in self.playlists if p.media_type == "audio" or p.quality.audio_only
        ]

    @property
    def qualities(self) -> List[StreamQuality]:
        """Get all available video qualities sorted by bandwidth."""
        qualities = [p.quality for p in self.video_playlists]
        return sorted(qualities, key=lambda q: q.bandwidth, reverse=True)

    def get_best_quality(
        self, max_bandwidth: Optional[int] = None
    ) -> Optional[HLSPlaylist]:
        """Get the best quality playlist within bandwidth constraints."""
        video_playlists = self.video_playlists
        if not video_playlists:
            return None

        if max_bandwidth:
            video_playlists = [
                p for p in video_playlists if p.quality.bandwidth <= max_bandwidth
            ]
            if not video_playlists:
                # If no playlist fits, return the lowest quality
                return min(self.video_playlists, key=lambda p: p.quality.bandwidth)

        return max(video_playlists, key=lambda p: p.quality.bandwidth)

    def get_quality_by_height(self, target_height: int) -> Optional[HLSPlaylist]:
        """Get playlist closest to target height."""
        video_playlists = self.video_playlists
        if not video_playlists:
            return None

        return min(video_playlists, key=lambda p: abs(p.quality.height - target_height))


class HLSParser:
    """HLS manifest parser with YouTube-specific optimizations."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self._owned_session = session is None

        # YouTube-specific settings
        self.youtube_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # Cache for parsed manifests
        self._manifest_cache: Dict[str, StreamManifest] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)  # YouTube manifests change frequently

    async def __aenter__(self) -> "HLSParser":
        """Async context manager entry."""
        if self._owned_session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owned_session and self.session:
            await self.session.close()

    @property
    def http_session(self) -> aiohttp.ClientSession:
        """Get HTTP session, creating one if needed."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owned_session = True
        return self.session

    async def parse_manifest(
        self, manifest_uri: str, is_youtube: bool = False
    ) -> StreamManifest:
        """
        Parse HLS manifest from URI.

        Args:
            manifest_uri: URI to the HLS manifest
            is_youtube: Whether this is a YouTube stream (enables specific optimizations)

        Returns:
            StreamManifest object

        Raises:
            aiohttp.ClientError: If manifest cannot be fetched
            ValueError: If manifest is invalid
        """
        # Check cache first
        if manifest_uri in self._manifest_cache:
            cached_time = self._cache_expiry.get(manifest_uri)
            if cached_time and datetime.now() < cached_time:
                logger.debug(f"Using cached manifest for {manifest_uri}")
                return self._manifest_cache[manifest_uri]

        logger.info(f"Parsing HLS manifest: {manifest_uri}")

        try:
            # Fetch manifest content
            headers = self.youtube_headers if is_youtube else {}

            async with self.http_session.get(manifest_uri, headers=headers) as response:
                response.raise_for_status()
                manifest_content = await response.text()

            # Parse using m3u8 library
            parsed_m3u8 = m3u8.loads(manifest_content, uri=manifest_uri)

            # Create StreamManifest
            manifest = StreamManifest(
                master_playlist_uri=manifest_uri,
                is_live=not parsed_m3u8.is_endlist,
                version=parsed_m3u8.version or 3,
                independent_segments=getattr(
                    parsed_m3u8, "is_independent_segments", False
                ),
            )

            # Process playlists
            if parsed_m3u8.is_variant:
                # Master playlist with multiple qualities
                await self._parse_variant_playlists(parsed_m3u8, manifest, is_youtube)
            else:
                # Single media playlist
                playlist = self._parse_media_playlist(
                    parsed_m3u8, manifest_uri, is_youtube
                )
                manifest.playlists.append(playlist)

            # Cache the result
            self._manifest_cache[manifest_uri] = manifest
            self._cache_expiry[manifest_uri] = datetime.now() + self._cache_ttl

            logger.info(f"Parsed manifest with {len(manifest.playlists)} playlists")
            return manifest

        except Exception as e:
            logger.error(f"Failed to parse manifest {manifest_uri}: {e}")
            raise

    async def _parse_variant_playlists(
        self, parsed_m3u8: m3u8.M3U8, manifest: StreamManifest, is_youtube: bool
    ) -> None:
        """Parse variant playlists from master playlist."""
        tasks = []

        for playlist in parsed_m3u8.playlists:
            task = self._parse_variant_playlist(
                playlist, manifest.master_playlist_uri, is_youtube
            )
            tasks.append(task)

        # Parse playlists concurrently
        parsed_playlists = await asyncio.gather(*tasks, return_exceptions=True)

        for result in parsed_playlists:
            if isinstance(result, HLSPlaylist):
                manifest.playlists.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Failed to parse variant playlist: {result}")

    async def _parse_variant_playlist(
        self, playlist_info: m3u8.Playlist, base_uri: str, is_youtube: bool
    ) -> HLSPlaylist:
        """Parse individual variant playlist."""
        # Resolve playlist URI
        playlist_uri = urljoin(base_uri, playlist_info.uri)

        # Extract quality information
        stream_info = playlist_info.stream_info
        resolution = stream_info.resolution or "unknown"
        bandwidth = stream_info.bandwidth or 0
        codecs = stream_info.codecs or ""

        quality = StreamQuality(
            resolution=f"{resolution[0]}x{resolution[1]}"
            if isinstance(resolution, tuple)
            else str(resolution),
            bandwidth=bandwidth,
            codecs=codecs,
            fps=getattr(stream_info, "frame_rate", None),
            audio_only=stream_info.audio_only
            if hasattr(stream_info, "audio_only")
            else False,
            video_only=stream_info.video_only
            if hasattr(stream_info, "video_only")
            else False,
        )

        # Parse the actual media playlist
        return await self._parse_media_playlist_from_uri(
            playlist_uri, quality, is_youtube
        )

    async def _parse_media_playlist_from_uri(
        self, playlist_uri: str, quality: StreamQuality, is_youtube: bool
    ) -> HLSPlaylist:
        """Parse media playlist from URI."""
        try:
            headers = self.youtube_headers if is_youtube else {}

            async with self.http_session.get(playlist_uri, headers=headers) as response:
                response.raise_for_status()
                playlist_content = await response.text()

            parsed_playlist = m3u8.loads(playlist_content, uri=playlist_uri)
            return self._parse_media_playlist(
                parsed_playlist, playlist_uri, is_youtube, quality
            )

        except Exception as e:
            logger.error(f"Failed to parse media playlist {playlist_uri}: {e}")
            # Return empty playlist as fallback
            return HLSPlaylist(uri=playlist_uri, quality=quality, segments=[])

    def _parse_media_playlist(
        self,
        parsed_playlist: m3u8.M3U8,
        playlist_uri: str,
        is_youtube: bool,
        quality: Optional[StreamQuality] = None,
    ) -> HLSPlaylist:
        """Parse media playlist segments."""
        if quality is None:
            quality = StreamQuality(resolution="unknown", bandwidth=0, codecs="")

        playlist = HLSPlaylist(
            uri=playlist_uri,
            quality=quality,
            target_duration=parsed_playlist.target_duration or 6.0,
            sequence_number=parsed_playlist.media_sequence or 0,
            is_live=not parsed_playlist.is_endlist,
        )

        # Parse segments
        for i, segment in enumerate(parsed_playlist.segments):
            segment_uri = urljoin(playlist_uri, segment.uri)

            stream_segment = StreamSegment(
                uri=segment_uri,
                duration=segment.duration,
                sequence_number=playlist.sequence_number + i,
                byte_range=getattr(segment, "byterange", None),
                discontinuity=getattr(segment, "discontinuity", False),
                program_date_time=getattr(segment, "program_date_time", None),
            )

            playlist.segments.append(stream_segment)

        # Extract encryption info if present
        if hasattr(parsed_playlist, "keys") and parsed_playlist.keys:
            key = parsed_playlist.keys[0]  # Use first key
            playlist.encryption_method = getattr(key, "method", None)
            playlist.encryption_key_uri = getattr(key, "uri", None)
            playlist.encryption_iv = getattr(key, "iv", None)

        return playlist

    async def get_stream_segments(
        self,
        playlist: HLSPlaylist,
        max_segments: Optional[int] = None,
        start_sequence: Optional[int] = None,
    ) -> AsyncGenerator[StreamSegment, None]:
        """
        Get stream segments from playlist.

        Args:
            playlist: HLS playlist to read from
            max_segments: Maximum number of segments to yield
            start_sequence: Start from specific sequence number

        Yields:
            StreamSegment objects
        """
        segments = playlist.segments

        if start_sequence is not None:
            # Filter segments by sequence number
            segments = [s for s in segments if s.sequence_number >= start_sequence]

        if max_segments is not None:
            segments = segments[:max_segments]

        for segment in segments:
            yield segment

    async def download_segment(
        self, segment: StreamSegment, is_youtube: bool = False, chunk_size: int = 8192
    ) -> AsyncGenerator[bytes, None]:
        """
        Download segment data.

        Args:
            segment: Stream segment to download
            is_youtube: Whether this is a YouTube segment
            chunk_size: Chunk size for streaming download

        Yields:
            Bytes chunks from the segment
        """
        headers = self.youtube_headers if is_youtube else {}

        try:
            async with self.http_session.get(segment.uri, headers=headers) as response:
                response.raise_for_status()

                async for chunk in response.content.iter_chunked(chunk_size):
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to download segment {segment.uri}: {e}")
            raise

    async def refresh_manifest(
        self, manifest: StreamManifest, is_youtube: bool = False
    ) -> StreamManifest:
        """
        Refresh live manifest to get latest segments.

        Args:
            manifest: Existing manifest to refresh
            is_youtube: Whether this is a YouTube stream

        Returns:
            Updated manifest
        """
        if not manifest.is_live:
            logger.debug("Manifest is not live, no refresh needed")
            return manifest

        try:
            # Force cache refresh
            cache_key = manifest.master_playlist_uri
            if cache_key in self._manifest_cache:
                del self._manifest_cache[cache_key]
            if cache_key in self._cache_expiry:
                del self._cache_expiry[cache_key]

            return await self.parse_manifest(manifest.master_playlist_uri, is_youtube)

        except Exception as e:
            logger.error(f"Failed to refresh manifest: {e}")
            return manifest

    def clear_cache(self) -> None:
        """Clear manifest cache."""
        self._manifest_cache.clear()
        self._cache_expiry.clear()
        logger.info("Cleared manifest cache")


class YouTubeStreamExtractor:
    """Extract streaming URLs from YouTube video metadata."""

    # Common YouTube streaming URL patterns
    YOUTUBE_HLS_PATTERNS = [
        r'hlsManifestUrl["\']:\s*["\']([^"\']+)["\']',
        r'"hlsManifestUrl":"([^"]+)"',
        r'hls_manifest_url["\']:\s*["\']([^"\']+)["\']',
    ]

    YOUTUBE_DASH_PATTERNS = [
        r'dashManifestUrl["\']:\s*["\']([^"\']+)["\']',
        r'"dashManifestUrl":"([^"]+)"',
        r'dash_manifest_url["\']:\s*["\']([^"\']+)["\']',
    ]

    @staticmethod
    def extract_stream_urls(video_metadata: Dict) -> Dict[str, Optional[str]]:
        """
        Extract HLS and DASH URLs from YouTube video metadata.

        Args:
            video_metadata: YouTube video metadata from API

        Returns:
            Dict with hls_url and dash_url keys
        """
        result: Dict[str, Optional[str]] = {"hls_url": None, "dash_url": None}

        try:
            # Try to extract from liveStreamingDetails
            live_details = video_metadata.get("liveStreamingDetails", {})

            # Some live streams include manifest URLs directly
            if "hlsManifestUrl" in live_details:
                result["hls_url"] = live_details["hlsManifestUrl"]

            if "dashManifestUrl" in live_details:
                result["dash_url"] = live_details["dashManifestUrl"]

            # If not found, try to extract from snippet or other fields
            # Note: This is a simplified extraction - real YouTube URL extraction
            # would require more complex parsing of the video page HTML

            logger.debug(f"Extracted stream URLs: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract stream URLs: {e}")
            return result

    @staticmethod
    def is_live_stream_url_available(video_metadata: Dict) -> bool:
        """Check if live stream URLs are available in metadata."""
        live_details = video_metadata.get("liveStreamingDetails", {})
        return bool(
            live_details.get("hlsManifestUrl") or live_details.get("dashManifestUrl")
        )


# Utility functions for quality selection
def select_optimal_quality(
    qualities: List[StreamQuality],
    target_height: Optional[int] = None,
    max_bandwidth: Optional[int] = None,
    prefer_quality: str = "best",
) -> Optional[StreamQuality]:
    """
    Select optimal quality based on constraints.

    Args:
        qualities: Available stream qualities
        target_height: Target video height (e.g., 720, 1080)
        max_bandwidth: Maximum allowed bandwidth
        prefer_quality: "best", "worst", or specific quality (e.g., "720p")

    Returns:
        Selected StreamQuality or None
    """
    if not qualities:
        return None

    # Filter by bandwidth constraint
    if max_bandwidth:
        qualities = [q for q in qualities if q.bandwidth <= max_bandwidth]
        if not qualities:
            return None

    # Select based on preference (prioritize explicit target_height)
    if target_height:
        return min(qualities, key=lambda q: abs(q.height - target_height))
    elif prefer_quality == "best":
        return max(qualities, key=lambda q: q.bandwidth)
    elif prefer_quality == "worst":
        return min(qualities, key=lambda q: q.bandwidth)
    elif prefer_quality.endswith("p"):
        # Match quality name (e.g., "720p")
        target_height_from_name = int(prefer_quality[:-1])
        return min(qualities, key=lambda q: abs(q.height - target_height_from_name))
    else:
        # Default to best quality
        return max(qualities, key=lambda q: q.bandwidth)


def estimate_bandwidth_requirement(
    quality: StreamQuality, buffer_seconds: float = 10.0
) -> int:
    """
    Estimate bandwidth requirement for smooth playback.

    Args:
        quality: Stream quality
        buffer_seconds: Buffer duration in seconds

    Returns:
        Estimated bandwidth in bits per second
    """
    # Add 20% overhead for network variability
    base_bandwidth = quality.bandwidth
    overhead = int(base_bandwidth * 0.2)

    # Consider buffer requirements
    buffer_factor = max(1.0, buffer_seconds / 10.0)

    return int((base_bandwidth + overhead) * buffer_factor)
