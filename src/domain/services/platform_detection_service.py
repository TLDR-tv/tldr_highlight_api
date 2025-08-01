"""Platform detection service for stream URLs."""

from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from src.domain.value_objects import Url, StreamPlatform, ContentType


@runtime_checkable
class PlatformDetectionService(Protocol):
    """Service for detecting stream platform from URL."""

    def detect_platform(self, url: Url) -> StreamPlatform:
        """Detect the streaming platform from a URL.
        
        Args:
            url: Stream URL to analyze
            
        Returns:
            Detected platform
        """
        ...

    def detect_content_type(self, url: Url) -> ContentType:
        """Detect content type from URL.
        
        Args:
            url: Content URL to analyze
            
        Returns:
            Detected content type
        """
        ...


class StandardPlatformDetectionService:
    """Standard implementation of platform detection."""

    def detect_platform(self, url: Url) -> StreamPlatform:
        """Detect platform based on URL patterns."""
        url_str = str(url).lower()
        parsed = urlparse(url_str)
        domain = parsed.netloc.lower()

        # Check for known streaming protocols
        if parsed.scheme == "rtmp":
            return StreamPlatform.RTMP
        
        # Check for file extensions indicating stream types
        if url_str.endswith(".m3u8"):
            return StreamPlatform.HLS
        if url_str.endswith(".mpd"):
            return StreamPlatform.DASH
        
        # Check for known platforms by domain
        if "twitch.tv" in domain:
            return StreamPlatform.TWITCH
        if "youtube.com" in domain or "youtu.be" in domain:
            return StreamPlatform.YOUTUBE
        
        # Check for streaming paths
        if "/live/" in parsed.path or "/stream/" in parsed.path:
            if ".m3u8" in parsed.path:
                return StreamPlatform.HLS
            elif ".mpd" in parsed.path:
                return StreamPlatform.DASH
            else:
                return StreamPlatform.HTTP_STREAM
        
        # Check for local files
        if parsed.scheme == "file" or not parsed.scheme:
            return StreamPlatform.LOCAL
        
        # Default to HTTP stream for unknown patterns
        return StreamPlatform.HTTP_STREAM

    def detect_content_type(self, url: Url) -> ContentType:
        """Detect content type from URL patterns."""
        url_str = str(url).lower()
        
        # Check for live streaming indicators
        if any(indicator in url_str for indicator in [
            "/live/", "/stream/", ".m3u8", ".mpd", "rtmp://"
        ]):
            return ContentType.LIVESTREAM
        
        # Check for video file extensions
        video_extensions = [
            ".mp4", ".avi", ".mov", ".mkv", ".webm", 
            ".flv", ".wmv", ".mpg", ".mpeg"
        ]
        if any(url_str.endswith(ext) for ext in video_extensions):
            return ContentType.VIDEO
        
        # Default to livestream for unknown patterns
        return ContentType.LIVESTREAM