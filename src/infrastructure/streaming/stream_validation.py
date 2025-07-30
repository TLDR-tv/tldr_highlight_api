"""Stream validation utilities for URL validation and platform detection.

This module provides utilities for validating stream URLs and detecting
streaming platforms based on URL patterns. It supports Twitch, YouTube,
and generic RTMP streams.
"""

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

from src.infrastructure.persistence.models.stream import StreamPlatform


class ValidationError(Exception):
    """Custom exception for stream validation errors."""

    pass


class StreamUrlPatterns:
    """URL patterns for different streaming platforms."""

    # Twitch patterns
    TWITCH_STREAM_PATTERN = re.compile(
        r"^https?://(?:www\.)?twitch\.tv/([a-zA-Z0-9_]{4,25})/?$"
    )
    TWITCH_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{4,25}$")

    # YouTube patterns
    YOUTUBE_LIVE_PATTERNS = [
        re.compile(
            r"^https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:&.*)?$"
        ),
        re.compile(
            r"^https?://(?:www\.)?youtube\.com/live/([a-zA-Z0-9_-]{11})(?:[?&].*)?$"
        ),
        re.compile(r"^https?://youtu\.be/([a-zA-Z0-9_-]{11})(?:[?&].*)?$"),
        re.compile(r"^https?://(?:www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)/live/?$"),
        re.compile(
            r"^https?://(?:www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)/live/?$"
        ),
        re.compile(r"^https?://(?:www\.)?youtube\.com/@([a-zA-Z0-9_.-]+)/live/?$"),
    ]

    # RTMP patterns
    RTMP_PATTERN = re.compile(r"^rtmps?://[^/]+/.+$")

    # Generic stream patterns
    HLS_PATTERN = re.compile(r"^https?://.+\.m3u8(?:\?.*)?$")
    DASH_PATTERN = re.compile(r"^https?://.+\.mpd(?:\?.*)?$")


def detect_platform(url: str) -> StreamPlatform:
    """Detect the streaming platform from a URL.

    Args:
        url: The stream URL to analyze

    Returns:
        StreamPlatform: The detected platform

    Raises:
        ValidationError: If the platform cannot be detected
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")

    url = url.strip()

    # Check Twitch
    if StreamUrlPatterns.TWITCH_STREAM_PATTERN.match(url):
        return StreamPlatform.TWITCH

    # Check YouTube
    for pattern in StreamUrlPatterns.YOUTUBE_LIVE_PATTERNS:
        if pattern.match(url):
            return StreamPlatform.YOUTUBE

    # Check RTMP
    if StreamUrlPatterns.RTMP_PATTERN.match(url):
        return StreamPlatform.RTMP

    # Check other streaming formats
    if StreamUrlPatterns.HLS_PATTERN.match(url) or StreamUrlPatterns.DASH_PATTERN.match(
        url
    ):
        return StreamPlatform.CUSTOM

    raise ValidationError(f"Unsupported or invalid stream URL: {url}")


def validate_stream_url(
    url: str, platform: Optional[StreamPlatform] = None
) -> Dict[str, str]:
    """Validate a stream URL and extract relevant information.

    Args:
        url: The stream URL to validate
        platform: Expected platform (optional, will be detected if not provided)

    Returns:
        Dict containing validated URL information:
        - platform: The detected platform
        - url: The normalized URL
        - identifier: Platform-specific identifier (username, video_id, etc.)
        - additional platform-specific fields

    Raises:
        ValidationError: If the URL is invalid or doesn't match the expected platform
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")

    url = url.strip()

    # Detect platform if not provided
    if platform is None:
        platform = detect_platform(url)
    else:
        # Verify platform matches detected platform
        detected_platform = detect_platform(url)
        if platform != detected_platform:
            raise ValidationError(
                f"URL platform mismatch: expected {platform.value}, "
                f"detected {detected_platform.value}"
            )

    # Validate based on platform
    if platform == StreamPlatform.TWITCH:
        return _validate_twitch_url(url)
    elif platform == StreamPlatform.YOUTUBE:
        return _validate_youtube_url(url)
    elif platform == StreamPlatform.RTMP:
        return _validate_rtmp_url(url)
    elif platform == StreamPlatform.CUSTOM:
        return _validate_custom_url(url)
    else:
        raise ValidationError(f"Unsupported platform: {platform.value}")


def _validate_twitch_url(url: str) -> Dict[str, str]:
    """Validate a Twitch stream URL.

    Args:
        url: Twitch stream URL

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If the URL is invalid
    """
    match = StreamUrlPatterns.TWITCH_STREAM_PATTERN.match(url)
    if not match:
        raise ValidationError(f"Invalid Twitch stream URL: {url}")

    username = match.group(1).lower()  # Twitch usernames are case-insensitive

    if not StreamUrlPatterns.TWITCH_USERNAME_PATTERN.match(username):
        raise ValidationError(f"Invalid Twitch username: {username}")

    return {
        "platform": StreamPlatform.TWITCH.value,
        "url": f"https://www.twitch.tv/{username}",
        "identifier": username,
        "username": username,
    }


def _validate_youtube_url(url: str) -> Dict[str, str]:
    """Validate a YouTube Live stream URL.

    Args:
        url: YouTube stream URL

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If the URL is invalid
    """
    for pattern in StreamUrlPatterns.YOUTUBE_LIVE_PATTERNS:
        match = pattern.match(url)
        if match:
            identifier = match.group(1)

            # Determine identifier type based on pattern
            if "/watch?v=" in url or "youtu.be/" in url or "/live/" in url:
                # Video ID
                parsed_url = urlparse(url)
                if "v" in parse_qs(parsed_url.query):
                    video_id = parse_qs(parsed_url.query)["v"][0]
                else:
                    video_id = identifier

                return {
                    "platform": StreamPlatform.YOUTUBE.value,
                    "url": url,
                    "identifier": video_id,
                    "video_id": video_id,
                    "type": "video",
                }
            elif "/c/" in url:
                # Channel custom URL
                return {
                    "platform": StreamPlatform.YOUTUBE.value,
                    "url": url,
                    "identifier": identifier,
                    "channel_custom_url": identifier,
                    "type": "channel_live",
                }
            elif "/channel/" in url:
                # Channel ID
                return {
                    "platform": StreamPlatform.YOUTUBE.value,
                    "url": url,
                    "identifier": identifier,
                    "channel_id": identifier,
                    "type": "channel_live",
                }
            elif "/@" in url:
                # Handle format
                return {
                    "platform": StreamPlatform.YOUTUBE.value,
                    "url": url,
                    "identifier": identifier,
                    "handle": identifier,
                    "type": "handle_live",
                }

    raise ValidationError(f"Invalid YouTube Live stream URL: {url}")


def _validate_rtmp_url(url: str) -> Dict[str, Any]:
    """Validate an RTMP stream URL.

    Args:
        url: RTMP stream URL

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If the URL is invalid
    """
    if not StreamUrlPatterns.RTMP_PATTERN.match(url):
        raise ValidationError(f"Invalid RTMP stream URL: {url}")

    parsed_url = urlparse(url)

    if not parsed_url.hostname:
        raise ValidationError(f"RTMP URL must include hostname: {url}")

    if not parsed_url.path or parsed_url.path == "/":
        raise ValidationError(f"RTMP URL must include stream path: {url}")

    return {
        "platform": StreamPlatform.RTMP.value,
        "url": url,
        "identifier": f"{parsed_url.hostname}{parsed_url.path}",
        "hostname": parsed_url.hostname,
        "port": parsed_url.port or (1935 if parsed_url.scheme == "rtmp" else 443),
        "path": parsed_url.path,
        "scheme": parsed_url.scheme,
    }


def _validate_custom_url(url: str) -> Dict[str, str]:
    """Validate a custom stream URL (HLS, DASH, etc.).

    Args:
        url: Custom stream URL

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If the URL is invalid
    """
    parsed_url = urlparse(url)

    if not parsed_url.scheme or parsed_url.scheme not in ["http", "https"]:
        raise ValidationError(f"Custom stream URL must use HTTP/HTTPS: {url}")

    if not parsed_url.hostname:
        raise ValidationError(f"Custom stream URL must include hostname: {url}")

    # Determine stream type
    stream_type = "unknown"
    if StreamUrlPatterns.HLS_PATTERN.match(url):
        stream_type = "hls"
    elif StreamUrlPatterns.DASH_PATTERN.match(url):
        stream_type = "dash"

    return {
        "platform": StreamPlatform.CUSTOM.value,
        "url": url,
        "identifier": f"{parsed_url.hostname}{parsed_url.path}",
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "stream_type": stream_type,
    }


def extract_twitch_username(url: str) -> str:
    """Extract Twitch username from a Twitch URL.

    Args:
        url: Twitch stream URL

    Returns:
        str: The Twitch username

    Raises:
        ValidationError: If the URL is not a valid Twitch URL
    """
    validation_result = _validate_twitch_url(url)
    return validation_result["username"]


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from a YouTube URL.

    Args:
        url: YouTube stream URL

    Returns:
        Optional[str]: The video ID if present, None for channel/handle URLs

    Raises:
        ValidationError: If the URL is not a valid YouTube URL
    """
    validation_result = _validate_youtube_url(url)
    return validation_result.get("video_id")


def is_live_stream_url(url: str) -> bool:
    """Check if a URL appears to be a live stream.

    Args:
        url: The URL to check

    Returns:
        bool: True if it appears to be a live stream URL
    """
    try:
        platform = detect_platform(url)
        return platform in [
            StreamPlatform.TWITCH,
            StreamPlatform.YOUTUBE,
            StreamPlatform.RTMP,
            StreamPlatform.CUSTOM,
        ]
    except ValidationError:
        return False


def normalize_stream_url(url: str) -> str:
    """Normalize a stream URL to a canonical format.

    Args:
        url: The stream URL to normalize

    Returns:
        str: The normalized URL

    Raises:
        ValidationError: If the URL is invalid
    """
    validation_result = validate_stream_url(url)
    return validation_result["url"]
