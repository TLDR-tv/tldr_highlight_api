"""Unit tests for stream validation utilities.

Tests URL validation, platform detection, and stream URL normalization
for different streaming platforms.
"""

import pytest

from src.utils.stream_validation import (
    detect_platform,
    validate_stream_url,
    extract_twitch_username,
    extract_youtube_video_id,
    is_live_stream_url,
    normalize_stream_url,
    ValidationError,
)
from src.models.stream import StreamPlatform


class TestPlatformDetection:
    """Test platform detection from URLs."""

    def test_detect_twitch_platform(self):
        """Test Twitch platform detection."""
        urls = [
            "https://www.twitch.tv/shroud",
            "https://twitch.tv/ninja",
            "http://www.twitch.tv/pokimane",
            "http://twitch.tv/xqcow",
        ]

        for url in urls:
            assert detect_platform(url) == StreamPlatform.TWITCH

    def test_detect_youtube_platform(self):
        """Test YouTube platform detection."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "https://www.youtube.com/c/PewDiePie/live",
            "https://www.youtube.com/channel/UC-lHJZR3Gqxm24_Vd_AJ5Yw/live",
            "https://www.youtube.com/@pewdiepie/live",
        ]

        for url in urls:
            assert detect_platform(url) == StreamPlatform.YOUTUBE

    def test_detect_rtmp_platform(self):
        """Test RTMP platform detection."""
        urls = [
            "rtmp://live.twitch.tv/live/stream_key",
            "rtmps://ingest.twitch.tv/live/stream_key",
            "rtmp://a.rtmp.youtube.com/live2/stream_key",
        ]

        for url in urls:
            assert detect_platform(url) == StreamPlatform.RTMP

    def test_detect_custom_platform(self):
        """Test custom platform detection for HLS/DASH."""
        urls = [
            "https://example.com/stream.m3u8",
            "https://example.com/stream.m3u8?token=abc",
            "https://example.com/stream.mpd",
            "https://example.com/stream.mpd?token=abc",
        ]

        for url in urls:
            assert detect_platform(url) == StreamPlatform.CUSTOM

    def test_detect_platform_invalid_url(self):
        """Test platform detection with invalid URLs."""
        invalid_urls = [
            "",
            "not_a_url",
            "https://example.com",
            "ftp://example.com/file.txt",
            "https://unsupported-platform.com/stream",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                detect_platform(url)

    def test_detect_platform_none_input(self):
        """Test platform detection with None input."""
        with pytest.raises(ValidationError):
            detect_platform(None)

    def test_detect_platform_empty_string(self):
        """Test platform detection with empty string."""
        with pytest.raises(ValidationError):
            detect_platform("")


class TestTwitchValidation:
    """Test Twitch URL validation."""

    def test_validate_twitch_valid_urls(self):
        """Test validation of valid Twitch URLs."""
        test_cases = [
            ("https://www.twitch.tv/shroud", "shroud"),
            ("https://twitch.tv/ninja", "ninja"),
            ("http://www.twitch.tv/pokimane", "pokimane"),
            ("http://twitch.tv/xqcow", "xqcow"),
            ("https://www.twitch.tv/TSM_Myth/", "tsm_myth"),  # Case insensitive
        ]

        for url, expected_username in test_cases:
            result = validate_stream_url(url, StreamPlatform.TWITCH)

            assert result["platform"] == StreamPlatform.TWITCH.value
            assert result["username"] == expected_username
            assert result["identifier"] == expected_username
            assert result["url"] == f"https://www.twitch.tv/{expected_username}"

    def test_validate_twitch_invalid_urls(self):
        """Test validation of invalid Twitch URLs."""
        invalid_urls = [
            "https://www.twitch.tv/",
            "https://www.twitch.tv/ab",  # Too short
            "https://www.twitch.tv/a" * 26,  # Too long
            "https://www.twitch.tv/user-with-dash",  # Invalid characters
            "https://www.twitch.tv/user with space",  # Invalid characters
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_stream_url(url, StreamPlatform.TWITCH)

    def test_extract_twitch_username(self):
        """Test Twitch username extraction."""
        test_cases = [
            ("https://www.twitch.tv/shroud", "shroud"),
            ("https://twitch.tv/NINJA", "ninja"),  # Case normalization
            ("http://www.twitch.tv/pokimane/", "pokimane"),
        ]

        for url, expected_username in test_cases:
            assert extract_twitch_username(url) == expected_username


class TestYouTubeValidation:
    """Test YouTube URL validation."""

    def test_validate_youtube_video_urls(self):
        """Test validation of YouTube video URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_video_id in test_cases:
            result = validate_stream_url(url, StreamPlatform.YOUTUBE)

            assert result["platform"] == StreamPlatform.YOUTUBE.value
            assert result["video_id"] == expected_video_id
            assert result["identifier"] == expected_video_id
            assert result["type"] == "video"

    def test_validate_youtube_channel_urls(self):
        """Test validation of YouTube channel URLs."""
        test_cases = [
            ("https://www.youtube.com/c/PewDiePie/live", "PewDiePie", "channel_live"),
            (
                "https://www.youtube.com/channel/UC-lHJZR3Gqxm24_Vd_AJ5Yw/live",
                "UC-lHJZR3Gqxm24_Vd_AJ5Yw",
                "channel_live",
            ),
            ("https://www.youtube.com/@pewdiepie/live", "pewdiepie", "handle_live"),
        ]

        for url, expected_identifier, expected_type in test_cases:
            result = validate_stream_url(url, StreamPlatform.YOUTUBE)

            assert result["platform"] == StreamPlatform.YOUTUBE.value
            assert result["identifier"] == expected_identifier
            assert result["type"] == expected_type

    def test_validate_youtube_invalid_urls(self):
        """Test validation of invalid YouTube URLs."""
        invalid_urls = [
            "https://www.youtube.com/",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/watch?v=",
            "https://www.youtube.com/watch?v=invalid_id_too_short",
            "https://www.youtube.com/watch?v=invalid_id_way_too_long_to_be_valid",
            "https://youtu.be/",
            "https://www.youtube.com/c/",
            "https://www.youtube.com/channel/",
            "https://www.youtube.com/live/",
            "https://www.youtube.com/live/invalid_id_short",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_stream_url(url, StreamPlatform.YOUTUBE)

    def test_extract_youtube_video_id(self):
        """Test YouTube video ID extraction."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/c/PewDiePie/live", None),  # Channel URL
            ("https://www.youtube.com/@pewdiepie/live", None),  # Handle URL
        ]

        for url, expected_video_id in test_cases:
            assert extract_youtube_video_id(url) == expected_video_id


class TestRTMPValidation:
    """Test RTMP URL validation."""

    def test_validate_rtmp_valid_urls(self):
        """Test validation of valid RTMP URLs."""
        test_cases = [
            "rtmp://live.twitch.tv/live/stream_key",
            "rtmps://ingest.twitch.tv/live/stream_key",
            "rtmp://a.rtmp.youtube.com/live2/stream_key",
            "rtmp://example.com:1935/live/stream",
        ]

        for url in test_cases:
            result = validate_stream_url(url, StreamPlatform.RTMP)

            assert result["platform"] == StreamPlatform.RTMP.value
            assert "hostname" in result
            assert "port" in result
            assert "path" in result
            assert "scheme" in result

    def test_validate_rtmp_invalid_urls(self):
        """Test validation of invalid RTMP URLs."""
        invalid_urls = [
            "rtmp://",
            "rtmp://hostname",
            "rtmp://hostname/",
            "http://example.com/rtmp",  # Wrong protocol
            "rtmp://:1935/live/stream",  # Missing hostname
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_stream_url(url, StreamPlatform.RTMP)


class TestCustomValidation:
    """Test custom stream URL validation."""

    def test_validate_custom_hls_urls(self):
        """Test validation of HLS URLs."""
        test_cases = [
            "https://example.com/stream.m3u8",
            "https://example.com/stream.m3u8?token=abc",
            "http://example.com/live/stream.m3u8",
        ]

        for url in test_cases:
            result = validate_stream_url(url, StreamPlatform.CUSTOM)

            assert result["platform"] == StreamPlatform.CUSTOM.value
            assert result["stream_type"] == "hls"
            assert "hostname" in result
            assert "path" in result

    def test_validate_custom_dash_urls(self):
        """Test validation of DASH URLs."""
        test_cases = [
            "https://example.com/stream.mpd",
            "https://example.com/stream.mpd?token=abc",
            "http://example.com/live/stream.mpd",
        ]

        for url in test_cases:
            result = validate_stream_url(url, StreamPlatform.CUSTOM)

            assert result["platform"] == StreamPlatform.CUSTOM.value
            assert result["stream_type"] == "dash"
            assert "hostname" in result
            assert "path" in result

    def test_validate_custom_invalid_urls(self):
        """Test validation of invalid custom URLs."""
        invalid_urls = [
            "ftp://example.com/stream.m3u8",  # Wrong protocol
            "https:///stream.m3u8",  # Missing hostname
            "stream.m3u8",  # Missing protocol and hostname
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_stream_url(url, StreamPlatform.CUSTOM)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_live_stream_url_valid(self):
        """Test live stream URL detection with valid URLs."""
        valid_urls = [
            "https://www.twitch.tv/shroud",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "rtmp://live.twitch.tv/live/stream_key",
            "https://example.com/stream.m3u8",
        ]

        for url in valid_urls:
            assert is_live_stream_url(url) is True

    def test_is_live_stream_url_invalid(self):
        """Test live stream URL detection with invalid URLs."""
        invalid_urls = [
            "https://example.com",
            "ftp://example.com/file.txt",
            "not_a_url",
            "",
        ]

        for url in invalid_urls:
            assert is_live_stream_url(url) is False

    def test_normalize_stream_url(self):
        """Test stream URL normalization."""
        test_cases = [
            ("https://twitch.tv/SHROUD", "https://www.twitch.tv/shroud"),
            ("https://www.twitch.tv/ninja/", "https://www.twitch.tv/ninja"),
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ),
        ]

        for input_url, expected_output in test_cases:
            assert normalize_stream_url(input_url) == expected_output

    def test_validate_stream_url_platform_mismatch(self):
        """Test URL validation with platform mismatch."""
        with pytest.raises(ValidationError, match="platform mismatch"):
            validate_stream_url("https://www.twitch.tv/shroud", StreamPlatform.YOUTUBE)

    def test_validate_stream_url_auto_detect(self):
        """Test URL validation with automatic platform detection."""
        result = validate_stream_url("https://www.twitch.tv/shroud")

        assert result["platform"] == StreamPlatform.TWITCH.value
        assert result["username"] == "shroud"


class TestValidationErrorHandling:
    """Test error handling in validation functions."""

    def test_validation_error_types(self):
        """Test different types of validation errors."""
        # Test with None input
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_stream_url(None)

        # Test with empty string
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_stream_url("")

        # Test with non-string input
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            validate_stream_url(123)

    def test_unsupported_platform_error(self):
        """Test error handling for unsupported platforms."""
        # This would need to be tested if we add more platforms
        pass  # Currently all enum values are supported
