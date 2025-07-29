"""
Unit tests for HLS parser utilities.

Tests for HLS manifest parsing, segment handling, and YouTube stream extraction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Apply patches before imports to ensure they take effect
import sys
from unittest.mock import Mock

# Mock the entire aiohttp module
mock_aiohttp = Mock()
mock_client_session = Mock()
mock_aiohttp.ClientSession = Mock(return_value=mock_client_session)
sys.modules['aiohttp'] = mock_aiohttp

from src.utils.hls_parser import (
    StreamQuality,
    StreamSegment,
    HLSPlaylist,
    StreamManifest,
    HLSParser,
    YouTubeStreamExtractor,
    select_optimal_quality,
    estimate_bandwidth_requirement,
)


class TestStreamQuality:
    """Test StreamQuality data class."""
    
    def test_quality_initialization(self):
        """Test StreamQuality initialization."""
        quality = StreamQuality(
            resolution="1920x1080",
            bandwidth=5000000,
            codecs="avc1.64001f,mp4a.40.2",
            fps=30.0
        )
        
        assert quality.resolution == "1920x1080"
        assert quality.bandwidth == 5000000
        assert quality.codecs == "avc1.64001f,mp4a.40.2"
        assert quality.fps == 30.0
        assert quality.width == 1920
        assert quality.height == 1080
        assert quality.quality_name == "1080p"
    
    def test_quality_names(self):
        """Test quality name generation."""
        test_cases = [
            ("3840x2160", "4K"),
            ("2560x1440", "1440p"),
            ("1920x1080", "1080p"),
            ("1280x720", "720p"),
            ("854x480", "480p"),
            ("640x360", "360p"),
            ("426x240", "240p"),
            ("320x180", "180p"),
        ]
        
        for resolution, expected_name in test_cases:
            quality = StreamQuality(resolution=resolution, bandwidth=1000000, codecs="test")
            assert quality.quality_name == expected_name


class TestStreamSegment:
    """Test StreamSegment data class."""
    
    def test_segment_initialization(self):
        """Test StreamSegment initialization."""
        segment = StreamSegment(
            uri="https://example.com/segment001.ts",
            duration=10.0,
            sequence_number=1,
            byte_range="1000@0",
            discontinuity=True
        )
        
        assert segment.uri == "https://example.com/segment001.ts"
        assert segment.duration == 10.0
        assert segment.sequence_number == 1
        assert segment.byte_range == "1000@0"
        assert segment.discontinuity is True
    
    def test_segment_encryption_detection(self):
        """Test segment encryption detection."""
        # Non-encrypted segment
        segment1 = StreamSegment(
            uri="https://example.com/segment001.ts",
            duration=10.0,
            sequence_number=1
        )
        assert not segment1.is_encrypted
        
        # Encrypted segment (key parameter)
        segment2 = StreamSegment(
            uri="https://example.com/segment001.ts?key=abc123",
            duration=10.0,
            sequence_number=1
        )
        assert segment2.is_encrypted


class TestHLSPlaylist:
    """Test HLSPlaylist data class."""
    
    def test_playlist_initialization(self):
        """Test HLSPlaylist initialization."""
        quality = StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f")
        playlist = HLSPlaylist(
            uri="https://example.com/720p.m3u8",
            quality=quality,
            target_duration=6.0,
            sequence_number=100,
            is_live=True
        )
        
        assert playlist.uri == "https://example.com/720p.m3u8"
        assert playlist.quality == quality
        assert playlist.target_duration == 6.0
        assert playlist.sequence_number == 100
        assert playlist.is_live is True
        assert playlist.media_type == "video"
    
    def test_playlist_duration_calculation(self):
        """Test playlist duration calculation."""
        quality = StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f")
        playlist = HLSPlaylist(uri="test.m3u8", quality=quality)
        
        # Add segments
        segments = [
            StreamSegment(uri="seg1.ts", duration=10.0, sequence_number=1),
            StreamSegment(uri="seg2.ts", duration=8.5, sequence_number=2),
            StreamSegment(uri="seg3.ts", duration=9.2, sequence_number=3),
        ]
        playlist.segments = segments
        
        expected_duration = 10.0 + 8.5 + 9.2
        assert playlist.duration == expected_duration
    
    def test_latest_segment(self):
        """Test latest segment retrieval."""
        quality = StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f")
        playlist = HLSPlaylist(uri="test.m3u8", quality=quality)
        
        # Empty playlist
        assert playlist.latest_segment is None
        
        # Add segments
        seg1 = StreamSegment(uri="seg1.ts", duration=10.0, sequence_number=1)
        seg2 = StreamSegment(uri="seg2.ts", duration=10.0, sequence_number=2)
        playlist.segments = [seg1, seg2]
        
        assert playlist.latest_segment == seg2


class TestStreamManifest:
    """Test StreamManifest data class."""
    
    def test_manifest_initialization(self):
        """Test StreamManifest initialization."""
        manifest = StreamManifest(
            master_playlist_uri="https://example.com/master.m3u8",
            is_live=True,
            version=6
        )
        
        assert manifest.master_playlist_uri == "https://example.com/master.m3u8"
        assert manifest.is_live is True
        assert manifest.version == 6
        assert len(manifest.playlists) == 0
    
    def test_manifest_playlist_filtering(self):
        """Test manifest playlist filtering."""
        manifest = StreamManifest(master_playlist_uri="test.m3u8")
        
        # Create video and audio qualities
        video_quality = StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f")
        audio_quality = StreamQuality(resolution="unknown", bandwidth=128000, codecs="mp4a.40.2", audio_only=True)
        
        video_playlist = HLSPlaylist(uri="video.m3u8", quality=video_quality, media_type="video")
        audio_playlist = HLSPlaylist(uri="audio.m3u8", quality=audio_quality, media_type="audio")
        
        manifest.playlists = [video_playlist, audio_playlist]
        
        # Test filtering
        video_playlists = manifest.video_playlists
        audio_playlists = manifest.audio_playlists
        
        assert len(video_playlists) == 1
        assert video_playlists[0] == video_playlist
        assert len(audio_playlists) == 1
        assert audio_playlists[0] == audio_playlist
    
    def test_quality_selection(self):
        """Test quality selection methods."""
        manifest = StreamManifest(master_playlist_uri="test.m3u8")
        
        # Create multiple video qualities
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        playlists = [
            HLSPlaylist(uri="1080p.m3u8", quality=qualities[0]),
            HLSPlaylist(uri="720p.m3u8", quality=qualities[1]),
            HLSPlaylist(uri="480p.m3u8", quality=qualities[2]),
        ]
        
        manifest.playlists = playlists
        
        # Test best quality selection
        best_playlist = manifest.get_best_quality()
        assert best_playlist == playlists[0]  # 1080p (highest bandwidth)
        
        # Test bandwidth-constrained selection
        constrained_playlist = manifest.get_best_quality(max_bandwidth=3000000)
        assert constrained_playlist == playlists[1]  # 720p
        
        # Test quality by height
        target_playlist = manifest.get_quality_by_height(720)
        assert target_playlist == playlists[1]  # 720p


class TestYouTubeStreamExtractor:
    """Test YouTubeStreamExtractor utility."""
    
    def test_extract_stream_urls_with_hls(self):
        """Test extraction with HLS manifest URL."""
        video_metadata = {
            "liveStreamingDetails": {
                "hlsManifestUrl": "https://manifest.googlevideo.com/test.m3u8",
                "dashManifestUrl": "https://manifest.googlevideo.com/test.mpd",
            }
        }
        
        result = YouTubeStreamExtractor.extract_stream_urls(video_metadata)
        
        assert result["hls_url"] == "https://manifest.googlevideo.com/test.m3u8"
        assert result["dash_url"] == "https://manifest.googlevideo.com/test.mpd"
    
    def test_extract_stream_urls_no_manifests(self):
        """Test extraction without manifest URLs."""
        video_metadata = {
            "liveStreamingDetails": {
                "actualStartTime": "2023-01-01T12:00:00Z",
            }
        }
        
        result = YouTubeStreamExtractor.extract_stream_urls(video_metadata)
        
        assert result["hls_url"] is None
        assert result["dash_url"] is None
    
    def test_extract_stream_urls_empty_metadata(self):
        """Test extraction with empty metadata."""
        result = YouTubeStreamExtractor.extract_stream_urls({})
        
        assert result["hls_url"] is None
        assert result["dash_url"] is None
    
    def test_is_live_stream_url_available(self):
        """Test live stream URL availability check."""
        # With HLS URL
        metadata_with_hls = {
            "liveStreamingDetails": {
                "hlsManifestUrl": "https://example.com/test.m3u8"
            }
        }
        assert YouTubeStreamExtractor.is_live_stream_url_available(metadata_with_hls)
        
        # With DASH URL
        metadata_with_dash = {
            "liveStreamingDetails": {
                "dashManifestUrl": "https://example.com/test.mpd"
            }
        }
        assert YouTubeStreamExtractor.is_live_stream_url_available(metadata_with_dash)
        
        # Without URLs
        metadata_without_urls = {
            "liveStreamingDetails": {
                "actualStartTime": "2023-01-01T12:00:00Z"
            }
        }
        assert not YouTubeStreamExtractor.is_live_stream_url_available(metadata_without_urls)


class TestQualitySelection:
    """Test quality selection utilities."""
    
    def test_select_optimal_quality_best(self):
        """Test selecting best quality."""
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        selected = select_optimal_quality(qualities, prefer_quality="best")
        assert selected == qualities[0]  # Highest bandwidth
    
    def test_select_optimal_quality_worst(self):
        """Test selecting worst quality."""
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        selected = select_optimal_quality(qualities, prefer_quality="worst")
        assert selected == qualities[2]  # Lowest bandwidth
    
    def test_select_optimal_quality_by_height(self):
        """Test selecting quality by target height."""
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        selected = select_optimal_quality(qualities, target_height=720)
        assert selected == qualities[1]  # 720p
        
        # Test closest match
        selected = select_optimal_quality(qualities, target_height=600)
        assert selected == qualities[1]  # 720p is closest to 600p
    
    def test_select_optimal_quality_by_name(self):
        """Test selecting quality by name."""
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        selected = select_optimal_quality(qualities, prefer_quality="720p")
        assert selected == qualities[1]  # 720p
    
    def test_select_optimal_quality_with_bandwidth_limit(self):
        """Test selecting quality with bandwidth constraint."""
        qualities = [
            StreamQuality(resolution="1920x1080", bandwidth=5000000, codecs="avc1.64001f"),
            StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f"),
            StreamQuality(resolution="854x480", bandwidth=1500000, codecs="avc1.42001e"),
        ]
        
        selected = select_optimal_quality(qualities, max_bandwidth=3000000, prefer_quality="best")
        assert selected == qualities[1]  # 720p (within bandwidth limit)
    
    def test_select_optimal_quality_empty_list(self):
        """Test selecting from empty quality list."""
        selected = select_optimal_quality([], prefer_quality="best")
        assert selected is None


class TestBandwidthEstimation:
    """Test bandwidth estimation utilities."""
    
    def test_estimate_bandwidth_requirement(self):
        """Test bandwidth requirement estimation."""
        quality = StreamQuality(resolution="1280x720", bandwidth=2500000, codecs="avc1.42001f")
        
        # Default buffer (10 seconds)
        estimated = estimate_bandwidth_requirement(quality)
        expected = int(2500000 * 1.2)  # 20% overhead
        assert estimated == expected
        
        # Custom buffer
        estimated_custom = estimate_bandwidth_requirement(quality, buffer_seconds=20.0)
        expected_custom = int(2500000 * 1.2 * 2.0)  # 20% overhead + 2x buffer factor
        assert estimated_custom == expected_custom
    
    def test_estimate_bandwidth_minimum(self):
        """Test bandwidth estimation with minimum buffer."""
        quality = StreamQuality(resolution="854x480", bandwidth=1000000, codecs="avc1.42001e")
        
        # Small buffer (should use minimum factor of 1.0)
        estimated = estimate_bandwidth_requirement(quality, buffer_seconds=5.0)
        expected = int(1000000 * 1.2 * 1.0)  # 20% overhead, no additional buffer factor
        assert estimated == expected


@pytest.mark.asyncio
class TestHLSParser:
    """Test HLSParser class (async tests)."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock HTTP session."""
        session = AsyncMock()
        return session
    
    @pytest.fixture
    def hls_parser(self, mock_session):
        """Create an HLS parser with mock session."""
        return HLSParser(session=mock_session)
    
    def test_parser_initialization(self, hls_parser):
        """Test HLS parser initialization."""
        assert hls_parser.session is not None
        assert isinstance(hls_parser.youtube_headers, dict)
        assert "User-Agent" in hls_parser.youtube_headers
        assert len(hls_parser._manifest_cache) == 0
    
    async def test_parse_simple_manifest(self, hls_parser, mock_session):
        """Test parsing a simple HLS manifest."""
        manifest_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:10.0,
segment000.ts
#EXTINF:10.0,
segment001.ts
#EXT-X-ENDLIST
"""
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = manifest_content
        mock_response.raise_for_status = Mock()
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Parse manifest
        manifest = await hls_parser.parse_manifest("https://example.com/test.m3u8")
        
        assert manifest.master_playlist_uri == "https://example.com/test.m3u8"
        assert manifest.is_live is False  # Has EXT-X-ENDLIST
        assert len(manifest.playlists) == 1
        
        playlist = manifest.playlists[0]
        assert len(playlist.segments) == 2
        assert playlist.segments[0].uri.endswith("segment000.ts")
        assert playlist.segments[1].uri.endswith("segment001.ts")
    
    async def test_parse_manifest_caching(self, hls_parser, mock_session):
        """Test manifest caching functionality."""
        manifest_content = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:10.0,
segment000.ts
"""
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = manifest_content
        mock_response.raise_for_status = Mock()
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Parse manifest twice
        manifest_uri = "https://example.com/test.m3u8"
        manifest1 = await hls_parser.parse_manifest(manifest_uri)
        manifest2 = await hls_parser.parse_manifest(manifest_uri)
        
        # Should only make one HTTP request due to caching
        assert mock_session.get.call_count == 1
        assert manifest1.master_playlist_uri == manifest2.master_playlist_uri
    
    async def test_download_segment(self, hls_parser, mock_session):
        """Test segment download functionality."""
        segment = StreamSegment(
            uri="https://example.com/segment001.ts",
            duration=10.0,
            sequence_number=1
        )
        
        segment_data = b"mock_segment_data_chunk"
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = Mock()
        mock_response.content.iter_chunked.return_value = AsyncMock()
        mock_response.content.iter_chunked.return_value.__aiter__ = AsyncMock(
            return_value=iter([segment_data])
        )
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Download segment
        chunks = []
        async for chunk in hls_parser.download_segment(segment):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0] == segment_data
    
    def test_clear_cache(self, hls_parser):
        """Test cache clearing functionality."""
        # Add something to cache
        hls_parser._manifest_cache["test"] = Mock()
        hls_parser._cache_expiry["test"] = datetime.now()
        
        assert len(hls_parser._manifest_cache) == 1
        assert len(hls_parser._cache_expiry) == 1
        
        # Clear cache
        hls_parser.clear_cache()
        
        assert len(hls_parser._manifest_cache) == 0
        assert len(hls_parser._cache_expiry) == 0