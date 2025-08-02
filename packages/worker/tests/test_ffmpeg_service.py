"""Tests for FFmpeg service."""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
from pathlib import Path
import pytest

from worker.services.ffmpeg_processor import FFmpegProcessor as FFmpegService


class TestFFmpegService:
    """Test FFmpeg service functionality."""
    
    @pytest.fixture
    def ffmpeg_service(self):
        """Create FFmpeg service instance."""
        return FFmpegService()
    
    @pytest.mark.asyncio
    async def test_segment_stream_success(self, ffmpeg_service):
        """Test successful stream segmentation."""
        stream_url = "https://example.com/stream.m3u8"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("worker.services.ffmpeg_service.asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock successful FFmpeg execution
                mock_process = AsyncMock()
                mock_process.wait = AsyncMock(return_value=0)
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                # Create fake segment files
                segment_files = []
                for i in range(3):
                    segment_path = Path(tmpdir) / f"segment_{i:03d}.mp4"
                    segment_path.write_text("fake video data")
                    segment_files.append(segment_path)
                
                # Mock glob to return our fake segments
                with patch("worker.services.ffmpeg_service.Path.glob") as mock_glob:
                    mock_glob.return_value = segment_files
                    
                    # Run segmentation
                    segments = await ffmpeg_service.segment_stream(stream_url, tmpdir)
                    
                    # Verify results
                    assert len(segments) == 3
                    for i, segment in enumerate(segments):
                        assert segment["path"] == str(segment_files[i])
                        assert segment["start_time"] == i * 10  # 10 seconds per segment
                        assert segment["duration"] == 10
                    
                    # Verify FFmpeg was called correctly
                    mock_subprocess.assert_called_once()
                    args = mock_subprocess.call_args[0]
                    assert args[0] == "ffmpeg"
                    assert "-i" in args
                    assert stream_url in args
                    assert "-segment_time" in args
                    assert "10" in args
    
    @pytest.mark.asyncio
    async def test_segment_stream_ffmpeg_error(self, ffmpeg_service):
        """Test handling FFmpeg errors."""
        stream_url = "https://example.com/stream.m3u8"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("worker.services.ffmpeg_service.asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock FFmpeg failure
                mock_process = AsyncMock()
                mock_process.wait = AsyncMock(return_value=1)
                mock_process.returncode = 1
                mock_process.stderr = AsyncMock()
                mock_process.stderr.read = AsyncMock(return_value=b"FFmpeg error: Invalid stream")
                mock_subprocess.return_value = mock_process
                
                # Should raise exception
                with pytest.raises(Exception, match="FFmpeg segmentation failed"):
                    await ffmpeg_service.segment_stream(stream_url, tmpdir)
    
    @pytest.mark.asyncio
    async def test_segment_stream_no_segments(self, ffmpeg_service):
        """Test handling when no segments are created."""
        stream_url = "https://example.com/stream.m3u8"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("worker.services.ffmpeg_service.asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock successful FFmpeg but no segments created
                mock_process = AsyncMock()
                mock_process.wait = AsyncMock(return_value=0)
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                # Mock glob to return empty list
                with patch("worker.services.ffmpeg_service.Path.glob") as mock_glob:
                    mock_glob.return_value = []
                    
                    # Should raise exception
                    with pytest.raises(Exception, match="No segments created"):
                        await ffmpeg_service.segment_stream(stream_url, tmpdir)
    
    @pytest.mark.asyncio
    async def test_segment_stream_custom_duration(self, ffmpeg_service):
        """Test segmentation with custom segment duration."""
        stream_url = "https://example.com/stream.m3u8"
        segment_duration = 30  # 30 seconds
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("worker.services.ffmpeg_service.asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.wait = AsyncMock(return_value=0)
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                # Create fake segment
                segment_path = Path(tmpdir) / "segment_000.mp4"
                segment_path.write_text("fake video data")
                
                with patch("worker.services.ffmpeg_service.Path.glob") as mock_glob:
                    mock_glob.return_value = [segment_path]
                    
                    # Run segmentation with custom duration
                    segments = await ffmpeg_service.segment_stream(
                        stream_url, tmpdir, segment_duration=segment_duration
                    )
                    
                    # Verify segment duration
                    assert segments[0]["duration"] == segment_duration
                    
                    # Verify FFmpeg was called with custom duration
                    args = mock_subprocess.call_args[0]
                    segment_time_index = args.index("-segment_time")
                    assert args[segment_time_index + 1] == "30"
    
    @pytest.mark.asyncio
    async def test_segment_stream_max_duration(self, ffmpeg_service):
        """Test segmentation with maximum duration limit."""
        stream_url = "https://example.com/stream.m3u8"
        max_duration = 60  # 1 minute
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("worker.services.ffmpeg_service.asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.wait = AsyncMock(return_value=0)
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                # Create fake segments
                segment_files = []
                for i in range(10):  # Would be 100 seconds total
                    segment_path = Path(tmpdir) / f"segment_{i:03d}.mp4"
                    segment_path.write_text("fake video data")
                    segment_files.append(segment_path)
                
                with patch("worker.services.ffmpeg_service.Path.glob") as mock_glob:
                    mock_glob.return_value = segment_files
                    
                    # Run segmentation with max duration
                    segments = await ffmpeg_service.segment_stream(
                        stream_url, tmpdir, max_duration=max_duration
                    )
                    
                    # Should only have 6 segments (60 seconds / 10 seconds per segment)
                    assert len(segments) == 6
                    
                    # Verify FFmpeg was called with duration limit
                    args = mock_subprocess.call_args[0]
                    assert "-t" in args
                    t_index = args.index("-t")
                    assert args[t_index + 1] == "60"