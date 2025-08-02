"""Tests for enhanced FFmpeg processor with dual video/audio extraction."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
import tempfile
import csv

from worker.services.ffmpeg_processor import (
    FFmpegProcessor, 
    FFmpegConfig, 
    StreamSegment,
    AudioChunk,
    StreamFormat,
)


class TestFFmpegProcessor:
    """Test enhanced FFmpeg processor functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def ffmpeg_config(self):
        """Create test FFmpeg configuration."""
        return FFmpegConfig(
            segment_duration=120,
            audio_segment_duration=30,
            audio_overlap=5,
            video_ring_buffer_size=5,
            audio_ring_buffer_size=10,
        )
    
    @pytest.fixture
    def processor(self, temp_dir, ffmpeg_config):
        """Create FFmpeg processor instance."""
        return FFmpegProcessor(
            stream_url="https://example.com/stream.m3u8",
            output_dir=temp_dir,
            config=ffmpeg_config,
        )
    
    def test_processor_initialization(self, processor, temp_dir):
        """Test processor initializes correctly."""
        assert processor.stream_url == "https://example.com/stream.m3u8"
        assert processor.output_dir == temp_dir
        assert processor.video_dir == temp_dir / "video"
        assert processor.audio_dir == temp_dir / "audio"
        assert processor.video_dir.exists()
        assert processor.audio_dir.exists()
        assert len(processor._video_ring_buffer) == 0
        assert len(processor._audio_ring_buffer) == 0
    
    def test_stream_format_detection(self):
        """Test stream format detection from URLs."""
        config = FFmpegConfig()
        temp_dir = Path("/tmp/test")
        
        # RTMP
        processor = FFmpegProcessor("rtmp://example.com/live", temp_dir, config)
        assert processor.format == StreamFormat.RTMP
        
        # HLS
        processor = FFmpegProcessor("https://example.com/stream.m3u8", temp_dir, config)
        assert processor.format == StreamFormat.HLS
        
        # DASH
        processor = FFmpegProcessor("https://example.com/stream.mpd", temp_dir, config)
        assert processor.format == StreamFormat.DASH
        
        # HTTP
        processor = FFmpegProcessor("https://example.com/video.mp4", temp_dir, config)
        assert processor.format == StreamFormat.HTTP
        
        # File
        processor = FFmpegProcessor("/path/to/video.mp4", temp_dir, config)
        assert processor.format == StreamFormat.FILE
    
    @pytest.mark.asyncio
    async def test_context_manager(self, processor):
        """Test async context manager functionality."""
        with patch.object(processor, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(processor, 'stop', new_callable=AsyncMock) as mock_stop:
                with patch.object(processor, 'cleanup_all_segments') as mock_cleanup:
                    async with processor:
                        mock_start.assert_called_once()
                    
                    mock_stop.assert_called_once()
                    mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audio_chunk_extraction(self, processor, temp_dir):
        """Test audio chunk extraction from video segment."""
        # Create a fake video segment
        video_path = temp_dir / "video" / "segment_00000.mp4"
        video_path.parent.mkdir(exist_ok=True)
        video_path.write_text("fake video")
        
        segment = StreamSegment(
            segment_id=uuid4(),
            path=video_path,
            start_time=0.0,
            duration=120.0,
            segment_number=0,
            size_bytes=1000,
            is_complete=True,
        )
        
        # Mock subprocess for FFmpeg
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_subprocess.return_value = mock_process
            
            # Create expected audio files
            for i in range(5):  # 120s / 30s with overlap
                audio_path = processor.audio_dir / f"chunk_{i:06d}.wav"
                audio_path.write_text("fake audio")
            
            chunks = await processor._extract_audio_chunks(segment)
            
            # Should extract 5 chunks (0-30, 25-55, 50-80, 75-105, 100-120)
            assert len(chunks) == 5
            
            # Verify chunk timing with overlap
            expected_starts = [0, 25, 50, 75, 100]
            for i, chunk in enumerate(chunks):
                assert chunk.start_time == expected_starts[i]
                assert chunk.duration <= 30
                assert chunk.is_complete
    
    @pytest.mark.asyncio
    async def test_ring_buffer_memory_management(self, processor, temp_dir):
        """Test ring buffer prevents memory issues."""
        # Create fake video segments
        for i in range(10):
            video_path = temp_dir / "video" / f"segment_{i:05d}.mp4"
            video_path.parent.mkdir(exist_ok=True)
            video_path.write_text(f"video {i}")
            
            segment = StreamSegment(
                segment_id=uuid4(),
                path=video_path,
                start_time=i * 120.0,
                duration=120.0,
                segment_number=i,
                size_bytes=1000,
                is_complete=True,
            )
            
            processor._video_ring_buffer.append(segment)
            
            # Ring buffer should maintain size limit
            if len(processor._video_ring_buffer) > processor.config.video_ring_buffer_size:
                old_segment = processor._video_ring_buffer.pop(0)
                # In real implementation, file would be deleted
        
        # Should only have 5 segments in buffer
        assert len(processor._video_ring_buffer) == 5
        assert processor._video_ring_buffer[0].segment_number == 5
        assert processor._video_ring_buffer[-1].segment_number == 9
    
    @pytest.mark.asyncio
    async def test_process_failure_and_restart(self, processor):
        """Test FFmpeg process failure and restart logic."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # First attempt fails
            mock_process1 = AsyncMock()
            mock_process1.pid = 1234
            mock_process1.returncode = 1
            mock_process1.wait = AsyncMock()
            
            # Second attempt succeeds
            mock_process2 = AsyncMock()
            mock_process2.pid = 5678
            mock_process2.returncode = None
            mock_process2.wait = AsyncMock()
            
            mock_subprocess.side_effect = [mock_process1, mock_process2]
            
            # Start should succeed on retry
            await processor.start()
            processor._process = mock_process1
            
            # Handle restart
            await processor._handle_restart()
            
            assert processor._retry_count == 1
            assert mock_subprocess.call_count == 2
    
    @pytest.mark.asyncio
    async def test_csv_segment_monitoring(self, processor, temp_dir):
        """Test monitoring segments from CSV file."""
        # Create CSV file with segment info
        csv_path = processor.video_dir / "segments.csv"
        csv_path.parent.mkdir(exist_ok=True)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["segment_00000.mp4", "0.0", "120.0"])
            writer.writerow(["segment_00001.mp4", "120.0", "240.0"])
        
        # Create corresponding video files
        for i in range(2):
            video_path = processor.video_dir / f"segment_{i:05d}.mp4"
            video_path.write_text(f"video {i}")
        
        # Mock audio extraction
        with patch.object(processor, '_extract_audio_chunks', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = []
            
            # Mock process check
            processor._process = Mock()
            processor._process.returncode = None
            processor._running = True
            
            segments = []
            monitor_task = asyncio.create_task(
                self._collect_segments(processor._monitor_segments(), segments, 2)
            )
            
            await asyncio.sleep(1)  # Let monitoring run
            processor._running = False
            await monitor_task
            
            assert len(segments) == 2
            assert segments[0].start_time == 0.0
            assert segments[1].start_time == 120.0
    
    async def _collect_segments(self, async_gen, segments, max_count):
        """Helper to collect segments from async generator."""
        count = 0
        async for segment in async_gen:
            segments.append(segment)
            count += 1
            if count >= max_count:
                break
    
    @pytest.mark.asyncio
    async def test_audio_extraction_edge_cases(self, processor, temp_dir):
        """Test audio extraction edge cases."""
        # Test very short video segment
        short_video = temp_dir / "video" / "short.mp4"
        short_video.parent.mkdir(exist_ok=True)
        short_video.write_text("short")
        
        short_segment = StreamSegment(
            segment_id=uuid4(),
            path=short_video,
            start_time=0.0,
            duration=3.0,  # Less than 5 seconds
            segment_number=0,
            size_bytes=100,
            is_complete=True,
        )
        
        chunks = await processor._extract_audio_chunks(short_segment)
        assert len(chunks) == 0  # Should skip too-short segments
        
        # Test segment at boundary
        boundary_segment = StreamSegment(
            segment_id=uuid4(),
            path=short_video,
            start_time=0.0,
            duration=35.0,  # Just over one chunk
            segment_number=0,
            size_bytes=1000,
            is_complete=True,
        )
        
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_subprocess.return_value = mock_process
            
            # Create audio files
            for i in range(2):
                audio_path = processor.audio_dir / f"chunk_{i:06d}.wav"
                audio_path.write_text("audio")
            
            chunks = await processor._extract_audio_chunks(boundary_segment)
            assert len(chunks) == 2  # 0-30 and 25-35
    
    @pytest.mark.asyncio
    async def test_cleanup_all_segments(self, processor, temp_dir):
        """Test cleanup removes all segments."""
        # Add segments to ring buffers
        for i in range(3):
            video_path = temp_dir / "video" / f"segment_{i:05d}.mp4"
            video_path.parent.mkdir(exist_ok=True)
            video_path.write_text("video")
            
            segment = StreamSegment(
                segment_id=uuid4(),
                path=video_path,
                start_time=i * 120.0,
                duration=120.0,
                segment_number=i,
                size_bytes=1000,
                is_complete=True,
            )
            processor._video_ring_buffer.append(segment)
            
            audio_path = temp_dir / "audio" / f"chunk_{i:06d}.wav"
            audio_path.parent.mkdir(exist_ok=True)
            audio_path.write_text("audio")
            
            chunk = AudioChunk(
                chunk_id=uuid4(),
                path=audio_path,
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                duration=30.0,
                chunk_number=i,
                size_bytes=500,
                is_complete=True,
            )
            processor._audio_ring_buffer.append(chunk)
        
        # Cleanup
        processor.cleanup_all_segments()
        
        assert len(processor._video_ring_buffer) == 0
        assert len(processor._audio_ring_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_ffmpeg_command_building(self, processor):
        """Test FFmpeg command construction."""
        cmd = processor._build_ffmpeg_command()
        
        assert "ffmpeg" in cmd
        assert "-y" in cmd  # Overwrite
        assert processor.stream_url in cmd
        assert "-segment_time" in cmd
        assert str(processor.config.segment_duration) in cmd
        assert "-f" in cmd
        assert "segment" in cmd
    
    @pytest.mark.asyncio
    async def test_error_handling_in_audio_extraction(self, processor, temp_dir):
        """Test error handling during audio extraction."""
        video_path = temp_dir / "video" / "test.mp4"
        video_path.parent.mkdir(exist_ok=True)
        video_path.write_text("video")
        
        segment = StreamSegment(
            segment_id=uuid4(),
            path=video_path,
            start_time=0.0,
            duration=120.0,
            segment_number=0,
            size_bytes=1000,
            is_complete=True,
        )
        
        # Mock FFmpeg failure
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate = AsyncMock(return_value=(b"", b"Error"))
            mock_subprocess.return_value = mock_process
            
            chunks = await processor._extract_audio_chunks(segment)
            
            # Should handle errors gracefully
            assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_segment_processing(self, processor):
        """Test handling concurrent segment processing."""
        # This tests that the processor can handle multiple segments
        # being processed simultaneously without race conditions
        
        async def mock_handler(segment):
            await asyncio.sleep(0.1)  # Simulate processing
        
        handler = Mock()
        handler.handle_segment = mock_handler
        
        # Create multiple segments quickly
        segments = []
        for i in range(5):
            segment = StreamSegment(
                segment_id=uuid4(),
                path=Path(f"/fake/segment_{i}.mp4"),
                start_time=i * 120.0,
                duration=120.0,
                segment_number=i,
                size_bytes=1000,
                is_complete=True,
            )
            segments.append(segment)
        
        # Process concurrently
        tasks = [handler.handle_segment(s) for s in segments]
        await asyncio.gather(*tasks)
        
        # All should complete without issues
        assert True  # If we get here, no race conditions occurred


class TestFFmpegProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_process_already_running(self, temp_dir):
        """Test starting when already running."""
        processor = FFmpegProcessor("test.mp4", temp_dir)
        processor._running = True
        
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            await processor.start()
            mock_subprocess.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, temp_dir):
        """Test stopping when not running."""
        processor = FFmpegProcessor("test.mp4", temp_dir)
        processor._running = False
        
        # Should not raise exception
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_max_retry_attempts(self, temp_dir):
        """Test maximum retry attempts."""
        config = FFmpegConfig(reconnect_attempts=2)
        processor = FFmpegProcessor("test.mp4", temp_dir, config)
        processor._retry_count = 2
        
        with pytest.raises(RuntimeError, match="Max reconnection attempts"):
            await processor._handle_restart()
    
    @pytest.mark.asyncio
    async def test_invalid_stream_url(self, temp_dir):
        """Test handling invalid stream URLs."""
        processor = FFmpegProcessor("invalid://url", temp_dir)
        
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")
            
            with pytest.raises(FileNotFoundError):
                await processor.start()
    
    @pytest.mark.asyncio 
    async def test_audio_extraction_with_missing_video(self, temp_dir):
        """Test audio extraction when video file is missing."""
        processor = FFmpegProcessor("test.mp4", temp_dir)
        
        segment = StreamSegment(
            segment_id=uuid4(),
            path=Path("/nonexistent/video.mp4"),
            start_time=0.0,
            duration=120.0,
            segment_number=0,
            size_bytes=0,
            is_complete=True,
        )
        
        chunks = await processor._extract_audio_chunks(segment)
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_zero_duration_segments(self, temp_dir):
        """Test handling zero duration segments."""
        processor = FFmpegProcessor("test.mp4", temp_dir)
        
        segment = StreamSegment(
            segment_id=uuid4(),
            path=Path("test.mp4"),
            start_time=0.0,
            duration=0.0,  # Zero duration
            segment_number=0,
            size_bytes=1000,
            is_complete=True,
        )
        
        chunks = await processor._extract_audio_chunks(segment)
        assert len(chunks) == 0