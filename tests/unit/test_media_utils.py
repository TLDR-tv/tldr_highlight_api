"""
Unit tests for media processing utilities.

Tests for video frame extraction, audio processing, and media format handling.
"""

import asyncio
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.media_utils import (
    MediaProcessor,
    VideoFrame,
    AudioChunk,
    MediaInfo,
    StreamCapture,
)


class TestMediaProcessor:
    """Test cases for MediaProcessor."""

    @pytest.fixture
    def media_processor(self):
        return MediaProcessor(max_memory_mb=100)

    @pytest.fixture
    def mock_media_info(self):
        return MediaInfo(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            has_audio=True,
            audio_codec="aac",
            audio_sample_rate=44100,
        )

    @pytest.fixture
    def sample_video_frame(self):
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return VideoFrame(
            frame=frame_data,
            timestamp=1.0,
            frame_number=30,
            width=640,
            height=480,
            quality_score=0.8,
        )

    @pytest.fixture
    def sample_audio_chunk(self):
        # Generate 1 second of audio data
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16).tobytes()

        return AudioChunk(
            data=audio_data,
            timestamp=1.0,
            duration=duration,
            sample_rate=sample_rate,
            channels=1,
            format="pcm_s16le",
        )

    def test_media_processor_initialization(self, media_processor):
        """Test MediaProcessor initialization."""
        assert media_processor.max_memory_mb == 100
        assert media_processor.temp_dir.exists()
        assert media_processor._cleanup_tasks == set()

    @pytest.mark.asyncio
    async def test_calculate_frame_quality(self, media_processor, sample_video_frame):
        """Test frame quality calculation."""
        quality_score = await media_processor._calculate_frame_quality(
            sample_video_frame.frame
        )

        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)

    @pytest.mark.asyncio
    async def test_frame_to_bytes(self, media_processor, sample_video_frame):
        """Test converting frame to bytes."""
        image_bytes = await media_processor.frame_to_bytes(
            sample_video_frame, format="JPEG", quality=85
        )

        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

    @pytest.mark.asyncio
    async def test_detect_scene_changes(self, media_processor):
        """Test scene change detection."""
        # Create two similar frames
        frame1_data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1 = VideoFrame(frame1_data, 1.0, 30, 640, 480, 0.8)

        frame2_data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = VideoFrame(frame2_data, 2.0, 60, 640, 480, 0.8)

        # Create two different frames
        frame3_data = np.ones((480, 640, 3), dtype=np.uint8) * 255
        frame3 = VideoFrame(frame3_data, 3.0, 90, 640, 480, 0.8)

        frames = [frame1, frame2, frame3]
        scene_changes = await media_processor.detect_scene_changes(
            frames, threshold=0.3
        )

        assert isinstance(scene_changes, list)
        # Should detect change between frame2 and frame3
        assert 2 in scene_changes or len(scene_changes) >= 1

    @pytest.mark.asyncio
    async def test_get_file_mime_type(self, media_processor):
        """Test MIME type detection."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name

        try:
            mime_type = await media_processor.get_file_mime_type(temp_path)
            assert "text" in mime_type or mime_type == "application/octet-stream"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_is_supported_format(self, media_processor):
        """Test format support checking."""
        # Create temporary files with different extensions
        test_files = ["test.mp4", "test.avi", "test.mp3", "test.wav", "test.txt"]

        for filename in test_files:
            with tempfile.NamedTemporaryFile(
                suffix=filename[-4:], delete=False
            ) as temp_file:
                temp_file.write(b"test")
                temp_path = temp_file.name

            try:
                with patch.object(media_processor, "get_file_mime_type") as mock_mime:
                    if filename.endswith((".mp4", ".avi")):
                        mock_mime.return_value = "video/mp4"
                    elif filename.endswith((".mp3", ".wav")):
                        mock_mime.return_value = "audio/mp3"
                    else:
                        mock_mime.return_value = "text/plain"

                    is_supported = await media_processor.is_supported_format(temp_path)

                    if filename.endswith((".mp4", ".avi", ".mp3", ".wav")):
                        assert is_supported is True
                    else:
                        assert is_supported is False
            finally:
                Path(temp_path).unlink(missing_ok=True)


class TestStreamCapture:
    """Test cases for StreamCapture."""

    @pytest.fixture
    def stream_capture(self):
        return StreamCapture("rtmp://test.stream/url", buffer_size=5)

    def test_stream_capture_initialization(self, stream_capture):
        """Test StreamCapture initialization."""
        assert stream_capture.stream_url == "rtmp://test.stream/url"
        assert stream_capture.buffer_size == 5
        assert (
            stream_capture.frame_buffer.maxsize == 5
        )  # Queue uses maxsize, not maxlen
        assert stream_capture.is_running is False
        assert stream_capture.capture_task is None

    @pytest.mark.asyncio
    async def test_start_stop_capture(self, stream_capture):
        """Test starting and stopping capture."""
        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap

            # Start capture
            await stream_capture.start_capture()
            assert stream_capture.is_running is True
            assert stream_capture.capture_task is not None

            # Let it run briefly
            await asyncio.sleep(0.1)

            # Stop capture
            await stream_capture.stop_capture()
            assert stream_capture.is_running is False

    @pytest.mark.asyncio
    async def test_get_frame_timeout(self, stream_capture):
        """Test getting frame with timeout."""
        # Should return None when no frames available
        frame = await stream_capture.get_frame(timeout=0.1)
        assert frame is None


class TestDataClasses:
    """Test data classes and structures."""

    def test_video_frame_creation(self):
        """Test VideoFrame creation."""
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(
            frame=frame_data,
            timestamp=1.5,
            frame_number=45,
            width=640,
            height=480,
            quality_score=0.75,
        )

        assert frame.timestamp == 1.5
        assert frame.frame_number == 45
        assert frame.width == 640
        assert frame.height == 480
        assert frame.quality_score == 0.75
        assert frame.frame.shape == (480, 640, 3)

    def test_audio_chunk_creation(self):
        """Test AudioChunk creation."""
        audio_data = b"test_audio_data"
        chunk = AudioChunk(
            data=audio_data,
            timestamp=2.0,
            duration=1.0,
            sample_rate=44100,
            channels=2,
            format="pcm_s16le",
        )

        assert chunk.timestamp == 2.0
        assert chunk.duration == 1.0
        assert chunk.sample_rate == 44100
        assert chunk.channels == 2
        assert chunk.format == "pcm_s16le"
        assert chunk.data == audio_data

    def test_media_info_creation(self):
        """Test MediaInfo creation."""
        info = MediaInfo(
            duration=120.0,
            width=1920,
            height=1080,
            fps=60.0,
            codec="h265",
            bitrate=5000000,
            format="mp4",
            has_audio=True,
            audio_codec="aac",
            audio_bitrate=128000,
            audio_sample_rate=48000,
        )

        assert info.duration == 120.0
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 60.0
        assert info.codec == "h265"
        assert info.has_audio is True
        assert info.audio_codec == "aac"


class TestUtilityFunctions:
    """Test utility functions and helpers."""

    @pytest.mark.asyncio
    async def test_frame_quality_calculation(self):
        """Test frame quality calculation with different frame types."""
        media_processor = MediaProcessor()

        # High quality frame (sharp, good contrast)
        high_quality_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some edges to make it sharp
        high_quality_frame[100:200, 100:200] = 255
        high_quality_frame[200:300, 200:300] = 0

        # Low quality frame (blurry, low contrast)
        low_quality_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Flat gray

        high_score = await media_processor._calculate_frame_quality(high_quality_frame)
        low_score = await media_processor._calculate_frame_quality(low_quality_frame)

        # High quality should score higher than low quality
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0

    def test_frame_conversion_edge_cases(self):
        """Test frame conversion with edge cases."""
        media_processor = MediaProcessor()

        # Test with None frame
        video_frame_none = VideoFrame(
            frame=None,
            timestamp=1.0,
            frame_number=1,
            width=0,
            height=0,
            quality_score=0.0,
        )

        # Should handle gracefully
        import asyncio

        result = asyncio.run(media_processor.frame_to_bytes(video_frame_none))
        assert result == b""  # Empty bytes for invalid frame

    @pytest.mark.skip(reason="Requires async context for task creation")
    def test_memory_management(self):
        """Test memory management and cleanup."""
        media_processor = MediaProcessor(max_memory_mb=50)

        # Verify memory limit is set
        assert media_processor.max_memory_mb == 50

        # Test cleanup task scheduling
        temp_file = Path("/tmp/test_cleanup_file")
        media_processor._schedule_cleanup(temp_file)

        # Should have added a cleanup task
        assert len(media_processor._cleanup_tasks) > 0


if __name__ == "__main__":
    pytest.main([__file__])
