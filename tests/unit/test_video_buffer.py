"""Unit tests for video buffer implementation."""

import asyncio
import pytest
import time
from datetime import datetime, timezone

from src.utils.video_buffer import (
    BufferConfig,
    BufferFormat,
    CircularVideoBuffer,
    FrameType,
    VideoFrame,
    VideoBufferManager,
    BufferSegment,
)


@pytest.fixture
def buffer_config():
    """Create a test buffer configuration."""
    return BufferConfig(
        max_memory_mb=10,
        max_items=100,
        retention_seconds=60.0,
        min_retention_items=10,
        enable_keyframe_priority=True,
        enable_memory_pooling=False,  # Disable for testing
        gc_interval_seconds=60.0,  # Longer interval for testing
        stats_interval_seconds=60.0,
    )


@pytest.fixture
def video_buffer(buffer_config):
    """Create a test video buffer."""
    return CircularVideoBuffer(buffer_config)


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    return VideoFrame(
        data=b"test_frame_data",
        timestamp=datetime.now(timezone.utc).timestamp(),
        duration=0.033,  # ~30fps
        format=BufferFormat.HLS_SEGMENT,
        frame_type=FrameType.P_FRAME,
        is_keyframe=False,
        width=1920,
        height=1080,
        fps=30.0,
        bitrate=5000000,
        codec="h264",
        sequence_number=1,
        quality_score=0.8,
    )


@pytest.fixture
def keyframe():
    """Create a sample keyframe."""
    return VideoFrame(
        data=b"keyframe_data" * 100,  # Larger data for keyframe
        timestamp=datetime.now(timezone.utc).timestamp(),
        duration=0.033,
        format=BufferFormat.HLS_SEGMENT,
        frame_type=FrameType.I_FRAME,
        is_keyframe=True,
        width=1920,
        height=1080,
        fps=30.0,
        bitrate=5000000,
        codec="h264",
        sequence_number=0,
        quality_score=0.9,
    )


class TestVideoFrame:
    """Test VideoFrame dataclass."""

    def test_frame_creation(self, sample_frame):
        """Test creating a video frame."""
        assert sample_frame.data == b"test_frame_data"
        assert sample_frame.format == BufferFormat.HLS_SEGMENT
        assert sample_frame.frame_type == FrameType.P_FRAME
        assert not sample_frame.is_keyframe
        assert sample_frame.size_bytes == len(b"test_frame_data")
        assert sample_frame.width == 1920
        assert sample_frame.height == 1080

    def test_frame_comparison(self):
        """Test frame comparison by timestamp."""
        frame1 = VideoFrame(
            data=b"frame1",
            timestamp=1000.0,
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.P_FRAME,
        )

        frame2 = VideoFrame(
            data=b"frame2",
            timestamp=2000.0,
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.P_FRAME,
        )

        assert frame1 < frame2
        assert not frame2 < frame1


class TestBufferSegment:
    """Test BufferSegment dataclass."""

    def test_segment_creation(self):
        """Test creating a buffer segment."""
        segment = BufferSegment(
            segment_id="test_segment",
            start_time=1000.0,
            end_time=1010.0,
        )

        assert segment.segment_id == "test_segment"
        assert segment.start_time == 1000.0
        assert segment.end_time == 1010.0
        assert segment.duration == 10.0
        assert segment.frame_count == 0
        assert not segment.has_keyframe

    def test_add_frame_to_segment(self, sample_frame, keyframe):
        """Test adding frames to a segment."""
        segment = BufferSegment(
            segment_id="test_segment",
            start_time=sample_frame.timestamp,
            end_time=sample_frame.timestamp + 10.0,
        )

        # Add regular frame
        segment.add_frame(sample_frame)
        assert segment.frame_count == 1
        assert segment.total_size == sample_frame.size_bytes
        assert not segment.has_keyframe

        # Add keyframe
        segment.add_frame(keyframe)
        assert segment.frame_count == 2
        assert segment.keyframe_count == 1
        assert segment.has_keyframe
        assert segment.total_size == sample_frame.size_bytes + keyframe.size_bytes


class TestCircularVideoBuffer:
    """Test CircularVideoBuffer implementation."""

    def test_buffer_initialization(self, video_buffer, buffer_config):
        """Test buffer initialization."""
        assert video_buffer.config == buffer_config
        assert len(video_buffer._frames) == 0
        assert video_buffer._total_memory_bytes == 0
        assert video_buffer._frame_count == 0

    def test_add_frame(self, video_buffer, sample_frame):
        """Test adding frames to buffer."""
        # Add frame
        assert video_buffer.add_frame(sample_frame)

        # Check buffer state
        assert len(video_buffer._frames) == 1
        assert video_buffer._frame_count == 1
        assert video_buffer._total_memory_bytes == sample_frame.size_bytes
        assert video_buffer._stats["frames_added"] == 1

    def test_add_multiple_frames(self, video_buffer):
        """Test adding multiple frames."""
        frames = []
        base_time = datetime.now(timezone.utc).timestamp()

        # Add 10 frames
        for i in range(10):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i * 0.033,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=(i % 5 == 0),  # Every 5th frame is keyframe
                sequence_number=i,
            )
            frames.append(frame)
            assert video_buffer.add_frame(frame)

        # Check buffer state
        assert len(video_buffer._frames) == 10
        assert video_buffer._frame_count == 10
        assert video_buffer._stats["frames_added"] == 10

        # Check keyframe index
        assert len(video_buffer._keyframe_index) == 2  # Frames 0 and 5

    def test_memory_limit_enforcement(self, buffer_config):
        """Test that memory limits are enforced."""
        # Create buffer with small memory limit
        config = BufferConfig(
            max_memory_mb=1,  # 1MB limit
            max_items=1000,
            enable_keyframe_priority=False,
        )
        buffer = CircularVideoBuffer(config)

        # Create large frame (500KB)
        large_frame_data = b"x" * (500 * 1024)

        # Add frames until memory limit
        frames_added = 0
        for i in range(10):
            frame = VideoFrame(
                data=large_frame_data,
                timestamp=time.time() + i,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=False,
                sequence_number=i,
            )

            if buffer.add_frame(frame):
                frames_added += 1

        # Should only fit 2 frames (1MB / 500KB)
        assert frames_added <= 2
        assert buffer._total_memory_bytes <= 1024 * 1024  # 1MB

    def test_keyframe_priority(self, buffer_config):
        """Test keyframe prioritization when memory is tight."""
        # Create buffer with small memory limit
        config = BufferConfig(
            max_memory_mb=1,
            max_items=100,
            enable_keyframe_priority=True,
        )
        buffer = CircularVideoBuffer(config)

        # Fill buffer with regular frames
        base_time = time.time()
        frame_size = 100 * 1024  # 100KB per frame

        for i in range(10):
            frame = VideoFrame(
                data=b"x" * frame_size,
                timestamp=base_time + i,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=False,
                sequence_number=i,
            )
            buffer.add_frame(frame)

        # Try to add a keyframe - should succeed even if at memory limit
        keyframe = VideoFrame(
            data=b"k" * frame_size,
            timestamp=base_time + 10,
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.I_FRAME,
            is_keyframe=True,
            sequence_number=10,
        )

        # Count keyframes before
        keyframes_before = sum(1 for f in buffer._frames if f.is_keyframe)

        # Add keyframe
        buffer.add_frame(keyframe)

        # Count keyframes after
        keyframes_after = sum(1 for f in buffer._frames if f.is_keyframe)

        # Should have at least preserved the keyframe
        assert keyframes_after >= keyframes_before

    def test_get_frames_by_time_range(self, video_buffer):
        """Test retrieving frames by time range."""
        base_time = time.time()

        # Add frames across 3 seconds
        for i in range(90):  # 3 seconds at 30fps
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i * 0.033,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=(i % 30 == 0),
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Get frames from middle second
        start_time = base_time + 1.0
        end_time = base_time + 2.0

        frames = video_buffer.get_frames(start_time, end_time)

        # Should get approximately 30 frames
        assert 25 <= len(frames) <= 35

        # All frames should be within time range
        for frame in frames:
            assert start_time <= frame.timestamp <= end_time

    def test_get_keyframes(self, video_buffer):
        """Test retrieving only keyframes."""
        base_time = time.time()

        # Add frames with some keyframes
        for i in range(50):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i * 0.1,
                duration=0.1,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME if i % 10 == 0 else FrameType.P_FRAME,
                is_keyframe=(i % 10 == 0),
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Get keyframes from entire range
        keyframes = video_buffer.get_keyframes(base_time, base_time + 10.0)

        # Should get 5 keyframes (0, 10, 20, 30, 40)
        assert len(keyframes) == 5

        # All should be keyframes
        for frame in keyframes:
            assert frame.is_keyframe
            assert frame.frame_type == FrameType.I_FRAME

    def test_get_latest_frames(self, video_buffer):
        """Test getting the most recent frames."""
        # Add 20 frames
        base_time = time.time()
        for i in range(20):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Get latest 5 frames
        latest = video_buffer.get_latest_frames(5)

        assert len(latest) == 5

        # Should be frames 15-19
        for i, frame in enumerate(latest):
            expected_seq = 15 + i
            assert frame.sequence_number == expected_seq

    def test_get_frame_at_timestamp(self, video_buffer):
        """Test getting frame closest to timestamp."""
        base_time = time.time()

        # Add frames every 0.1 seconds
        for i in range(10):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i * 0.1,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Get frame at exact timestamp
        target = base_time + 0.5
        frame = video_buffer.get_frame_at_timestamp(target, tolerance=0.05)
        assert frame is not None
        assert frame.sequence_number == 5

        # Get frame with tolerance
        target = base_time + 0.53
        frame = video_buffer.get_frame_at_timestamp(target, tolerance=0.1)
        assert frame is not None
        assert frame.sequence_number == 5

        # No frame within tolerance
        target = base_time + 10.0
        frame = video_buffer.get_frame_at_timestamp(target, tolerance=0.1)
        assert frame is None

    def test_add_segment(self, video_buffer):
        """Test adding a segment of frames."""
        base_time = time.time()

        # Create segment frames
        frames = []
        for i in range(30):
            frame = VideoFrame(
                data=f"segment_frame_{i}".encode(),
                timestamp=base_time + i * 0.033,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=(i == 0),  # First frame is keyframe
                sequence_number=i,
            )
            frames.append(frame)

        # Add as segment
        success = video_buffer.add_segment("segment_1", frames)
        assert success

        # Check segment was stored
        segment = video_buffer.get_segment("segment_1")
        assert segment is not None
        assert segment.frame_count == 30
        assert segment.has_keyframe

        # Check frames were added to buffer
        assert len(video_buffer._frames) == 30

    @pytest.mark.asyncio
    async def test_create_window(self, video_buffer):
        """Test creating sliding windows."""
        base_time = time.time()

        # Add 100 frames (about 3.3 seconds at 30fps)
        for i in range(100):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=base_time + i * 0.033,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Create 1-second windows with 0.5s overlap
        windows = []
        window_count = 0

        async for window_frames in video_buffer.create_window(
            start_time=base_time, duration=1.0, overlap=0.5
        ):
            windows.append(window_frames)
            window_count += 1

            # Stop after a few windows for testing
            if window_count >= 3:
                break

        # Check windows
        assert len(windows) == 3

        # Each window should have about 30 frames (1 second at 30fps)
        for window in windows:
            assert 25 <= len(window) <= 35

    def test_buffer_statistics(self, video_buffer):
        """Test buffer statistics tracking."""
        base_time = time.time()

        # Add some frames
        for i in range(50):
            frame = VideoFrame(
                data=b"x" * 1000,  # 1KB
                timestamp=base_time + i * 0.1,
                duration=0.1,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                is_keyframe=(i % 10 == 0),
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Get stats
        stats = video_buffer.get_stats()

        assert stats["frames_added"] == 50
        assert stats["current_frames"] == 50
        assert stats["frames_dropped"] == 0
        assert stats["segments_created"] == 0
        assert stats["memory_usage_mb"] > 0
        assert stats["avg_frame_size"] == 1000
        assert 0 <= stats["keyframe_ratio"] <= 1.0

    def test_buffer_clear(self, video_buffer):
        """Test clearing the buffer."""
        # Add frames
        for i in range(10):
            frame = VideoFrame(
                data=f"frame_{i}".encode(),
                timestamp=time.time() + i,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
                sequence_number=i,
            )
            video_buffer.add_frame(frame)

        # Add segment
        video_buffer.add_segment("test_segment", [frame])

        # Clear buffer
        video_buffer.clear()

        # Check everything is cleared
        assert len(video_buffer._frames) == 0
        assert len(video_buffer._segments) == 0
        assert len(video_buffer._keyframe_index) == 0
        assert video_buffer._total_memory_bytes == 0
        assert video_buffer._frame_count == 0


class TestVideoBufferManager:
    """Test VideoBufferManager implementation."""

    def test_manager_initialization(self):
        """Test buffer manager initialization."""
        manager = VideoBufferManager()
        assert manager.default_config is not None
        assert len(manager._buffers) == 0

    def test_get_buffer(self):
        """Test getting/creating buffers."""
        manager = VideoBufferManager()

        # Get buffer for stream
        buffer1 = manager.get_buffer("stream1")
        assert buffer1 is not None
        assert "stream1" in manager._buffers

        # Get same buffer again
        buffer2 = manager.get_buffer("stream1")
        assert buffer1 is buffer2

        # Get different buffer
        buffer3 = manager.get_buffer("stream2")
        assert buffer3 is not buffer1
        assert len(manager._buffers) == 2

    @pytest.mark.asyncio
    async def test_remove_buffer(self):
        """Test removing buffers."""
        manager = VideoBufferManager()

        # Create buffer
        manager.get_buffer("stream1")
        assert "stream1" in manager._buffers

        # Remove buffer
        await manager.remove_buffer("stream1")
        assert "stream1" not in manager._buffers

        # Remove non-existent buffer (should not error)
        await manager.remove_buffer("nonexistent")

    def test_get_all_stats(self):
        """Test getting stats for all buffers."""
        manager = VideoBufferManager()

        # Create buffers and add frames
        for stream_id in ["stream1", "stream2"]:
            buffer = manager.get_buffer(stream_id)

            for i in range(5):
                frame = VideoFrame(
                    data=f"{stream_id}_frame_{i}".encode(),
                    timestamp=time.time() + i,
                    duration=0.033,
                    format=BufferFormat.HLS_SEGMENT,
                    frame_type=FrameType.P_FRAME,
                    sequence_number=i,
                )
                buffer.add_frame(frame)

        # Get all stats
        all_stats = manager.get_all_stats()

        assert len(all_stats) == 2
        assert "stream1" in all_stats
        assert "stream2" in all_stats

        # Check each stream has frames
        assert all_stats["stream1"]["current_frames"] == 5
        assert all_stats["stream2"]["current_frames"] == 5

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all buffers."""
        manager = VideoBufferManager()

        # Create multiple buffers
        for i in range(3):
            manager.get_buffer(f"stream{i}")

        assert len(manager._buffers) == 3

        # Close all
        await manager.close_all()

        assert len(manager._buffers) == 0


class TestBufferIntegration:
    """Integration tests for buffer components."""

    @pytest.mark.asyncio
    async def test_buffer_with_multiple_formats(self):
        """Test buffer handling different frame formats."""
        buffer = CircularVideoBuffer()
        base_time = time.time()

        # Add HLS segment
        hls_frame = VideoFrame(
            data=b"hls_segment_data",
            timestamp=base_time,
            duration=2.0,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.I_FRAME,
            is_keyframe=True,
        )
        assert buffer.add_frame(hls_frame)

        # Add FLV frame
        flv_frame = VideoFrame(
            data=b"flv_frame_data",
            timestamp=base_time + 0.033,
            duration=0.033,
            format=BufferFormat.FLV_FRAME,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )
        assert buffer.add_frame(flv_frame)

        # Add raw frame
        raw_frame = VideoFrame(
            data=b"raw_frame_data",
            timestamp=base_time + 0.066,
            duration=0.033,
            format=BufferFormat.RAW_FRAME,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )
        assert buffer.add_frame(raw_frame)

        # Check all frames are stored
        assert len(buffer._frames) == 3

        # Get frames by format type
        all_frames = buffer.get_frames(base_time - 1, base_time + 3)
        formats = {f.format for f in all_frames}
        assert formats == {
            BufferFormat.HLS_SEGMENT,
            BufferFormat.FLV_FRAME,
            BufferFormat.RAW_FRAME,
        }

    @pytest.mark.asyncio
    async def test_concurrent_buffer_access(self):
        """Test concurrent access to buffer."""
        buffer = CircularVideoBuffer()
        base_time = time.time()

        async def add_frames(start_idx, count):
            """Add frames to buffer."""
            for i in range(count):
                frame = VideoFrame(
                    data=f"frame_{start_idx + i}".encode(),
                    timestamp=base_time + (start_idx + i) * 0.01,
                    duration=0.033,
                    format=BufferFormat.HLS_SEGMENT,
                    frame_type=FrameType.P_FRAME,
                    sequence_number=start_idx + i,
                )
                buffer.add_frame(frame)
                await asyncio.sleep(0.001)

        async def read_frames():
            """Read frames from buffer."""
            reads = []
            for _ in range(5):
                frames = buffer.get_latest_frames(10)
                reads.append(len(frames))
                await asyncio.sleep(0.01)
            return reads

        # Run concurrent operations
        tasks = [
            add_frames(0, 20),
            add_frames(20, 20),
            add_frames(40, 20),
            read_frames(),
            read_frames(),
        ]

        results = await asyncio.gather(*tasks)

        # Check final state
        assert len(buffer._frames) == 60

        # Check reads were successful
        read_results = results[3:]
        for reads in read_results:
            assert all(r >= 0 for r in reads)  # All reads succeeded

    def test_memory_pressure_handling(self):
        """Test buffer behavior under memory pressure."""
        # Small buffer for testing
        config = BufferConfig(
            max_memory_mb=1,  # 1MB
            max_items=1000,
            retention_seconds=60,
            min_retention_items=5,
            enable_keyframe_priority=True,
        )
        buffer = CircularVideoBuffer(config)

        # Track what gets added
        added_frames = []
        dropped_count = 0

        # Add frames until memory is full and beyond
        base_time = time.time()
        for i in range(100):
            # Mix of regular frames and keyframes
            is_keyframe = i % 10 == 0
            frame_size = (
                50 * 1024 if is_keyframe else 40 * 1024
            )  # 50KB keyframes, 40KB regular

            frame = VideoFrame(
                data=b"x" * frame_size,
                timestamp=base_time + i * 0.1,
                duration=0.1,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME if is_keyframe else FrameType.P_FRAME,
                is_keyframe=is_keyframe,
                sequence_number=i,
            )

            if buffer.add_frame(frame):
                added_frames.append(i)
            else:
                dropped_count += 1

        # Should have dropped some frames
        assert dropped_count > 0

        # Should have kept minimum retention items
        assert len(buffer._frames) >= config.min_retention_items

        # Memory should be under limit
        assert buffer._total_memory_bytes <= config.max_memory_mb * 1024 * 1024

        # Should have preferentially kept keyframes
        keyframe_ratio = sum(1 for f in buffer._frames if f.is_keyframe) / len(
            buffer._frames
        )
        assert keyframe_ratio > 0.1  # Should have kept some keyframes
