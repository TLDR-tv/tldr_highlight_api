"""Unit tests for frame synchronization."""

import pytest
import time
from datetime import datetime, timezone

from src.infrastructure.streaming.frame_synchronizer import (
    FrameSynchronizer,
    SyncConfig,
    TimestampFormat,
    TimestampMapping,
    ClockDriftDetector,
)
from src.infrastructure.streaming.video_buffer import (
    BufferFormat,
    VideoFrame,
    FrameType,
)


@pytest.fixture
def sync_config():
    """Create test synchronization configuration."""
    return SyncConfig(
        max_clock_drift_ms=100.0,
        sync_window_ms=50.0,
        interpolation_method="linear",
        timestamp_history_size=100,
        sync_buffer_size=50,
        enable_drift_correction=True,
        enable_interpolation=True,
        enable_outlier_detection=True,
        min_sync_confidence=0.7,
        outlier_threshold_sigma=2.0,
        drift_check_interval_seconds=60.0,
        sync_update_interval_seconds=60.0,
    )


@pytest.fixture
def synchronizer(sync_config):
    """Create test frame synchronizer."""
    return FrameSynchronizer(sync_config)


@pytest.fixture
def sample_frame():
    """Create sample video frame."""
    return VideoFrame(
        data=b"test_frame",
        timestamp=datetime.now(timezone.utc).timestamp(),
        duration=0.033,
        format=BufferFormat.HLS_SEGMENT,
        frame_type=FrameType.P_FRAME,
        is_keyframe=False,
    )


class TestTimestampMapping:
    """Test timestamp format mapping."""

    def test_basic_conversion(self):
        """Test basic timestamp conversion."""
        mapping = TimestampMapping(
            source_format=TimestampFormat.EPOCH_MILLIS,
            target_format=TimestampFormat.EPOCH_SECONDS,
            scale=0.001,
        )

        # Convert milliseconds to seconds
        assert mapping.convert(1000.0) == 1.0
        assert mapping.convert(1500.0) == 1.5
        assert mapping.convert(2000.0) == 2.0

    def test_conversion_with_offset(self):
        """Test conversion with offset."""
        base_time = 1000.0
        mapping = TimestampMapping(
            source_format=TimestampFormat.RELATIVE_SECONDS,
            target_format=TimestampFormat.EPOCH_SECONDS,
            offset=base_time,
        )

        # Convert relative to absolute
        assert mapping.convert(0.0) == base_time
        assert mapping.convert(10.0) == base_time + 10.0
        assert mapping.convert(60.0) == base_time + 60.0

    def test_conversion_with_drift(self):
        """Test conversion with drift correction."""
        mapping = TimestampMapping(
            source_format=TimestampFormat.EPOCH_SECONDS,
            target_format=TimestampFormat.EPOCH_SECONDS,
            drift_rate=0.001,  # 1ms drift per second
        )

        current_time = mapping.last_update + 10.0  # 10 seconds elapsed

        # Should apply drift correction
        result = mapping.convert(100.0, current_time)
        expected = 100.0 + (0.001 * 10.0)  # 10ms drift
        assert abs(result - expected) < 0.0001


class TestClockDriftDetector:
    """Test clock drift detection."""

    def test_drift_detection(self):
        """Test detecting clock drift between streams."""
        detector = ClockDriftDetector(window_size=50)

        # Add timestamp pairs with consistent drift
        base_ref = 1000.0
        base_stream = 1000.0
        drift_rate = 0.002  # 2ms per second

        for i in range(20):
            ref_time = base_ref + i
            stream_time = base_stream + i + (drift_rate * i)
            detector.add_timestamp_pair(ref_time, stream_time)

        # Calculate drift
        calculated_drift = detector.calculate_drift()

        # Should detect the drift rate
        assert abs(calculated_drift - drift_rate) < 0.0005

    def test_drift_correction(self):
        """Test applying drift correction."""
        detector = ClockDriftDetector()
        detector.drift_rate = 0.001  # 1ms per second

        # Correct timestamp
        timestamp = 1000.0
        elapsed = 10.0
        corrected = detector.correct_timestamp(timestamp, elapsed)

        # Should subtract accumulated drift
        assert corrected == timestamp - (0.001 * 10.0)

    def test_insufficient_data(self):
        """Test drift calculation with insufficient data."""
        detector = ClockDriftDetector()

        # Add only a few pairs
        for i in range(5):
            detector.add_timestamp_pair(i, i)

        # Should return 0 drift
        assert detector.calculate_drift() == 0.0


class TestFrameSynchronizer:
    """Test FrameSynchronizer implementation."""

    def test_stream_registration(self, synchronizer):
        """Test registering streams."""
        # Register first stream (should be reference)
        synchronizer.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
            is_reference=True,
        )

        assert "stream1" in synchronizer._streams
        assert synchronizer._reference_stream == "stream1"

        # Register second stream
        synchronizer.register_stream(
            "stream2",
            BufferFormat.FLV_FRAME,
            TimestampFormat.RTMP_TIMESTAMP,
            base_timestamp=1000.0,
        )

        assert "stream2" in synchronizer._streams
        assert len(synchronizer._streams) == 2

    def test_timestamp_normalization(self, synchronizer):
        """Test normalizing timestamps to common format."""
        # Register streams with different formats
        synchronizer.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        synchronizer.register_stream(
            "stream2",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_MILLIS,
        )

        # Normalize timestamps
        ts1 = synchronizer.normalize_timestamp("stream1", 1000.0)
        ts2 = synchronizer.normalize_timestamp(
            "stream2", 1000000.0
        )  # 1000 seconds in millis

        # Should normalize to same value
        assert abs(ts1 - ts2) < 0.001

    def test_unknown_stream_normalization(self, synchronizer):
        """Test normalizing timestamp for unknown stream."""
        # Should return original timestamp
        ts = synchronizer.normalize_timestamp("unknown", 1234.5)
        assert ts == 1234.5

    @pytest.mark.asyncio
    async def test_frame_synchronization(self, synchronizer):
        """Test synchronizing frames across streams."""
        # Register streams
        synchronizer.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
            is_reference=True,
        )

        synchronizer.register_stream(
            "stream2",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Create frames
        base_time = time.time()
        frame1 = VideoFrame(
            data=b"frame1",
            timestamp=base_time,
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )

        frame2 = VideoFrame(
            data=b"frame2",
            timestamp=base_time + 0.01,  # Slightly offset
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )

        # Synchronize frames
        sync_frame1, confidence1 = await synchronizer.synchronize_frame(
            "stream1", frame1
        )
        sync_frame2, confidence2 = await synchronizer.synchronize_frame(
            "stream2", frame2
        )

        # Frames should be synchronized
        assert sync_frame1.timestamp == base_time
        assert sync_frame2.timestamp == base_time + 0.01

        # Should have sync buffers
        assert len(synchronizer._sync_buffer["stream1"]) == 1
        assert len(synchronizer._sync_buffer["stream2"]) == 1

    @pytest.mark.asyncio
    async def test_sync_point_creation(self, synchronizer):
        """Test creating synchronization points."""
        # Register streams
        for i in range(3):
            synchronizer.register_stream(
                f"stream{i}",
                BufferFormat.HLS_SEGMENT,
                TimestampFormat.EPOCH_SECONDS,
                is_reference=(i == 0),
            )

        # Add synchronized frames (all at same time)
        base_time = time.time()
        for i in range(3):
            frame = VideoFrame(
                data=f"frame{i}".encode(),
                timestamp=base_time,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME,
                is_keyframe=True,
            )

            await synchronizer.synchronize_frame(f"stream{i}", frame)

        # Should have created sync point
        assert len(synchronizer._sync_points) >= 1

        # Check sync point
        sync_point = synchronizer._sync_points[-1]
        assert sync_point.is_keyframe
        assert len(sync_point.stream_timestamps) == 3
        assert sync_point.confidence >= synchronizer.config.min_sync_confidence

    def test_outlier_detection(self, synchronizer):
        """Test detecting timestamp outliers."""
        # Register stream
        synchronizer.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Add regular timestamps
        base_time = 1000.0
        for i in range(20):
            _ = synchronizer.normalize_timestamp("stream1", base_time + i * 0.1)

        # Add outlier
        _ = synchronizer.normalize_timestamp("stream1", base_time + 100.0)

        # Add more regular timestamps
        for i in range(20, 25):
            _ = synchronizer.normalize_timestamp("stream1", base_time + i * 0.1)

        # Detect outliers
        outliers = synchronizer.detect_outliers("stream1")

        # Should detect the outlier
        assert len(outliers) > 0
        assert any(abs(o - (base_time + 100.0)) < 0.001 for o in outliers)

    def test_timestamp_interpolation(self, synchronizer):
        """Test interpolating missing timestamps."""
        # Register stream
        synchronizer.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Add timestamps with gap
        base_time = 1000.0
        timestamps = [base_time + i for i in [0, 1, 2, 3, 5, 6, 7, 8]]  # Missing 4

        for ts in timestamps:
            synchronizer.normalize_timestamp("stream1", ts)

        # Interpolate missing timestamp
        target = base_time + 4.0
        interpolated = synchronizer.interpolate_timestamp("stream1", target)

        # Should be close to expected value
        assert interpolated is not None
        assert abs(interpolated - target) < 0.1

    def test_interpolation_disabled(self):
        """Test interpolation when disabled."""
        config = SyncConfig(enable_interpolation=False)
        sync = FrameSynchronizer(config)

        sync.register_stream(
            "stream1",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Add some timestamps
        for i in range(10):
            sync.normalize_timestamp("stream1", 1000.0 + i)

        # Try to interpolate
        result = sync.interpolate_timestamp("stream1", 1004.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_align_frames(self, synchronizer):
        """Test aligning frames from multiple streams."""
        # Register streams
        for i in range(3):
            synchronizer.register_stream(
                f"stream{i}",
                BufferFormat.HLS_SEGMENT,
                TimestampFormat.EPOCH_SECONDS,
            )

        # Add frames at slightly different times
        base_time = time.time()
        offsets = [0.0, 0.02, 0.04]  # Within 50ms window

        for i in range(3):
            frame = VideoFrame(
                data=f"frame{i}".encode(),
                timestamp=base_time + offsets[i],
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.P_FRAME,
            )

            await synchronizer.synchronize_frame(f"stream{i}", frame)

        # Get aligned frames
        aligned = await synchronizer.align_frames(base_time + 0.02, tolerance=0.05)

        # Should get frames from all streams
        assert len(aligned) == 3
        assert all(aligned[f"stream{i}"] is not None for i in range(3))

    @pytest.mark.asyncio
    async def test_align_frames_with_missing(self, synchronizer):
        """Test frame alignment with missing frames."""
        # Register streams
        synchronizer.register_stream(
            "stream1", BufferFormat.HLS_SEGMENT, TimestampFormat.EPOCH_SECONDS
        )
        synchronizer.register_stream(
            "stream2", BufferFormat.HLS_SEGMENT, TimestampFormat.EPOCH_SECONDS
        )

        # Add frame only to stream1
        base_time = time.time()
        frame = VideoFrame(
            data=b"frame1",
            timestamp=base_time,
            duration=0.033,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.P_FRAME,
        )

        await synchronizer.synchronize_frame("stream1", frame)

        # Try to align - stream2 has no frames
        aligned = await synchronizer.align_frames(base_time, tolerance=0.1)

        assert aligned["stream1"] is not None
        assert aligned["stream2"] is None

    def test_sync_statistics(self, synchronizer):
        """Test synchronization statistics."""
        # Register streams
        synchronizer.register_stream(
            "stream1", BufferFormat.HLS_SEGMENT, TimestampFormat.EPOCH_SECONDS
        )
        synchronizer.register_stream(
            "stream2", BufferFormat.HLS_SEGMENT, TimestampFormat.EPOCH_MILLIS
        )

        # Get initial stats
        stats = synchronizer.get_sync_stats()

        assert stats["registered_streams"] == 2
        assert stats["reference_stream"] == "stream1"
        assert stats["frames_synchronized"] == 0
        assert stats["drift_corrections"] == 0
        assert stats["outliers_detected"] == 0
        assert stats["interpolations"] == 0
        assert stats["sync_points_created"] == 0

        # Check stream details
        assert "stream1" in stats["stream_details"]
        assert "stream2" in stats["stream_details"]
        assert stats["stream_details"]["stream1"]["format"] == BufferFormat.HLS_SEGMENT
        assert (
            stats["stream_details"]["stream2"]["timestamp_format"]
            == TimestampFormat.EPOCH_MILLIS
        )

    @pytest.mark.asyncio
    async def test_multi_format_sync(self, synchronizer):
        """Test synchronizing different stream formats."""
        # Register different format streams
        synchronizer.register_stream(
            "hls_stream",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.HLS_TIMESTAMP,
            base_timestamp=1000.0,
        )

        synchronizer.register_stream(
            "rtmp_stream",
            BufferFormat.FLV_FRAME,
            TimestampFormat.RTMP_TIMESTAMP,
            base_timestamp=1000.0,
        )

        synchronizer.register_stream(
            "raw_stream",
            BufferFormat.RAW_FRAME,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Create frames with different timestamp formats
        hls_frame = VideoFrame(
            data=b"hls",
            timestamp=100.0,  # Relative to base
            duration=2.0,
            format=BufferFormat.HLS_SEGMENT,
            frame_type=FrameType.I_FRAME,
            is_keyframe=True,
        )

        rtmp_frame = VideoFrame(
            data=b"rtmp",
            timestamp=100000.0,  # Milliseconds relative to base
            duration=0.033,
            format=BufferFormat.FLV_FRAME,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )

        raw_frame = VideoFrame(
            data=b"raw",
            timestamp=1100.0,  # Absolute epoch seconds
            duration=0.033,
            format=BufferFormat.RAW_FRAME,
            frame_type=FrameType.P_FRAME,
            is_keyframe=False,
        )

        # Synchronize all frames
        sync_hls, _ = await synchronizer.synchronize_frame("hls_stream", hls_frame)
        sync_rtmp, _ = await synchronizer.synchronize_frame("rtmp_stream", rtmp_frame)
        sync_raw, _ = await synchronizer.synchronize_frame("raw_stream", raw_frame)

        # All should be normalized to same time range
        assert abs(sync_hls.timestamp - 1100.0) < 1.0
        assert abs(sync_rtmp.timestamp - 1100.0) < 1.0
        assert abs(sync_raw.timestamp - 1100.0) < 1.0

    @pytest.mark.asyncio
    async def test_cleanup(self, synchronizer):
        """Test synchronizer cleanup."""
        # Register stream
        synchronizer.register_stream(
            "stream1", BufferFormat.HLS_SEGMENT, TimestampFormat.EPOCH_SECONDS
        )

        # Add some data
        for i in range(10):
            synchronizer.normalize_timestamp("stream1", 1000.0 + i)

        # Close synchronizer
        await synchronizer.close()

        # Background tasks should be cancelled
        assert (
            synchronizer._drift_check_task is None
            or synchronizer._drift_check_task.done()
        )
        assert (
            synchronizer._sync_update_task is None
            or synchronizer._sync_update_task.done()
        )


class TestSyncIntegration:
    """Integration tests for synchronization."""

    @pytest.mark.asyncio
    async def test_multi_stream_synchronization(self):
        """Test synchronizing multiple streams with drift."""
        sync = FrameSynchronizer()

        # Register three streams with different characteristics
        sync.register_stream(
            "fast_stream",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
            is_reference=True,
        )

        sync.register_stream(
            "slow_stream",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        sync.register_stream(
            "drifting_stream",
            BufferFormat.HLS_SEGMENT,
            TimestampFormat.EPOCH_SECONDS,
        )

        # Simulate streams with different timing
        base_time = time.time()
        duration = 10  # 10 seconds of data

        # Fast stream: normal timing
        # Slow stream: 50ms behind
        # Drifting stream: starts aligned but drifts 1ms per second

        frames_added = 0
        for i in range(duration * 30):  # 30 fps
            t = i / 30.0

            # Fast stream frame
            fast_frame = VideoFrame(
                data=f"fast_{i}".encode(),
                timestamp=base_time + t,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME if i % 30 == 0 else FrameType.P_FRAME,
                is_keyframe=(i % 30 == 0),
            )
            await sync.synchronize_frame("fast_stream", fast_frame)

            # Slow stream frame (50ms behind)
            slow_frame = VideoFrame(
                data=f"slow_{i}".encode(),
                timestamp=base_time + t - 0.05,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME if i % 30 == 0 else FrameType.P_FRAME,
                is_keyframe=(i % 30 == 0),
            )
            await sync.synchronize_frame("slow_stream", slow_frame)

            # Drifting stream frame
            drift = t * 0.001  # 1ms per second
            drift_frame = VideoFrame(
                data=f"drift_{i}".encode(),
                timestamp=base_time + t + drift,
                duration=0.033,
                format=BufferFormat.HLS_SEGMENT,
                frame_type=FrameType.I_FRAME if i % 30 == 0 else FrameType.P_FRAME,
                is_keyframe=(i % 30 == 0),
            )
            await sync.synchronize_frame("drifting_stream", drift_frame)

            frames_added += 3

        # Check synchronization
        stats = sync.get_sync_stats()
        assert stats["frames_synchronized"] == frames_added

        # Should have created sync points
        assert stats["sync_points_created"] > 0

        # Try to get aligned frames at different points
        for t in [2.0, 5.0, 8.0]:
            aligned = await sync.align_frames(base_time + t, tolerance=0.1)

            # Should get frames from all streams
            assert all(
                aligned[stream] is not None
                for stream in ["fast_stream", "slow_stream", "drifting_stream"]
            )

            # Check timestamps are close
            timestamps = [
                aligned[stream].timestamp for stream in aligned if aligned[stream]
            ]
            if timestamps:
                max_diff = max(timestamps) - min(timestamps)
                assert max_diff < 0.2  # Within 200ms
