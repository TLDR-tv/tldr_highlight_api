"""Unit tests for segment buffer module."""

import asyncio
from pathlib import Path
from uuid import uuid4
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import tempfile
import shutil

from worker.services.segment_buffer import (
    BufferStats,
    SegmentRingBuffer,
    SegmentFileManager,
    ProcessingQueue,
)
from worker.services.ffmpeg_processor import StreamSegment


class TestBufferStats:
    """Test BufferStats dataclass."""

    def test_buffer_stats_creation(self):
        """Test creating buffer stats with defaults."""
        stats = BufferStats()
        
        assert stats.total_segments == 0
        assert stats.dropped_segments == 0
        assert stats.current_size == 0
        assert stats.max_size == 0
        assert stats.total_bytes == 0

    def test_buffer_stats_with_values(self):
        """Test creating buffer stats with values."""
        stats = BufferStats(
            total_segments=10,
            dropped_segments=2,
            current_size=5,
            max_size=10,
            total_bytes=1024000
        )
        
        assert stats.total_segments == 10
        assert stats.dropped_segments == 2
        assert stats.current_size == 5
        assert stats.max_size == 10
        assert stats.total_bytes == 1024000


class TestSegmentRingBuffer:
    """Test SegmentRingBuffer class."""

    @pytest.fixture
    def sample_segment(self):
        """Create a sample stream segment."""
        return StreamSegment(
            path=Path("/tmp/test.mp4"),
            segment_number=1,
            segment_id=uuid4(),
            start_time=0.0,
            duration=10.0,
            size_bytes=1024000,
        )

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return SegmentRingBuffer(max_size=3)

    @pytest.mark.asyncio
    async def test_buffer_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.max_size == 3
        assert buffer.size == 0
        assert buffer.is_empty
        assert not buffer.is_full
        assert buffer.stats.max_size == 3

    @pytest.mark.asyncio
    async def test_add_segment(self, buffer, sample_segment):
        """Test adding a segment to buffer."""
        await buffer.add_segment(sample_segment)
        
        assert buffer.size == 1
        assert not buffer.is_empty
        assert not buffer.is_full
        assert buffer.stats.total_segments == 1
        assert buffer.stats.current_size == 1
        assert buffer.stats.total_bytes == 1024000

    @pytest.mark.asyncio
    async def test_add_multiple_segments(self, buffer):
        """Test adding multiple segments."""
        segments = []
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            segments.append(segment)
            await buffer.add_segment(segment)
        
        assert buffer.size == 3
        assert buffer.is_full
        assert buffer.stats.total_segments == 3
        assert buffer.stats.dropped_segments == 0

    @pytest.mark.asyncio
    async def test_buffer_overflow(self, buffer):
        """Test buffer behavior when full."""
        # Fill buffer
        for i in range(4):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            await buffer.add_segment(segment)
        
        # Buffer should still be at max size
        assert buffer.size == 3
        assert buffer.is_full
        assert buffer.stats.total_segments == 4
        assert buffer.stats.dropped_segments == 1

    @pytest.mark.asyncio
    async def test_get_segment(self, buffer, sample_segment):
        """Test getting a segment from buffer."""
        await buffer.add_segment(sample_segment)
        
        retrieved = await buffer.get_segment()
        
        assert retrieved == sample_segment
        assert buffer.size == 0
        assert buffer.is_empty

    @pytest.mark.asyncio
    async def test_get_segment_empty_buffer(self, buffer):
        """Test getting segment from empty buffer with timeout."""
        segment = await buffer.get_segment(timeout=0.1)
        assert segment is None

    @pytest.mark.asyncio
    async def test_get_segment_wait_for_new(self, buffer, sample_segment):
        """Test waiting for new segment."""
        async def add_segment_later():
            await asyncio.sleep(0.1)
            await buffer.add_segment(sample_segment)
        
        # Start adding segment in background
        asyncio.create_task(add_segment_later())
        
        # Get should wait and return the segment
        segment = await buffer.get_segment(timeout=1.0)
        assert segment == sample_segment

    @pytest.mark.asyncio
    async def test_peek_segments(self, buffer):
        """Test peeking at segments without removal."""
        segments = []
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            segments.append(segment)
            await buffer.add_segment(segment)
        
        # Peek at first 2 segments
        peeked = await buffer.peek(2)
        assert len(peeked) == 2
        assert peeked[0].segment_number == 0
        assert peeked[1].segment_number == 1
        
        # Buffer should still be full
        assert buffer.size == 3

    @pytest.mark.asyncio
    async def test_get_all_segments(self, buffer):
        """Test getting all segments at once."""
        segments = []
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            segments.append(segment)
            await buffer.add_segment(segment)
        
        all_segments = await buffer.get_all()
        
        assert len(all_segments) == 3
        assert buffer.size == 0
        assert buffer.is_empty

    @pytest.mark.asyncio
    async def test_clear_buffer(self, buffer):
        """Test clearing the buffer."""
        # Add segments
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            await buffer.add_segment(segment)
        
        await buffer.clear()
        
        assert buffer.size == 0
        assert buffer.is_empty
        assert buffer.stats.dropped_segments == 3

    @pytest.mark.asyncio
    async def test_close_buffer(self, buffer, sample_segment):
        """Test closing the buffer."""
        await buffer.close()
        
        # Should not be able to add segments
        with pytest.raises(RuntimeError, match="Buffer is closed"):
            await buffer.add_segment(sample_segment)
        
        # Get should return None
        segment = await buffer.get_segment(timeout=0.1)
        assert segment is None

    @pytest.mark.asyncio
    async def test_segments_async_iterator(self, buffer):
        """Test async iterator for segments."""
        segments_to_add = []
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            segments_to_add.append(segment)
        
        # Add segments and close buffer
        for segment in segments_to_add:
            await buffer.add_segment(segment)
        await buffer.close()
        
        # Iterate through segments
        retrieved = []
        async for segment in buffer.segments_async():
            retrieved.append(segment)
        
        assert len(retrieved) == 3
        for i, segment in enumerate(retrieved):
            assert segment.segment_number == i


class TestSegmentFileManager:
    """Test SegmentFileManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def file_manager(self, temp_dir):
        """Create a test file manager."""
        return SegmentFileManager(temp_dir, max_segments=3)

    @pytest.fixture
    def sample_segment(self, temp_dir):
        """Create a sample segment with actual file."""
        # Create a test file
        test_file = temp_dir / "source.mp4"
        test_file.write_text("test video data")
        
        return StreamSegment(
            path=test_file,
            segment_number=1,
            segment_id=uuid4(),
            start_time=0.0,
            duration=10.0,
            size_bytes=1024000,
        )

    @pytest.mark.asyncio
    async def test_file_manager_initialization(self, file_manager, temp_dir):
        """Test file manager initialization."""
        assert file_manager.storage_dir == temp_dir
        assert file_manager.max_segments == 3
        assert temp_dir.exists()

    @pytest.mark.asyncio
    async def test_store_segment(self, file_manager, sample_segment):
        """Test storing a segment."""
        stored_path = await file_manager.store_segment(sample_segment)
        
        assert stored_path.exists()
        assert stored_path.name.startswith("stream_")
        assert stored_path.name.endswith("_00001.mp4")
        assert sample_segment.path == stored_path

    @pytest.mark.asyncio
    async def test_store_multiple_segments(self, file_manager, temp_dir):
        """Test storing multiple segments."""
        stored_paths = []
        
        for i in range(3):
            # Create source file
            source = temp_dir / f"source{i}.mp4"
            source.write_text(f"test video {i}")
            
            segment = StreamSegment(
                path=source,
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            
            stored_path = await file_manager.store_segment(segment)
            stored_paths.append(stored_path)
        
        # All paths should exist
        for path in stored_paths:
            assert path.exists()

    @pytest.mark.asyncio
    async def test_max_segments_cleanup(self, file_manager, temp_dir):
        """Test automatic cleanup when max segments exceeded."""
        stored_paths = []
        
        # Store 4 segments (max is 3)
        for i in range(4):
            source = temp_dir / f"source{i}.mp4"
            source.write_text(f"test video {i}")
            
            segment = StreamSegment(
                path=source,
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            
            stored_path = await file_manager.store_segment(segment)
            stored_paths.append(stored_path)
        
        # First segment should be deleted
        assert not stored_paths[0].exists()
        # Last 3 should exist
        for path in stored_paths[1:]:
            assert path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_segment(self, file_manager, sample_segment):
        """Test cleaning up a specific segment."""
        stored_path = await file_manager.store_segment(sample_segment)
        assert stored_path.exists()
        
        await file_manager.cleanup_segment(sample_segment)
        assert not stored_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_all(self, file_manager, temp_dir):
        """Test cleaning up all segments."""
        # Store multiple segments
        for i in range(3):
            source = temp_dir / f"source{i}.mp4"
            source.write_text(f"test video {i}")
            
            segment = StreamSegment(
                path=source,
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            
            await file_manager.store_segment(segment)
        
        # Cleanup all
        await file_manager.cleanup_all()
        
        # Check no segment files remain
        segment_files = list(temp_dir.glob("stream_*.mp4"))
        assert len(segment_files) == 0

    @pytest.mark.asyncio
    async def test_delete_file_error_handling(self, file_manager):
        """Test error handling in file deletion."""
        # Test deleting non-existent file
        fake_path = Path("/tmp/non_existent_file.mp4")
        
        # Should not raise exception
        await file_manager._delete_file(fake_path)


class TestProcessingQueue:
    """Test ProcessingQueue class."""

    @pytest.fixture
    def queue(self):
        """Create a test processing queue."""
        return ProcessingQueue(max_size=5)

    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment."""
        return StreamSegment(
            path=Path("/tmp/test.mp4"),
            segment_number=1,
            segment_id=uuid4(),
            start_time=0.0,
            duration=10.0,
            size_bytes=1024000,
        )

    @pytest.mark.asyncio
    async def test_queue_initialization(self, queue):
        """Test queue initialization."""
        assert queue.max_size == 5
        assert queue.size == 0
        assert not queue.is_full
        assert queue.stats["queued"] == 0
        assert queue.stats["processing"] == 0
        assert queue.stats["completed"] == 0
        assert queue.stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_put_and_get(self, queue, sample_segment):
        """Test putting and getting segments."""
        await queue.put(sample_segment)
        assert queue.size == 1
        
        retrieved = await queue.get()
        assert retrieved == sample_segment
        assert queue.size == 0
        assert sample_segment.segment_id in queue._processing

    @pytest.mark.asyncio
    async def test_mark_completed(self, queue, sample_segment):
        """Test marking segment as completed."""
        await queue.put(sample_segment)
        retrieved = await queue.get()
        
        queue.mark_completed(retrieved.segment_id)
        
        assert retrieved.segment_id not in queue._processing
        assert retrieved.segment_id in queue._completed
        assert queue.stats["completed"] == 1

    @pytest.mark.asyncio
    async def test_mark_failed(self, queue, sample_segment):
        """Test marking segment as failed."""
        await queue.put(sample_segment)
        retrieved = await queue.get()
        
        queue.mark_failed(retrieved.segment_id, "Processing error")
        
        assert retrieved.segment_id not in queue._processing
        assert retrieved.segment_id in queue._failed
        assert queue._failed[retrieved.segment_id] == "Processing error"
        assert queue.stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_queue_full(self, sample_segment):
        """Test queue full behavior."""
        queue = ProcessingQueue(max_size=2)
        
        # Fill queue
        for i in range(2):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            await queue.put(segment)
        
        assert queue.is_full
        assert queue.size == 2

    @pytest.mark.asyncio
    async def test_unlimited_queue(self, sample_segment):
        """Test queue with no size limit."""
        queue = ProcessingQueue(max_size=None)
        
        # Add many segments
        for i in range(100):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            await queue.put(segment)
        
        assert not queue.is_full
        assert queue.size == 100

    @pytest.mark.asyncio
    async def test_queue_stats(self, queue):
        """Test queue statistics tracking."""
        segments = []
        for i in range(3):
            segment = StreamSegment(
                path=Path(f"/tmp/test{i}.mp4"),
                segment_number=i,
                segment_id=uuid4(),
                start_time=i * 10.0,
                duration=10.0,
                size_bytes=1024000,
            )
            segments.append(segment)
            await queue.put(segment)
        
        # Get first segment
        seg1 = await queue.get()
        
        # Get second segment and mark completed
        seg2 = await queue.get()
        queue.mark_completed(seg2.segment_id)
        
        # Get third segment and mark failed
        seg3 = await queue.get()
        queue.mark_failed(seg3.segment_id, "Error")
        
        stats = queue.stats
        assert stats["queued"] == 0
        assert stats["processing"] == 1  # seg1 still processing
        assert stats["completed"] == 1   # seg2 completed
        assert stats["failed"] == 1      # seg3 failed