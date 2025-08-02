"""Ring buffer implementation for video segments."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Deque, Optional
from uuid import UUID

from .ffmpeg_processor import StreamSegment

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics for the segment buffer."""

    total_segments: int = 0
    dropped_segments: int = 0
    current_size: int = 0
    max_size: int = 0
    total_bytes: int = 0


class SegmentRingBuffer:
    """Thread-safe ring buffer for video segments.

    Maintains a fixed-size buffer of video segments, automatically
    removing old segments when the buffer is full.
    """

    def __init__(self, max_size: int = 10):
        """Initialize ring buffer.

        Args:
            max_size: Maximum number of segments to keep in buffer

        """
        self.max_size = max_size
        self._buffer: Deque[StreamSegment] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._new_segment_event = asyncio.Event()
        self._stats = BufferStats(max_size=max_size)
        self._closed = False

    async def add_segment(self, segment: StreamSegment) -> None:
        """Add a segment to the buffer.

        Args:
            segment: Segment to add

        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("Buffer is closed")

            # Check if buffer is full
            if len(self._buffer) >= self.max_size:
                dropped = self._buffer[0]
                logger.warning(
                    f"Buffer full, dropping segment {dropped.segment_number} "
                    f"to make room for {segment.segment_number}"
                )
                self._stats.dropped_segments += 1

            # Add segment
            self._buffer.append(segment)
            self._stats.total_segments += 1
            self._stats.current_size = len(self._buffer)
            self._stats.total_bytes += segment.size_bytes

            # Notify waiters
            self._new_segment_event.set()

            logger.debug(
                f"Added segment {segment.segment_number} to buffer "
                f"(size: {len(self._buffer)}/{self.max_size})"
            )

    async def get_segment(
        self, timeout: Optional[float] = None
    ) -> Optional[StreamSegment]:
        """Get the oldest segment from the buffer.

        Args:
            timeout: Maximum time to wait for a segment

        Returns:
            Oldest segment or None if buffer is empty/closed

        """
        end_time = asyncio.get_event_loop().time() + timeout if timeout else None

        while True:
            async with self._lock:
                if self._buffer:
                    segment = self._buffer.popleft()
                    self._stats.current_size = len(self._buffer)
                    logger.debug(
                        f"Retrieved segment {segment.segment_number} from buffer"
                    )
                    return segment

                if self._closed:
                    return None

                # Clear event for next wait
                self._new_segment_event.clear()

            # Wait for new segment
            try:
                remaining = None
                if end_time:
                    remaining = end_time - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return None

                await asyncio.wait_for(
                    self._new_segment_event.wait(), timeout=remaining
                )
            except asyncio.TimeoutError:
                return None

    async def peek(self, n: int = 1) -> list[StreamSegment]:
        """Peek at segments without removing them.

        Args:
            n: Number of segments to peek at

        Returns:
            List of segments (may be less than n if buffer is smaller)

        """
        async with self._lock:
            return list(self._buffer)[:n]

    async def get_all(self) -> list[StreamSegment]:
        """Get all segments from the buffer.

        Returns:
            All segments in the buffer (oldest first)

        """
        async with self._lock:
            segments = list(self._buffer)
            self._buffer.clear()
            self._stats.current_size = 0
            return segments

    async def clear(self) -> None:
        """Clear all segments from the buffer."""
        async with self._lock:
            dropped_count = len(self._buffer)
            self._buffer.clear()
            self._stats.current_size = 0
            self._stats.dropped_segments += dropped_count
            logger.info(f"Cleared {dropped_count} segments from buffer")

    async def close(self) -> None:
        """Close the buffer, preventing new segments."""
        async with self._lock:
            self._closed = True
            self._new_segment_event.set()  # Wake up any waiters

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.max_size

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    @property
    def stats(self) -> BufferStats:
        """Get buffer statistics."""
        return self._stats

    async def segments_async(self) -> AsyncIterator[StreamSegment]:
        """Async iterator for consuming segments.

        Yields:
            Segments as they become available

        """
        while True:
            segment = await self.get_segment(timeout=1.0)
            if segment is None:
                if self._closed:
                    break
                continue
            yield segment


class SegmentFileManager:
    """Manages segment files on disk with automatic cleanup."""

    def __init__(self, storage_dir: Path, max_segments: int = 50):
        """Initialize file manager.

        Args:
            storage_dir: Directory for storing segments
            max_segments: Maximum segments to keep on disk

        """
        self.storage_dir = storage_dir
        self.max_segments = max_segments
        self._segment_paths: Deque[Path] = deque(maxlen=max_segments)
        self._lock = asyncio.Lock()

        # Ensure directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def store_segment(self, segment: StreamSegment) -> Path:
        """Store a segment file, managing disk space.

        Args:
            segment: Segment to store

        Returns:
            Path where segment was stored

        """
        async with self._lock:
            # Generate storage path
            filename = f"stream_{segment.segment_id}_{segment.segment_number:05d}.mp4"
            storage_path = self.storage_dir / filename

            # Copy segment to storage
            if segment.path != storage_path:
                # Use async file operations
                await asyncio.to_thread(self._copy_file, segment.path, storage_path)

            # Track path
            if len(self._segment_paths) >= self.max_segments:
                # Remove oldest segment
                old_path = self._segment_paths[0]
                await self._delete_file(old_path)
                logger.debug(f"Deleted old segment: {old_path}")

            self._segment_paths.append(storage_path)

            # Update segment path
            segment.path = storage_path

            return storage_path

    async def cleanup_segment(self, segment: StreamSegment) -> None:
        """Clean up a segment file.

        Args:
            segment: Segment to clean up

        """
        async with self._lock:
            if segment.path in self._segment_paths:
                self._segment_paths.remove(segment.path)

            await self._delete_file(segment.path)

    async def cleanup_all(self) -> None:
        """Clean up all managed segment files."""
        async with self._lock:
            for path in list(self._segment_paths):
                await self._delete_file(path)
            self._segment_paths.clear()

    @staticmethod
    def _copy_file(src: Path, dst: Path) -> None:
        """Copy file (sync operation for thread)."""
        import shutil

        shutil.copy2(str(src), str(dst))

    @staticmethod
    async def _delete_file(path: Path) -> None:
        """Delete file safely."""
        try:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")


class ProcessingQueue:
    """Queue for segments awaiting processing with priority support."""

    def __init__(self, max_size: Optional[int] = None):
        """Initialize processing queue.

        Args:
            max_size: Maximum queue size (None for unlimited)

        """
        self.max_size = max_size
        self._queue: asyncio.Queue[StreamSegment] = asyncio.Queue(maxsize=max_size or 0)
        self._processing: set[UUID] = set()
        self._completed: set[UUID] = set()
        self._failed: dict[UUID, str] = {}

    async def put(self, segment: StreamSegment) -> None:
        """Add segment to processing queue.

        Args:
            segment: Segment to process

        """
        await self._queue.put(segment)

    async def get(self) -> StreamSegment:
        """Get next segment for processing.

        Returns:
            Next segment to process

        """
        segment = await self._queue.get()
        self._processing.add(segment.segment_id)
        return segment

    def mark_completed(self, segment_id: UUID) -> None:
        """Mark segment as completed.

        Args:
            segment_id: ID of completed segment

        """
        self._processing.discard(segment_id)
        self._completed.add(segment_id)

    def mark_failed(self, segment_id: UUID, error: str) -> None:
        """Mark segment as failed.

        Args:
            segment_id: ID of failed segment
            error: Error message

        """
        self._processing.discard(segment_id)
        self._failed[segment_id] = error

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.max_size is not None and self.size >= self.max_size

    @property
    def stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queued": self.size,
            "processing": len(self._processing),
            "completed": len(self._completed),
            "failed": len(self._failed),
        }
