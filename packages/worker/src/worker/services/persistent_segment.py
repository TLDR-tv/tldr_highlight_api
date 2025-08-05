"""Context manager for handling persistent segment files during processing."""

import asyncio
import shutil
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional
from uuid import UUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ffmpeg_processor import StreamSegment, AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class PersistentAudioChunk:
    """Audio chunk with persistent file path."""
    
    id: UUID
    path: Path
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for task serialization."""
        return {
            "id": str(self.id),
            "path": str(self.path),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass 
class PersistentSegment:
    """Video segment with persistent file paths."""
    
    id: UUID
    video_path: Path
    start_time: float
    end_time: float
    audio_chunks: List[PersistentAudioChunk]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for task serialization."""
        return {
            "id": str(self.id),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "video_path": str(self.video_path),
            "audio_chunks": [chunk.to_dict() for chunk in self.audio_chunks],
        }


class PersistentSegmentManager:
    """Context manager for persistent segment file lifecycle.
    
    Handles copying segment files to persistent location for async processing,
    and ensures cleanup after processing is complete.
    """
    
    def __init__(self, segment: "StreamSegment", base_dir: Path, auto_cleanup: bool = True):
        """Initialize persistent segment manager.
        
        Args:
            segment: Original segment from FFmpeg processor
            base_dir: Base directory for persistent files
            auto_cleanup: Whether to auto-cleanup files on context exit
        """
        self.segment = segment
        self.base_dir = base_dir
        self.persistent_dir = base_dir / "processing"
        self.persistent_segment: Optional[PersistentSegment] = None
        self.auto_cleanup = auto_cleanup
        self._cleanup_paths: List[Path] = []
    
    async def __aenter__(self) -> PersistentSegment:
        """Create persistent copies of segment files."""
        # Create persistent directory
        self.persistent_dir.mkdir(exist_ok=True)
        
        # Copy video segment
        video_filename = f"segment_{self.segment.segment_number:05d}.mp4"
        persistent_video_path = self.persistent_dir / video_filename
        
        await asyncio.to_thread(shutil.copy2, self.segment.path, persistent_video_path)
        self._cleanup_paths.append(persistent_video_path)
        
        logger.debug(f"Copied video segment to persistent location: {persistent_video_path}")
        
        # Copy audio chunks
        persistent_audio_chunks = []
        for chunk in self.segment.audio_chunks:
            audio_filename = f"audio_{chunk.chunk_number:05d}.wav"
            persistent_audio_path = self.persistent_dir / audio_filename
            
            await asyncio.to_thread(shutil.copy2, chunk.path, persistent_audio_path)
            self._cleanup_paths.append(persistent_audio_path)
            
            persistent_chunk = PersistentAudioChunk(
                id=chunk.chunk_id,
                path=persistent_audio_path,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
            persistent_audio_chunks.append(persistent_chunk)
            
            logger.debug(f"Copied audio chunk to persistent location: {persistent_audio_path}")
        
        # Create persistent segment
        self.persistent_segment = PersistentSegment(
            id=self.segment.segment_id,
            video_path=persistent_video_path,
            start_time=self.segment.start_time,
            end_time=self.segment.start_time + self.segment.duration,
            audio_chunks=persistent_audio_chunks,
        )
        
        return self.persistent_segment
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up persistent files."""
        cleanup_errors = []
        
        for path in self._cleanup_paths:
            try:
                if path.exists():
                    await asyncio.to_thread(path.unlink)
                    logger.debug(f"Cleaned up persistent file: {path}")
            except Exception as e:
                cleanup_errors.append(f"Failed to delete {path}: {e}")
                logger.warning(f"Failed to cleanup {path}: {e}")
        
        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
        else:
            logger.debug(f"Successfully cleaned up {len(self._cleanup_paths)} persistent files")
        
        self._cleanup_paths.clear()


@asynccontextmanager
async def persistent_segment_files(segment: "StreamSegment", base_dir: Path) -> AsyncIterator[PersistentSegment]:
    """Async context manager for persistent segment files.
    
    Usage:
        async with persistent_segment_files(segment, output_dir) as persistent:
            # Use persistent.video_path and persistent.audio_chunks
            task.delay(segment_data=persistent.to_dict())
            # Files automatically cleaned up on exit
    """
    manager = PersistentSegmentManager(segment, base_dir)
    async with manager as persistent_segment:
        yield persistent_segment


class ProcessingFileRegistry:
    """Registry to track files being processed and coordinate cleanup.
    
    This helps prevent race conditions where files are cleaned up
    while still being processed by async tasks.
    """
    
    def __init__(self):
        self._processing_files: Dict[str, int] = {}  # file_path -> ref_count
        self._lock = asyncio.Lock()
    
    async def register_file(self, file_path: Path) -> None:
        """Register a file as being processed."""
        path_str = str(file_path)
        async with self._lock:
            self._processing_files[path_str] = self._processing_files.get(path_str, 0) + 1
            logger.debug(f"Registered file for processing: {path_str} (refs: {self._processing_files[path_str]})")
    
    async def unregister_file(self, file_path: Path) -> bool:
        """Unregister a file and return True if it can be safely deleted."""
        path_str = str(file_path)
        async with self._lock:
            if path_str in self._processing_files:
                self._processing_files[path_str] -= 1
                refs = self._processing_files[path_str]
                
                if refs <= 0:
                    del self._processing_files[path_str]
                    logger.debug(f"File can be safely deleted: {path_str}")
                    return True
                else:
                    logger.debug(f"File still being processed: {path_str} (refs: {refs})")
                    return False
            else:
                logger.debug(f"File not found in registry: {path_str}")
                return True
    
    async def get_processing_count(self) -> int:
        """Get the number of files currently being processed."""
        async with self._lock:
            return len(self._processing_files)


# Global registry for coordinating file cleanup
processing_registry = ProcessingFileRegistry()