"""Comprehensive tests for persistent segment manager."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
from pathlib import Path
import tempfile
import shutil

from worker.services.persistent_segment import (
    PersistentAudioChunk,
    PersistentSegment,
    PersistentSegmentManager
)


class TestPersistentAudioChunk:
    """Test PersistentAudioChunk dataclass."""

    def test_persistent_audio_chunk_creation(self):
        """Test creating a persistent audio chunk."""
        chunk_id = uuid4()
        path = Path("/tmp/audio.wav")
        start_time = 10.0
        end_time = 40.0
        
        chunk = PersistentAudioChunk(
            id=chunk_id,
            path=path,
            start_time=start_time,
            end_time=end_time
        )
        
        assert chunk.id == chunk_id
        assert chunk.path == path
        assert chunk.start_time == start_time
        assert chunk.end_time == end_time

    def test_persistent_audio_chunk_to_dict(self):
        """Test converting persistent audio chunk to dictionary."""
        chunk_id = uuid4()
        path = Path("/tmp/test_audio.wav")
        start_time = 5.0
        end_time = 35.0
        
        chunk = PersistentAudioChunk(
            id=chunk_id,
            path=path,
            start_time=start_time,
            end_time=end_time
        )
        
        result = chunk.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == str(chunk_id)
        assert result["path"] == str(path)
        assert result["start_time"] == start_time
        assert result["end_time"] == end_time

    def test_persistent_audio_chunk_to_dict_with_complex_path(self):
        """Test to_dict with complex path structures."""
        chunk_id = uuid4()
        path = Path("/tmp/audio_processing/segments/chunk_001_audio.wav")
        
        chunk = PersistentAudioChunk(
            id=chunk_id,
            path=path,
            start_time=0.0,
            end_time=30.0
        )
        
        result = chunk.to_dict()
        
        assert result["path"] == str(path)
        assert "/" in result["path"]


class TestPersistentSegment:
    """Test PersistentSegment dataclass."""

    @pytest.fixture
    def sample_audio_chunks(self):
        """Create sample audio chunks."""
        return [
            PersistentAudioChunk(
                id=uuid4(),
                path=Path("/tmp/audio1.wav"),
                start_time=0.0,
                end_time=30.0
            ),
            PersistentAudioChunk(
                id=uuid4(),
                path=Path("/tmp/audio2.wav"),
                start_time=25.0,
                end_time=55.0
            )
        ]

    def test_persistent_segment_creation(self, sample_audio_chunks):
        """Test creating a persistent segment."""
        segment_id = uuid4()
        video_path = Path("/tmp/segment.mp4")
        start_time = 0.0
        end_time = 60.0
        
        segment = PersistentSegment(
            id=segment_id,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            audio_chunks=sample_audio_chunks
        )
        
        assert segment.id == segment_id
        assert segment.video_path == video_path
        assert segment.start_time == start_time
        assert segment.end_time == end_time
        assert len(segment.audio_chunks) == 2

    def test_persistent_segment_to_dict(self, sample_audio_chunks):
        """Test converting persistent segment to dictionary."""
        segment_id = uuid4()
        video_path = Path("/tmp/test_segment.mp4")
        start_time = 10.0
        end_time = 70.0
        
        segment = PersistentSegment(
            id=segment_id,
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            audio_chunks=sample_audio_chunks
        )
        
        result = segment.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == str(segment_id)
        assert result["video_path"] == str(video_path)
        assert result["start_time"] == start_time
        assert result["end_time"] == end_time
        assert len(result["audio_chunks"]) == 2
        assert all(isinstance(chunk, dict) for chunk in result["audio_chunks"])

    def test_persistent_segment_to_dict_empty_chunks(self):
        """Test to_dict with no audio chunks."""
        segment_id = uuid4()
        video_path = Path("/tmp/segment_no_audio.mp4")
        
        segment = PersistentSegment(
            id=segment_id,
            video_path=video_path,
            start_time=0.0,
            end_time=120.0,
            audio_chunks=[]
        )
        
        result = segment.to_dict()
        
        assert result["audio_chunks"] == []
        assert len(result["audio_chunks"]) == 0


class TestPersistentSegmentManager:
    """Test PersistentSegmentManager context manager."""

    @pytest.fixture
    def mock_stream_segment(self):
        """Create a mock stream segment."""
        segment = Mock()
        segment.segment_id = uuid4()
        segment.path = Path("/tmp/original_segment.mp4")
        segment.start_time = 0.0
        segment.duration = 120.0
        segment.segment_number = 1
        segment.audio_chunks = []
        return segment

    @pytest.fixture
    def mock_audio_chunk(self):
        """Create a mock audio chunk."""
        chunk = Mock()
        chunk.chunk_id = uuid4()
        chunk.path = Path("/tmp/original_audio.wav")
        chunk.start_time = 0.0
        chunk.end_time = 30.0
        chunk.chunk_number = 1
        return chunk

    @pytest.fixture
    def temp_base_dir(self):
        """Create a temporary base directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_manager_initialization(self, mock_stream_segment, temp_base_dir):
        """Test manager initialization."""
        manager = PersistentSegmentManager(
            segment=mock_stream_segment,
            base_dir=temp_base_dir
        )
        
        assert manager.segment == mock_stream_segment
        assert manager.base_dir == temp_base_dir
        assert manager.persistent_dir == temp_base_dir / "processing"
        assert manager.persistent_segment is None
        assert manager.auto_cleanup is True
        assert manager._cleanup_paths == []

    def test_manager_initialization_with_custom_params(self, mock_stream_segment, temp_base_dir):
        """Test manager initialization with custom parameters."""
        manager = PersistentSegmentManager(
            segment=mock_stream_segment,
            base_dir=temp_base_dir,
            auto_cleanup=False
        )
        
        assert manager.auto_cleanup is False

    @pytest.mark.asyncio
    async def test_context_manager_enter_basic(self, mock_stream_segment, temp_base_dir):
        """Test context manager entry with basic segment."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None  # shutil.copy2 returns None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            persistent_segment = await manager.__aenter__()
            
            assert isinstance(persistent_segment, PersistentSegment)
            assert persistent_segment.id == mock_stream_segment.segment_id
            assert persistent_segment.start_time == mock_stream_segment.start_time
            assert persistent_segment.end_time == mock_stream_segment.start_time + mock_stream_segment.duration
            assert len(persistent_segment.audio_chunks) == 0
            
            # Verify directory was created
            assert manager.persistent_dir.exists()
            
            # Verify shutil.copy2 was called
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_enter_with_audio_chunks(self, mock_stream_segment, mock_audio_chunk, temp_base_dir):
        """Test context manager entry with audio chunks."""
        mock_stream_segment.audio_chunks = [mock_audio_chunk]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            persistent_segment = await manager.__aenter__()
            
            assert len(persistent_segment.audio_chunks) == 1
            assert persistent_segment.audio_chunks[0].id == mock_audio_chunk.chunk_id
            assert persistent_segment.audio_chunks[0].start_time == mock_audio_chunk.start_time
            assert persistent_segment.audio_chunks[0].end_time == mock_audio_chunk.end_time
            
            # Should have called to_thread for video + audio
            assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager_exit_with_cleanup(self, mock_stream_segment, temp_base_dir):
        """Test context manager exit with auto cleanup."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir,
                auto_cleanup=True
            )
            
            # Enter context
            await manager.__aenter__()
            
            # Create some fake cleanup paths
            fake_path1 = temp_base_dir / "fake1.mp4"
            fake_path2 = temp_base_dir / "fake2.wav"
            fake_path1.touch()
            fake_path2.touch()
            manager._cleanup_paths = [fake_path1, fake_path2]
            
            # Exit context
            await manager.__aexit__(None, None, None)
            
            # Files should be deleted
            assert not fake_path1.exists()
            assert not fake_path2.exists()

    @pytest.mark.asyncio
    async def test_context_manager_exit_no_cleanup(self, mock_stream_segment, temp_base_dir):
        """Test context manager exit without auto cleanup."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir,
                auto_cleanup=False
            )
            
            # Enter context
            await manager.__aenter__()
            
            # Create some fake cleanup paths
            fake_path = temp_base_dir / "fake.mp4"
            fake_path.touch()
            manager._cleanup_paths = [fake_path]
            
            # Exit context
            await manager.__aexit__(None, None, None)
            
            # File should still exist
            assert fake_path.exists()

    @pytest.mark.asyncio
    async def test_context_manager_exit_handles_missing_files(self, mock_stream_segment, temp_base_dir):
        """Test context manager exit handles missing files gracefully."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir,
                auto_cleanup=True
            )
            
            # Enter context
            await manager.__aenter__()
            
            # Add non-existent paths to cleanup
            fake_path = temp_base_dir / "nonexistent.mp4"
            manager._cleanup_paths = [fake_path]
            
            # Should not raise exception
            await manager.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_context_manager_with_statement(self, mock_stream_segment, temp_base_dir):
        """Test using manager with async with statement."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            async with manager as persistent_segment:
                assert isinstance(persistent_segment, PersistentSegment)
                assert persistent_segment.id == mock_stream_segment.segment_id
                assert manager.persistent_segment == persistent_segment

    @pytest.mark.asyncio
    async def test_context_manager_file_copy_error(self, mock_stream_segment, temp_base_dir):
        """Test handling of file copy errors."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = OSError("Permission denied")
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            with pytest.raises(OSError, match="Permission denied"):
                await manager.__aenter__()

    @pytest.mark.asyncio
    async def test_persistent_directory_creation(self, mock_stream_segment, temp_base_dir):
        """Test that persistent directory is created if it doesn't exist."""
        # Ensure directory doesn't exist
        processing_dir = temp_base_dir / "processing"
        assert not processing_dir.exists()
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            await manager.__aenter__()
            
            assert processing_dir.exists()
            assert processing_dir.is_dir()

    @pytest.mark.asyncio
    async def test_video_filename_formatting(self, mock_stream_segment, temp_base_dir):
        """Test video filename formatting."""
        mock_stream_segment.segment_number = 42
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            persistent_segment = await manager.__aenter__()
            
            # Check filename format
            expected_filename = "segment_00042.mp4"
            assert persistent_segment.video_path.name == expected_filename

    @pytest.mark.asyncio
    async def test_audio_filename_formatting(self, mock_stream_segment, mock_audio_chunk, temp_base_dir):
        """Test audio filename formatting."""
        mock_audio_chunk.chunk_number = 7
        mock_stream_segment.audio_chunks = [mock_audio_chunk]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            persistent_segment = await manager.__aenter__()
            
            # Check audio filename format
            expected_filename = "audio_00007.wav"
            assert persistent_segment.audio_chunks[0].path.name == expected_filename

    @pytest.mark.asyncio
    async def test_cleanup_paths_tracking(self, mock_stream_segment, mock_audio_chunk, temp_base_dir):
        """Test that cleanup paths are properly tracked."""
        mock_stream_segment.audio_chunks = [mock_audio_chunk]
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            await manager.__aenter__()
            
            # Should track video + audio paths
            assert len(manager._cleanup_paths) == 2
            assert all(isinstance(path, Path) for path in manager._cleanup_paths)

    @pytest.mark.asyncio
    async def test_multiple_audio_chunks(self, mock_stream_segment, temp_base_dir):
        """Test handling multiple audio chunks."""
        audio_chunks = []
        for i in range(3):
            chunk = Mock()
            chunk.chunk_id = uuid4()
            chunk.path = Path(f"/tmp/audio_{i}.wav")
            chunk.start_time = i * 30.0
            chunk.end_time = (i + 1) * 30.0
            chunk.chunk_number = i + 1
            audio_chunks.append(chunk)
        
        mock_stream_segment.audio_chunks = audio_chunks
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            manager = PersistentSegmentManager(
                segment=mock_stream_segment,
                base_dir=temp_base_dir
            )
            
            persistent_segment = await manager.__aenter__()
            
            assert len(persistent_segment.audio_chunks) == 3
            # Should copy video + 3 audio files
            assert mock_to_thread.call_count == 4
            assert len(manager._cleanup_paths) == 4