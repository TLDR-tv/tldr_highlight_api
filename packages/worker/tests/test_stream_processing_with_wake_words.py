"""Tests for stream processing with wake word detection integration."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import tempfile

from worker.tasks.stream_processing import process_stream_task, _process_stream_async
from worker.services.ffmpeg_processor import FFmpegConfig, StreamSegment, AudioChunk
from shared.domain.models.stream import Stream, StreamStatus


class TestStreamProcessingWithWakeWords:
    """Test stream processing task with wake word detection."""
    
    @pytest.fixture
    def mock_stream(self):
        """Create mock stream."""
        return Stream(
            id=uuid4(),
            organization_id=uuid4(),
            url="https://example.com/stream.m3u8",
            status=StreamStatus.PENDING,
        )
    
    @pytest.fixture
    def processing_options(self):
        """Create processing options."""
        return {
            "video_segment_duration": 120,
            "audio_segment_duration": 30,
            "audio_overlap": 5,
            "dimension_set_id": str(uuid4()),
            "type_registry_id": str(uuid4()),
            "fusion_strategy": "weighted",
            "enabled_modalities": ["video", "audio"],
            "confidence_threshold": 0.7,
        }
    
    @pytest.mark.asyncio
    async def test_stream_processing_queues_wake_word_detection(
        self, mock_stream, processing_options
    ):
        """Test that stream processing properly queues wake word detection tasks."""
        # Mock database
        with patch("worker.tasks.stream_processing.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.connect = AsyncMock()
            mock_db.return_value.disconnect = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Mock repositories
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                mock_stream_repo.return_value.get_by_id.return_value = mock_stream
                mock_stream_repo.return_value.update = AsyncMock()
                
                # Mock Celery tasks
                with patch("worker.tasks.stream_processing.detect_highlights_task") as mock_highlight_task:
                    with patch("worker.tasks.stream_processing.detect_wake_words_task") as mock_wake_word_task:
                        
                        # Create test segments with audio chunks
                        test_segments = []
                        for i in range(2):
                            segment = StreamSegment(
                                segment_id=uuid4(),
                                path=Path(f"/fake/segment_{i}.mp4"),
                                start_time=i * 120.0,
                                duration=120.0,
                                segment_number=i,
                                size_bytes=1000000,
                                is_complete=True,
                            )
                            
                            # Add audio chunks
                            for j in range(4):  # 4 chunks per 2-minute segment
                                chunk = AudioChunk(
                                    chunk_id=uuid4(),
                                    path=Path(f"/fake/chunk_{i}_{j}.wav"),
                                    start_time=i * 120.0 + j * 25.0,  # With overlap
                                    end_time=i * 120.0 + j * 25.0 + 30.0,
                                    duration=30.0,
                                    chunk_number=i * 4 + j,
                                    size_bytes=50000,
                                    is_complete=True,
                                )
                                segment.audio_chunks.append(chunk)
                            
                            test_segments.append(segment)
                        
                        # Mock FFmpeg processor
                        with patch("worker.tasks.stream_processing.FFmpegProcessor") as mock_ffmpeg:
                            mock_processor = AsyncMock()
                            mock_processor.__aenter__.return_value = mock_processor
                            mock_processor.__aexit__.return_value = None
                            
                            # Make process_stream return our test segments
                            async def mock_process_stream(handler):
                                for segment in test_segments:
                                    yield segment
                            
                            mock_processor.process_stream = mock_process_stream
                            mock_ffmpeg.return_value = mock_processor
                            
                            # Run stream processing
                            result = await _process_stream_async(
                                str(mock_stream.id),
                                processing_options,
                                "test-task-id",
                            )
                            
                            # Verify results
                            assert result["segments_processed"] == 2
                            assert result["status"] == "completed"
                            
                            # Verify highlight detection was queued for each segment
                            assert mock_highlight_task.delay.call_count == 2
                            
                            # Verify wake word detection was queued for each audio chunk
                            assert mock_wake_word_task.delay.call_count == 8  # 2 segments * 4 chunks each
                            
                            # Check wake word task arguments
                            wake_word_calls = mock_wake_word_task.delay.call_args_list
                            for call in wake_word_calls:
                                args = call.kwargs
                                assert "stream_id" in args
                                assert "audio_chunk" in args
                                assert "organization_id" in args
                                
                                audio_chunk = args["audio_chunk"]
                                assert "id" in audio_chunk
                                assert "path" in audio_chunk
                                assert "start_time" in audio_chunk
                                assert "end_time" in audio_chunk
                                assert "video_segment_number" in audio_chunk
    
    @pytest.mark.asyncio
    async def test_stream_processing_handles_no_audio_chunks(
        self, mock_stream, processing_options
    ):
        """Test stream processing when segments have no audio chunks."""
        with patch("worker.tasks.stream_processing.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.connect = AsyncMock()
            mock_db.return_value.disconnect = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                mock_stream_repo.return_value.get_by_id.return_value = mock_stream
                mock_stream_repo.return_value.update = AsyncMock()
                
                with patch("worker.tasks.stream_processing.detect_highlights_task") as mock_highlight_task:
                    with patch("worker.tasks.stream_processing.detect_wake_words_task") as mock_wake_word_task:
                        
                        # Create segment without audio chunks
                        test_segment = StreamSegment(
                            segment_id=uuid4(),
                            path=Path("/fake/segment.mp4"),
                            start_time=0.0,
                            duration=120.0,
                            segment_number=0,
                            size_bytes=1000000,
                            is_complete=True,
                            audio_chunks=[],  # No audio chunks
                        )
                        
                        with patch("worker.tasks.stream_processing.FFmpegProcessor") as mock_ffmpeg:
                            mock_processor = AsyncMock()
                            mock_processor.__aenter__.return_value = mock_processor
                            mock_processor.__aexit__.return_value = None
                            
                            async def mock_process_stream(handler):
                                yield test_segment
                            
                            mock_processor.process_stream = mock_process_stream
                            mock_ffmpeg.return_value = mock_processor
                            
                            result = await _process_stream_async(
                                str(mock_stream.id),
                                processing_options,
                                "test-task-id",
                            )
                            
                            # Should still process video segment
                            assert mock_highlight_task.delay.call_count == 1
                            
                            # But no wake word detection tasks
                            assert mock_wake_word_task.delay.call_count == 0
    
    @pytest.mark.asyncio
    async def test_stream_processing_error_handling(self, mock_stream, processing_options):
        """Test error handling in stream processing."""
        # Test database connection failure
        with patch("worker.tasks.stream_processing.Database") as mock_db:
            mock_db.return_value.connect.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await _process_stream_async(
                    str(mock_stream.id),
                    processing_options,
                    "test-task-id",
                )
        
        # Test stream not found
        with patch("worker.tasks.stream_processing.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.connect = AsyncMock()
            mock_db.return_value.disconnect = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                mock_stream_repo.return_value.get_by_id.return_value = None
                
                with pytest.raises(ValueError, match="Stream .* not found"):
                    await _process_stream_async(
                        str(uuid4()),
                        processing_options,
                        "test-task-id",
                    )
    
    @pytest.mark.asyncio
    async def test_temporary_directory_cleanup(self, mock_stream, processing_options):
        """Test that temporary directories are cleaned up."""
        temp_dirs_created = []
        
        # Track temp directory creation
        original_temp_dir = tempfile.TemporaryDirectory
        
        class TrackingTempDir:
            def __init__(self):
                self.dir = original_temp_dir()
                temp_dirs_created.append(self.dir.name)
            
            def __enter__(self):
                return self.dir.__enter__()
            
            def __exit__(self, *args):
                return self.dir.__exit__(*args)
        
        with patch("worker.tasks.stream_processing.TemporaryDirectory", TrackingTempDir):
            with patch("worker.tasks.stream_processing.Database") as mock_db:
                mock_session = AsyncMock()
                mock_db.return_value.connect = AsyncMock()
                mock_db.return_value.disconnect = AsyncMock()
                mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
                
                with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                    mock_stream_repo.return_value.get_by_id.return_value = mock_stream
                    mock_stream_repo.return_value.update = AsyncMock()
                    
                    with patch("worker.tasks.stream_processing.FFmpegProcessor") as mock_ffmpeg:
                        # Simulate error during processing
                        mock_ffmpeg.side_effect = Exception("Processing failed")
                        
                        try:
                            await _process_stream_async(
                                str(mock_stream.id),
                                processing_options,
                                "test-task-id",
                            )
                        except Exception:
                            pass
        
        # Verify temp directory was created
        assert len(temp_dirs_created) == 1
        
        # Verify it was cleaned up (directory should not exist)
        assert not Path(temp_dirs_created[0]).exists()
    
    def test_celery_task_configuration(self):
        """Test Celery task configuration."""
        # Verify task is properly configured
        assert process_stream_task.name == "process_stream"
        assert process_stream_task.max_retries == 3
        assert process_stream_task.default_retry_delay == 300
        
        # Test task binding
        assert hasattr(process_stream_task, "bind")
        
        # Test base task class
        from worker.tasks.stream_processing import StreamProcessingTask
        assert process_stream_task.__class__.__bases__[0] == StreamProcessingTask