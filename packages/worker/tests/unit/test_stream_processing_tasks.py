"""Tests for stream processing tasks."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import uuid4, UUID
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from worker.tasks.stream_processing import (
    process_stream_task,
    _process_stream_async,
    StreamProcessingTask
)
from shared.domain.models.stream import Stream, StreamStatus


class TestStreamProcessingTask:
    """Test the StreamProcessingTask base class."""

    @pytest.fixture
    def task_instance(self):
        """Create a StreamProcessingTask instance."""
        task = StreamProcessingTask()
        task.request = Mock()
        task.request.retries = 0
        task.default_retry_delay = 300
        task.retry = Mock()
        return task

    @pytest.mark.asyncio
    async def test_update_stream_status(self, task_instance):
        """Test updating stream status."""
        stream_id = str(uuid4())
        status = StreamStatus.COMPLETED

        mock_stream = Mock(spec=Stream)
        mock_stream.id = UUID(stream_id)
        mock_stream.status = StreamStatus.PROCESSING

        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class:
            
            # Setup mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = mock_stream
            mock_repo.update.return_value = None
            mock_repo_class.return_value = mock_repo

            # Test
            await task_instance._update_stream_status(stream_id, status)

            # Verify
            mock_repo.get.assert_called_once_with(UUID(stream_id))
            assert mock_stream.status == status
            mock_repo.update.assert_called_once_with(mock_stream)

    @pytest.mark.asyncio
    async def test_update_stream_status_stream_not_found(self, task_instance):
        """Test updating stream status when stream not found."""
        stream_id = str(uuid4())
        status = StreamStatus.COMPLETED

        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class:
            
            # Setup mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = None
            mock_repo_class.return_value = mock_repo

            # Test - should not raise exception
            await task_instance._update_stream_status(stream_id, status)

            # Verify
            mock_repo.get.assert_called_once_with(UUID(stream_id))
            mock_repo.update.assert_not_called()

    def test_on_success(self, task_instance):
        """Test on_success callback."""
        stream_id = str(uuid4())
        
        with patch.object(asyncio, 'run') as mock_run:
            task_instance.on_success(
                retval={},
                task_id="test_task_id",
                args=[stream_id],
                kwargs={}
            )
            
            mock_run.assert_called_once()
            # Verify the coroutine passed to asyncio.run
            call_args = mock_run.call_args[0]
            assert len(call_args) == 1

    def test_on_success_with_kwargs(self, task_instance):
        """Test on_success callback with stream_id in kwargs."""
        stream_id = str(uuid4())
        
        with patch.object(asyncio, 'run') as mock_run:
            task_instance.on_success(
                retval={},
                task_id="test_task_id", 
                args=[],
                kwargs={"stream_id": stream_id}
            )
            
            mock_run.assert_called_once()

    def test_on_failure(self, task_instance):
        """Test on_failure callback."""
        stream_id = str(uuid4())
        exc = Exception("Test error")
        einfo = Mock()
        einfo.__str__ = Mock(return_value="Test traceback")
        
        with patch.object(asyncio, 'run') as mock_run, \
             patch('worker.tasks.stream_processing.logger') as mock_logger:
            
            task_instance.on_failure(
                exc=exc,
                task_id="test_task_id",
                args=[stream_id],
                kwargs={},
                einfo=einfo
            )
            
            mock_run.assert_called_once()
            mock_logger.error.assert_called_once()


class TestProcessStreamTask:
    """Test the process_stream_task function."""

    @patch('worker.tasks.stream_processing.asyncio.run')
    def test_process_stream_task_success(self, mock_run):
        """Test successful stream processing task."""
        stream_id = str(uuid4())
        processing_options = {
            "dimension_set_id": "test_dimension_set",
            "type_registry_id": "test_registry",
            "fusion_strategy": "weighted",
            "enabled_modalities": ["video", "audio"],
            "confidence_threshold": 0.8
        }
        
        expected_result = {
            "status": "completed",
            "segments_processed": 5,
            "highlights_found": 2
        }
        
        mock_run.return_value = expected_result
        
        # Create a mock task instance
        mock_task = Mock()
        mock_task.request.id = "test_task_id"
        mock_task.default_retry_delay = 300
        mock_task.request.retries = 0
        
        # Test
        result = process_stream_task(mock_task, stream_id, processing_options)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        assert len(call_args) == 1

    @patch('worker.tasks.stream_processing.asyncio.run')
    def test_process_stream_task_error_with_retry(self, mock_run):
        """Test stream processing task with error and retry."""
        stream_id = str(uuid4())
        processing_options = {}
        
        # Mock exception
        test_exception = Exception("Processing failed")
        mock_run.side_effect = test_exception
        
        # Create a mock task instance
        mock_task = Mock()
        mock_task.request.id = "test_task_id"
        mock_task.default_retry_delay = 300
        mock_task.request.retries = 1
        mock_task.retry.side_effect = Exception("Retry called")
        
        with patch('worker.tasks.stream_processing.logger') as mock_logger:
            with pytest.raises(Exception, match="Retry called"):
                process_stream_task(mock_task, stream_id, processing_options)
            
            mock_logger.error.assert_called_once()
            mock_task.retry.assert_called_once()
            # Verify exponential backoff
            retry_kwargs = mock_task.retry.call_args[1]
            assert retry_kwargs['exc'] == test_exception
            assert retry_kwargs['countdown'] == 600  # 300 * (2^1)


class TestProcessStreamAsync:
    """Test the _process_stream_async function."""

    @pytest.fixture
    def mock_stream(self):
        """Create a mock stream."""
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        stream.url = "rtmp://test.stream.com/live"
        stream.organization_id = uuid4()
        stream.status = StreamStatus.PENDING
        stream.celery_task_id = None
        return stream

    @pytest.fixture
    def processing_options(self):
        """Create processing options."""
        return {
            "video_segment_duration": 120,
            "audio_segment_duration": 30,
            "audio_overlap": 5,
            "use_readrate": False,
            "dimension_set_id": "test_dimension_set",
            "type_registry_id": "test_registry"
        }

    @pytest.mark.asyncio
    async def test_process_stream_async_success(self, mock_stream, processing_options):
        """Test successful async stream processing."""
        stream_id = str(mock_stream.id)
        task_id = "test_task_id"
        
        # Mock segments from FFmpeg processor
        mock_segments = [
            Mock(segment_id=uuid4(), start_time=0.0, duration=120.0, segment_number=1),
            Mock(segment_id=uuid4(), start_time=120.0, duration=120.0, segment_number=2)
        ]
        
        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class, \
             patch('worker.tasks.stream_processing.FFmpegProcessor') as mock_ffmpeg_class, \
             patch('worker.tasks.stream_processing.SegmentRingBuffer') as mock_buffer_class, \
             patch('worker.tasks.stream_processing.PersistentSegmentManager') as mock_persistent, \
             patch('worker.tasks.stream_processing.process_segment_for_highlights') as mock_highlight_task, \
             patch('worker.tasks.stream_processing.TemporaryDirectory') as mock_temp_dir:
            
            # Setup database mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = mock_stream
            mock_repo.update.return_value = None
            mock_repo_class.return_value = mock_repo

            # Setup temporary directory
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            # Setup FFmpeg processor
            mock_ffmpeg = AsyncMock()
            mock_ffmpeg.__aenter__ = AsyncMock(return_value=mock_ffmpeg)
            mock_ffmpeg.__aexit__ = AsyncMock(return_value=None)
            async def mock_process_stream(handler):
                for segment in mock_segments:
                    yield segment
            mock_ffmpeg.process_stream = mock_process_stream
            mock_ffmpeg_class.return_value = mock_ffmpeg
            
            # Setup segment buffer
            mock_buffer = AsyncMock()
            mock_buffer.add_segment = AsyncMock()
            mock_buffer.peek.return_value = []
            mock_buffer_class.return_value = mock_buffer
            
            # Setup persistent segment manager
            mock_persistent_manager = AsyncMock()
            mock_persistent_manager.to_dict.return_value = {"segment_id": "test"}
            mock_persistent_manager.__aenter__ = AsyncMock(return_value=mock_persistent_manager)
            mock_persistent_manager.__aexit__ = AsyncMock(return_value=None)
            mock_persistent.return_value = mock_persistent_manager
            
            # Setup highlight processing
            mock_highlight_task.return_value = []
            
            # Test
            result = await _process_stream_async(stream_id, processing_options, task_id)
            
            # Verify stream was updated
            assert mock_stream.status == StreamStatus.PROCESSING
            assert mock_stream.celery_task_id == task_id
            mock_repo.update.assert_called_with(mock_stream)
            
            # Verify processing components were used
            mock_ffmpeg_class.assert_called_once()
            mock_buffer_class.assert_called_once_with(max_size=10)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "segments_processed" in result
            assert result["segments_processed"] == len(mock_segments)

    @pytest.mark.asyncio
    async def test_process_stream_async_stream_not_found(self, processing_options):
        """Test async processing when stream not found."""
        stream_id = str(uuid4())
        task_id = "test_task_id"
        
        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class:
            
            # Setup database mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = None  # Stream not found
            mock_repo_class.return_value = mock_repo
            
            # Test
            with pytest.raises(ValueError, match=f"Stream {stream_id} not found"):
                await _process_stream_async(stream_id, processing_options, task_id)

    @pytest.mark.asyncio
    async def test_process_stream_async_with_processing_error(self, mock_stream, processing_options):
        """Test async processing with FFmpeg processing error."""
        stream_id = str(mock_stream.id)
        task_id = "test_task_id"
        
        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class, \
             patch('worker.tasks.stream_processing.FFmpegProcessor') as mock_ffmpeg_class, \
             patch('worker.tasks.stream_processing.TemporaryDirectory') as mock_temp_dir:
            
            # Setup database mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = mock_stream
            mock_repo.update.return_value = None
            mock_repo_class.return_value = mock_repo

            # Setup temporary directory
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            # Setup FFmpeg processor to raise error
            mock_ffmpeg = AsyncMock()
            mock_ffmpeg.__aenter__.side_effect = Exception("FFmpeg failed")
            mock_ffmpeg_class.return_value = mock_ffmpeg
            
            # Test
            with pytest.raises(Exception, match="FFmpeg failed"):
                await _process_stream_async(stream_id, processing_options, task_id)
            
            # Verify stream status was still updated to processing
            assert mock_stream.status == StreamStatus.PROCESSING
            assert mock_stream.celery_task_id == task_id

    @pytest.mark.asyncio
    async def test_process_stream_async_highlight_processing_error(self, mock_stream, processing_options):
        """Test async processing with highlight processing error."""
        stream_id = str(mock_stream.id)
        task_id = "test_task_id"
        
        # Mock single segment
        mock_segment = Mock(segment_id=uuid4(), start_time=0.0, duration=120.0, segment_number=1)
        
        with patch('worker.tasks.stream_processing.get_settings') as mock_settings, \
             patch('worker.tasks.stream_processing.Database') as mock_database, \
             patch('worker.tasks.stream_processing.StreamRepository') as mock_repo_class, \
             patch('worker.tasks.stream_processing.FFmpegProcessor') as mock_ffmpeg_class, \
             patch('worker.tasks.stream_processing.SegmentRingBuffer') as mock_buffer_class, \
             patch('worker.tasks.stream_processing.PersistentSegmentManager') as mock_persistent, \
             patch('worker.tasks.stream_processing.process_segment_for_highlights') as mock_highlight_task, \
             patch('worker.tasks.stream_processing.TemporaryDirectory') as mock_temp_dir, \
             patch('worker.tasks.stream_processing.logger') as mock_logger:
            
            # Setup database mocks
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_repo = AsyncMock()
            mock_repo.get.return_value = mock_stream
            mock_repo.update.return_value = None
            mock_repo_class.return_value = mock_repo

            # Setup temporary directory
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            # Setup FFmpeg processor
            mock_ffmpeg = AsyncMock()
            mock_ffmpeg.__aenter__ = AsyncMock(return_value=mock_ffmpeg)
            mock_ffmpeg.__aexit__ = AsyncMock(return_value=None)
            async def mock_process_stream(handler):
                yield mock_segment
            mock_ffmpeg.process_stream = mock_process_stream
            mock_ffmpeg_class.return_value = mock_ffmpeg
            
            # Setup segment buffer
            mock_buffer = AsyncMock()
            mock_buffer.add_segment = AsyncMock()
            mock_buffer.peek.return_value = []
            mock_buffer_class.return_value = mock_buffer
            
            # Setup persistent segment manager
            mock_persistent_manager = AsyncMock()
            mock_persistent_manager.to_dict.return_value = {"segment_id": "test"}
            mock_persistent_manager.__aenter__ = AsyncMock(return_value=mock_persistent_manager)
            mock_persistent_manager.__aexit__ = AsyncMock(return_value=None)
            mock_persistent.return_value = mock_persistent_manager
            
            # Setup highlight processing to fail
            mock_highlight_task.side_effect = Exception("Highlight processing failed")
            
            # Test - should continue processing despite highlight error
            result = await _process_stream_async(stream_id, processing_options, task_id)
            
            # Verify error was logged but processing continued
            mock_logger.error.assert_called()
            assert isinstance(result, dict)
            assert "segments_processed" in result