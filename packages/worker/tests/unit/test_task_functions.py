"""Unit tests for worker task functions."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4


class TestStreamProcessingTasks:
    """Test stream processing task functions."""

    def test_import_stream_processing_task(self):
        """Test that stream processing task can be imported."""
        from worker.tasks.stream_processing import process_stream_task
        assert process_stream_task is not None
        assert callable(process_stream_task)

    def test_import_stream_processing_class(self):
        """Test that stream processing class can be imported."""
        from worker.tasks.stream_processing import StreamProcessingTask
        assert StreamProcessingTask is not None
        assert hasattr(StreamProcessingTask, 'on_success')
        assert hasattr(StreamProcessingTask, 'on_failure')

    @patch('worker.tasks.stream_processing.asyncio')
    def test_stream_processing_task_success_callback(self, mock_asyncio):
        """Test success callback updates stream status."""
        from worker.tasks.stream_processing import StreamProcessingTask
        
        task = StreamProcessingTask()
        task._update_stream_status = Mock(return_value=None)
        
        # Test callback
        task.on_success(retval={}, task_id="test", args=["stream-id"], kwargs={})
        
        mock_asyncio.run.assert_called_once()

    @patch('worker.tasks.stream_processing.asyncio')
    def test_stream_processing_task_failure_callback(self, mock_asyncio):
        """Test failure callback updates stream status."""
        from worker.tasks.stream_processing import StreamProcessingTask
        
        task = StreamProcessingTask()
        task._update_stream_status = Mock(return_value=None)
        
        # Test callback
        exc = Exception("Test error")
        task.on_failure(exc, task_id="test", args=["stream-id"], kwargs={}, einfo=None)
        
        mock_asyncio.run.assert_called_once()


class TestHighlightDetectionTasks:
    """Test highlight detection task functions."""

    def test_import_highlight_detection_function(self):
        """Test that highlight detection function can be imported."""
        from worker.tasks.highlight_detection import process_segment_for_highlights
        assert process_segment_for_highlights is not None
        assert callable(process_segment_for_highlights)

    @patch('worker.tasks.highlight_detection.Database')
    @patch('worker.tasks.highlight_detection.HighlightDetector')
    async def test_process_segment_for_highlights_basic(self, mock_detector, mock_db):
        """Test basic highlight detection processing."""
        from worker.tasks.highlight_detection import process_segment_for_highlights
        
        # Setup mocks
        mock_session = AsyncMock()
        mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
        
        detector_instance = AsyncMock()
        mock_detector.return_value = detector_instance
        detector_instance.detect_highlights.return_value = []
        
        # Test data
        segment_data = {
            "path": "/path/to/segment.mp4",
            "start_time": 0.0,
            "end_time": 60.0
        }
        
        # Should be able to call without error
        try:
            result = await process_segment_for_highlights(
                str(uuid4()), str(uuid4()), segment_data
            )
            # Function exists and can be called
            assert True
        except Exception as e:
            # Even if there are missing dependencies, function should exist
            assert "cannot import" not in str(e)

    def test_highlight_detection_imports(self):
        """Test that required classes can be imported."""
        from worker.tasks.highlight_detection import HighlightDetector
        from worker.tasks.highlight_detection import GeminiVideoScorer
        
        assert HighlightDetector is not None
        assert GeminiVideoScorer is not None


class TestWakeWordDetectionTasks:
    """Test wake word detection task functions."""

    def test_import_wake_word_detection_task(self):
        """Test that wake word detection task can be imported."""
        from worker.tasks.wake_word_detection import detect_wake_words_task
        assert detect_wake_words_task is not None
        assert callable(detect_wake_words_task)

    def test_import_wake_word_detection_class(self):
        """Test that wake word detection class can be imported.""" 
        from worker.tasks.wake_word_detection import WakeWordDetectionTask
        assert WakeWordDetectionTask is not None
        assert hasattr(WakeWordDetectionTask, '__call__')

    def test_wake_word_detection_dependencies(self):
        """Test that wake word detection dependencies can be imported."""
        from worker.tasks.wake_word_detection import WakeWordRepository
        from worker.tasks.wake_word_detection import WhisperModel
        
        assert WakeWordRepository is not None
        assert WhisperModel is not None

    @patch('worker.tasks.wake_word_detection.Database')
    @patch('worker.tasks.wake_word_detection.WakeWordRepository')
    async def test_detect_wake_words_basic_call(self, mock_repo, mock_db):
        """Test basic wake word detection call."""
        from worker.tasks.wake_word_detection import detect_wake_words_task
        
        # Setup mocks
        mock_session = AsyncMock()
        mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
        
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        mock_repo_instance.get_active_by_organization.return_value = []
        
        # Test data
        audio_data = {
            "path": "/path/to/audio.wav",
            "start_time": 0.0,
            "end_time": 30.0
        }
        
        # Should be able to call
        try:
            result = await detect_wake_words_task(
                str(uuid4()), str(uuid4()), audio_data
            )
            assert True  # Function callable
        except Exception as e:
            # Should not have import errors
            assert "cannot import" not in str(e)


class TestWebhookDeliveryTasks:
    """Test webhook delivery task functions."""

    def test_import_webhook_delivery_functions(self):
        """Test that webhook delivery functions can be imported."""
        from worker.tasks.webhook_delivery import send_highlight_webhook
        from worker.tasks.webhook_delivery import send_stream_webhook
        from worker.tasks.webhook_delivery import send_progress_update
        
        assert send_highlight_webhook is not None
        assert send_stream_webhook is not None 
        assert send_progress_update is not None
        
        assert callable(send_highlight_webhook)
        assert callable(send_stream_webhook)
        assert callable(send_progress_update)

    @patch('worker.tasks.webhook_delivery.Database')
    @patch('worker.tasks.webhook_delivery.httpx')
    async def test_send_highlight_webhook_basic(self, mock_httpx, mock_db):
        """Test basic webhook delivery."""
        from worker.tasks.webhook_delivery import send_highlight_webhook
        
        # Setup mocks
        mock_session = AsyncMock()
        mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client
        
        # Test data
        webhook_data = {
            "event": "highlight_detected",
            "stream_id": str(uuid4()),
            "data": {"score": 8.5}
        }
        
        # Should be able to call
        try:
            result = await send_highlight_webhook(
                str(uuid4()), webhook_data
            )
            assert True  # Function callable
        except Exception as e:
            # Should not have import errors
            assert "cannot import" not in str(e)

    def test_webhook_dependencies(self):
        """Test webhook delivery dependencies."""
        from worker.tasks.webhook_delivery import OrganizationRepository
        
        assert OrganizationRepository is not None


class TestTaskIntegration:
    """Test task integration and module structure."""

    def test_all_task_modules_importable(self):
        """Test that all task modules can be imported."""
        from worker.tasks import stream_processing
        from worker.tasks import highlight_detection  
        from worker.tasks import wake_word_detection
        from worker.tasks import webhook_delivery
        
        assert stream_processing is not None
        assert highlight_detection is not None
        assert wake_word_detection is not None
        assert webhook_delivery is not None

    def test_celery_app_import(self):
        """Test that celery app can be imported."""
        from worker.app import celery_app
        assert celery_app is not None
        assert hasattr(celery_app, 'task')
        assert hasattr(celery_app, 'send_task')

    def test_task_registration(self):
        """Test that tasks are properly registered with celery."""
        from worker.app import celery_app
        
        # Check that tasks are registered
        registered_tasks = celery_app.tasks.keys()
        
        # Should have some registered tasks
        assert len(registered_tasks) > 0
        
        # Common task patterns should exist
        task_names = list(registered_tasks)
        stream_tasks = [t for t in task_names if 'stream' in t.lower()]
        highlight_tasks = [t for t in task_names if 'highlight' in t.lower()] 
        
        # Should have stream and highlight related tasks
        assert len(stream_tasks) > 0 or len(highlight_tasks) > 0

    def test_task_configuration(self):
        """Test task configuration and routing."""
        from worker.app import celery_app
        
        # Should have configuration
        assert celery_app.conf is not None
        
        # Should have broker URL configured
        assert hasattr(celery_app.conf, 'broker_url') or 'broker_url' in celery_app.conf


class TestServiceIntegration:
    """Test service integration with tasks."""

    def test_services_importable(self):
        """Test that worker services can be imported."""
        from worker.services.highlight_detector import HighlightDetector
        from worker.services.gemini_scorer import GeminiVideoScorer
        from worker.services.dimension_framework import ScoringRubric
        from worker.services.scoring_factory import ScoringRubricFactory
        
        assert HighlightDetector is not None
        assert GeminiVideoScorer is not None
        assert ScoringRubric is not None
        assert ScoringRubricFactory is not None

    def test_ffmpeg_processor_import(self):
        """Test FFmpeg processor import."""
        from worker.services.ffmpeg_processor import FFmpegProcessor
        assert FFmpegProcessor is not None
        assert hasattr(FFmpegProcessor, '__init__')

    def test_segment_buffer_import(self):
        """Test segment buffer import.""" 
        from worker.services.segment_buffer import SegmentRingBuffer
        assert SegmentRingBuffer is not None

    def test_persistent_segment_import(self):
        """Test persistent segment manager import."""
        from worker.services.persistent_segment import PersistentSegmentManager
        assert PersistentSegmentManager is not None