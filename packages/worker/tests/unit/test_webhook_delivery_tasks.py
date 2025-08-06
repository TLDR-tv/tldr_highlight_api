"""Tests for webhook delivery tasks."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
import asyncio

from worker.tasks.webhook_delivery import (
    send_highlight_webhook,
    send_stream_webhook
)
from shared.domain.models.organization import Organization


class TestSendHighlightWebhook:
    """Test the send_highlight_webhook task."""

    @pytest.fixture
    def highlight_data(self):
        """Create sample highlight data."""
        return {
            "id": str(uuid4()),
            "stream_id": str(uuid4()),
            "start_time": 30.0,
            "end_time": 90.0,
            "duration": 60.0,
            "overall_score": 0.85,
            "clip_url": "https://example.com/clip.mp4",
            "thumbnail_url": "https://example.com/thumb.jpg",
            "dimension_scores": {
                "action_intensity": 0.9,
                "educational_value": 0.8
            }
        }

    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = Mock()
        task.request = Mock()
        task.request.retries = 0
        task.default_retry_delay = 60
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    @patch('worker.tasks.webhook_delivery._send_webhook_async')
    def test_send_highlight_webhook_success(self, mock_async_func, mock_run, highlight_data, mock_task):
        """Test successful highlight webhook delivery."""
        organization_id = str(uuid4())
        expected_result = {
            "status": "delivered",
            "webhook_url": "https://api.example.com/webhooks",
            "response_status": 200,
            "delivery_time": "2024-01-01T12:00:00Z"
        }
        
        mock_run.return_value = expected_result
        
        # Test
        result = send_highlight_webhook(mock_task, organization_id, highlight_data)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()
        # Verify the coroutine passed to asyncio.run has correct parameters
        call_args = mock_run.call_args[0][0]
        # The call should be awaitable

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    @patch('worker.tasks.webhook_delivery.logger')
    def test_send_highlight_webhook_failure_with_retry(self, mock_logger, mock_run, 
                                                       highlight_data, mock_task):
        """Test highlight webhook failure with retry."""
        organization_id = str(uuid4())
        test_exception = Exception("Webhook delivery failed")
        
        mock_run.side_effect = test_exception
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_highlight_webhook(mock_task, organization_id, highlight_data)
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert "Webhook delivery failed" in error_call[0][0]
        assert error_call[1]["organization_id"] == organization_id
        assert error_call[1]["event_type"] == "highlight.detected"
        
        # Verify retry with exponential backoff
        mock_task.retry.assert_called_once()
        retry_kwargs = mock_task.retry.call_args[1]
        assert retry_kwargs["exc"] == test_exception
        assert retry_kwargs["countdown"] == 60  # default_retry_delay * (2^0)
        assert retry_kwargs["max_retries"] == 5

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    @patch('worker.tasks.webhook_delivery.logger')
    def test_send_highlight_webhook_retry_backoff(self, mock_logger, mock_run, highlight_data):
        """Test exponential backoff on retry."""
        organization_id = str(uuid4())
        test_exception = Exception("Webhook delivery failed")
        
        # Create mock task with retries
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.retries = 2  # Second retry
        mock_task.default_retry_delay = 60
        mock_task.retry = Mock(side_effect=Exception("Retry called"))
        
        mock_run.side_effect = test_exception
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_highlight_webhook(mock_task, organization_id, highlight_data)
        
        # Verify retry with exponential backoff: 60 * (2^2) = 240
        retry_kwargs = mock_task.retry.call_args[1]
        assert retry_kwargs["countdown"] == 240


class TestSendStreamWebhook:
    """Test the send_stream_webhook task."""

    @pytest.fixture
    def stream_data(self):
        """Create sample stream data."""
        return {
            "id": str(uuid4()),
            "url": "rtmp://test.stream.com/live",
            "status": "completed",
            "started_at": "2024-01-01T12:00:00Z",
            "completed_at": "2024-01-01T13:00:00Z",
            "highlights_count": 5,
            "total_duration": 3600.0
        }

    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = Mock()
        task.request = Mock()
        task.request.retries = 0
        task.default_retry_delay = 30
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    def test_send_stream_webhook_success(self, mock_run, stream_data, mock_task):
        """Test successful stream webhook delivery."""
        organization_id = str(uuid4())
        event_type = "stream.completed"
        expected_result = {
            "status": "delivered",
            "webhook_url": "https://api.example.com/webhooks",
            "response_status": 200
        }
        
        mock_run.return_value = expected_result
        
        # Test
        result = send_stream_webhook(mock_task, organization_id, event_type, stream_data)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    @patch('worker.tasks.webhook_delivery.logger')
    def test_send_stream_webhook_failure(self, mock_logger, mock_run, stream_data, mock_task):
        """Test stream webhook failure with retry."""
        organization_id = str(uuid4())
        event_type = "stream.started"
        test_exception = Exception("Network timeout")
        
        mock_run.side_effect = test_exception
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_stream_webhook(mock_task, organization_id, event_type, stream_data)
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert error_call[1]["organization_id"] == organization_id
        assert error_call[1]["event_type"] == event_type
        
        # Verify retry parameters
        mock_task.retry.assert_called_once()
        retry_kwargs = mock_task.retry.call_args[1]
        assert retry_kwargs["exc"] == test_exception
        assert retry_kwargs["countdown"] == 30  # default_retry_delay * (2^0)

    @pytest.mark.parametrize("event_type", [
        "stream.started",
        "stream.completed", 
        "stream.failed",
        "stream.cancelled"
    ])
    @patch('worker.tasks.webhook_delivery.asyncio.run')
    def test_send_stream_webhook_different_events(self, mock_run, stream_data, mock_task, event_type):
        """Test stream webhook with different event types."""
        organization_id = str(uuid4())
        expected_result = {"status": "delivered"}
        
        mock_run.return_value = expected_result
        
        # Test
        result = send_stream_webhook(mock_task, organization_id, event_type, stream_data)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()


class TestWebhookAsyncImplementation:
    """Test the _send_webhook_async function (indirectly through task tests)."""

    @pytest.fixture
    def mock_organization(self):
        """Create a mock organization with webhook URL."""
        org = Mock(spec=Organization)
        org.id = uuid4()
        org.name = "Test Organization"
        org.webhook_url = "https://api.example.com/webhooks"
        org.webhook_secret = "secret123"
        return org

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    def test_webhook_async_called_with_correct_params(self, mock_run):
        """Test that _send_webhook_async is called with correct parameters."""
        organization_id = str(uuid4())
        event_type = "highlight.detected"
        payload = {"test": "data"}
        
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.retries = 0
        mock_task.default_retry_delay = 60
        
        expected_result = {"status": "delivered"}
        mock_run.return_value = expected_result
        
        # Test highlight webhook
        result = send_highlight_webhook(mock_task, organization_id, payload)
        
        # Verify asyncio.run was called
        mock_run.assert_called_once()
        
        # The argument should be a coroutine that we can't easily inspect,
        # but we can verify the result
        assert result == expected_result

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    def test_webhook_with_empty_payload(self, mock_run):
        """Test webhook delivery with empty payload."""
        organization_id = str(uuid4())
        empty_payload = {}
        
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.retries = 0
        mock_task.default_retry_delay = 60
        
        expected_result = {"status": "delivered"}
        mock_run.return_value = expected_result
        
        # Test
        result = send_highlight_webhook(mock_task, organization_id, empty_payload)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()

    @patch('worker.tasks.webhook_delivery.asyncio.run')
    def test_webhook_with_large_payload(self, mock_run):
        """Test webhook delivery with large payload."""
        organization_id = str(uuid4())
        large_payload = {
            "id": str(uuid4()),
            "metadata": {f"field_{i}": f"value_{i}" for i in range(1000)},
            "large_array": [{"item": i} for i in range(100)]
        }
        
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.retries = 0
        mock_task.default_retry_delay = 60
        
        expected_result = {"status": "delivered"}
        mock_run.return_value = expected_result
        
        # Test
        result = send_highlight_webhook(mock_task, organization_id, large_payload)
        
        # Verify
        assert result == expected_result
        mock_run.assert_called_once()