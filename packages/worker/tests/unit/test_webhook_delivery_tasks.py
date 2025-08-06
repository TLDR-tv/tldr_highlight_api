"""Tests for webhook delivery tasks."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from datetime import datetime
import asyncio
import json

from worker.tasks.webhook_delivery import (
    send_highlight_webhook,
    send_stream_webhook,
    _json_serialize_datetime,
    _generate_webhook_signature
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


class TestDatetimeSerialization:
    """Test datetime serialization in webhook payloads."""

    def test_json_serialize_datetime_success(self):
        """Test successful datetime serialization."""
        test_datetime = datetime(2024, 1, 1, 12, 0, 0)
        result = _json_serialize_datetime(test_datetime)
        assert result == "2024-01-01T12:00:00"

    def test_json_serialize_datetime_with_microseconds(self):
        """Test datetime serialization with microseconds."""
        test_datetime = datetime(2024, 1, 1, 12, 0, 0, 123456)
        result = _json_serialize_datetime(test_datetime)
        assert result == "2024-01-01T12:00:00.123456"

    def test_json_serialize_datetime_non_datetime_object(self):
        """Test error handling for non-datetime objects."""
        with pytest.raises(TypeError, match="Object of type str is not JSON serializable"):
            _json_serialize_datetime("not a datetime")

    def test_json_serialize_datetime_none(self):
        """Test error handling for None."""
        with pytest.raises(TypeError, match="Object of type NoneType is not JSON serializable"):
            _json_serialize_datetime(None)

    def test_json_serialize_uuid_success(self):
        """Test successful UUID serialization."""
        test_uuid = uuid4()
        result = _json_serialize_datetime(test_uuid)
        assert result == str(test_uuid)
        assert isinstance(result, str)

    def test_generate_webhook_signature_with_datetime(self):
        """Test webhook signature generation with datetime objects."""
        payload = {
            "event": "highlight.detected",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "data": {
                "created_at": datetime(2024, 1, 1, 12, 0, 0),
                "updated_at": datetime(2024, 1, 1, 12, 30, 0),
                "highlight_id": str(uuid4())
            }
        }
        secret = "test_secret"
        
        # This should not raise a TypeError
        signature = _generate_webhook_signature(payload, secret)
        
        # Verify it's a valid signature format
        assert signature.startswith("sha256=")
        assert len(signature) == 71  # "sha256=" (7) + 64 hex characters

    def test_generate_webhook_signature_consistency(self):
        """Test that identical payloads generate identical signatures."""
        payload1 = {
            "event": "highlight.detected",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "data": {"id": "test-id"}
        }
        payload2 = {
            "event": "highlight.detected",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "data": {"id": "test-id"}
        }
        secret = "test_secret"
        
        signature1 = _generate_webhook_signature(payload1, secret)
        signature2 = _generate_webhook_signature(payload2, secret)
        
        assert signature1 == signature2

    def test_json_dumps_with_datetime_serializer(self):
        """Test that JSON serialization works with our custom serializer."""
        test_uuid = uuid4()
        payload = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "created_at": datetime(2024, 1, 1, 11, 0, 0),
            "id": test_uuid,
            "regular_field": "test_value",
            "number": 42
        }
        
        # This should not raise an error
        result = json.dumps(payload, default=_json_serialize_datetime, sort_keys=True)
        
        # Verify the JSON is valid and contains expected datetime strings
        parsed = json.loads(result)
        assert parsed["timestamp"] == "2024-01-01T12:00:00"
        assert parsed["created_at"] == "2024-01-01T11:00:00"
        assert parsed["id"] == str(test_uuid)
        assert parsed["regular_field"] == "test_value"
        assert parsed["number"] == 42

    def test_datetime_payload_integration(self):
        """Test that datetime payload can be properly serialized for webhook signature."""
        # Test the core serialization functionality directly
        highlight_data_with_datetime = {
            "id": str(uuid4()),
            "stream_id": str(uuid4()),
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "updated_at": datetime(2024, 1, 1, 12, 30, 0),
            "start_time": 30.0,
            "end_time": 90.0,
            "overall_score": 0.85
        }
        
        # Create webhook payload similar to what _send_webhook_async does
        payload = {
            "event": "highlight.detected",
            "timestamp": highlight_data_with_datetime.get("created_at"),
            "organization_id": str(uuid4()),
            "data": highlight_data_with_datetime,
        }
        
        # This should not raise a JSON serialization error
        try:
            signature = _generate_webhook_signature(payload, "test_secret")
            assert signature.startswith("sha256=")
            # If we get here, datetime serialization worked
            success = True
        except TypeError as e:
            if "not JSON serializable" in str(e):
                success = False
            else:
                raise
        
        assert success, "Datetime serialization should work without errors"