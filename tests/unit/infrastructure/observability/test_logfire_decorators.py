"""Tests for Logfire decorators."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from src.infrastructure.observability.logfire_decorators import (
    traced,
    timed,
    with_span,
    log_event,
    traced_api_endpoint,
    traced_service_method,
    traced_use_case,
)


class TestTracedDecorator:
    """Test the traced decorator."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_sync_function_success(self, mock_logfire):
        """Test tracing a synchronous function that succeeds."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced(name="test_operation", capture_args=True)
        def test_func(x, y):
            return x + y

        # Execute
        result = test_func(1, 2)

        # Verify
        assert result == 3
        mock_logfire.span.assert_called_once_with("test_operation")
        mock_span.set_attribute.assert_any_call("function", "test_func")
        mock_span.set_attribute.assert_any_call("args", "(1, 2)")
        mock_span.set_attribute.assert_any_call("result", "3")

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_sync_function_error(self, mock_logfire):
        """Test tracing a synchronous function that raises an error."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced(name="test_operation")
        def test_func():
            raise ValueError("Test error")

        # Execute and verify exception is raised
        with pytest.raises(ValueError, match="Test error"):
            test_func()

        # Verify error attributes
        mock_span.set_attribute.assert_any_call("error", True)
        mock_span.set_attribute.assert_any_call("error_type", "ValueError")
        mock_span.set_attribute.assert_any_call("error_message", "Test error")

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @pytest.mark.asyncio
    async def test_traced_async_function_success(self, mock_logfire):
        """Test tracing an asynchronous function that succeeds."""
        mock_span = Mock()
        mock_span.__aenter__ = Mock(return_value=mock_span)
        mock_span.__aexit__ = Mock(return_value=None)
        mock_logfire.span.return_value = mock_span

        @traced(capture_result=False)
        async def test_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        # Execute
        result = await test_func(5)

        # Verify
        assert result == 10
        mock_logfire.span.assert_called_once()
        mock_span.set_attribute.assert_any_call("function", "test_func")
        # Result should not be captured
        assert not any(
            call[0][0] == "result" for call in mock_span.set_attribute.call_args_list
        )

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_with_extra_attributes(self, mock_logfire):
        """Test traced decorator with extra attributes."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced(service="test-service", operation="test-op", version="1.0")
        def test_func():
            return "result"

        # Execute
        test_func()

        # Verify extra attributes
        mock_span.set_attribute.assert_any_call("service", "test-service")
        mock_span.set_attribute.assert_any_call("operation", "test-op")
        mock_span.set_attribute.assert_any_call("version", "1.0")


class TestTimedDecorator:
    """Test the timed decorator."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @patch("src.infrastructure.observability.logfire_decorators.time.time")
    def test_timed_sync_function_success(self, mock_time, mock_logfire):
        """Test timing a synchronous function."""
        # Setup time mock
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 seconds duration

        @timed(metric_name="test.duration")
        def test_func():
            return "done"

        # Execute
        result = test_func()

        # Verify
        assert result == "done"

        # Check metric was logged
        metric_calls = [
            call
            for call in mock_logfire.info.call_args_list
            if "metric.test.duration" in call[0]
        ]
        assert len(metric_calls) == 1
        assert metric_calls[0][1]["value"] == 2.5
        assert metric_calls[0][1]["unit"] == "seconds"

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @pytest.mark.asyncio
    async def test_timed_async_function_with_error(self, mock_logfire):
        """Test timing an async function that errors."""

        @timed(metric_name="test.duration", include_errors=True)
        async def test_func():
            await asyncio.sleep(0.01)
            raise RuntimeError("Test error")

        # Execute
        with pytest.raises(RuntimeError):
            await test_func()

        # Verify error counter was incremented
        error_calls = [
            call
            for call in mock_logfire.info.call_args_list
            if "metric.test.duration.errors" in call[0]
        ]
        assert len(error_calls) == 1
        assert error_calls[0][1]["value"] == 1
        assert error_calls[0][1]["error_type"] == "RuntimeError"


class TestWithSpanDecorator:
    """Test the with_span decorator/context manager."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_with_span_as_context_manager(self, mock_logfire):
        """Test with_span as a context manager."""
        mock_span = Mock()
        mock_logfire.span.return_value = mock_span

        # Execute
        result = with_span("test_operation", user_id=123)

        # Verify
        assert result == mock_span
        mock_logfire.span.assert_called_once_with("test_operation", user_id=123)

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_with_span_as_decorator(self, mock_logfire):
        """Test with_span as a decorator."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @with_span
        def test_func():
            return "result"

        # Execute
        result = test_func()

        # Verify
        assert result == "result"
        # Should use function name as span name
        expected_span_name = f"{test_func.__module__}.{test_func.__name__}"
        mock_logfire.span.assert_called()


class TestLogEventDecorator:
    """Test the log_event decorator."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_log_event_sync(self, mock_logfire):
        """Test logging events from sync functions."""

        @log_event("user_action", action="create_stream")
        def test_func(stream_id):
            return stream_id

        # Execute
        result = test_func(123)

        # Verify
        assert result == 123
        mock_logfire.info.assert_called_once()
        call_args = mock_logfire.info.call_args
        assert "event.user_action" in call_args[0][0]
        assert call_args[1]["event_type"] == "user_action"
        assert call_args[1]["action"] == "create_stream"

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @pytest.mark.asyncio
    async def test_log_event_async(self, mock_logfire):
        """Test logging events from async functions."""

        @log_event("api_call", endpoint="/streams")
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        # Execute
        result = await test_func()

        # Verify
        assert result == "success"
        mock_logfire.info.assert_called_once()


class TestConvenienceDecorators:
    """Test the pre-configured convenience decorators."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_api_endpoint(self, mock_logfire):
        """Test traced_api_endpoint decorator."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced_api_endpoint(name="get_streams")
        def test_endpoint():
            return {"streams": []}

        # Execute
        result = test_endpoint()

        # Verify
        assert result == {"streams": []}
        mock_span.set_attribute.assert_any_call("operation_type", "api_endpoint")

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_service_method(self, mock_logfire):
        """Test traced_service_method decorator."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced_service_method()
        def process_stream():
            return True

        # Execute
        result = process_stream()

        # Verify
        assert result is True
        mock_span.set_attribute.assert_any_call("operation_type", "service_method")

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_use_case(self, mock_logfire):
        """Test traced_use_case decorator."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced_use_case(name="start_stream")
        def test_use_case():
            return {"status": "started"}

        # Execute
        result = test_use_case()

        # Verify
        assert result == {"status": "started"}
        mock_span.set_attribute.assert_any_call("operation_type", "use_case")


class TestDecoratorEdgeCases:
    """Test edge cases and error handling in decorators."""

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_with_none_arguments(self, mock_logfire):
        """Test traced decorator handles None arguments gracefully."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced(capture_args=True)
        def test_func(x, y=None):
            return x

        # Execute
        result = test_func(1, None)

        # Verify
        assert result == 1
        # Should handle None without errors
        mock_span.set_attribute.assert_any_call("args", "(1, None)")

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    def test_traced_with_large_result(self, mock_logfire):
        """Test traced decorator truncates large results."""
        mock_span = Mock()
        mock_logfire.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = Mock(return_value=None)

        @traced(capture_result=True)
        def test_func():
            return "x" * 1000  # Large string

        # Execute
        test_func()

        # Verify result was truncated
        result_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "result"
        ]
        assert len(result_calls) == 1
        assert len(result_calls[0][0][1]) <= 203  # 200 + "..."
