"""Tests for Logfire middleware."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.datastructures import Headers

from src.infrastructure.observability.logfire_middleware import LogfireMiddleware


class TestLogfireMiddleware:
    """Test the Logfire middleware for FastAPI."""

    @pytest.fixture
    def middleware(self):
        """Create a LogfireMiddleware instance for testing."""
        with patch("src.infrastructure.observability.logfire_middleware.logfire"):
            middleware = LogfireMiddleware(app=Mock())
            yield middleware

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/v1/streams"
        request.url._url = "http://localhost:8000/api/v1/streams"
        request.headers = Headers(
            {
                "host": "localhost:8000",
                "user-agent": "test-client",
                "x-api-key": "test-key-123",
                "x-correlation-id": "corr-123",
            }
        )
        request.path_params = {}
        request.query_params = {}
        request.get.return_value = None  # No route
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = Response(
            content=b'{"status": "ok"}',
            status_code=200,
            headers={"content-type": "application/json"},
        )
        return response

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @patch("src.infrastructure.observability.logfire_middleware.time.time")
    @pytest.mark.asyncio
    async def test_dispatch_successful_request(
        self, mock_time, mock_logfire, middleware, mock_request, mock_response
    ):
        """Test middleware handling a successful request."""
        # Setup
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms request
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        # Create async call_next
        async def call_next(request):
            return mock_response

        # Execute
        response = await middleware.dispatch(mock_request, call_next)

        # Verify
        assert response == mock_response
        mock_logfire.span.assert_called_once_with("http.request GET /api/v1/streams")

        # Verify span attributes
        expected_attributes = [
            ("http.method", "GET"),
            ("http.path", "/api/v1/streams"),
            ("http.url", "http://localhost:8000/api/v1/streams"),
            ("http.status_code", 200),
            ("http.duration_ms", 500.0),
            ("user.api_key_id", "test-key-123"),
        ]

        for attr_name, attr_value in expected_attributes:
            mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @patch("src.infrastructure.observability.logfire_middleware.settings")
    @pytest.mark.asyncio
    async def test_dispatch_with_headers_capture(
        self, mock_settings, mock_logfire, middleware, mock_request
    ):
        """Test middleware captures headers when enabled."""
        # Setup
        mock_settings.logfire_capture_headers = True
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return Response(content=b"OK", headers={"x-custom-header": "value"})

        # Execute
        await middleware.dispatch(mock_request, call_next)

        # Verify request headers were captured
        mock_span.set_attribute.assert_any_call(
            "http.request.headers",
            {
                "host": "localhost:8000",
                "user-agent": "test-client",
                "x-api-key": "[REDACTED]",
                "x-correlation-id": "corr-123",
            },
        )

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @patch("src.infrastructure.observability.logfire_middleware.settings")
    @pytest.mark.asyncio
    async def test_dispatch_with_body_capture(
        self, mock_settings, mock_logfire, middleware
    ):
        """Test middleware captures body when enabled."""
        # Setup
        mock_settings.logfire_capture_body = True
        mock_settings.logfire_max_body_size = 1000

        # Create request with body
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/v1/streams"
        request.headers = Headers({"content-type": "application/json"})
        request.body = AsyncMock(return_value=b'{"title": "Test Stream"}')
        request.get.return_value = None

        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return JSONResponse({"id": 123})

        # Execute
        await middleware.dispatch(request, call_next)

        # Verify body was captured
        mock_span.set_attribute.assert_any_call(
            "http.request.body", '{"title": "Test Stream"}'
        )

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @pytest.mark.asyncio
    async def test_dispatch_error_handling(
        self, mock_logfire, middleware, mock_request
    ):
        """Test middleware handles errors properly."""
        # Setup
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        # Create call_next that raises an exception
        async def call_next(request):
            raise ValueError("Test error")

        # Execute and verify exception is propagated
        with pytest.raises(ValueError, match="Test error"):
            await middleware.dispatch(mock_request, call_next)

        # Verify error attributes were set
        mock_span.set_attribute.assert_any_call("error", True)
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.set_attribute.assert_any_call("error.message", "Test error")

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @pytest.mark.asyncio
    async def test_dispatch_with_organization_context(self, mock_logfire, middleware):
        """Test middleware extracts organization context."""
        # Setup
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/v1/streams"
        request.headers = Headers({})
        request.get.return_value = Mock(
            scope={"organization": {"id": "org-123", "name": "Test Org"}}
        )

        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return Response(content=b"OK")

        # Execute
        await middleware.dispatch(request, call_next)

        # Verify organization context was set
        mock_span.set_attribute.assert_any_call("user.organization_id", "org-123")
        mock_span.set_attribute.assert_any_call("user.organization_name", "Test Org")

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @pytest.mark.asyncio
    async def test_dispatch_health_check_minimal_logging(
        self, mock_logfire, middleware
    ):
        """Test middleware uses minimal logging for health checks."""
        # Setup
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/health"
        request.headers = Headers({})
        request.get.return_value = None

        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return Response(content=b"OK")

        # Execute
        await middleware.dispatch(request, call_next)

        # Verify minimal attributes for health check
        set_attribute_calls = mock_span.set_attribute.call_args_list
        # Should have fewer attributes than normal requests
        assert len(set_attribute_calls) < 10

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @patch("src.infrastructure.observability.logfire_middleware.settings")
    @pytest.mark.asyncio
    async def test_dispatch_large_body_truncation(
        self, mock_settings, mock_logfire, middleware
    ):
        """Test middleware truncates large request bodies."""
        # Setup
        mock_settings.logfire_capture_body = True
        mock_settings.logfire_max_body_size = 100

        # Create request with large body
        large_body = b'{"data": "' + b"x" * 200 + b'"}'
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/v1/data"
        request.headers = Headers({"content-type": "application/json"})
        request.body = AsyncMock(return_value=large_body)
        request.get.return_value = None

        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return Response(content=b"OK")

        # Execute
        await middleware.dispatch(request, call_next)

        # Verify body was truncated
        body_calls = [
            call
            for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "http.request.body"
        ]
        assert len(body_calls) == 1
        captured_body = body_calls[0][0][1]
        assert len(captured_body) <= 103  # 100 + "..."
        assert captured_body.endswith("...")

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @pytest.mark.asyncio
    async def test_dispatch_correlation_id_propagation(
        self, mock_logfire, middleware, mock_request
    ):
        """Test middleware propagates correlation ID."""
        # Setup
        mock_logfire.with_tags.return_value = Mock()
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        async def call_next(request):
            return Response(content=b"OK")

        # Execute
        await middleware.dispatch(mock_request, call_next)

        # Verify correlation ID was set
        mock_logfire.with_tags.assert_called_once_with(correlation_id="corr-123")

    @patch("src.infrastructure.observability.logfire_middleware.logfire")
    @pytest.mark.asyncio
    async def test_dispatch_streaming_response(
        self, mock_logfire, middleware, mock_request
    ):
        """Test middleware handles streaming responses."""
        # Setup
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        # Create streaming response
        async def generate():
            yield b"chunk1"
            yield b"chunk2"

        from starlette.responses import StreamingResponse

        streaming_response = StreamingResponse(generate(), media_type="text/plain")

        async def call_next(request):
            return streaming_response

        # Execute
        response = await middleware.dispatch(mock_request, call_next)

        # Verify
        assert isinstance(response, StreamingResponse)
        mock_span.set_attribute.assert_any_call("http.response.streaming", True)
