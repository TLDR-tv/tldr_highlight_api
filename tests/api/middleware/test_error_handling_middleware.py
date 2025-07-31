"""Tests for error handling middleware."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.api.middleware.error_handling import ErrorHandlingMiddleware
from src.infrastructure.config import settings


@pytest.mark.asyncio
class TestErrorHandlingMiddleware:
    """Test cases for error handling middleware."""

    async def test_successful_request_passthrough(self):
        """Test that successful requests pass through without modification."""
        # Create mock app and middleware
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Create mock request and response
        request = MagicMock(spec=Request)
        expected_response = Response(content="Success", status_code=200)

        # Mock call_next to return successful response
        async def call_next(req):
            return expected_response

        response = await middleware.dispatch(request, call_next)

        assert response == expected_response
        assert response.status_code == 200

    async def test_exception_handling_production(self):
        """Test exception handling in production mode."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Create mock request
        request = MagicMock(spec=Request)
        request.state.request_id = "test-request-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"
        request.headers = {"user-agent": "TestClient/1.0"}
        request.client = MagicMock(host="192.168.1.1")

        # Mock call_next to raise exception
        test_exception = ValueError("Test error message")

        async def call_next(req):
            raise test_exception

        with patch.object(settings, "is_production", True):
            response = await middleware.dispatch(request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Check response content
        content = response.body.decode()
        assert "An internal server error occurred" in content
        assert (
            "Test error message" not in content
        )  # Should not expose details in production

    async def test_exception_handling_development(self):
        """Test exception handling in development mode."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Create mock request
        request = MagicMock(spec=Request)
        request.state.request_id = "test-request-123"
        request.url = "https://api.example.com/test"
        request.method = "POST"
        request.headers = {"user-agent": "TestClient/1.0"}
        request.client = MagicMock(host="192.168.1.1")

        # Mock call_next to raise exception
        test_exception = ValueError("Detailed error message")

        async def call_next(req):
            raise test_exception

        with patch.object(settings, "is_production", False):
            response = await middleware.dispatch(request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Check response content
        content = response.body.decode()
        assert (
            "ValueError: Detailed error message" in content
        )  # Should expose details in development
        assert "exception_type" in content
        assert "exception_message" in content

    async def test_logging_on_exception(self):
        """Test that exceptions are properly logged."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Create mock request
        request = MagicMock(spec=Request)
        request.state.request_id = "log-test-123"
        request.url = "https://api.example.com/error"
        request.method = "PUT"
        request.headers = {"user-agent": "TestLogger/1.0"}
        request.client = MagicMock(host="10.0.0.1")

        test_exception = RuntimeError("Log this error")

        async def call_next(req):
            raise test_exception

        with patch("src.api.middleware.error_handling.logger") as mock_logger:
            await middleware.dispatch(request, call_next)

            # Verify logging was called
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args

            # Check log message
            assert "Unhandled exception in middleware: RuntimeError" in call_args[0][0]

            # Check log extra data
            extra = call_args.kwargs["extra"]
            assert extra["request_id"] == "log-test-123"
            assert extra["path"] == "https://api.example.com/error"
            assert extra["method"] == "PUT"
            assert extra["exception_type"] == "RuntimeError"
            assert extra["exception_message"] == "Log this error"
            assert extra["client_ip"] == "10.0.0.1"
            assert extra["user_agent"] == "TestLogger/1.0"

    async def test_request_without_request_id(self):
        """Test handling requests without request_id in state."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Create mock request without request_id
        request = MagicMock(spec=Request)
        request.state = MagicMock(spec=[])  # Empty state
        request.url = "https://api.example.com/test"
        request.method = "GET"
        request.headers = {}
        request.client = MagicMock(host="localhost")

        test_exception = Exception("No request ID")

        async def call_next(req):
            raise test_exception

        response = await middleware.dispatch(request, call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

    async def test_client_ip_extraction_forwarded_for(self):
        """Test client IP extraction from X-Forwarded-For header."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Test X-Forwarded-For header
        request = MagicMock(spec=Request)
        request.headers = {"x-forwarded-for": "203.0.113.1, 198.51.100.2, 172.16.0.1"}
        request.client = MagicMock(host="127.0.0.1")

        client_ip = middleware._get_client_ip(request)
        assert client_ip == "203.0.113.1"  # Should get first IP

    async def test_client_ip_extraction_real_ip(self):
        """Test client IP extraction from X-Real-IP header."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Test X-Real-IP header
        request = MagicMock(spec=Request)
        request.headers = {"x-real-ip": "203.0.113.5"}
        request.client = MagicMock(host="127.0.0.1")

        client_ip = middleware._get_client_ip(request)
        assert client_ip == "203.0.113.5"

    async def test_client_ip_extraction_direct(self):
        """Test client IP extraction from direct connection."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Test direct client connection
        request = MagicMock(spec=Request)
        request.headers = {}
        request.client = MagicMock(host="192.168.1.100")

        client_ip = middleware._get_client_ip(request)
        assert client_ip == "192.168.1.100"

    async def test_client_ip_extraction_no_client(self):
        """Test client IP extraction when client is None."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        # Test no client info
        request = MagicMock(spec=Request)
        request.headers = {}
        request.client = None

        client_ip = middleware._get_client_ip(request)
        assert client_ip == "unknown"

    async def test_different_exception_types(self):
        """Test handling of different exception types."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        exceptions = [
            ValueError("Value error"),
            KeyError("key"),
            AttributeError("attribute"),
            RuntimeError("Runtime error"),
            Exception("Generic exception"),
        ]

        for test_exception in exceptions:
            request = MagicMock(spec=Request)
            request.state.request_id = "test-123"
            request.url = "https://api.example.com/test"
            request.method = "GET"
            request.headers = {}
            request.client = MagicMock(host="localhost")

            async def call_next(req):
                raise test_exception

            with patch.object(settings, "is_production", False):
                response = await middleware.dispatch(request, call_next)

            assert response.status_code == 500
            content = response.body.decode()
            assert type(test_exception).__name__ in content

    async def test_error_response_format(self):
        """Test that error responses have the correct format."""
        app = MagicMock()
        middleware = ErrorHandlingMiddleware(app)

        request = MagicMock(spec=Request)
        request.state.request_id = "format-test-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"
        request.headers = {}
        request.client = MagicMock(host="localhost")

        async def call_next(req):
            raise ValueError("Test error")

        with patch(
            "src.api.middleware.error_handling.create_error_response"
        ) as mock_create:
            mock_create.return_value = {
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An error occurred",
                    "status_code": 500,
                },
                "request_id": "format-test-123",
            }

            await middleware.dispatch(request, call_next)

            # Verify create_error_response was called with correct parameters
            mock_create.assert_called_once_with(
                status_code=500,
                message=mock_create.call_args.kwargs["message"],
                error_code="INTERNAL_SERVER_ERROR",
                details=mock_create.call_args.kwargs["details"],
                request_id="format-test-123",
            )
