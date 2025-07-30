"""Tests for API exception handling."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
    StatementError,
    DatabaseError,
)

from src.api.exceptions import (
    TLDRException,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    ResourceConflictError,
    RateLimitExceededError,
    TLDRValidationError,
    ExternalServiceError,
    ProcessingError,
    create_error_response,
    tldr_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    database_exception_handler,
    generic_exception_handler,
    setup_exception_handlers,
)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_tldr_exception_base(self):
        """Test base TLDRException."""
        exc = TLDRException(
            message="Test error",
            status_code=400,
            error_code="TEST_ERROR",
            details={"key": "value"},
        )

        assert exc.message == "Test error"
        assert exc.status_code == 400
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == {"key": "value"}
        assert str(exc) == "Test error"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError("Invalid credentials", {"reason": "expired"})

        assert exc.message == "Invalid credentials"
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_FAILED"
        assert exc.details == {"reason": "expired"}

    def test_authentication_error_defaults(self):
        """Test AuthenticationError with defaults."""
        exc = AuthenticationError()

        assert exc.message == "Authentication failed"
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_FAILED"

    def test_authorization_error(self):
        """Test AuthorizationError."""
        exc = AuthorizationError("Admin access required", {"required_role": "admin"})

        assert exc.message == "Admin access required"
        assert exc.status_code == 403
        assert exc.error_code == "AUTHORIZATION_FAILED"
        assert exc.details == {"required_role": "admin"}

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError."""
        exc = ResourceNotFoundError("User", 123, {"checked_tables": ["users"]})

        assert exc.message == "User with ID '123' not found"
        assert exc.status_code == 404
        assert exc.error_code == "RESOURCE_NOT_FOUND"
        assert exc.details["resource"] == "User"
        assert exc.details["resource_id"] == "123"
        assert exc.details["checked_tables"] == ["users"]

    def test_resource_conflict_error(self):
        """Test ResourceConflictError."""
        exc = ResourceConflictError(
            "Email already exists", {"email": "test@example.com"}
        )

        assert exc.message == "Email already exists"
        assert exc.status_code == 409
        assert exc.error_code == "RESOURCE_CONFLICT"
        assert exc.details == {"email": "test@example.com"}

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        exc = RateLimitExceededError("API rate limit exceeded", {"retry_after": 60})

        assert exc.message == "API rate limit exceeded"
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.details == {"retry_after": 60}

    def test_tldr_validation_error(self):
        """Test TLDRValidationError."""
        field_errors = {
            "email": "Invalid email format",
            "age": "Must be greater than 0",
        }
        exc = TLDRValidationError("Validation failed", field_errors)

        assert exc.message == "Validation failed"
        assert exc.status_code == 422
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.details["field_errors"] == field_errors

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError(
            "AWS S3", "Connection timeout", {"region": "us-east-1"}
        )

        assert exc.message == "Connection timeout"
        assert exc.status_code == 503
        assert exc.error_code == "EXTERNAL_SERVICE_ERROR"
        assert exc.details == {"region": "us-east-1"}

    def test_external_service_error_default_message(self):
        """Test ExternalServiceError with default message."""
        exc = ExternalServiceError("Redis")

        assert exc.message == "External service 'Redis' is unavailable"
        assert exc.details == {"service": "Redis"}

    def test_processing_error(self):
        """Test ProcessingError."""
        exc = ProcessingError("Video format not supported", {"format": "avi"})

        assert exc.message == "Video format not supported"
        assert exc.status_code == 422
        assert exc.error_code == "PROCESSING_ERROR"
        assert exc.details == {"format": "avi"}


class TestErrorResponse:
    """Test error response creation."""

    def test_create_error_response_basic(self):
        """Test basic error response creation."""
        response = create_error_response(
            status_code=404, message="Not found", error_code="NOT_FOUND"
        )

        assert response["success"] is False
        assert response["error"]["code"] == "NOT_FOUND"
        assert response["error"]["message"] == "Not found"
        assert response["error"]["status_code"] == 404
        assert "details" not in response["error"]
        assert "request_id" not in response

    def test_create_error_response_with_details(self):
        """Test error response with details."""
        response = create_error_response(
            status_code=400,
            message="Bad request",
            error_code="BAD_REQUEST",
            details={"field": "email", "reason": "invalid"},
        )

        assert response["error"]["details"] == {"field": "email", "reason": "invalid"}

    def test_create_error_response_with_request_id(self):
        """Test error response with request ID."""
        response = create_error_response(
            status_code=500,
            message="Server error",
            error_code="ERROR",
            request_id="req-123",
        )

        assert response["request_id"] == "req-123"


@pytest.mark.asyncio
class TestExceptionHandlers:
    """Test exception handler functions."""

    async def test_tldr_exception_handler(self):
        """Test handling of TLDRException."""
        request = MagicMock(spec=Request)
        request.state.request_id = "test-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"

        exc = ResourceNotFoundError("User", 123)

        with patch("src.api.exceptions.logger") as mock_logger:
            response = await tldr_exception_handler(request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 404

            # Check logging
            mock_logger.error.assert_called_once()
            log_call = mock_logger.error.call_args
            assert "RESOURCE_NOT_FOUND" in log_call[0][0]

    async def test_http_exception_handler(self):
        """Test handling of HTTPException."""
        request = MagicMock(spec=Request)
        request.state.request_id = "http-123"
        request.url = "https://api.example.com/test"
        request.method = "POST"

        exc = HTTPException(status_code=401, detail="Unauthorized access")

        with patch("src.api.exceptions.logger") as mock_logger:
            response = await http_exception_handler(request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 401

            # Check error code mapping
            mock_logger.error.assert_called_once()
            log_extra = mock_logger.error.call_args.kwargs["extra"]
            assert log_extra["error_code"] == "UNAUTHORIZED"

    async def test_validation_exception_handler(self):
        """Test handling of validation errors."""
        request = MagicMock(spec=Request)
        request.state.request_id = "val-123"
        request.url = "https://api.example.com/test"
        request.method = "POST"

        # Mock validation error
        mock_errors = [
            {
                "loc": ("body", "email"),
                "msg": "invalid email format",
                "type": "value_error.email",
                "input": "not-an-email",
            },
            {
                "loc": ("body", "age"),
                "msg": "ensure this value is greater than 0",
                "type": "value_error.number.not_gt",
                "input": -5,
            },
        ]

        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = mock_errors

        with patch("src.api.exceptions.logger") as mock_logger:
            response = await validation_exception_handler(request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 422

            # Check field errors are extracted
            mock_logger.error.assert_called_once()
            log_extra = mock_logger.error.call_args.kwargs["extra"]
            assert "body.email" in log_extra["field_errors"]
            assert "body.age" in log_extra["field_errors"]

    async def test_database_exception_handler_integrity_error(self):
        """Test handling of database integrity errors."""
        request = MagicMock(spec=Request)
        request.state.request_id = "db-123"
        request.url = "https://api.example.com/test"
        request.method = "POST"

        # Mock IntegrityError
        exc = MagicMock(spec=IntegrityError)
        exc.__class__ = IntegrityError

        with patch("src.api.exceptions.logger") as mock_logger:
            response = await database_exception_handler(request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 409

            # Check error categorization
            mock_logger.error.assert_called_once()
            log_call = mock_logger.error.call_args
            assert "DATABASE_INTEGRITY_ERROR" in log_call[0][0]

    async def test_database_exception_handler_operational_error(self):
        """Test handling of database operational errors."""
        request = MagicMock(spec=Request)
        request.state.request_id = "db-op-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"

        # Mock OperationalError
        exc = MagicMock(spec=OperationalError)
        exc.__class__ = OperationalError

        response = await database_exception_handler(request, exc)

        assert response.status_code == 503

    async def test_database_exception_handler_statement_error(self):
        """Test handling of database statement errors."""
        request = MagicMock(spec=Request)
        request.state.request_id = "db-stmt-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"

        # Mock StatementError
        exc = MagicMock(spec=StatementError)
        exc.__class__ = StatementError

        response = await database_exception_handler(request, exc)

        assert response.status_code == 400

    async def test_generic_exception_handler_production(self):
        """Test handling of generic exceptions in production."""
        request = MagicMock(spec=Request)
        request.state.request_id = "gen-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"
        request.app.state.settings.is_production = True

        exc = RuntimeError("Secret internal error")

        with patch("src.api.exceptions.logger") as mock_logger:
            response = await generic_exception_handler(request, exc)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 500

            # Should not expose internal details
            # We need to decode the response body to check the message
            import json

            content = json.loads(response.body.decode())
            assert content["error"]["message"] == "Internal server error"
            assert "Secret internal error" not in content["error"]["message"]

    async def test_generic_exception_handler_development(self):
        """Test handling of generic exceptions in development."""
        request = MagicMock(spec=Request)
        request.state.request_id = "gen-dev-123"
        request.url = "https://api.example.com/test"
        request.method = "GET"
        request.app.state.settings.is_production = False

        exc = ValueError("Detailed error for debugging")

        response = await generic_exception_handler(request, exc)

        # Should expose details in development
        import json

        content = json.loads(response.body.decode())
        assert "ValueError: Detailed error for debugging" in content["error"]["message"]


class TestExceptionHandlerSetup:
    """Test exception handler setup."""

    def test_setup_exception_handlers(self):
        """Test setting up exception handlers on FastAPI app."""
        app = FastAPI()
        app.extra = {}

        # Mock settings
        mock_settings = MagicMock(is_production=False)
        with patch("src.core.config.settings", mock_settings):
            setup_exception_handlers(app)

            # Check that handlers were added
            assert TLDRException in app.exception_handlers
            assert HTTPException in app.exception_handlers
            assert RequestValidationError in app.exception_handlers
            assert DatabaseError in app.exception_handlers
            assert Exception in app.exception_handlers

            # Check settings were stored
            assert hasattr(app.state, "settings")

    def test_setup_exception_handlers_with_existing_settings(self):
        """Test setup with pre-existing settings."""
        app = FastAPI()
        mock_settings = MagicMock(is_production=True)
        app.extra = {"settings": mock_settings}

        setup_exception_handlers(app)

        assert app.state.settings == mock_settings


@pytest.mark.asyncio
class TestExceptionHandlerIntegration:
    """Test exception handler integration scenarios."""

    async def test_request_without_request_id(self):
        """Test handlers when request has no request_id."""
        request = MagicMock(spec=Request)
        request.state = MagicMock(spec=[])  # No request_id attribute
        request.url = "https://api.example.com/test"
        request.method = "GET"

        exc = AuthenticationError("No auth")

        response = await tldr_exception_handler(request, exc)

        assert response.status_code == 401
        # Should handle missing request_id gracefully

    async def test_multiple_validation_errors(self):
        """Test handling multiple validation errors."""
        request = MagicMock(spec=Request)
        request.state.request_id = "multi-val-123"
        request.url = "https://api.example.com/test"
        request.method = "POST"

        # Multiple validation errors
        mock_errors = [
            {
                "loc": ("body", "user", "email"),
                "msg": "invalid email",
                "type": "value_error.email",
            },
            {
                "loc": ("body", "user", "password"),
                "msg": "password too short",
                "type": "value_error.str.min_length",
            },
            {
                "loc": ("query", "page"),
                "msg": "ensure this value is greater than 0",
                "type": "value_error.number.not_gt",
                "input": -1,
            },
        ]

        exc = MagicMock(spec=RequestValidationError)
        exc.errors.return_value = mock_errors

        response = await validation_exception_handler(request, exc)

        import json

        content = json.loads(response.body.decode())
        field_errors = content["error"]["details"]["field_errors"]

        assert "body.user.email" in field_errors
        assert "body.user.password" in field_errors
        assert "query.page" in field_errors
