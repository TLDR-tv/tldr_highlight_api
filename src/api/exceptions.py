"""Exception handlers and custom exceptions for the TL;DR Highlight API.

This module defines custom exception classes and global exception handlers
for consistent error responses across the API.
"""

import logging
from typing import Any, Dict, Union

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import (
    DatabaseError,
    IntegrityError,
    OperationalError,
    StatementError,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class TLDRException(Exception):
    """Base exception class for TL;DR Highlight API."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Dict[str, Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(TLDRException):
    """Exception raised for authentication failures."""

    def __init__(
        self, message: str = "Authentication failed", details: Dict[str, Any] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_FAILED",
            details=details,
        )


class AuthorizationError(TLDRException):
    """Exception raised for authorization failures."""

    def __init__(
        self, message: str = "Insufficient permissions", details: Dict[str, Any] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_FAILED",
            details=details,
        )


class ResourceNotFoundError(TLDRException):
    """Exception raised when a resource is not found."""

    def __init__(
        self,
        resource: str,
        resource_id: Union[str, int],
        details: Dict[str, Any] = None,
    ):
        message = f"{resource} with ID '{resource_id}' not found"
        # Merge provided details with default ones
        default_details = {"resource": resource, "resource_id": str(resource_id)}
        if details:
            default_details.update(details)
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details=default_details,
        )


class ResourceConflictError(TLDRException):
    """Exception raised when a resource conflict occurs."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="RESOURCE_CONFLICT",
            details=details,
        )


class RateLimitExceededError(TLDRException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", details: Dict[str, Any] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
        )


class TLDRValidationError(TLDRException):
    """Exception raised for validation errors."""

    def __init__(self, message: str, field_errors: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details={"field_errors": field_errors or {}},
        )


class ExternalServiceError(TLDRException):
    """Exception raised when external service calls fail."""

    def __init__(
        self, service: str, message: str = None, details: Dict[str, Any] = None
    ):
        message = message or f"External service '{service}' is unavailable"
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details or {"service": service},
        )


class ProcessingError(TLDRException):
    """Exception raised during content processing."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="PROCESSING_ERROR",
            details=details,
        )


def create_error_response(
    status_code: int,
    message: str,
    error_code: str = "ERROR",
    details: Dict[str, Any] = None,
    request_id: str = None,
) -> Dict[str, Any]:
    """Create standardized error response format.

    Args:
        status_code: HTTP status code
        message: Error message
        error_code: Application-specific error code
        details: Additional error details
        request_id: Request ID for tracing

    Returns:
        Dict: Standardized error response
    """
    response = {
        "error": {"code": error_code, "message": message, "status_code": status_code},
        "success": False,
    }

    if details:
        response["error"]["details"] = details

    if request_id:
        response["request_id"] = request_id

    return response


async def tldr_exception_handler(request: Request, exc: TLDRException) -> JSONResponse:
    """Handle custom TL;DR exceptions."""
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        f"TLDRException: {exc.error_code} - {exc.message}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "details": exc.details,
            "path": str(request.url),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=request_id,
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Map HTTP status codes to error codes
    error_code_mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }

    error_code = error_code_mapping.get(exc.status_code, "HTTP_ERROR")

    logger.error(
        f"HTTPException: {error_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "error_code": error_code,
            "path": str(request.url),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=error_code,
            request_id=request_id,
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", None)

    # Extract field errors
    field_errors = {}
    for error in exc.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        field_errors[field_path] = {
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input"),
        }

    logger.error(
        "ValidationError: Request validation failed",
        extra={
            "request_id": request_id,
            "field_errors": field_errors,
            "path": str(request.url),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            details={"field_errors": field_errors},
            request_id=request_id,
        ),
    )


async def database_exception_handler(
    request: Request, exc: DatabaseError
) -> JSONResponse:
    """Handle database exceptions."""
    request_id = getattr(request.state, "request_id", None)

    # Determine error type and message
    if isinstance(exc, IntegrityError):
        error_code = "DATABASE_INTEGRITY_ERROR"
        message = "Database constraint violation"
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, OperationalError):
        error_code = "DATABASE_OPERATIONAL_ERROR"
        message = "Database operation failed"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, StatementError):
        error_code = "DATABASE_STATEMENT_ERROR"
        message = "Invalid database query"
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        error_code = "DATABASE_ERROR"
        message = "Database error occurred"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    logger.error(
        f"DatabaseError: {error_code} - {str(exc)}",
        extra={
            "request_id": request_id,
            "error_code": error_code,
            "exception_type": type(exc).__name__,
            "path": str(request.url),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status_code,
        content=create_error_response(
            status_code=status_code,
            message=message,
            error_code=error_code,
            request_id=request_id,
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", None)

    logger.error(
        f"UnhandledException: {type(exc).__name__} - {str(exc)}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "path": str(request.url),
            "method": request.method,
        },
        exc_info=True,
    )

    # Don't expose internal error details in production
    if request.app.state.settings.is_production:
        message = "Internal server error"
    else:
        message = f"{type(exc).__name__}: {str(exc)}"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            error_code="INTERNAL_SERVER_ERROR",
            request_id=request_id,
        ),
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Store settings in app state for access in handlers
    app.state.settings = (
        app.extra.get("settings")
        or __import__("src.infrastructure.config", fromlist=["settings"]).settings
    )

    # Custom exception handlers
    app.add_exception_handler(TLDRException, tldr_exception_handler)

    # Standard HTTP exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Validation exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)

    # Database exception handlers
    app.add_exception_handler(DatabaseError, database_exception_handler)
    app.add_exception_handler(IntegrityError, database_exception_handler)
    app.add_exception_handler(OperationalError, database_exception_handler)
    app.add_exception_handler(StatementError, database_exception_handler)

    # Catch-all exception handler
    app.add_exception_handler(Exception, generic_exception_handler)
