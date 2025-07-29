"""Common response models and schemas for the TL;DR Highlight API.

This module contains standardized response wrappers, error models,
pagination schemas, and common validation models used across the API.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields."""

    created_at: datetime = Field(description="Timestamp when the resource was created")
    updated_at: datetime = Field(
        description="Timestamp when the resource was last updated"
    )


class PaginationParams(BaseModel):
    """Query parameters for pagination."""

    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of items to return"
    )


class PaginationMeta(BaseModel):
    """Pagination metadata for responses."""

    offset: int = Field(description="Current offset")
    limit: int = Field(description="Current limit")
    total: int = Field(description="Total number of items")
    has_next: bool = Field(description="Whether there are more items")
    has_previous: bool = Field(description="Whether there are previous items")

    @classmethod
    def create(cls, offset: int, limit: int, total: int) -> "PaginationMeta":
        """Create pagination metadata.

        Args:
            offset: Current offset
            limit: Current limit
            total: Total number of items

        Returns:
            PaginationMeta: Pagination metadata
        """
        return cls(
            offset=offset,
            limit=limit,
            total=total,
            has_next=offset + limit < total,
            has_previous=offset > 0,
        )


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper."""

    success: bool = Field(default=True, description="Indicates successful operation")
    data: T = Field(description="Response data")
    message: Optional[str] = Field(default=None, description="Optional success message")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    success: bool = Field(default=True, description="Indicates successful operation")
    data: List[T] = Field(description="List of items")
    pagination: PaginationMeta = Field(description="Pagination metadata")
    message: Optional[str] = Field(default=None, description="Optional message")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(description="Application-specific error code")
    message: str = Field(description="Human-readable error message")
    status_code: int = Field(description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""

    success: bool = Field(default=False, description="Indicates failed operation")
    error: ErrorDetail = Field(description="Error information")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class ValidationErrorDetail(BaseModel):
    """Validation error detail for a specific field."""

    message: str = Field(description="Validation error message")
    type: str = Field(description="Validation error type")
    input: Optional[Any] = Field(default=None, description="Invalid input value")


class ValidationErrorResponse(BaseModel):
    """Response for validation errors."""

    success: bool = Field(default=False, description="Indicates failed operation")
    error: ErrorDetail = Field(description="Error information")
    field_errors: Optional[Dict[str, ValidationErrorDetail]] = Field(
        default=None, description="Field-specific validation errors"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class StatusResponse(BaseModel):
    """Simple status response."""

    status: str = Field(description="Status message")
    timestamp: datetime = Field(description="Response timestamp")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Overall health status")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="Health check timestamp")
    services: Dict[str, Dict[str, Any]] = Field(description="Service health details")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class IdResponse(BaseModel):
    """Response containing a resource ID."""

    id: Union[str, int] = Field(description="Resource identifier")
    message: Optional[str] = Field(default=None, description="Optional message")


class BulkOperationResponse(BaseModel):
    """Response for bulk operations."""

    success: bool = Field(description="Indicates overall operation success")
    total: int = Field(description="Total number of items processed")
    successful: int = Field(description="Number of successfully processed items")
    failed: int = Field(description="Number of failed items")
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Details of failed items"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracing"
    )


class SearchParams(BaseModel):
    """Common search parameters."""

    query: Optional[str] = Field(
        default=None, max_length=1000, description="Search query string"
    )
    sort_by: Optional[str] = Field(default="created_at", description="Field to sort by")
    sort_order: Optional[str] = Field(
        default="desc", pattern="^(asc|desc)$", description="Sort order (asc or desc)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize search query."""
        if v:
            # Basic sanitization - remove potentially dangerous characters
            sanitized = v.strip()
            if not sanitized:
                return None
            return sanitized
        return v


class DateRangeFilter(BaseModel):
    """Date range filter parameters."""

    start_date: Optional[datetime] = Field(
        default=None, description="Start date for filtering (inclusive)"
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date for filtering (inclusive)"
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end_date is after start_date."""
        if v and info.data.get("start_date") and v < info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class FilterParams(SearchParams, DateRangeFilter):
    """Combined filter parameters for advanced filtering."""

    tags: Optional[List[str]] = Field(
        default=None, max_length=50, description="Filter by tags"
    )
    status: Optional[str] = Field(default=None, description="Filter by status")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and sanitize tags."""
        if v:
            # Remove empty tags and limit length
            cleaned_tags = [tag.strip()[:50] for tag in v if tag and tag.strip()]
            return cleaned_tags[:10] if cleaned_tags else None  # Limit to 10 tags
        return v


class WebhookPayload(BaseModel):
    """Base webhook payload structure."""

    event: str = Field(description="Event type")
    timestamp: datetime = Field(description="Event timestamp")
    data: Dict[str, Any] = Field(description="Event data")
    webhook_id: str = Field(description="Webhook configuration ID")
    delivery_id: str = Field(description="Unique delivery ID")


class APIUsageStats(BaseModel):
    """API usage statistics."""

    requests_count: int = Field(description="Total number of requests")
    requests_per_minute: float = Field(description="Average requests per minute")
    requests_per_hour: float = Field(description="Average requests per hour")
    error_rate: float = Field(description="Error rate percentage")
    avg_response_time_ms: float = Field(
        description="Average response time in milliseconds"
    )

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v: float) -> float:
        """Ensure error rate is between 0 and 100."""
        return max(0.0, min(100.0, v))


class ResourceQuota(BaseModel):
    """Resource quota information."""

    used: int = Field(description="Currently used amount")
    limit: int = Field(description="Maximum allowed amount")
    remaining: int = Field(description="Remaining amount")
    reset_at: Optional[datetime] = Field(
        default=None, description="When the quota resets"
    )

    @field_validator("remaining")
    @classmethod
    def calculate_remaining(cls, v: int, info) -> int:
        """Calculate remaining quota."""
        used = info.data.get("used", 0)
        limit = info.data.get("limit", 0)
        return max(0, limit - used)


# Common HTTP status responses for OpenAPI documentation
COMMON_RESPONSES = {
    400: {"description": "Bad Request", "model": ErrorResponse},
    401: {"description": "Unauthorized", "model": ErrorResponse},
    403: {"description": "Forbidden", "model": ErrorResponse},
    404: {"description": "Not Found", "model": ErrorResponse},
    409: {"description": "Conflict", "model": ErrorResponse},
    422: {"description": "Unprocessable Entity", "model": ValidationErrorResponse},
    429: {"description": "Too Many Requests", "model": ErrorResponse},
    500: {"description": "Internal Server Error", "model": ErrorResponse},
    503: {"description": "Service Unavailable", "model": ErrorResponse},
}
