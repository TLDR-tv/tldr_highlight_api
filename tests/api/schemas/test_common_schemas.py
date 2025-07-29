"""Tests for common API schemas."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from pydantic import ValidationError

from src.api.schemas.common import (
    TimestampMixin,
    PaginationParams,
    PaginationMeta,
    SuccessResponse,
    PaginatedResponse,
    ErrorDetail,
    ErrorResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
    StatusResponse,
    HealthCheckResponse,
    IdResponse,
    BulkOperationResponse,
    SearchParams,
    DateRangeFilter,
    FilterParams,
    WebhookPayload,
    APIUsageStats,
    ResourceQuota,
)


class TestTimestampMixin:
    """Test cases for TimestampMixin."""

    def test_timestamp_mixin_creation(self):
        """Test creating a model with timestamp fields."""
        now = datetime.utcnow()
        
        class TestModel(TimestampMixin):
            name: str
        
        obj = TestModel(name="test", created_at=now, updated_at=now)
        assert obj.created_at == now
        assert obj.updated_at == now
        assert obj.name == "test"


class TestPaginationSchemas:
    """Test cases for pagination schemas."""

    def test_pagination_params_defaults(self):
        """Test pagination params with default values."""
        params = PaginationParams()
        assert params.offset == 0
        assert params.limit == 20

    def test_pagination_params_custom(self):
        """Test pagination params with custom values."""
        params = PaginationParams(offset=50, limit=100)
        assert params.offset == 50
        assert params.limit == 100

    def test_pagination_params_validation(self):
        """Test pagination params validation."""
        # Negative offset
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)
        
        # Zero limit
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)
        
        # Limit too high
        with pytest.raises(ValidationError):
            PaginationParams(limit=101)

    def test_pagination_meta_create(self):
        """Test creating pagination metadata."""
        meta = PaginationMeta.create(offset=20, limit=10, total=100)
        
        assert meta.offset == 20
        assert meta.limit == 10
        assert meta.total == 100
        assert meta.has_next is True
        assert meta.has_previous is True

    def test_pagination_meta_boundaries(self):
        """Test pagination metadata at boundaries."""
        # First page
        meta = PaginationMeta.create(offset=0, limit=10, total=50)
        assert meta.has_previous is False
        assert meta.has_next is True
        
        # Last page
        meta = PaginationMeta.create(offset=40, limit=10, total=50)
        assert meta.has_previous is True
        assert meta.has_next is False
        
        # Single page
        meta = PaginationMeta.create(offset=0, limit=10, total=5)
        assert meta.has_previous is False
        assert meta.has_next is False


class TestResponseWrappers:
    """Test cases for response wrapper schemas."""

    def test_success_response(self):
        """Test success response wrapper."""
        response = SuccessResponse(
            data={"id": 123, "name": "Test"},
            message="Operation successful",
            request_id="req-123"
        )
        
        assert response.success is True
        assert response.data == {"id": 123, "name": "Test"}
        assert response.message == "Operation successful"
        assert response.request_id == "req-123"

    def test_paginated_response(self):
        """Test paginated response wrapper."""
        meta = PaginationMeta.create(offset=0, limit=2, total=5)
        response = PaginatedResponse(
            data=[{"id": 1}, {"id": 2}],
            pagination=meta,
            message="Retrieved items",
            request_id="req-456"
        )
        
        assert response.success is True
        assert len(response.data) == 2
        assert response.pagination.total == 5
        assert response.pagination.has_next is True

    def test_error_response(self):
        """Test error response format."""
        error_detail = ErrorDetail(
            code="AUTH_001",
            message="Invalid API key",
            status_code=401,
            details={"key": "test-key"}
        )
        
        response = ErrorResponse(
            error=error_detail,
            request_id="req-789"
        )
        
        assert response.success is False
        assert response.error.code == "AUTH_001"
        assert response.error.status_code == 401
        assert response.error.details["key"] == "test-key"

    def test_validation_error_response(self):
        """Test validation error response."""
        field_error = ValidationErrorDetail(
            message="Invalid email format",
            type="email_validation",
            input="not-an-email"
        )
        
        error_detail = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            status_code=422
        )
        
        response = ValidationErrorResponse(
            error=error_detail,
            field_errors={"email": field_error},
            request_id="req-val-123"
        )
        
        assert response.success is False
        assert response.error.status_code == 422
        assert "email" in response.field_errors
        assert response.field_errors["email"].type == "email_validation"


class TestStatusResponses:
    """Test cases for status response schemas."""

    def test_status_response(self):
        """Test simple status response."""
        now = datetime.utcnow()
        response = StatusResponse(
            status="Service operational",
            timestamp=now,
            request_id="req-status-123"
        )
        
        assert response.status == "Service operational"
        assert response.timestamp == now
        assert response.request_id == "req-status-123"

    def test_health_check_response(self):
        """Test health check response."""
        now = datetime.utcnow()
        services = {
            "database": {"status": "healthy", "response_time_ms": 5.5},
            "redis": {"status": "healthy", "response_time_ms": 2.1},
            "storage": {"status": "degraded", "response_time_ms": 100.0}
        }
        
        response = HealthCheckResponse(
            status="degraded",
            version="1.0.0",
            timestamp=now,
            services=services
        )
        
        assert response.status == "degraded"
        assert response.version == "1.0.0"
        assert len(response.services) == 3
        assert response.services["database"]["status"] == "healthy"
        assert response.services["storage"]["status"] == "degraded"


class TestOperationResponses:
    """Test cases for operation response schemas."""

    def test_id_response(self):
        """Test ID response with different ID types."""
        # String ID
        response = IdResponse(id="uuid-123", message="Resource created")
        assert response.id == "uuid-123"
        assert response.message == "Resource created"
        
        # Integer ID
        response = IdResponse(id=12345)
        assert response.id == 12345
        assert response.message is None

    def test_bulk_operation_response(self):
        """Test bulk operation response."""
        errors = [
            {"id": "item-1", "error": "Invalid format"},
            {"id": "item-2", "error": "Duplicate entry"}
        ]
        
        response = BulkOperationResponse(
            success=False,
            total=10,
            successful=8,
            failed=2,
            errors=errors,
            request_id="bulk-123"
        )
        
        assert response.success is False
        assert response.total == 10
        assert response.successful == 8
        assert response.failed == 2
        assert len(response.errors) == 2


class TestSearchAndFilterSchemas:
    """Test cases for search and filter schemas."""

    def test_search_params_defaults(self):
        """Test search params with defaults."""
        params = SearchParams()
        assert params.query is None
        assert params.sort_by == "created_at"
        assert params.sort_order == "desc"

    def test_search_params_validation(self):
        """Test search params validation."""
        # Valid params
        params = SearchParams(
            query="test search",
            sort_by="name",
            sort_order="asc"
        )
        assert params.query == "test search"
        
        # Invalid sort order
        with pytest.raises(ValidationError):
            SearchParams(sort_order="invalid")
        
        # Empty query after stripping
        params = SearchParams(query="   ")
        assert params.query is None

    def test_date_range_filter(self):
        """Test date range filter validation."""
        start = datetime.utcnow()
        end = start + timedelta(days=7)
        
        # Valid range
        filter_params = DateRangeFilter(start_date=start, end_date=end)
        assert filter_params.start_date == start
        assert filter_params.end_date == end
        
        # Invalid range (end before start)
        with pytest.raises(ValidationError) as exc_info:
            DateRangeFilter(start_date=end, end_date=start)
        assert "end_date must be after start_date" in str(exc_info.value)

    def test_filter_params(self):
        """Test combined filter params."""
        params = FilterParams(
            query="search term",
            tags=["tag1", "tag2", "  tag3  "],
            status="active",
            start_date=datetime.utcnow()
        )
        
        assert params.query == "search term"
        assert params.tags == ["tag1", "tag2", "tag3"]  # Trimmed
        assert params.status == "active"

    def test_filter_params_tag_limits(self):
        """Test filter params tag validation and limits."""
        # Too many tags - should be limited to 10
        many_tags = [f"tag{i}" for i in range(20)]
        params = FilterParams(tags=many_tags)
        assert len(params.tags) == 10
        
        # Long tag names - should be truncated
        long_tag = "x" * 100
        params = FilterParams(tags=[long_tag])
        assert len(params.tags[0]) == 50
        
        # Empty tags should be removed
        params = FilterParams(tags=["valid", "", "  ", "another"])
        assert params.tags == ["valid", "another"]


class TestWebhookAndUsageSchemas:
    """Test cases for webhook and usage schemas."""

    def test_webhook_payload(self):
        """Test webhook payload structure."""
        now = datetime.utcnow()
        payload = WebhookPayload(
            event="highlight.created",
            timestamp=now,
            data={"highlight_id": "123", "stream_id": "456"},
            webhook_id="webhook-789",
            delivery_id="delivery-abc"
        )
        
        assert payload.event == "highlight.created"
        assert payload.timestamp == now
        assert payload.data["highlight_id"] == "123"
        assert payload.webhook_id == "webhook-789"
        assert payload.delivery_id == "delivery-abc"

    def test_api_usage_stats(self):
        """Test API usage statistics."""
        stats = APIUsageStats(
            requests_count=1000,
            requests_per_minute=16.67,
            requests_per_hour=1000.0,
            error_rate=2.5,
            avg_response_time_ms=125.5
        )
        
        assert stats.requests_count == 1000
        assert stats.error_rate == 2.5
        
        # Error rate validation (clamped to 0-100)
        stats = APIUsageStats(
            requests_count=100,
            requests_per_minute=1.67,
            requests_per_hour=100.0,
            error_rate=150.0,  # Should be clamped to 100
            avg_response_time_ms=50.0
        )
        assert stats.error_rate == 100.0
        
        # Negative error rate
        stats = APIUsageStats(
            requests_count=100,
            requests_per_minute=1.67,
            requests_per_hour=100.0,
            error_rate=-10.0,  # Should be clamped to 0
            avg_response_time_ms=50.0
        )
        assert stats.error_rate == 0.0

    def test_resource_quota(self):
        """Test resource quota information."""
        reset_time = datetime.utcnow() + timedelta(hours=1)
        
        # Quota with manual remaining
        quota = ResourceQuota(
            used=75,
            limit=100,
            remaining=25,
            reset_at=reset_time
        )
        
        assert quota.used == 75
        assert quota.limit == 100
        assert quota.remaining == 25
        assert quota.reset_at == reset_time
        
        # Quota with calculated remaining
        quota = ResourceQuota(
            used=80,
            limit=100,
            remaining=50  # Should be recalculated to 20
        )
        assert quota.remaining == 20
        
        # Over quota
        quota = ResourceQuota(
            used=120,
            limit=100,
            remaining=0  # Should be 0, not negative
        )
        assert quota.remaining == 0


class TestSchemaIntegration:
    """Test schema integration and complex scenarios."""

    def test_nested_response_serialization(self):
        """Test serializing nested response structures."""
        # Create a complex paginated response
        items = [
            {"id": i, "name": f"Item {i}", "created_at": datetime.utcnow()}
            for i in range(3)
        ]
        
        meta = PaginationMeta.create(offset=0, limit=10, total=3)
        response = PaginatedResponse(
            data=items,
            pagination=meta,
            message="Items retrieved successfully",
            request_id="test-request"
        )
        
        # Convert to dict (simulating JSON serialization)
        response_dict = response.model_dump()
        
        assert response_dict["success"] is True
        assert len(response_dict["data"]) == 3
        assert response_dict["pagination"]["total"] == 3
        assert response_dict["message"] == "Items retrieved successfully"

    def test_error_response_serialization(self):
        """Test error response serialization with validation errors."""
        field_errors = {
            "email": ValidationErrorDetail(
                message="Invalid email format",
                type="email",
                input="bad-email"
            ),
            "age": ValidationErrorDetail(
                message="Must be at least 18",
                type="min_value",
                input=16
            )
        }
        
        error = ErrorDetail(
            code="VALIDATION_FAILED",
            message="Multiple validation errors",
            status_code=422,
            details={"fields_count": 2}
        )
        
        response = ValidationErrorResponse(
            error=error,
            field_errors=field_errors
        )
        
        # Serialize to dict
        response_dict = response.model_dump()
        
        assert response_dict["success"] is False
        assert response_dict["error"]["code"] == "VALIDATION_FAILED"
        assert len(response_dict["field_errors"]) == 2
        assert response_dict["field_errors"]["email"]["type"] == "email"