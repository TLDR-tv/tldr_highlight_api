"""Tests for Logfire setup and configuration."""

import pytest
from unittest.mock import Mock, patch, call

from src.infrastructure.observability.logfire_setup import (
    configure_logfire,
    get_logfire,
    create_span,
    log_metric,
    log_event,
    set_correlation_id,
    add_user_context,
    add_processing_context,
)


class TestLogfireSetup:
    """Test Logfire configuration and setup."""

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_setup.settings")
    def test_configure_logfire_enabled(self, mock_settings, mock_logfire):
        """Test Logfire configuration when enabled."""
        # Setup
        mock_settings.logfire_enabled = True
        mock_settings.logfire_service_name = "test-service"
        mock_settings.logfire_version = "1.0.0"
        mock_settings.logfire_env = "test"
        mock_settings.logfire_console_enabled = True
        mock_settings.logfire_log_level = "INFO"
        mock_settings.logfire_api_key = None
        mock_settings.logfire_capture_headers = True
        mock_settings.logfire_capture_body = False
        mock_settings.logfire_sql_enabled = True
        mock_settings.logfire_redis_enabled = True
        mock_settings.logfire_celery_enabled = True
        mock_settings.environment = "test"
        mock_settings.app_version = "1.0.0"
        mock_settings.is_development = False

        mock_app = Mock()

        # Execute
        configure_logfire(mock_app)

        # Verify
        mock_logfire.configure.assert_called_once_with(
            service_name="test-service",
            service_version="1.0.0",
            environment="test",
            console=True,
            log_level="INFO",
        )

        # Verify integrations
        mock_logfire.instrument_fastapi.assert_called_once_with(
            mock_app,
            capture_headers=True,
            capture_body=False,
        )
        mock_logfire.instrument_sqlalchemy.assert_called_once()
        mock_logfire.instrument_redis.assert_called_once()
        mock_logfire.instrument_celery.assert_called_once()
        mock_logfire.instrument_httpx.assert_called_once()

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_setup.settings")
    def test_configure_logfire_disabled(self, mock_settings, mock_logfire):
        """Test Logfire configuration when disabled."""
        # Setup
        mock_settings.logfire_enabled = False

        # Execute
        configure_logfire()

        # Verify
        mock_logfire.configure.assert_not_called()

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_setup.settings")
    def test_configure_logfire_with_api_key(self, mock_settings, mock_logfire):
        """Test Logfire configuration with API key."""
        # Setup
        mock_settings.logfire_enabled = True
        mock_settings.logfire_service_name = "test-service"
        mock_settings.logfire_version = "1.0.0"
        mock_settings.logfire_env = "prod"
        mock_settings.logfire_console_enabled = False
        mock_settings.logfire_log_level = "INFO"

        mock_api_key = Mock()
        mock_api_key.get_secret_value.return_value = "secret-api-key"
        mock_settings.logfire_api_key = mock_api_key

        # Execute
        configure_logfire()

        # Verify
        expected_config = {
            "service_name": "test-service",
            "service_version": "1.0.0",
            "environment": "prod",
            "console": False,
            "log_level": "INFO",
            "token": "secret-api-key",
        }
        mock_logfire.configure.assert_called_once_with(**expected_config)

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_get_logfire(self, mock_logfire):
        """Test getting Logfire instance."""
        result = get_logfire()
        assert result == mock_logfire

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_create_span(self, mock_logfire):
        """Test creating a span."""
        mock_span = Mock()
        mock_logfire.span.return_value = mock_span

        # Execute
        result = create_span("test_span", span_type="custom", user_id=123)

        # Verify
        mock_logfire.span.assert_called_once_with(
            "test_span", _span_type="custom", user_id=123
        )
        assert result == mock_span

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_log_metric(self, mock_logfire):
        """Test logging a metric."""
        # Execute
        log_metric(
            "request_count", 42, unit="requests", tags={"endpoint": "/api/v1/streams"}
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.request_count",
            metric_name="request_count",
            value=42,
            unit="requests",
            endpoint="/api/v1/streams",
        )

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_log_event(self, mock_logfire):
        """Test logging an event."""
        # Execute
        log_event("stream_started", "processing", stream_id=123, platform="twitch")

        # Verify
        mock_logfire.info.assert_called_once_with(
            "event.processing.stream_started",
            event_type="processing",
            event_name="stream_started",
            stream_id=123,
            platform="twitch",
        )

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_set_correlation_id(self, mock_logfire):
        """Test setting correlation ID."""
        mock_logfire.with_tags.return_value = "tagged_context"

        # Execute
        result = set_correlation_id("test-correlation-123")

        # Verify
        mock_logfire.with_tags.assert_called_once_with(
            correlation_id="test-correlation-123"
        )
        assert result == "tagged_context"

    def test_add_user_context(self):
        """Test adding user context to span."""
        mock_span = Mock()

        # Execute
        add_user_context(
            mock_span,
            organization_id="org-123",
            user_id="user-456",
            api_key_id="key-789",
        )

        # Verify
        mock_span.set_attribute.assert_has_calls(
            [
                call("user.organization_id", "org-123"),
                call("user.id", "user-456"),
                call("user.api_key_id", "key-789"),
            ]
        )

    def test_add_processing_context(self):
        """Test adding processing context to span."""
        mock_span = Mock()

        # Execute
        add_processing_context(
            mock_span,
            stream_id="stream-123",
            batch_id="batch-456",
            platform="youtube",
            processing_stage="ingestion",
        )

        # Verify
        mock_span.set_attribute.assert_has_calls(
            [
                call("processing.stream_id", "stream-123"),
                call("processing.batch_id", "batch-456"),
                call("processing.platform", "youtube"),
                call("processing.stage", "ingestion"),
            ]
        )

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_setup.settings")
    def test_configure_logfire_integration_failures(self, mock_settings, mock_logfire):
        """Test handling of integration failures."""
        # Setup
        mock_settings.logfire_enabled = True
        mock_settings.logfire_service_name = "test-service"
        mock_settings.logfire_version = "1.0.0"
        mock_settings.logfire_env = "test"
        mock_settings.logfire_console_enabled = True
        mock_settings.logfire_log_level = "INFO"
        mock_settings.logfire_api_key = None
        mock_settings.logfire_capture_headers = True
        mock_settings.logfire_capture_body = False
        mock_settings.logfire_sql_enabled = True
        mock_settings.logfire_redis_enabled = True
        mock_settings.logfire_celery_enabled = True
        mock_settings.is_development = True

        # Make integrations fail
        mock_logfire.instrument_fastapi.side_effect = Exception("FastAPI failed")
        mock_logfire.instrument_sqlalchemy.side_effect = Exception("SQLAlchemy failed")

        mock_app = Mock()

        # Execute - should not raise
        configure_logfire(mock_app)

        # Verify configure was called
        mock_logfire.configure.assert_called_once()

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_setup.settings")
    def test_configure_logfire_complete_failure_production(
        self, mock_settings, mock_logfire
    ):
        """Test complete configuration failure in production."""
        # Setup
        mock_settings.logfire_enabled = True
        mock_settings.is_development = False
        mock_logfire.configure.side_effect = Exception("Configuration failed")

        # Execute and verify exception is raised
        with pytest.raises(Exception, match="Configuration failed"):
            configure_logfire()
