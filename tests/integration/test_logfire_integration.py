"""Integration tests for Logfire observability."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.infrastructure.observability import (
    configure_logfire,
    traced_service_method,
    traced_use_case,
    metrics,
)
from src.domain.services.stream_processing_service import StreamProcessingService
from src.application.use_cases.stream_processing import (
    StreamProcessingUseCase,
    StreamStartRequest,
)


class TestLogfireIntegration:
    """Test end-to-end Logfire integration."""

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    @pytest.mark.asyncio
    async def test_stream_processing_full_trace(
        self, mock_metrics_logfire, mock_decorators_logfire, mock_setup_logfire
    ):
        """Test full trace from use case to service with metrics."""
        # Setup mocks
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_span.set_attribute = Mock()

        # Configure all logfire mocks to return the same span
        mock_setup_logfire.span.return_value = mock_span
        mock_decorators_logfire.span.return_value = mock_span
        mock_metrics_logfire.info = Mock()

        # Create mocked repositories and services
        mock_user_repo = Mock()
        mock_user_repo.get = AsyncMock(return_value=Mock(id=1, name="Test User"))

        mock_stream_repo = Mock()
        mock_stream_repo.save = AsyncMock(
            return_value=Mock(
                id=123,
                user_id=1,
                platform=Mock(value="twitch"),
                status=Mock(value="processing"),
                url=Mock(value="https://twitch.tv/test"),
                organization_id=1,
            )
        )

        mock_org_repo = Mock()
        mock_org_repo.get_by_owner = AsyncMock(
            return_value=[Mock(id=1, name="Test Org")]
        )

        mock_usage_repo = Mock()
        mock_usage_repo.save = AsyncMock()

        mock_highlight_repo = Mock()
        mock_agent_config_repo = Mock()

        # Create service instance
        stream_service = StreamProcessingService(
            stream_repo=mock_stream_repo,
            user_repo=mock_user_repo,
            org_repo=mock_org_repo,
            usage_repo=mock_usage_repo,
            highlight_repo=mock_highlight_repo,
            agent_config_repo=mock_agent_config_repo,
        )

        # Mock the async task triggering
        with patch(
            "src.domain.services.stream_processing_service.ingest_stream_with_ffmpeg"
        ) as mock_task:
            mock_task.delay.return_value = Mock(id="task-123")

            # Create use case instance
            use_case = StreamProcessingUseCase(
                user_repo=mock_user_repo,
                stream_repo=mock_stream_repo,
                highlight_repo=mock_highlight_repo,
                agent_config_repo=mock_agent_config_repo,
                stream_service=stream_service,
                highlight_service=Mock(),
                webhook_service=Mock(trigger_event=AsyncMock()),
                usage_service=Mock(track_api_call=AsyncMock()),
            )

            # Execute the use case
            request = StreamStartRequest(
                user_id=1,
                url="https://twitch.tv/test",
                title="Test Stream",
                agent_config_id=None,
            )

            result = await use_case.start_stream(request)

            # Verify the result
            assert result.stream_id == 123
            assert result.stream_status == "processing"

            # Verify spans were created
            assert (
                mock_decorators_logfire.span.call_count >= 2
            )  # At least use case and service

            # Verify metrics were recorded
            metric_calls = mock_metrics_logfire.info.call_args_list

            # Should have recorded stream started metric
            stream_started_calls = [
                c for c in metric_calls if "metric.streams.started" in str(c)
            ]
            assert len(stream_started_calls) > 0

            # Should have recorded API call metric
            api_call_calls = [c for c in metric_calls if "metric.api.calls" in str(c)]
            assert len(api_call_calls) > 0

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @pytest.mark.asyncio
    async def test_nested_spans_hierarchy(self, mock_logfire):
        """Test that nested spans maintain proper hierarchy."""
        # Track span enter/exit order
        span_stack = []

        class MockSpan:
            def __init__(self, name):
                self.name = name
                self.attributes = {}

            async def __aenter__(self):
                span_stack.append(f"enter:{self.name}")
                return self

            async def __aexit__(self, *args):
                span_stack.append(f"exit:{self.name}")

            def set_attribute(self, key, value):
                self.attributes[key] = value

        # Configure mock to return different spans
        def create_span(name, **kwargs):
            return MockSpan(name)

        mock_logfire.span.side_effect = create_span

        # Create nested traced functions
        @traced_service_method(name="outer")
        async def outer_function():
            with mock_logfire.span("inner_span") as span:
                span.set_attribute("test", "value")
                await inner_function()
            return "done"

        @traced_service_method(name="inner")
        async def inner_function():
            await asyncio.sleep(0.01)
            return "inner_done"

        # Execute
        result = await outer_function()

        # Verify proper nesting
        assert result == "done"
        assert span_stack == [
            "enter:outer",
            "enter:inner_span",
            "enter:inner",
            "exit:inner",
            "exit:inner_span",
            "exit:outer",
        ]

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_metrics_collector_thread_safety(self, mock_logfire):
        """Test metrics collector is thread-safe."""
        import threading

        mock_logfire.info = Mock()
        collector = metrics()

        # Track calls
        call_count = {"api": 0, "stream": 0}

        def increment_api_calls():
            for _ in range(100):
                collector.increment_api_calls("/api/v1/streams", "POST", 200, "org-1")
                call_count["api"] += 1

        def increment_stream_started():
            for _ in range(100):
                collector.increment_stream_started("twitch", "org-1", "live")
                call_count["stream"] += 1

        # Run in threads
        threads = [
            threading.Thread(target=increment_api_calls),
            threading.Thread(target=increment_stream_started),
            threading.Thread(target=increment_api_calls),
            threading.Thread(target=increment_stream_started),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify all calls were made
        assert call_count["api"] == 200
        assert call_count["stream"] == 200

        # Verify logfire was called correctly
        assert mock_logfire.info.call_count == 400

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @pytest.mark.asyncio
    async def test_error_propagation_through_traces(self, mock_logfire):
        """Test that errors are properly propagated through traces."""
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        mock_logfire.span.return_value = mock_span

        error_recorded = []

        def record_error(*args, **kwargs):
            if args[0] == "error":
                error_recorded.append(kwargs)

        mock_span.set_attribute.side_effect = record_error

        @traced_use_case(name="failing_use_case")
        async def failing_use_case():
            await failing_service()

        @traced_service_method(name="failing_service")
        async def failing_service():
            raise ValueError("Service failure")

        # Execute and verify error propagation
        with pytest.raises(ValueError, match="Service failure"):
            await failing_use_case()

        # Verify error was recorded in spans
        assert len(error_recorded) >= 2  # Both use case and service spans

    @patch("src.infrastructure.observability.logfire_setup.logfire")
    def test_logfire_configuration_with_environment(self, mock_logfire):
        """Test Logfire configuration respects environment settings."""
        with patch(
            "src.infrastructure.observability.logfire_setup.settings"
        ) as mock_settings:
            # Configure settings
            mock_settings.logfire_enabled = True
            mock_settings.logfire_service_name = "test-api"
            mock_settings.logfire_version = "2.0.0"
            mock_settings.logfire_env = "production"
            mock_settings.logfire_console_enabled = False
            mock_settings.logfire_log_level = "WARNING"
            mock_settings.logfire_api_key = Mock(get_secret_value=lambda: "prod-key")
            mock_settings.logfire_capture_headers = False
            mock_settings.logfire_capture_body = False
            mock_settings.logfire_sql_enabled = False
            mock_settings.logfire_redis_enabled = False
            mock_settings.logfire_celery_enabled = False
            mock_settings.environment = "production"
            mock_settings.app_version = "2.0.0"
            mock_settings.is_development = False

            # Configure
            configure_logfire()

            # Verify production configuration
            mock_logfire.configure.assert_called_once()
            config = mock_logfire.configure.call_args[1]

            assert config["service_name"] == "test-api"
            assert config["environment"] == "production"
            assert config["console"] is False
            assert config["token"] == "prod-key"
            assert config["log_level"] == "WARNING"

            # Verify integrations were not enabled
            mock_logfire.instrument_sqlalchemy.assert_not_called()
            mock_logfire.instrument_redis.assert_not_called()
            mock_logfire.instrument_celery.assert_not_called()

    @patch("src.infrastructure.observability.logfire_decorators.logfire")
    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    @pytest.mark.asyncio
    async def test_complete_stream_lifecycle_observability(
        self, mock_metrics_logfire, mock_decorators_logfire
    ):
        """Test observability through complete stream lifecycle."""
        # Setup spans
        spans_created = []

        class TrackingSpan:
            def __init__(self, name):
                self.name = name
                self.attributes = {}
                spans_created.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def set_attribute(self, key, value):
                self.attributes[key] = value

        mock_decorators_logfire.span.side_effect = lambda name, **kwargs: TrackingSpan(
            name
        )
        mock_metrics_logfire.info = Mock()

        # Simulate stream lifecycle
        @traced_use_case(name="stream_lifecycle")
        async def simulate_stream_lifecycle():
            # Start stream
            stream_id = await start_stream()

            # Process stream
            await process_stream(stream_id)

            # Complete stream
            await complete_stream(stream_id)

            return stream_id

        @traced_service_method(name="start_stream")
        async def start_stream():
            metrics().increment_stream_started("twitch", "org-1", "live")
            return 123

        @traced_service_method(name="process_stream")
        async def process_stream(stream_id):
            metrics().increment_highlights_detected(5, "twitch", "org-1", "ai_agent")
            await asyncio.sleep(0.01)

        @traced_service_method(name="complete_stream")
        async def complete_stream(stream_id):
            metrics().increment_stream_completed("twitch", "org-1", "live", True)
            metrics().record_stream_duration(45.5, "twitch", 5)

        # Execute
        await simulate_stream_lifecycle()

        # Verify spans were created
        assert len(spans_created) == 4  # lifecycle + 3 operations
        span_names = [s.name for s in spans_created]
        assert "stream_lifecycle" in span_names
        assert "start_stream" in span_names
        assert "process_stream" in span_names
        assert "complete_stream" in span_names

        # Verify metrics were recorded
        metric_names = [call[0][0] for call in mock_metrics_logfire.info.call_args_list]
        assert "metric.streams.started" in metric_names
        assert "metric.highlights.detected" in metric_names
        assert "metric.streams.completed" in metric_names
        assert "metric.stream.duration" in metric_names
