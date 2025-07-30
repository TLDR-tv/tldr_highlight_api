"""Tests for Logfire metrics collection."""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime

from src.infrastructure.observability.logfire_metrics import MetricsCollector


class TestMetricsCollector:
    """Test the MetricsCollector class."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a MetricsCollector instance for testing."""
        with patch("src.infrastructure.observability.logfire_metrics.logfire"):
            collector = MetricsCollector()
            yield collector

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_api_calls(self, mock_logfire, metrics_collector):
        """Test incrementing API call metrics."""
        # Execute
        metrics_collector.increment_api_calls(
            endpoint="/api/v1/streams",
            method="POST",
            status_code=201,
            organization_id="org-123"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.api.calls",
            metric_name="api.calls",
            value=1,
            metric_type="counter",
            endpoint="/api/v1/streams",
            method="POST",
            status_code=201,
            organization_id="org-123"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_api_latency(self, mock_logfire, metrics_collector):
        """Test recording API latency metrics."""
        # Execute
        metrics_collector.record_api_latency(
            endpoint="/api/v1/highlights",
            method="GET",
            duration_ms=125.5,
            organization_id="org-456"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.api.latency",
            metric_name="api.latency",
            value=125.5,
            unit="milliseconds",
            metric_type="histogram",
            endpoint="/api/v1/highlights",
            method="GET",
            organization_id="org-456"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_stream_started(self, mock_logfire, metrics_collector):
        """Test incrementing stream started metrics."""
        # Execute
        metrics_collector.increment_stream_started(
            platform="twitch",
            organization_id="org-789",
            stream_type="live"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.streams.started",
            metric_name="streams.started",
            value=1,
            metric_type="counter",
            platform="twitch",
            organization_id="org-789",
            stream_type="live"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_stream_completed(self, mock_logfire, metrics_collector):
        """Test incrementing stream completed metrics."""
        # Execute
        metrics_collector.increment_stream_completed(
            platform="youtube",
            organization_id="org-111",
            stream_type="vod",
            success=True
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.streams.completed",
            metric_name="streams.completed",
            value=1,
            metric_type="counter",
            platform="youtube",
            organization_id="org-111",
            stream_type="vod",
            success=True
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_stream_duration(self, mock_logfire, metrics_collector):
        """Test recording stream duration metrics."""
        # Execute
        metrics_collector.record_stream_duration(
            duration_minutes=45.5,
            platform="rtmp",
            highlights_count=12
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.stream.duration",
            metric_name="stream.duration",
            value=45.5,
            unit="minutes",
            metric_type="histogram",
            platform="rtmp",
            highlights_count=12
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_highlights_detected(self, mock_logfire, metrics_collector):
        """Test incrementing highlights detected metrics."""
        # Execute
        metrics_collector.increment_highlights_detected(
            count=5,
            platform="twitch",
            organization_id="org-222",
            detection_method="ai_agent"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.highlights.detected",
            metric_name="highlights.detected",
            value=5,
            metric_type="counter",
            platform="twitch",
            organization_id="org-222",
            detection_method="ai_agent"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_highlight_confidence(self, mock_logfire, metrics_collector):
        """Test recording highlight confidence metrics."""
        # Execute
        metrics_collector.record_highlight_confidence(
            confidence=0.92,
            detection_method="b2b_agent",
            platform="youtube"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.highlight.confidence",
            metric_name="highlight.confidence",
            value=0.92,
            metric_type="histogram",
            detection_method="b2b_agent",
            platform="youtube"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_highlight_processing_time(self, mock_logfire, metrics_collector):
        """Test recording highlight processing time metrics."""
        # Execute
        metrics_collector.record_highlight_processing_time(
            duration_seconds=3.75,
            stage="ai_analysis",
            platform="twitch"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.highlight.processing.time",
            metric_name="highlight.processing.time",
            value=3.75,
            unit="seconds",
            metric_type="histogram",
            stage="ai_analysis",
            platform="twitch"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_task_executed(self, mock_logfire, metrics_collector):
        """Test incrementing task executed metrics."""
        # Execute
        metrics_collector.increment_task_executed(
            task_name="detect_highlights_with_ai",
            organization_id="org-333",
            success=True
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.celery.task.executed",
            metric_name="celery.task.executed",
            value=1,
            metric_type="counter",
            task_name="detect_highlights_with_ai",
            organization_id="org-333",
            success=True
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_task_duration(self, mock_logfire, metrics_collector):
        """Test recording task duration metrics."""
        # Execute
        metrics_collector.record_task_duration(
            duration_seconds=15.25,
            task_name="ingest_stream_with_ffmpeg",
            organization_id="org-444"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.celery.task.duration",
            metric_name="celery.task.duration",
            value=15.25,
            unit="seconds",
            metric_type="histogram",
            task_name="ingest_stream_with_ffmpeg",
            organization_id="org-444"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_gauge_active_streams(self, mock_logfire, metrics_collector):
        """Test setting active streams gauge."""
        # Execute
        metrics_collector.gauge_active_streams(
            count=42,
            platform="twitch",
            organization_id="org-555"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.streams.active",
            metric_name="streams.active",
            value=42,
            metric_type="gauge",
            platform="twitch",
            organization_id="org-555"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_gauge_queue_size(self, mock_logfire, metrics_collector):
        """Test setting queue size gauge."""
        # Execute
        metrics_collector.gauge_queue_size(
            queue_name="celery",
            size=150
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.queue.size",
            metric_name="queue.size",
            value=150,
            metric_type="gauge",
            queue_name="celery"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_database_query_time(self, mock_logfire, metrics_collector):
        """Test recording database query time."""
        # Execute
        metrics_collector.record_database_query_time(
            duration_ms=25.5,
            operation="select",
            table="streams"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.db.query.time",
            metric_name="db.query.time",
            value=25.5,
            unit="milliseconds",
            metric_type="histogram",
            operation="select",
            table="streams"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_redis_operation_time(self, mock_logfire, metrics_collector):
        """Test recording Redis operation time."""
        # Execute
        metrics_collector.record_redis_operation_time(
            duration_ms=2.5,
            operation="get",
            key_pattern="stream:*"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.redis.operation.time",
            metric_name="redis.operation.time",
            value=2.5,
            unit="milliseconds",
            metric_type="histogram",
            operation="get",
            key_pattern="stream:*"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_stream_cost(self, mock_logfire, metrics_collector):
        """Test recording stream cost metrics."""
        # Execute
        metrics_collector.record_stream_cost(
            cost=12.50,
            duration_minutes=45,
            platform="youtube"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.stream.cost",
            metric_name="stream.cost",
            value=12.50,
            unit="dollars",
            metric_type="histogram",
            duration_minutes=45,
            platform="youtube"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_batch_processing_stats(self, mock_logfire, metrics_collector):
        """Test recording batch processing statistics."""
        # Execute
        metrics_collector.record_batch_processing_stats(
            batch_id="batch-123",
            total_items=100,
            processed_items=95,
            failed_items=5,
            duration_seconds=300
        )

        # Verify
        expected_calls = [
            call(
                "metric.batch.size",
                metric_name="batch.size",
                value=100,
                metric_type="gauge",
                batch_id="batch-123"
            ),
            call(
                "metric.batch.processed",
                metric_name="batch.processed",
                value=95,
                metric_type="counter",
                batch_id="batch-123"
            ),
            call(
                "metric.batch.failed",
                metric_name="batch.failed",
                value=5,
                metric_type="counter",
                batch_id="batch-123"
            ),
            call(
                "metric.batch.duration",
                metric_name="batch.duration",
                value=300,
                unit="seconds",
                metric_type="histogram",
                batch_id="batch-123"
            ),
            call(
                "metric.batch.success_rate",
                metric_name="batch.success_rate",
                value=0.95,
                metric_type="gauge",
                batch_id="batch-123"
            )
        ]
        mock_logfire.info.assert_has_calls(expected_calls)

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_increment_error_count(self, mock_logfire, metrics_collector):
        """Test incrementing error count metrics."""
        # Execute
        metrics_collector.increment_error_count(
            error_type="ValueError",
            service="stream_processor",
            severity="error"
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.errors.count",
            metric_name="errors.count",
            value=1,
            metric_type="counter",
            error_type="ValueError",
            service="stream_processor",
            severity="error"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_record_webhook_delivery(self, mock_logfire, metrics_collector):
        """Test recording webhook delivery metrics."""
        # Execute
        metrics_collector.record_webhook_delivery(
            event_type="stream.started",
            status_code=200,
            duration_ms=150,
            organization_id="org-666"
        )

        # Verify
        expected_calls = [
            call(
                "metric.webhook.delivery",
                metric_name="webhook.delivery",
                value=1,
                metric_type="counter",
                event_type="stream.started",
                status_code=200,
                organization_id="org-666"
            ),
            call(
                "metric.webhook.delivery.time",
                metric_name="webhook.delivery.time",
                value=150,
                unit="milliseconds",
                metric_type="histogram",
                event_type="stream.started",
                organization_id="org-666"
            )
        ]
        mock_logfire.info.assert_has_calls(expected_calls)

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_gauge_memory_usage(self, mock_logfire, metrics_collector):
        """Test setting memory usage gauge."""
        # Execute
        metrics_collector.gauge_memory_usage(
            service_name="api",
            memory_mb=512.5
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.memory.usage",
            metric_name="memory.usage",
            value=512.5,
            unit="megabytes",
            metric_type="gauge",
            service_name="api"
        )

    @patch("src.infrastructure.observability.logfire_metrics.logfire")
    def test_gauge_cpu_usage(self, mock_logfire, metrics_collector):
        """Test setting CPU usage gauge."""
        # Execute
        metrics_collector.gauge_cpu_usage(
            service_name="worker",
            cpu_percent=75.5
        )

        # Verify
        mock_logfire.info.assert_called_once_with(
            "metric.cpu.usage",
            metric_name="cpu.usage",
            value=75.5,
            unit="percent",
            metric_type="gauge",
            service_name="worker"
        )