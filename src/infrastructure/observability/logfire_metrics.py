"""Custom metrics collection for TL;DR Highlight API using Logfire.

This module provides a centralized metrics collection system for tracking
business metrics, performance indicators, and system health.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from threading import Lock

import logfire


class MetricsCollector:
    """Centralized metrics collector for the application."""

    def __init__(self):
        """Initialize the metrics collector."""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
        self._metric_callbacks: Dict[str, Callable] = {}
        self._background_task: Optional[asyncio.Task] = None

    # Stream Processing Metrics

    def increment_stream_started(
        self, platform: str, organization_id: str, stream_type: str = "live"
    ) -> None:
        """Increment stream started counter."""
        self._increment_counter(
            "streams_started_total",
            tags={
                "platform": platform,
                "organization_id": organization_id,
                "stream_type": stream_type,
            },
        )
        
        # Send to Logfire
        logfire.info(
            "Stream started",
            platform=platform,
            organization_id=organization_id,
            stream_type=stream_type,
        )

    def increment_stream_completed(
        self,
        platform: str,
        organization_id: str,
        stream_type: str = "live",
        success: bool = True,
    ) -> None:
        """Increment stream completed counter."""
        status = "success" if success else "failed"
        self._increment_counter(
            "streams_completed_total",
            tags={
                "platform": platform,
                "organization_id": organization_id,
                "stream_type": stream_type,
                "status": status,
            },
        )

    def record_stream_duration(
        self, duration_seconds: float, platform: str, organization_id: str
    ) -> None:
        """Record stream processing duration."""
        self._record_histogram(
            "stream_duration_seconds",
            duration_seconds,
            tags={
                "platform": platform,
                "organization_id": organization_id,
            },
        )

    def set_active_streams(self, count: int, platform: Optional[str] = None) -> None:
        """Set the number of active streams."""
        tags = {"platform": platform} if platform else {}
        self._set_gauge("active_streams", count, tags)

    # Highlight Detection Metrics

    def increment_highlights_detected(
        self, count: int, platform: str, organization_id: str, detection_method: str
    ) -> None:
        """Increment highlights detected counter."""
        self._increment_counter(
            "highlights_detected_total",
            value=count,
            tags={
                "platform": platform,
                "organization_id": organization_id,
                "detection_method": detection_method,
            },
        )

    def record_highlight_confidence(
        self, confidence: float, detection_method: str, platform: str
    ) -> None:
        """Record highlight confidence score."""
        self._record_histogram(
            "highlight_confidence_score",
            confidence,
            tags={
                "detection_method": detection_method,
                "platform": platform,
            },
        )

    def record_highlight_processing_time(
        self, duration_seconds: float, stage: str, platform: str
    ) -> None:
        """Record highlight processing time by stage."""
        self._record_histogram(
            "highlight_processing_time_seconds",
            duration_seconds,
            tags={
                "stage": stage,
                "platform": platform,
            },
        )

    # API Metrics

    def increment_api_key_usage(
        self, api_key_id: str, organization_id: str, endpoint: str
    ) -> None:
        """Increment API key usage counter."""
        self._increment_counter(
            "api_key_usage_total",
            tags={
                "api_key_id": api_key_id,
                "organization_id": organization_id,
                "endpoint": endpoint,
            },
        )

    def record_api_latency(
        self, latency_seconds: float, endpoint: str, method: str, status_code: int
    ) -> None:
        """Record API endpoint latency."""
        self._record_histogram(
            "api_latency_seconds",
            latency_seconds,
            tags={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
            },
        )

    def increment_rate_limit_exceeded(
        self, api_key_id: str, organization_id: str, endpoint: str
    ) -> None:
        """Increment rate limit exceeded counter."""
        self._increment_counter(
            "rate_limit_exceeded_total",
            tags={
                "api_key_id": api_key_id,
                "organization_id": organization_id,
                "endpoint": endpoint,
            },
        )

    # Webhook Metrics

    def increment_webhook_sent(
        self, organization_id: str, event_type: str, success: bool
    ) -> None:
        """Increment webhook sent counter."""
        status = "success" if success else "failed"
        self._increment_counter(
            "webhooks_sent_total",
            tags={
                "organization_id": organization_id,
                "event_type": event_type,
                "status": status,
            },
        )

    def record_webhook_latency(
        self, latency_seconds: float, organization_id: str, event_type: str
    ) -> None:
        """Record webhook delivery latency."""
        self._record_histogram(
            "webhook_latency_seconds",
            latency_seconds,
            tags={
                "organization_id": organization_id,
                "event_type": event_type,
            },
        )

    def increment_webhook_retry(
        self, organization_id: str, event_type: str, retry_count: int
    ) -> None:
        """Increment webhook retry counter."""
        self._increment_counter(
            "webhook_retries_total",
            tags={
                "organization_id": organization_id,
                "event_type": event_type,
                "retry_count": str(retry_count),
            },
        )

    # Storage Metrics

    def record_storage_usage(
        self, bytes_used: int, organization_id: str, storage_type: str
    ) -> None:
        """Record storage usage in bytes."""
        self._set_gauge(
            "storage_usage_bytes",
            bytes_used,
            tags={
                "organization_id": organization_id,
                "storage_type": storage_type,
            },
        )

    def increment_storage_operations(
        self, operation: str, organization_id: str, success: bool
    ) -> None:
        """Increment storage operation counter."""
        status = "success" if success else "failed"
        self._increment_counter(
            "storage_operations_total",
            tags={
                "operation": operation,
                "organization_id": organization_id,
                "status": status,
            },
        )

    # Usage Tracking Metrics

    def track_api_call_usage(
        self,
        user_id: str,
        organization_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
    ) -> None:
        """Track API call usage for billing and analytics."""
        self._increment_counter(
            "api_calls_total",
            tags={
                "user_id": user_id,
                "organization_id": organization_id,
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
            },
        )
        
        # Record response time
        self._record_histogram("api_response_time_ms", response_time_ms)
        
        # Send to Logfire
        logfire.info(
            "API call tracked",
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
        )

    def track_stream_processing_usage(
        self,
        user_id: str,
        organization_id: str,
        stream_id: str,
        processing_minutes: float,
    ) -> None:
        """Track stream processing usage."""
        self._increment_counter(
            "stream_processing_minutes_total",
            tags={
                "user_id": user_id,
                "organization_id": organization_id,
            },
        )
        
        # Record processing time
        self._record_histogram("stream_processing_minutes", processing_minutes)
        
        # Send to Logfire
        logfire.info(
            "Stream processing tracked",
            user_id=user_id,
            organization_id=organization_id,
            stream_id=stream_id,
            processing_minutes=processing_minutes,
        )

    def track_webhook_delivery_usage(
        self,
        user_id: str,
        organization_id: str,
        webhook_id: str,
        success: bool = True,
    ) -> None:
        """Track webhook delivery usage."""
        status = "success" if success else "failed"
        self._increment_counter(
            "webhook_deliveries_total",
            tags={
                "user_id": user_id,
                "organization_id": organization_id,
                "status": status,
            },
        )
        
        # Send to Logfire
        logfire.info(
            "Webhook delivery tracked",
            user_id=user_id,
            organization_id=organization_id,
            webhook_id=webhook_id,
            success=success,
        )

    # AI/ML Metrics

    def increment_ai_api_calls(
        self, provider: str, model: str, organization_id: str, success: bool
    ) -> None:
        """Increment AI API call counter."""
        status = "success" if success else "failed"
        self._increment_counter(
            "ai_api_calls_total",
            tags={
                "provider": provider,
                "model": model,
                "organization_id": organization_id,
                "status": status,
            },
        )

    def record_ai_tokens_used(
        self,
        tokens: int,
        provider: str,
        model: str,
        organization_id: str,
        token_type: str = "total",
    ) -> None:
        """Record AI tokens used."""
        self._increment_counter(
            "ai_tokens_used_total",
            value=tokens,
            tags={
                "provider": provider,
                "model": model,
                "organization_id": organization_id,
                "token_type": token_type,
            },
        )

    def record_ai_latency(
        self, latency_seconds: float, provider: str, model: str, operation: str
    ) -> None:
        """Record AI API latency."""
        self._record_histogram(
            "ai_api_latency_seconds",
            latency_seconds,
            tags={
                "provider": provider,
                "model": model,
                "operation": operation,
            },
        )

    # Celery Task Metrics

    def increment_task_executed(
        self, task_name: str, organization_id: Optional[str], success: bool
    ) -> None:
        """Increment task execution counter."""
        status = "success" if success else "failed"
        tags = {
            "task_name": task_name,
            "status": status,
        }
        if organization_id:
            tags["organization_id"] = organization_id

        self._increment_counter("tasks_executed_total", tags=tags)

    def record_task_duration(
        self, duration_seconds: float, task_name: str, organization_id: Optional[str]
    ) -> None:
        """Record task execution duration."""
        tags = {"task_name": task_name}
        if organization_id:
            tags["organization_id"] = organization_id

        self._record_histogram("task_duration_seconds", duration_seconds, tags)

    def set_task_queue_depth(self, queue_name: str, depth: int) -> None:
        """Set task queue depth gauge."""
        self._set_gauge("task_queue_depth", depth, tags={"queue_name": queue_name})

    # System Metrics

    def set_system_metric(self, metric_name: str, value: float, **tags: Any) -> None:
        """Set a system metric gauge."""
        self._set_gauge(f"system_{metric_name}", value, tags)

    # Core metric operations

    def _increment_counter(
        self, name: str, value: float = 1.0, tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """Increment a counter metric."""
        metric_key = self._get_metric_key(name, tags)

        with self._lock:
            self._counters[metric_key] += value

        # Log to Logfire
        logfire.info(
            f"metric.{name}", value=value, metric_type="counter", **(tags or {})
        )

    def _set_gauge(
        self, name: str, value: float, tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set a gauge metric."""
        metric_key = self._get_metric_key(name, tags)

        with self._lock:
            self._gauges[metric_key] = value

        # Log to Logfire
        logfire.info(f"metric.{name}", value=value, metric_type="gauge", **(tags or {}))

    def _record_histogram(
        self, name: str, value: float, tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a histogram value."""
        metric_key = self._get_metric_key(name, tags)

        with self._lock:
            if metric_key not in self._histograms:
                self._histograms[metric_key] = []
            self._histograms[metric_key].append(value)

            # Keep only last 1000 values to prevent memory issues
            if len(self._histograms[metric_key]) > 1000:
                self._histograms[metric_key] = self._histograms[metric_key][-1000:]

        # Log to Logfire
        logfire.info(
            f"metric.{name}", value=value, metric_type="histogram", **(tags or {})
        )

    def _get_metric_key(self, name: str, tags: Optional[Dict[str, Any]]) -> str:
        """Generate a unique key for a metric with tags."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"

    def register_metric_callback(
        self,
        name: str,
        callback: Callable[[], Dict[str, float]],
        interval_seconds: int = 60,
    ) -> None:
        """Register a callback to collect metrics periodically."""
        self._metric_callbacks[name] = (callback, interval_seconds)

    async def start_background_collection(self) -> None:
        """Start background metric collection."""
        if self._background_task:
            return

        self._background_task = asyncio.create_task(self._collect_metrics_loop())

    async def stop_background_collection(self) -> None:
        """Stop background metric collection."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

    async def _collect_metrics_loop(self) -> None:
        """Background loop for collecting metrics."""
        last_collection_times = {}

        while True:
            try:
                current_time = time.time()

                for name, (callback, interval) in self._metric_callbacks.items():
                    last_time = last_collection_times.get(name, 0)

                    if current_time - last_time >= interval:
                        try:
                            metrics = callback()
                            for metric_name, value in metrics.items():
                                self._set_gauge(metric_name, value)

                            last_collection_times[name] = current_time
                        except Exception as e:
                            logfire.error(
                                f"Error collecting metric {name}: {e}",
                                metric_name=name,
                                exc_info=True,
                            )

                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                logfire.error("Error in metric collection loop", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error


# Global metrics instance
metrics = MetricsCollector()


# Convenience functions
def increment(name: str, value: float = 1.0, **tags: Any) -> None:
    """Increment a counter metric."""
    metrics._increment_counter(name, value, tags)


def gauge(name: str, value: float, **tags: Any) -> None:
    """Set a gauge metric."""
    metrics._set_gauge(name, value, tags)


def histogram(name: str, value: float, **tags: Any) -> None:
    """Record a histogram value."""
    metrics._record_histogram(name, value, tags)
