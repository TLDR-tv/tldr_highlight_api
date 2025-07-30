"""Dashboard configurations for Logfire monitoring.

This module defines the dashboard layouts and queries for monitoring
the TL;DR Highlight API in production.
"""

from typing import Dict, List, Any


class DashboardConfig:
    """Configuration for Logfire dashboards."""

    @staticmethod
    def get_overview_dashboard() -> Dict[str, Any]:
        """Main overview dashboard configuration."""
        return {
            "name": "TL;DR Highlight API Overview",
            "description": "High-level overview of API health and performance",
            "panels": [
                {
                    "title": "Request Rate",
                    "type": "line_chart",
                    "query": {
                        "metric": "http.request.count",
                        "aggregation": "rate",
                        "group_by": ["method", "endpoint"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Response Time (p95)",
                    "type": "line_chart",
                    "query": {
                        "metric": "http.request.duration",
                        "aggregation": "percentile",
                        "percentile": 95,
                        "group_by": ["endpoint"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Error Rate",
                    "type": "line_chart",
                    "query": {
                        "metric": "http.request.errors",
                        "aggregation": "rate",
                        "group_by": ["status_code", "endpoint"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 3, "w": 6, "h": 3},
                },
                {
                    "title": "Active Streams",
                    "type": "gauge",
                    "query": {
                        "metric": "streams.active.count",
                        "aggregation": "last",
                        "group_by": ["platform"],
                    },
                    "position": {"x": 6, "y": 3, "w": 3, "h": 3},
                },
                {
                    "title": "Highlights Detected (Last Hour)",
                    "type": "counter",
                    "query": {
                        "metric": "highlights.detected.count",
                        "aggregation": "sum",
                        "time_window": "1h",
                        "group_by": ["platform", "detection_method"],
                    },
                    "position": {"x": 9, "y": 3, "w": 3, "h": 3},
                },
            ],
        }

    @staticmethod
    def get_stream_processing_dashboard() -> Dict[str, Any]:
        """Stream processing specific dashboard."""
        return {
            "name": "Stream Processing Monitor",
            "description": "Detailed view of stream processing pipeline",
            "panels": [
                {
                    "title": "Stream Start Rate",
                    "type": "line_chart",
                    "query": {
                        "metric": "streams.started.count",
                        "aggregation": "rate",
                        "group_by": ["platform", "organization_id"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Stream Completion Rate",
                    "type": "line_chart",
                    "query": {
                        "metric": "streams.completed.count",
                        "aggregation": "rate",
                        "group_by": ["platform", "success"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Processing Duration Distribution",
                    "type": "histogram",
                    "query": {
                        "metric": "stream.duration.minutes",
                        "buckets": [1, 5, 10, 30, 60, 120],
                        "group_by": ["platform"],
                    },
                    "position": {"x": 0, "y": 3, "w": 6, "h": 3},
                },
                {
                    "title": "Highlight Detection Time",
                    "type": "line_chart",
                    "query": {
                        "metric": "highlight.processing.duration",
                        "aggregation": "avg",
                        "group_by": ["stage", "platform"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 3, "w": 6, "h": 3},
                },
                {
                    "title": "FFmpeg Task Queue",
                    "type": "gauge",
                    "query": {
                        "metric": "celery.queue.size",
                        "filter": {"queue_name": "ingest_stream_with_ffmpeg"},
                        "aggregation": "last",
                    },
                    "position": {"x": 0, "y": 6, "w": 3, "h": 2},
                },
                {
                    "title": "AI Detection Task Queue",
                    "type": "gauge",
                    "query": {
                        "metric": "celery.queue.size",
                        "filter": {"queue_name": "detect_highlights_with_ai"},
                        "aggregation": "last",
                    },
                    "position": {"x": 3, "y": 6, "w": 3, "h": 2},
                },
            ],
        }

    @staticmethod
    def get_business_metrics_dashboard() -> Dict[str, Any]:
        """Business metrics dashboard."""
        return {
            "name": "Business Metrics",
            "description": "Key business metrics and usage patterns",
            "panels": [
                {
                    "title": "API Usage by Organization",
                    "type": "bar_chart",
                    "query": {
                        "metric": "api.calls.count",
                        "aggregation": "sum",
                        "group_by": ["organization_id", "endpoint"],
                        "time_window": "1d",
                        "top_k": 10,
                    },
                    "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                },
                {
                    "title": "Stream Processing Minutes",
                    "type": "line_chart",
                    "query": {
                        "metric": "usage.stream.minutes",
                        "aggregation": "sum",
                        "group_by": ["organization_id"],
                        "time_window": "1h",
                    },
                    "position": {"x": 6, "y": 0, "w": 6, "h": 4},
                },
                {
                    "title": "Highlight Confidence Distribution",
                    "type": "histogram",
                    "query": {
                        "metric": "highlight.confidence.score",
                        "buckets": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        "group_by": ["detection_method", "platform"],
                    },
                    "position": {"x": 0, "y": 4, "w": 6, "h": 3},
                },
                {
                    "title": "Cost per Stream",
                    "type": "scatter_plot",
                    "query": {
                        "x_metric": "stream.duration.minutes",
                        "y_metric": "stream.cost.total",
                        "group_by": ["platform"],
                        "time_window": "1d",
                    },
                    "position": {"x": 6, "y": 4, "w": 6, "h": 3},
                },
                {
                    "title": "Quota Usage by Organization",
                    "type": "table",
                    "query": {
                        "metrics": [
                            {
                                "name": "concurrent_streams",
                                "metric": "streams.active.count",
                                "aggregation": "max",
                            },
                            {
                                "name": "daily_minutes",
                                "metric": "usage.stream.minutes",
                                "aggregation": "sum",
                            },
                            {
                                "name": "api_calls",
                                "metric": "api.calls.count",
                                "aggregation": "sum",
                            },
                        ],
                        "group_by": ["organization_id"],
                        "time_window": "1d",
                        "sort_by": "daily_minutes",
                        "limit": 20,
                    },
                    "position": {"x": 0, "y": 7, "w": 12, "h": 4},
                },
            ],
        }

    @staticmethod
    def get_infrastructure_dashboard() -> Dict[str, Any]:
        """Infrastructure and performance dashboard."""
        return {
            "name": "Infrastructure Health",
            "description": "System performance and infrastructure metrics",
            "panels": [
                {
                    "title": "Database Query Time",
                    "type": "line_chart",
                    "query": {
                        "metric": "db.query.duration",
                        "aggregation": "avg",
                        "group_by": ["operation", "table"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Redis Command Rate",
                    "type": "line_chart",
                    "query": {
                        "metric": "redis.command.count",
                        "aggregation": "rate",
                        "group_by": ["command"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 0, "w": 6, "h": 3},
                },
                {
                    "title": "Celery Task Execution Time",
                    "type": "line_chart",
                    "query": {
                        "metric": "celery.task.duration",
                        "aggregation": "percentile",
                        "percentile": 95,
                        "group_by": ["task_name"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 3, "w": 6, "h": 3},
                },
                {
                    "title": "S3 Operation Latency",
                    "type": "line_chart",
                    "query": {
                        "metric": "s3.operation.duration",
                        "aggregation": "avg",
                        "group_by": ["operation", "bucket"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 3, "w": 6, "h": 3},
                },
                {
                    "title": "Memory Usage by Service",
                    "type": "area_chart",
                    "query": {
                        "metric": "process.memory.usage",
                        "aggregation": "avg",
                        "group_by": ["service_name"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 6, "w": 6, "h": 3},
                },
                {
                    "title": "CPU Usage by Service",
                    "type": "area_chart",
                    "query": {
                        "metric": "process.cpu.usage",
                        "aggregation": "avg",
                        "group_by": ["service_name"],
                        "time_window": "5m",
                    },
                    "position": {"x": 6, "y": 6, "w": 6, "h": 3},
                },
            ],
        }

    @staticmethod
    def get_error_tracking_dashboard() -> Dict[str, Any]:
        """Error tracking and debugging dashboard."""
        return {
            "name": "Error Tracking",
            "description": "Track errors and exceptions across the system",
            "panels": [
                {
                    "title": "Error Rate by Type",
                    "type": "stacked_bar",
                    "query": {
                        "metric": "error.count",
                        "aggregation": "rate",
                        "group_by": ["error_type", "service"],
                        "time_window": "5m",
                    },
                    "position": {"x": 0, "y": 0, "w": 12, "h": 3},
                },
                {
                    "title": "Recent Errors",
                    "type": "log_table",
                    "query": {
                        "filter": {"level": "error"},
                        "fields": [
                            "timestamp",
                            "service",
                            "error_type",
                            "message",
                            "trace_id",
                        ],
                        "limit": 50,
                        "sort": "timestamp desc",
                    },
                    "position": {"x": 0, "y": 3, "w": 12, "h": 4},
                },
                {
                    "title": "Failed Stream Processing",
                    "type": "table",
                    "query": {
                        "metric": "stream.failed",
                        "group_by": ["stream_id", "error_message", "platform"],
                        "time_window": "1h",
                        "limit": 20,
                    },
                    "position": {"x": 0, "y": 7, "w": 6, "h": 3},
                },
                {
                    "title": "Agent Errors by Organization",
                    "type": "heatmap",
                    "query": {
                        "metric": "agent.error.count",
                        "group_by": ["organization_id", "error_type"],
                        "time_window": "1d",
                    },
                    "position": {"x": 6, "y": 7, "w": 6, "h": 3},
                },
            ],
        }

    @staticmethod
    def get_alerts_config() -> List[Dict[str, Any]]:
        """Alert configurations for monitoring."""
        return [
            {
                "name": "High Error Rate",
                "condition": {
                    "metric": "http.request.errors",
                    "aggregation": "rate",
                    "threshold": 0.05,  # 5% error rate
                    "duration": "5m",
                    "comparison": ">",
                },
                "severity": "critical",
                "notification_channels": ["pagerduty", "slack"],
            },
            {
                "name": "Stream Processing Failure Rate",
                "condition": {
                    "metric": "streams.completed.count",
                    "filter": {"success": False},
                    "aggregation": "rate",
                    "threshold": 0.1,  # 10% failure rate
                    "duration": "10m",
                    "comparison": ">",
                },
                "severity": "warning",
                "notification_channels": ["slack"],
            },
            {
                "name": "High Response Time",
                "condition": {
                    "metric": "http.request.duration",
                    "aggregation": "percentile",
                    "percentile": 95,
                    "threshold": 5.0,  # 5 seconds
                    "duration": "5m",
                    "comparison": ">",
                },
                "severity": "warning",
                "notification_channels": ["slack"],
            },
            {
                "name": "Celery Queue Backup",
                "condition": {
                    "metric": "celery.queue.size",
                    "aggregation": "max",
                    "threshold": 1000,
                    "duration": "5m",
                    "comparison": ">",
                },
                "severity": "warning",
                "notification_channels": ["slack", "email"],
            },
            {
                "name": "Database Connection Pool Exhausted",
                "condition": {
                    "metric": "db.pool.available",
                    "aggregation": "min",
                    "threshold": 2,
                    "duration": "2m",
                    "comparison": "<",
                },
                "severity": "critical",
                "notification_channels": ["pagerduty", "slack"],
            },
            {
                "name": "Quota Exceeded",
                "condition": {
                    "metric": "quota.exceeded.count",
                    "aggregation": "sum",
                    "threshold": 10,
                    "duration": "5m",
                    "comparison": ">",
                },
                "severity": "info",
                "notification_channels": ["email"],
            },
        ]

    @classmethod
    def export_all_dashboards(cls) -> Dict[str, Any]:
        """Export all dashboard configurations."""
        return {
            "dashboards": [
                cls.get_overview_dashboard(),
                cls.get_stream_processing_dashboard(),
                cls.get_business_metrics_dashboard(),
                cls.get_infrastructure_dashboard(),
                cls.get_error_tracking_dashboard(),
            ],
            "alerts": cls.get_alerts_config(),
            "metadata": {
                "version": "1.0.0",
                "created_at": "2024-01-01",
                "description": "TL;DR Highlight API Monitoring Configuration",
            },
        }
