"""Comprehensive tests for metrics module."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from threading import Thread
import time

from src.utils.metrics import (
    MetricType,
    HealthStatus,
    MetricPoint,
    TimerMetrics,
    HistogramMetrics,
    Metric,
    MetricsRegistry,
    HealthCheck,
    HealthMonitor,
    MetricsContext,
    counter,
    gauge,
    histogram,
    timer,
    timed,
    metrics_registry,
    health_monitor,
)


class TestMetricPoint:
    """Test cases for MetricPoint class."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        timestamp = datetime.utcnow()
        point = MetricPoint(timestamp=timestamp, value=42.5, labels={"env": "test"})
        
        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels == {"env": "test"}

    def test_metric_point_to_dict(self):
        """Test converting metric point to dictionary."""
        timestamp = datetime.utcnow()
        point = MetricPoint(timestamp=timestamp, value=10.0)
        
        result = point.to_dict()
        assert result["timestamp"] == timestamp.isoformat()
        assert result["value"] == 10.0
        assert result["labels"] == {}


class TestTimerMetrics:
    """Test cases for TimerMetrics class."""

    def test_timer_metrics_initialization(self):
        """Test timer metrics initialization."""
        metrics = TimerMetrics()
        
        assert metrics.count == 0
        assert metrics.total_time == 0.0
        assert metrics.min_time == float("inf")
        assert metrics.max_time == 0.0
        assert metrics.average_time == 0.0

    def test_add_time(self):
        """Test adding time measurements."""
        metrics = TimerMetrics()
        
        metrics.add_time(1.5)
        assert metrics.count == 1
        assert metrics.total_time == 1.5
        assert metrics.min_time == 1.5
        assert metrics.max_time == 1.5
        assert metrics.average_time == 1.5
        
        metrics.add_time(2.5)
        assert metrics.count == 2
        assert metrics.total_time == 4.0
        assert metrics.min_time == 1.5
        assert metrics.max_time == 2.5
        assert metrics.average_time == 2.0
        
        metrics.add_time(0.5)
        assert metrics.count == 3
        assert metrics.min_time == 0.5
        assert metrics.max_time == 2.5

    def test_reset(self):
        """Test resetting timer metrics."""
        metrics = TimerMetrics()
        metrics.add_time(1.0)
        metrics.add_time(2.0)
        
        metrics.reset()
        
        assert metrics.count == 0
        assert metrics.total_time == 0.0
        assert metrics.min_time == float("inf")
        assert metrics.max_time == 0.0


class TestHistogramMetrics:
    """Test cases for HistogramMetrics class."""

    def test_histogram_initialization(self):
        """Test histogram metrics initialization."""
        metrics = HistogramMetrics()
        
        assert metrics.count == 0
        assert metrics.sum == 0.0
        assert len(metrics.buckets) == 8  # Default buckets
        assert all(count == 0 for count in metrics.buckets.values())

    def test_observe_values(self):
        """Test observing values in histogram."""
        metrics = HistogramMetrics()
        
        # Observe values that fall into different buckets
        metrics.observe(0.05)  # <= 0.1
        metrics.observe(0.2)   # <= 0.25
        metrics.observe(0.4)   # <= 0.5
        metrics.observe(1.5)   # <= 2.5
        metrics.observe(10.0)  # <= inf
        
        assert metrics.count == 5
        assert metrics.sum == 12.15
        
        # Check bucket counts
        assert metrics.buckets[0.1] == 1
        assert metrics.buckets[0.25] == 1
        assert metrics.buckets[0.5] == 1
        assert metrics.buckets[1.0] == 0
        assert metrics.buckets[2.5] == 1
        assert metrics.buckets[float("inf")] == 1


class TestMetric:
    """Test cases for Metric class."""

    def test_counter_metric(self):
        """Test counter metric operations."""
        metric = Metric("test_counter", MetricType.COUNTER, "Test counter")
        
        # Initial value
        assert metric.current_value == 0.0
        
        # Increment
        metric.increment()
        assert metric.current_value == 1.0
        
        metric.increment(5.0)
        assert metric.current_value == 6.0
        
        # With labels
        metric.increment(2.0, {"env": "prod"})
        assert metric.current_value == 8.0
        
        # Check points are stored
        points = metric.get_points()
        assert len(points) == 3
        assert points[-1].labels == {"env": "prod"}

    def test_gauge_metric(self):
        """Test gauge metric operations."""
        metric = Metric("test_gauge", MetricType.GAUGE, "Test gauge")
        
        # Set value
        metric.set(42.5)
        assert metric.current_value == 42.5
        
        metric.set(10.0, {"region": "us-east"})
        assert metric.current_value == 10.0
        
        # Check points
        points = metric.get_points()
        assert len(points) == 2
        assert points[0].value == 42.5
        assert points[1].value == 10.0

    def test_histogram_metric(self):
        """Test histogram metric operations."""
        metric = Metric("test_histogram", MetricType.HISTOGRAM, "Test histogram")
        
        # Observe values
        metric.observe(0.1)
        metric.observe(0.5)
        metric.observe(1.2)
        
        # Check histogram stats
        stats = metric.get_stats()
        assert stats["histogram_stats"]["count"] == 3
        assert stats["histogram_stats"]["sum"] == 1.8

    def test_timer_metric(self):
        """Test timer metric operations."""
        metric = Metric("test_timer", MetricType.TIMER, "Test timer")
        
        # Time a function
        def slow_function():
            time.sleep(0.01)
            return "result"
        
        result = metric.time_function(slow_function)
        assert result == "result"
        
        stats = metric.get_stats()
        assert stats["timer_stats"]["count"] == 1
        assert stats["timer_stats"]["min_time"] > 0.009
        assert stats["timer_stats"]["max_time"] > 0.009

    @pytest.mark.asyncio
    async def test_timer_async_function(self):
        """Test timing async functions."""
        metric = Metric("test_async_timer", MetricType.TIMER, "Test async timer")
        
        async def async_slow_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await metric.time_async_function(async_slow_function)
        assert result == "async_result"
        
        stats = metric.get_stats()
        assert stats["timer_stats"]["count"] == 1
        assert stats["timer_stats"]["total_time"] > 0.009

    def test_metric_type_validation(self):
        """Test metric type validation."""
        counter_metric = Metric("counter", MetricType.COUNTER)
        gauge_metric = Metric("gauge", MetricType.GAUGE)
        histogram_metric = Metric("histogram", MetricType.HISTOGRAM)
        timer_metric = Metric("timer", MetricType.TIMER)
        
        # Wrong operations for counter
        with pytest.raises(ValueError, match="Cannot set non-gauge"):
            counter_metric.set(10.0)
        with pytest.raises(ValueError, match="Cannot observe non-histogram"):
            counter_metric.observe(10.0)
        with pytest.raises(ValueError, match="Cannot time with non-timer"):
            counter_metric.time_function(lambda: None)
        
        # Wrong operations for gauge
        with pytest.raises(ValueError, match="Cannot increment non-counter"):
            gauge_metric.increment()
        
        # Wrong operations for histogram
        with pytest.raises(ValueError, match="Cannot increment non-counter"):
            histogram_metric.increment()
        with pytest.raises(ValueError, match="Cannot set non-gauge"):
            histogram_metric.set(10.0)
        
        # Wrong operations for timer
        with pytest.raises(ValueError, match="Cannot increment non-counter"):
            timer_metric.increment()

    def test_get_points_with_since(self):
        """Test getting points since a specific time."""
        metric = Metric("test_metric", MetricType.GAUGE)
        
        # Add points with some delay
        metric.set(1.0)
        time.sleep(0.01)
        cutoff_time = datetime.utcnow()
        time.sleep(0.01)
        metric.set(2.0)
        metric.set(3.0)
        
        # Get all points
        all_points = metric.get_points()
        assert len(all_points) == 3
        
        # Get points since cutoff
        recent_points = metric.get_points(since=cutoff_time)
        assert len(recent_points) == 2
        assert all(p.value in [2.0, 3.0] for p in recent_points)

    def test_max_points_limit(self):
        """Test that metric respects max_points limit."""
        metric = Metric("test_limited", MetricType.COUNTER, max_points=5)
        
        # Add more than max points
        for i in range(10):
            metric.increment()
        
        points = metric.get_points()
        assert len(points) == 5  # Should keep only last 5
        assert points[-1].value == 10.0  # Last value should be 10

    def test_get_stats(self):
        """Test getting comprehensive metric statistics."""
        # Test timer stats
        timer_metric = Metric("timer", MetricType.TIMER)
        timer_metric.time_function(lambda: time.sleep(0.01))
        timer_metric.time_function(lambda: time.sleep(0.02))
        
        stats = timer_metric.get_stats()
        assert stats["name"] == "timer"
        assert stats["type"] == "timer"
        assert stats["timer_stats"]["count"] == 2
        assert stats["timer_stats"]["average_time"] > 0.01
        
        # Test histogram stats
        hist_metric = Metric("hist", MetricType.HISTOGRAM)
        hist_metric.observe(0.1)
        hist_metric.observe(0.5)
        
        stats = hist_metric.get_stats()
        assert stats["histogram_stats"]["count"] == 2
        assert stats["histogram_stats"]["sum"] == 0.6
        assert stats["histogram_stats"]["buckets"][0.5] == 1
        assert stats["histogram_stats"]["buckets"][1.0] == 0


class TestMetricsRegistry:
    """Test cases for MetricsRegistry class."""

    def test_create_metrics(self):
        """Test creating different types of metrics."""
        registry = MetricsRegistry()
        
        # Create metrics
        c = registry.create_counter("test_counter", "Test counter")
        g = registry.create_gauge("test_gauge", "Test gauge")
        h = registry.create_histogram("test_histogram", "Test histogram")
        t = registry.create_timer("test_timer", "Test timer")
        
        assert c.type == MetricType.COUNTER
        assert g.type == MetricType.GAUGE
        assert h.type == MetricType.HISTOGRAM
        assert t.type == MetricType.TIMER

    def test_get_existing_metric(self):
        """Test getting existing metrics."""
        registry = MetricsRegistry()
        
        # Create metric
        counter1 = registry.create_counter("my_counter")
        counter1.increment(5)
        
        # Get same metric
        counter2 = registry.create_counter("my_counter")
        assert counter2.current_value == 5  # Same instance
        
        # Also test get_metric
        counter3 = registry.get_metric("my_counter")
        assert counter3 is counter1

    def test_metric_type_conflict(self):
        """Test creating metric with conflicting type."""
        registry = MetricsRegistry()
        
        # Create counter
        registry.create_counter("my_metric")
        
        # Try to create same name with different type
        with pytest.raises(ValueError, match="already exists with different type"):
            registry.create_gauge("my_metric")

    def test_list_metrics(self):
        """Test listing all metrics."""
        registry = MetricsRegistry()
        
        registry.create_counter("counter1")
        registry.create_gauge("gauge1")
        registry.create_timer("timer1")
        
        metrics = registry.list_metrics()
        assert len(metrics) == 3
        assert "counter1" in metrics
        assert "gauge1" in metrics
        assert "timer1" in metrics

    def test_get_all_stats(self):
        """Test getting stats for all metrics."""
        registry = MetricsRegistry()
        
        counter = registry.create_counter("c1")
        counter.increment(10)
        
        gauge = registry.create_gauge("g1")
        gauge.set(42)
        
        all_stats = registry.get_all_stats()
        assert len(all_stats) == 2
        assert all_stats["c1"]["current_value"] == 10
        assert all_stats["g1"]["current_value"] == 42

    def test_remove_metric(self):
        """Test removing metrics."""
        registry = MetricsRegistry()
        
        registry.create_counter("to_remove")
        assert "to_remove" in registry.list_metrics()
        
        # Remove metric
        assert registry.remove_metric("to_remove") is True
        assert "to_remove" not in registry.list_metrics()
        
        # Remove non-existent metric
        assert registry.remove_metric("non_existent") is False

    def test_clear_all(self):
        """Test clearing all metrics."""
        registry = MetricsRegistry()
        
        registry.create_counter("c1")
        registry.create_gauge("g1")
        registry.create_timer("t1")
        
        assert len(registry.list_metrics()) == 3
        
        registry.clear_all()
        assert len(registry.list_metrics()) == 0

    def test_thread_safety(self):
        """Test thread safety of registry operations."""
        registry = MetricsRegistry()
        errors = []
        
        def create_metrics():
            try:
                for i in range(100):
                    registry.create_counter(f"counter_{i % 10}")
                    registry.create_gauge(f"gauge_{i % 10}")
            except Exception as e:
                errors.append(e)
        
        # Run in multiple threads
        threads = [Thread(target=create_metrics) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not have any errors
        assert len(errors) == 0
        # Should have created 20 metrics (10 counters + 10 gauges)
        assert len(registry.list_metrics()) == 20


class TestHealthCheck:
    """Test cases for HealthCheck class."""

    def test_health_check_creation(self):
        """Test creating a health check."""
        check_func = AsyncMock(return_value=True)
        check = HealthCheck(
            name="test_check",
            check_function=check_func,
            description="Test health check",
            timeout_seconds=5.0,
            critical=True,
            interval_seconds=30.0
        )
        
        assert check.name == "test_check"
        assert check.check_function == check_func
        assert check.description == "Test health check"
        assert check.timeout_seconds == 5.0
        assert check.critical is True
        assert check.interval_seconds == 30.0
        assert check.last_status == HealthStatus.UNKNOWN
        assert check.consecutive_failures == 0


class TestHealthMonitor:
    """Test cases for HealthMonitor class."""

    @pytest.mark.asyncio
    async def test_register_check(self):
        """Test registering health checks."""
        monitor = HealthMonitor()
        
        async def check_func():
            return True
        
        await monitor.register_check(
            "test_check",
            check_func,
            description="Test check",
            critical=True
        )
        
        # Check should be registered
        assert "test_check" in monitor._checks
        assert monitor._checks["test_check"].name == "test_check"

    @pytest.mark.asyncio
    async def test_remove_check(self):
        """Test removing health checks."""
        monitor = HealthMonitor()
        
        async def check_func():
            return True
        
        await monitor.register_check("test_check", check_func)
        assert "test_check" in monitor._checks
        
        # Remove check
        assert await monitor.remove_check("test_check") is True
        assert "test_check" not in monitor._checks
        
        # Remove non-existent check
        assert await monitor.remove_check("non_existent") is False

    @pytest.mark.asyncio
    async def test_run_check_healthy(self):
        """Test running a healthy check."""
        monitor = HealthMonitor()
        
        async def healthy_check():
            return True
        
        await monitor.register_check("healthy", healthy_check)
        
        status = await monitor.run_check("healthy")
        assert status == HealthStatus.HEALTHY
        assert monitor._checks["healthy"].last_status == HealthStatus.HEALTHY
        assert monitor._checks["healthy"].consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_check_unhealthy(self):
        """Test running an unhealthy check."""
        monitor = HealthMonitor()
        
        async def unhealthy_check():
            return False
        
        await monitor.register_check("unhealthy", unhealthy_check)
        
        status = await monitor.run_check("unhealthy")
        assert status == HealthStatus.UNHEALTHY
        assert monitor._checks["unhealthy"].last_status == HealthStatus.UNHEALTHY
        assert monitor._checks["unhealthy"].consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_check_exception(self):
        """Test running a check that raises exception."""
        monitor = HealthMonitor()
        
        async def failing_check():
            raise Exception("Check failed")
        
        await monitor.register_check("failing", failing_check)
        
        status = await monitor.run_check("failing")
        assert status == HealthStatus.UNHEALTHY
        assert monitor._checks["failing"].last_error == "Check failed"
        assert monitor._checks["failing"].consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_check_timeout(self):
        """Test running a check that times out."""
        monitor = HealthMonitor()
        
        async def slow_check():
            await asyncio.sleep(2.0)
            return True
        
        await monitor.register_check("slow", slow_check, timeout_seconds=0.1)
        
        status = await monitor.run_check("slow")
        assert status == HealthStatus.UNHEALTHY
        assert "timed out" in monitor._checks["slow"].last_error

    @pytest.mark.asyncio
    async def test_run_check_degraded(self):
        """Test running a check that returns degraded status."""
        monitor = HealthMonitor()
        
        async def degraded_check():
            return "degraded"  # Non-boolean indicates degraded
        
        await monitor.register_check("degraded", degraded_check)
        
        status = await monitor.run_check("degraded")
        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_sync_check_function(self):
        """Test running a synchronous check function."""
        monitor = HealthMonitor()
        
        def sync_check():
            return True
        
        await monitor.register_check("sync", sync_check)
        
        status = await monitor.run_check("sync")
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_nonexistent_check(self):
        """Test running a non-existent check."""
        monitor = HealthMonitor()
        
        with pytest.raises(ValueError, match="Health check 'nonexistent' not found"):
            await monitor.run_check("nonexistent")

    @pytest.mark.asyncio
    async def test_get_overall_health_healthy(self):
        """Test overall health when all checks are healthy."""
        monitor = HealthMonitor()
        
        async def healthy_check():
            return True
        
        await monitor.register_check("check1", healthy_check, critical=True)
        await monitor.register_check("check2", healthy_check, critical=True)
        await monitor.register_check("check3", healthy_check, critical=False)
        
        # Run all checks
        await monitor.run_check("check1")
        await monitor.run_check("check2")
        await monitor.run_check("check3")
        
        overall = await monitor.get_overall_health()
        assert overall == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_overall_health_unhealthy(self):
        """Test overall health when critical check fails."""
        monitor = HealthMonitor()
        
        async def healthy_check():
            return True
        
        async def unhealthy_check():
            return False
        
        await monitor.register_check("critical_fail", unhealthy_check, critical=True)
        await monitor.register_check("non_critical_ok", healthy_check, critical=False)
        
        await monitor.run_check("critical_fail")
        await monitor.run_check("non_critical_ok")
        
        overall = await monitor.get_overall_health()
        assert overall == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_get_overall_health_degraded(self):
        """Test overall health in degraded states."""
        monitor = HealthMonitor()
        
        async def healthy_check():
            return True
        
        async def degraded_check():
            return "degraded"
        
        async def unhealthy_check():
            return False
        
        # Test critical degraded
        await monitor.register_check("critical_degraded", degraded_check, critical=True)
        await monitor.register_check("non_critical_ok", healthy_check, critical=False)
        
        await monitor.run_check("critical_degraded")
        await monitor.run_check("non_critical_ok")
        
        overall = await monitor.get_overall_health()
        assert overall == HealthStatus.DEGRADED
        
        # Test non-critical unhealthy
        monitor = HealthMonitor()  # Fresh instance
        await monitor.register_check("critical_ok", healthy_check, critical=True)
        await monitor.register_check("non_critical_fail", unhealthy_check, critical=False)
        
        await monitor.run_check("critical_ok")
        await monitor.run_check("non_critical_fail")
        
        overall = await monitor.get_overall_health()
        assert overall == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_get_overall_health_no_checks(self):
        """Test overall health with no checks registered."""
        monitor = HealthMonitor()
        
        overall = await monitor.get_overall_health()
        assert overall == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_health_report(self):
        """Test getting comprehensive health report."""
        monitor = HealthMonitor()
        
        async def healthy_check():
            return True
        
        async def unhealthy_check():
            raise Exception("Connection failed")
        
        await monitor.register_check("database", healthy_check, "Database check", critical=True)
        await monitor.register_check("cache", unhealthy_check, "Cache check", critical=False)
        
        # Run checks
        await monitor.run_check("database")
        await monitor.run_check("cache")
        
        report = await monitor.get_health_report()
        
        assert report["overall_health"] == HealthStatus.DEGRADED.value
        assert "timestamp" in report
        assert len(report["checks"]) == 2
        
        # Check database check details
        assert report["checks"]["database"]["status"] == "healthy"
        assert report["checks"]["database"]["description"] == "Database check"
        assert report["checks"]["database"]["critical"] is True
        assert report["checks"]["database"]["consecutive_failures"] == 0
        
        # Check cache check details
        assert report["checks"]["cache"]["status"] == "unhealthy"
        assert report["checks"]["cache"]["last_error"] == "Connection failed"
        assert report["checks"]["cache"]["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping health monitoring."""
        monitor = HealthMonitor()
        call_count = 0
        
        async def counting_check():
            nonlocal call_count
            call_count += 1
            return True
        
        await monitor.register_check("counter", counting_check, interval_seconds=0.05)
        
        # Start monitoring
        await monitor.start()
        
        # Wait for a few check intervals
        await asyncio.sleep(0.15)
        
        # Stop monitoring
        await monitor.stop()
        
        # Should have run multiple times
        assert call_count >= 2
        
        # Check tasks are cleaned up
        assert len(monitor._check_tasks) == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_counter_convenience(self):
        """Test counter convenience function."""
        c = counter("test_counter", "Test description", {"env": "test"})
        
        assert c.type == MetricType.COUNTER
        assert c.description == "Test description"
        assert c.labels == {"env": "test"}
        
        # Should return same instance
        c2 = counter("test_counter")
        assert c2 is c

    def test_gauge_convenience(self):
        """Test gauge convenience function."""
        g = gauge("test_gauge", "Test gauge")
        assert g.type == MetricType.GAUGE

    def test_histogram_convenience(self):
        """Test histogram convenience function."""
        h = histogram("test_histogram")
        assert h.type == MetricType.HISTOGRAM

    def test_timer_convenience(self):
        """Test timer convenience function."""
        t = timer("test_timer")
        assert t.type == MetricType.TIMER


class TestTimedDecorator:
    """Test cases for timed decorator."""

    def test_timed_sync_function(self):
        """Test timing synchronous functions with decorator."""
        call_count = 0
        
        @timed("decorated_function", "Test function")
        def slow_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return x * 2
        
        result = slow_function(5)
        assert result == 10
        assert call_count == 1
        
        # Check metric was created and recorded
        metric = metrics_registry.get_metric("decorated_function")
        assert metric is not None
        assert metric.type == MetricType.TIMER
        stats = metric.get_stats()
        assert stats["timer_stats"]["count"] == 1
        assert stats["timer_stats"]["min_time"] > 0.009

    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        """Test timing asynchronous functions with decorator."""
        call_count = 0
        
        @timed("async_decorated_function")
        async def async_slow_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 3
        
        result = await async_slow_function(4)
        assert result == 12
        assert call_count == 1
        
        # Check metric was created and recorded
        metric = metrics_registry.get_metric("async_decorated_function")
        assert metric is not None
        stats = metric.get_stats()
        assert stats["timer_stats"]["count"] == 1


class TestMetricsContext:
    """Test cases for MetricsContext class."""

    def test_sync_context_success(self):
        """Test metrics context for successful sync operations."""
        with MetricsContext("test_operation", {"env": "test"}):
            time.sleep(0.01)
        
        # Check metrics were recorded
        timer_metric = metrics_registry.get_metric("test_operation_duration")
        success_metric = metrics_registry.get_metric("test_operation_success_total")
        error_metric = metrics_registry.get_metric("test_operation_error_total")
        
        assert timer_metric is not None
        assert success_metric is not None
        assert error_metric is not None
        
        assert success_metric.current_value == 1
        assert error_metric.current_value == 0

    def test_sync_context_error(self):
        """Test metrics context for failed sync operations."""
        try:
            with MetricsContext("failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check error was recorded
        success_metric = metrics_registry.get_metric("failing_operation_success_total")
        error_metric = metrics_registry.get_metric("failing_operation_error_total")
        
        assert success_metric.current_value == 0
        assert error_metric.current_value == 1
        
        # Check error type label
        points = error_metric.get_points()
        assert points[-1].labels["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_async_context_success(self):
        """Test metrics context for successful async operations."""
        async with MetricsContext("async_operation"):
            await asyncio.sleep(0.01)
        
        # Check metrics were recorded
        timer_metric = metrics_registry.get_metric("async_operation_duration")
        success_metric = metrics_registry.get_metric("async_operation_success_total")
        
        assert timer_metric is not None
        assert success_metric.current_value == 1

    @pytest.mark.asyncio
    async def test_async_context_error(self):
        """Test metrics context for failed async operations."""
        try:
            async with MetricsContext("async_failing"):
                raise RuntimeError("Async error")
        except RuntimeError:
            pass
        
        # Check error was recorded
        error_metric = metrics_registry.get_metric("async_failing_error_total")
        assert error_metric.current_value == 1
        
        # Check error type label
        points = error_metric.get_points()
        assert points[-1].labels["error_type"] == "RuntimeError"


class TestGlobalInstances:
    """Test cases for global instances."""

    def test_metrics_registry_global(self):
        """Test global metrics registry instance."""
        # Should be the same instance
        from src.utils.metrics import metrics_registry as registry1
        from src.utils.metrics import metrics_registry as registry2
        
        assert registry1 is registry2
        
        # Should work correctly
        registry1.create_counter("global_test")
        assert "global_test" in registry2.list_metrics()

    @pytest.mark.asyncio
    async def test_health_monitor_global(self):
        """Test global health monitor instance."""
        # Should be the same instance
        from src.utils.metrics import health_monitor as monitor1
        from src.utils.metrics import health_monitor as monitor2
        
        assert monitor1 is monitor2
        
        # Should work correctly
        async def test_check():
            return True
        
        await monitor1.register_check("global_health_test", test_check)
        assert "global_health_test" in monitor2._checks