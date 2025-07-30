"""Metrics collection and monitoring utilities for stream adapters.

This module provides comprehensive metrics collection, performance monitoring,
and health check capabilities for the stream processing system.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
from threading import Lock

from src.infrastructure.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class TimerMetrics:
    """Timer-specific metrics."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def average_time(self) -> float:
        """Calculate average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    def add_time(self, duration: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

    def reset(self) -> None:
        """Reset all metrics."""
        self.count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0


@dataclass
class HistogramMetrics:
    """Histogram-specific metrics."""

    buckets: Dict[float, int] = field(default_factory=dict)
    count: int = 0
    sum: float = 0.0

    def __post_init__(self):
        """Initialize default buckets."""
        if not self.buckets:
            # Default buckets for response times (in seconds)
            self.buckets = {
                0.1: 0,
                0.25: 0,
                0.5: 0,
                1.0: 0,
                2.5: 0,
                5.0: 0,
                10.0: 0,
                float("inf"): 0,
            }

    def observe(self, value: float) -> None:
        """Observe a value."""
        self.count += 1
        self.sum += value

        for bucket_limit in sorted(self.buckets.keys()):
            if value <= bucket_limit:
                self.buckets[bucket_limit] += 1


class Metric:
    """Base metric class."""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        max_points: int = 1000,
    ):
        """Initialize metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: Default labels
            max_points: Maximum number of data points to keep
        """
        self.name = name
        self.type = metric_type
        self.description = description
        self.labels = labels or {}
        self.max_points = max_points

        # Data storage
        self._points: Deque[MetricPoint] = deque(maxlen=max_points)
        self._current_value: float = 0.0
        self._timer_metrics = TimerMetrics()
        self._histogram_metrics = HistogramMetrics()
        self._lock = Lock()

    @property
    def current_value(self) -> float:
        """Get current metric value."""
        return self._current_value

    def increment(
        self, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter metric."""
        if self.type != MetricType.COUNTER:
            raise ValueError(f"Cannot increment non-counter metric: {self.name}")

        with self._lock:
            self._current_value += value
            point_labels = {**self.labels, **(labels or {})}
            self._points.append(
                MetricPoint(datetime.utcnow(), self._current_value, point_labels)
            )

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric value."""
        if self.type != MetricType.GAUGE:
            raise ValueError(f"Cannot set non-gauge metric: {self.name}")

        with self._lock:
            self._current_value = value
            point_labels = {**self.labels, **(labels or {})}
            self._points.append(MetricPoint(datetime.utcnow(), value, point_labels))

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value (for histograms)."""
        if self.type != MetricType.HISTOGRAM:
            raise ValueError(f"Cannot observe non-histogram metric: {self.name}")

        with self._lock:
            self._histogram_metrics.observe(value)
            point_labels = {**self.labels, **(labels or {})}
            self._points.append(MetricPoint(datetime.utcnow(), value, point_labels))

    def time_function(self, func: Callable, *args, **kwargs) -> Any:
        """Time a function execution."""
        if self.type != MetricType.TIMER:
            raise ValueError(f"Cannot time with non-timer metric: {self.name}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            with self._lock:
                self._timer_metrics.add_time(duration)
                self._points.append(MetricPoint(datetime.utcnow(), duration))

    async def time_async_function(self, func: Callable, *args, **kwargs) -> Any:
        """Time an async function execution."""
        if self.type != MetricType.TIMER:
            raise ValueError(f"Cannot time with non-timer metric: {self.name}")

        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            with self._lock:
                self._timer_metrics.add_time(duration)
                self._points.append(MetricPoint(datetime.utcnow(), duration))

    def get_points(self, since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric points since a given time."""
        with self._lock:
            if since is None:
                return list(self._points)
            else:
                return [p for p in self._points if p.timestamp >= since]

    def get_stats(self) -> Dict[str, Any]:
        """Get metric statistics."""
        with self._lock:
            stats = {
                "name": self.name,
                "type": self.type.value,
                "description": self.description,
                "labels": self.labels,
                "current_value": self._current_value,
                "point_count": len(self._points),
            }

            if self.type == MetricType.TIMER:
                stats.update(
                    {
                        "timer_stats": {
                            "count": self._timer_metrics.count,
                            "total_time": self._timer_metrics.total_time,
                            "average_time": self._timer_metrics.average_time,
                            "min_time": self._timer_metrics.min_time
                            if self._timer_metrics.min_time != float("inf")
                            else 0,
                            "max_time": self._timer_metrics.max_time,
                        }
                    }
                )

            elif self.type == MetricType.HISTOGRAM:
                stats.update(
                    {
                        "histogram_stats": {
                            "count": self._histogram_metrics.count,
                            "sum": self._histogram_metrics.sum,
                            "buckets": self._histogram_metrics.buckets,
                        }
                    }
                )

            return stats


class MetricsRegistry:
    """Registry for managing metrics."""

    def __init__(self):
        """Initialize metrics registry."""
        self._metrics: Dict[str, Metric] = {}
        self._lock = Lock()

    def create_counter(
        self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Create a counter metric."""
        return self._create_metric(name, MetricType.COUNTER, description, labels)

    def create_gauge(
        self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Create a gauge metric."""
        return self._create_metric(name, MetricType.GAUGE, description, labels)

    def create_histogram(
        self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Create a histogram metric."""
        return self._create_metric(name, MetricType.HISTOGRAM, description, labels)

    def create_timer(
        self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Create a timer metric."""
        return self._create_metric(name, MetricType.TIMER, description, labels)

    def _create_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[Dict[str, str]],
    ) -> Metric:
        """Create a metric of specified type."""
        with self._lock:
            if name in self._metrics:
                existing = self._metrics[name]
                if existing.type != metric_type:
                    raise ValueError(
                        f"Metric '{name}' already exists with different type: "
                        f"{existing.type} != {metric_type}"
                    )
                return existing

            metric = Metric(name, metric_type, description, labels)
            self._metrics[name] = metric
            logger.debug(f"Created {metric_type.value} metric: {name}")
            return metric

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def list_metrics(self) -> List[str]:
        """List all metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        with self._lock:
            return {name: metric.get_stats() for name, metric in self._metrics.items()}

    def remove_metric(self, name: str) -> bool:
        """Remove a metric."""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                logger.debug(f"Removed metric: {name}")
                return True
            return False

    def clear_all(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            logger.info("Cleared all metrics")


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_function: Callable
    description: str = ""
    timeout_seconds: float = 10.0
    critical: bool = True  # If True, failure marks overall system as unhealthy
    interval_seconds: float = 30.0

    # State
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    consecutive_failures: int = 0


class HealthMonitor:
    """Health monitoring system."""

    def __init__(self):
        """Initialize health monitor."""
        self._checks: Dict[str, HealthCheck] = {}
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False

    async def register_check(
        self,
        name: str,
        check_function: Callable,
        description: str = "",
        timeout_seconds: float = 10.0,
        critical: bool = True,
        interval_seconds: float = 30.0,
    ) -> None:
        """Register a health check.

        Args:
            name: Unique name for the check
            check_function: Async function that returns True if healthy
            description: Description of what this check does
            timeout_seconds: Timeout for the check
            critical: If True, failure affects overall health
            interval_seconds: How often to run the check
        """
        async with self._lock:
            check = HealthCheck(
                name=name,
                check_function=check_function,
                description=description,
                timeout_seconds=timeout_seconds,
                critical=critical,
                interval_seconds=interval_seconds,
            )

            self._checks[name] = check

            # Start background task for this check
            if not self._shutdown:
                self._check_tasks[name] = asyncio.create_task(
                    self._run_check_loop(check)
                )

            logger.info(f"Registered health check: {name}")

    async def remove_check(self, name: str) -> bool:
        """Remove a health check."""
        async with self._lock:
            if name in self._checks:
                # Cancel background task
                if name in self._check_tasks:
                    self._check_tasks[name].cancel()
                    try:
                        await self._check_tasks[name]
                    except asyncio.CancelledError:
                        pass
                    del self._check_tasks[name]

                del self._checks[name]
                logger.info(f"Removed health check: {name}")
                return True
            return False

    async def run_check(self, name: str) -> HealthStatus:
        """Run a specific health check manually."""
        async with self._lock:
            if name not in self._checks:
                raise ValueError(f"Health check '{name}' not found")

            check = self._checks[name]

        return await self._execute_check(check)

    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        async with self._lock:
            if not self._checks:
                return HealthStatus.UNKNOWN

            critical_checks = [c for c in self._checks.values() if c.critical]
            non_critical_checks = [c for c in self._checks.values() if not c.critical]

            # Check critical checks first
            critical_unhealthy = any(
                c.last_status == HealthStatus.UNHEALTHY for c in critical_checks
            )

            if critical_unhealthy:
                return HealthStatus.UNHEALTHY

            # Check for degraded state
            critical_degraded = any(
                c.last_status == HealthStatus.DEGRADED for c in critical_checks
            )

            non_critical_unhealthy = any(
                c.last_status == HealthStatus.UNHEALTHY for c in non_critical_checks
            )

            if critical_degraded or non_critical_unhealthy:
                return HealthStatus.DEGRADED

            # All critical checks healthy
            return HealthStatus.HEALTHY

    async def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report."""
        async with self._lock:
            overall_health = await self.get_overall_health()

            checks_status = {}
            for name, check in self._checks.items():
                checks_status[name] = {
                    "status": check.last_status.value,
                    "description": check.description,
                    "last_check": check.last_check.isoformat()
                    if check.last_check
                    else None,
                    "last_error": check.last_error,
                    "consecutive_failures": check.consecutive_failures,
                    "critical": check.critical,
                }

            return {
                "overall_health": overall_health.value,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": checks_status,
            }

    async def _execute_check(self, check: HealthCheck) -> HealthStatus:
        """Execute a single health check."""
        logger.debug(f"Running health check: {check.name}")

        try:
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(), timeout=check.timeout_seconds
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, check.check_function),
                    timeout=check.timeout_seconds,
                )

            if result is True:
                check.last_status = HealthStatus.HEALTHY
                check.consecutive_failures = 0
                check.last_error = None
            elif result is False:
                check.last_status = HealthStatus.UNHEALTHY
                check.consecutive_failures += 1
                check.last_error = "Check returned False"
            else:
                # Assume degraded for non-boolean results
                check.last_status = HealthStatus.DEGRADED
                check.last_error = f"Check returned non-boolean: {result}"

        except asyncio.TimeoutError:
            check.last_status = HealthStatus.UNHEALTHY
            check.consecutive_failures += 1
            check.last_error = f"Check timed out after {check.timeout_seconds}s"
            logger.warning(f"Health check '{check.name}' timed out")

        except Exception as e:
            check.last_status = HealthStatus.UNHEALTHY
            check.consecutive_failures += 1
            check.last_error = str(e)
            logger.error(f"Health check '{check.name}' failed: {e}")

        finally:
            check.last_check = datetime.utcnow()

        logger.debug(f"Health check '{check.name}' result: {check.last_status.value}")

        return check.last_status

    async def _run_check_loop(self, check: HealthCheck) -> None:
        """Run a health check in a loop."""
        while not self._shutdown:
            try:
                await self._execute_check(check)
                await asyncio.sleep(check.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for '{check.name}': {e}")
                await asyncio.sleep(check.interval_seconds)

    async def start(self) -> None:
        """Start all health check loops."""
        async with self._lock:
            self._shutdown = False

            for name, check in self._checks.items():
                if name not in self._check_tasks:
                    self._check_tasks[name] = asyncio.create_task(
                        self._run_check_loop(check)
                    )

            logger.info("Started health monitoring")

    async def stop(self) -> None:
        """Stop all health check loops."""
        async with self._lock:
            self._shutdown = True

            # Cancel all tasks
            for task in self._check_tasks.values():
                task.cancel()

            # Wait for tasks to complete
            if self._check_tasks:
                await asyncio.gather(
                    *self._check_tasks.values(), return_exceptions=True
                )

            self._check_tasks.clear()
            logger.info("Stopped health monitoring")


# Global instances
metrics_registry = MetricsRegistry()
health_monitor = HealthMonitor()


# Convenience functions
def counter(
    name: str, description: str = "", labels: Optional[Dict[str, str]] = None
) -> Metric:
    """Create or get a counter metric."""
    return metrics_registry.create_counter(name, description, labels)


def gauge(
    name: str, description: str = "", labels: Optional[Dict[str, str]] = None
) -> Metric:
    """Create or get a gauge metric."""
    return metrics_registry.create_gauge(name, description, labels)


def histogram(
    name: str, description: str = "", labels: Optional[Dict[str, str]] = None
) -> Metric:
    """Create or get a histogram metric."""
    return metrics_registry.create_histogram(name, description, labels)


def timer(
    name: str, description: str = "", labels: Optional[Dict[str, str]] = None
) -> Metric:
    """Create or get a timer metric."""
    return metrics_registry.create_timer(name, description, labels)


def timed(metric_name: str, description: str = ""):
    """Decorator for timing function execution."""

    def decorator(func):
        timer_metric = timer(metric_name, description)

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                return await timer_metric.time_async_function(func, *args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                return timer_metric.time_function(func, *args, **kwargs)

            return sync_wrapper

    return decorator


class MetricsContext:
    """Context manager for collecting metrics during a block of code."""

    def __init__(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Initialize metrics context.

        Args:
            operation_name: Name of the operation being measured
            labels: Optional labels to apply to metrics
        """
        self.operation_name = operation_name
        self.labels = labels or {}

        # Metrics
        self.timer_metric = histogram(
            f"{operation_name}_duration_seconds",
            f"Duration of {operation_name} operations in seconds",
            self.labels,
        )
        self.success_counter = counter(
            f"{operation_name}_success_total",
            f"Total successful {operation_name} operations",
            self.labels,
        )
        self.error_counter = counter(
            f"{operation_name}_error_total",
            f"Total failed {operation_name} operations",
            self.labels,
        )

        self.start_time = None
        self.success = False

    async def __aenter__(self):
        """Enter async context."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.timer_metric.observe(duration)

        if exc_type is None:
            self.success_counter.increment()
        else:
            error_labels = {**self.labels, "error_type": exc_type.__name__}
            self.error_counter.increment(labels=error_labels)

        return False  # Don't suppress exceptions

    def __enter__(self):
        """Enter sync context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sync context."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.timer_metric.observe(duration)

        if exc_type is None:
            self.success_counter.increment()
        else:
            error_labels = {**self.labels, "error_type": exc_type.__name__}
            self.error_counter.increment(labels=error_labels)

        return False  # Don't suppress exceptions
