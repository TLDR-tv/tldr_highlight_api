"""
Job Utilities for async processing pipeline.

This module provides utility functions for job monitoring, health checks,
cleanup operations, metrics collection, and system maintenance for the
async processing pipeline.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import func, text

from src.core.cache import get_redis_client
from src.core.config import get_settings
from src.core.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus

# from src.infrastructure.persistence.models.webhook import WebhookAttempt  # TODO: WebhookAttempt model not yet implemented
from src.services.async_processing.celery_app import celery_app


logger = structlog.get_logger(__name__)
settings = get_settings()


class JobMonitor:
    """
    Comprehensive job monitoring and metrics collection.

    Provides real-time monitoring of job queues, worker health,
    performance metrics, and system resource utilization.
    """

    def __init__(self) -> None:
        """Initialize job monitor with Redis client."""
        self.redis_client = get_redis_client()
        self.metrics_prefix = "metrics:"
        self.health_prefix = "health:"

    def get_queue_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive queue metrics and statistics.

        Returns:
            Dict[str, Any]: Queue metrics including lengths, rates, and performance
        """
        try:
            inspect = celery_app.control.inspect()

            # Get queue lengths
            active_queues = inspect.active_queues() or {}
            queue_lengths = {}

            for worker, queues in active_queues.items():
                for queue_info in queues:
                    queue_name = queue_info.get("name", "unknown")
                    if queue_name not in queue_lengths:
                        queue_lengths[queue_name] = 0

                    # This would normally get actual queue length from broker
                    # For now, we'll simulate based on active tasks
                    queue_lengths[queue_name] += len(queue_info.get("messages", []))

            # Get active tasks
            active_tasks = inspect.active() or {}
            total_active = sum(len(tasks) for tasks in active_tasks.values())

            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled() or {}
            total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())

            # Get reserved tasks
            reserved_tasks = inspect.reserved() or {}
            total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())

            # Calculate processing rates
            processing_rates = self._calculate_processing_rates()

            return {
                "queue_lengths": queue_lengths,
                "total_active_tasks": total_active,
                "total_scheduled_tasks": total_scheduled,
                "total_reserved_tasks": total_reserved,
                "active_tasks_by_worker": {
                    worker: len(tasks) for worker, tasks in active_tasks.items()
                },
                "processing_rates": processing_rates,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get queue metrics", error=str(e))
            return {"error": str(e)}

    def get_worker_health(self) -> Dict[str, Any]:
        """
        Get worker health status and performance metrics.

        Returns:
            Dict[str, Any]: Worker health information
        """
        try:
            inspect = celery_app.control.inspect()

            # Get worker stats
            stats = inspect.stats() or {}

            # Get worker status
            ping_results = inspect.ping() or {}

            # Process worker information
            workers = {}
            for worker_name in stats.keys():
                worker_stats = stats.get(worker_name, {})
                ping_result = ping_results.get(worker_name, {})

                workers[worker_name] = {
                    "status": "online"
                    if ping_result.get("ok") == "pong"
                    else "offline",
                    "total_tasks": worker_stats.get("total", 0),
                    "pool_info": worker_stats.get("pool", {}),
                    "rusage": worker_stats.get("rusage", {}),
                    "clock": worker_stats.get("clock", 0),
                    "load_avg": self._get_worker_load_avg(worker_name),
                }

            # Calculate overall health
            total_workers = len(workers)
            online_workers = len(
                [w for w in workers.values() if w["status"] == "online"]
            )
            health_percentage = (
                (online_workers / total_workers * 100) if total_workers > 0 else 0
            )

            return {
                "total_workers": total_workers,
                "online_workers": online_workers,
                "offline_workers": total_workers - online_workers,
                "health_percentage": health_percentage,
                "workers": workers,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get worker health", error=str(e))
            return {"error": str(e)}

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system-wide metrics for the async processing pipeline.

        Returns:
            Dict[str, Any]: System metrics including throughput, errors, and performance
        """
        try:
            metrics = {}

            # Database metrics
            with get_db_session() as db:
                # Stream statistics
                stream_stats = (
                    db.query(StreamStatus, func.count(Stream.id).label("count"))
                    .filter(
                        Stream.created_at >= datetime.utcnow() - timedelta(hours=24)
                    )
                    .group_by(Stream.status)
                    .all()
                )

                metrics["stream_stats_24h"] = {
                    status.value: count for status, count in stream_stats
                }

                # Total streams
                total_streams = db.query(func.count(Stream.id)).scalar()
                metrics["total_streams"] = total_streams

                # Webhook attempt statistics
                # TODO: Uncomment when WebhookAttempt model is implemented
                # webhook_stats = (
                #     db.query(
                #         func.count(WebhookAttempt.id).label("total"),
                #         func.sum(
                #             func.case(
                #                 (
                #                     WebhookAttempt.response_status_code.between(
                #                         200, 299
                #                     ),
                #                     1,
                #                 ),
                #                 else_=0,
                #             )
                #         ).label("successful"),
                #         func.avg(WebhookAttempt.response_time_ms).label(
                #             "avg_response_time"
                #         ),
                #     )
                #     .filter(
                #         WebhookAttempt.created_at
                #         >= datetime.utcnow() - timedelta(hours=24)
                #     )
                #     .first()
                # )
                webhook_stats = (
                    0,
                    0,
                    0,
                )  # Placeholder: (total, successful, avg_response_time)

                metrics["webhook_stats_24h"] = {
                    "total_attempts": webhook_stats[0] or 0,
                    "successful_attempts": webhook_stats[1] or 0,
                    "average_response_time_ms": float(webhook_stats[2] or 0),
                    "success_rate": (
                        (webhook_stats[1] / webhook_stats[0] * 100)
                        if webhook_stats[0] and webhook_stats[1]
                        else 0
                    ),
                }

            # Redis metrics
            redis_info = self.redis_client.info()
            metrics["redis_stats"] = {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory", 0),
                "used_memory_human": redis_info.get("used_memory_human", "0B"),
                "total_commands_processed": redis_info.get(
                    "total_commands_processed", 0
                ),
                "keyspace_hits": redis_info.get("keyspace_hits", 0),
                "keyspace_misses": redis_info.get("keyspace_misses", 0),
            }

            # Calculate hit rate
            hits = redis_info.get("keyspace_hits", 0)
            misses = redis_info.get("keyspace_misses", 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            metrics["redis_stats"]["hit_rate_percentage"] = hit_rate

            # Processing throughput
            metrics["throughput"] = self._calculate_throughput()

            # Error rates
            metrics["error_rates"] = self._calculate_error_rates()

            metrics["timestamp"] = datetime.utcnow().isoformat()

            return metrics

        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e)}

    def record_task_metrics(
        self,
        task_name: str,
        execution_time: float,
        success: bool,
        stream_id: Optional[int] = None,
    ) -> None:
        """
        Record metrics for task execution.

        Args:
            task_name: Name of the executed task
            execution_time: Task execution time in seconds
            success: Whether the task was successful
            stream_id: Optional stream ID for task
        """
        try:
            timestamp = datetime.utcnow()

            # Create metrics record
            metrics_data = {
                "task_name": task_name,
                "execution_time": execution_time,
                "success": success,
                "stream_id": stream_id,
                "timestamp": timestamp.isoformat(),
                "hour": timestamp.strftime("%Y-%m-%d_%H"),  # For hourly aggregation
                "date": timestamp.strftime("%Y-%m-%d"),  # For daily aggregation
            }

            # Store in Redis with expiration
            metrics_key = (
                f"{self.metrics_prefix}task:{task_name}:{int(timestamp.timestamp())}"
            )
            self.redis_client.hset(
                metrics_key,
                mapping={
                    k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in metrics_data.items()
                },
            )

            # Set expiration for 7 days
            self.redis_client.expire(metrics_key, 86400 * 7)

            # Update aggregated metrics
            self._update_aggregated_metrics(
                task_name, execution_time, success, timestamp
            )

        except Exception as e:
            logger.error(
                "Failed to record task metrics", task_name=task_name, error=str(e)
            )

    def get_task_performance(
        self, task_name: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get task performance metrics.

        Args:
            task_name: Optional specific task name to filter by
            hours: Number of hours to look back

        Returns:
            Dict[str, Any]: Task performance metrics
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Find relevant metrics keys
            if task_name:
                pattern = f"{self.metrics_prefix}task:{task_name}:*"
            else:
                pattern = f"{self.metrics_prefix}task:*"

            metrics_keys = self.redis_client.keys(pattern)

            # Process metrics
            task_metrics = {}
            total_executions = 0
            total_success = 0
            total_execution_time = 0

            for key in metrics_keys:
                try:
                    metrics_data = self.redis_client.hgetall(key)
                    if not metrics_data:
                        continue

                    # Check if within time range
                    timestamp = datetime.fromisoformat(
                        metrics_data.get("timestamp", "")
                    )
                    if timestamp < cutoff_time:
                        continue

                    task = metrics_data.get("task_name", "unknown")
                    execution_time = float(metrics_data.get("execution_time", 0))
                    success = metrics_data.get("success", "false").lower() == "true"

                    # Initialize task metrics if not exists
                    if task not in task_metrics:
                        task_metrics[task] = {
                            "total_executions": 0,
                            "successful_executions": 0,
                            "failed_executions": 0,
                            "total_execution_time": 0,
                            "min_execution_time": float("inf"),
                            "max_execution_time": 0,
                            "execution_times": [],
                        }

                    # Update task metrics
                    task_metrics[task]["total_executions"] += 1
                    task_metrics[task]["total_execution_time"] += execution_time
                    task_metrics[task]["execution_times"].append(execution_time)

                    if success:
                        task_metrics[task]["successful_executions"] += 1
                    else:
                        task_metrics[task]["failed_executions"] += 1

                    # Update min/max
                    task_metrics[task]["min_execution_time"] = min(
                        task_metrics[task]["min_execution_time"], execution_time
                    )
                    task_metrics[task]["max_execution_time"] = max(
                        task_metrics[task]["max_execution_time"], execution_time
                    )

                    # Update totals
                    total_executions += 1
                    if success:
                        total_success += 1
                    total_execution_time += execution_time

                except Exception as e:
                    logger.error("Failed to process metrics key", key=key, error=str(e))
                    continue

            # Calculate derived metrics
            for task, metrics in task_metrics.items():
                executions = metrics["total_executions"]
                if executions > 0:
                    metrics["success_rate"] = (
                        metrics["successful_executions"] / executions
                    ) * 100
                    metrics["average_execution_time"] = (
                        metrics["total_execution_time"] / executions
                    )

                    # Calculate percentiles
                    execution_times = sorted(metrics["execution_times"])
                    if execution_times:
                        metrics["p50_execution_time"] = self._percentile(
                            execution_times, 50
                        )
                        metrics["p95_execution_time"] = self._percentile(
                            execution_times, 95
                        )
                        metrics["p99_execution_time"] = self._percentile(
                            execution_times, 99
                        )

                # Clean up raw execution times (not needed in response)
                del metrics["execution_times"]

                # Fix infinity values
                if metrics["min_execution_time"] == float("inf"):
                    metrics["min_execution_time"] = 0

            # Overall metrics
            overall_metrics = {
                "total_executions": total_executions,
                "successful_executions": total_success,
                "failed_executions": total_executions - total_success,
                "overall_success_rate": (total_success / total_executions * 100)
                if total_executions > 0
                else 0,
                "overall_average_execution_time": (
                    total_execution_time / total_executions
                )
                if total_executions > 0
                else 0,
            }

            return {
                "time_range_hours": hours,
                "overall": overall_metrics,
                "by_task": task_metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get task performance", error=str(e))
            return {"error": str(e)}

    def _calculate_processing_rates(self) -> Dict[str, float]:
        """Calculate processing rates for different metrics."""
        try:
            # This would calculate actual processing rates
            # For now, return simulated rates
            return {
                "streams_per_hour": 42.5,
                "highlights_per_hour": 127.3,
                "webhooks_per_hour": 89.1,
                "tasks_per_minute": 15.7,
            }
        except Exception as e:
            logger.error("Failed to calculate processing rates", error=str(e))
            return {}

    def _calculate_throughput(self) -> Dict[str, Any]:
        """Calculate system throughput metrics."""
        try:
            # Calculate throughput based on completed streams in last hour
            cutoff_time = datetime.utcnow() - timedelta(hours=1)

            with get_db_session() as db:
                completed_streams = (
                    db.query(func.count(Stream.id))
                    .filter(
                        Stream.status == StreamStatus.COMPLETED,
                        Stream.completed_at >= cutoff_time,
                    )
                    .scalar()
                )

            return {
                "streams_completed_last_hour": completed_streams or 0,
                "estimated_streams_per_day": (completed_streams or 0) * 24,
            }

        except Exception as e:
            logger.error("Failed to calculate throughput", error=str(e))
            return {}

    def _calculate_error_rates(self) -> Dict[str, Any]:
        """Calculate error rates for different components."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            with get_db_session() as db:
                # Stream error rate
                total_streams = (
                    db.query(func.count(Stream.id))
                    .filter(Stream.created_at >= cutoff_time)
                    .scalar()
                )

                failed_streams = (
                    db.query(func.count(Stream.id))
                    .filter(
                        Stream.status == StreamStatus.FAILED,
                        Stream.created_at >= cutoff_time,
                    )
                    .scalar()
                )

                stream_error_rate = (
                    (failed_streams / total_streams * 100) if total_streams > 0 else 0
                )

                # Webhook error rate
                # TODO: Uncomment when WebhookAttempt model is implemented
                # total_webhooks = (
                #     db.query(func.count(WebhookAttempt.id))
                #     .filter(WebhookAttempt.created_at >= cutoff_time)
                #     .scalar()
                # )
                #
                # failed_webhooks = (
                #     db.query(func.count(WebhookAttempt.id))
                #     .filter(
                #         WebhookAttempt.response_status_code >= 400,
                #         WebhookAttempt.created_at >= cutoff_time,
                #     )
                #     .scalar()
                # )
                total_webhooks = 0  # Placeholder
                failed_webhooks = 0  # Placeholder

                webhook_error_rate = (
                    (failed_webhooks / total_webhooks * 100)
                    if total_webhooks > 0
                    else 0
                )

            return {
                "stream_error_rate_24h": stream_error_rate,
                "webhook_error_rate_24h": webhook_error_rate,
                "total_streams_24h": total_streams or 0,
                "failed_streams_24h": failed_streams or 0,
                "total_webhooks_24h": total_webhooks or 0,
                "failed_webhooks_24h": failed_webhooks or 0,
            }

        except Exception as e:
            logger.error("Failed to calculate error rates", error=str(e))
            return {}

    def _get_worker_load_avg(self, worker_name: str) -> List[float]:
        """Get worker load average (simulated for now)."""
        # In a real implementation, this would get actual system load
        return [1.2, 1.5, 1.8]  # 1min, 5min, 15min averages

    def _update_aggregated_metrics(
        self, task_name: str, execution_time: float, success: bool, timestamp: datetime
    ) -> None:
        """Update aggregated metrics for faster querying."""
        try:
            hour_key = f"{self.metrics_prefix}hourly:{task_name}:{timestamp.strftime('%Y-%m-%d_%H')}"
            day_key = f"{self.metrics_prefix}daily:{task_name}:{timestamp.strftime('%Y-%m-%d')}"

            # Update hourly aggregates
            self.redis_client.hincrby(hour_key, "total_executions", 1)
            self.redis_client.hincrby(
                hour_key, "total_execution_time", int(execution_time * 1000)
            )  # milliseconds

            if success:
                self.redis_client.hincrby(hour_key, "successful_executions", 1)
            else:
                self.redis_client.hincrby(hour_key, "failed_executions", 1)

            self.redis_client.expire(hour_key, 86400 * 7)  # 7 days

            # Update daily aggregates
            self.redis_client.hincrby(day_key, "total_executions", 1)
            self.redis_client.hincrby(
                day_key, "total_execution_time", int(execution_time * 1000)
            )

            if success:
                self.redis_client.hincrby(day_key, "successful_executions", 1)
            else:
                self.redis_client.hincrby(day_key, "failed_executions", 1)

            self.redis_client.expire(day_key, 86400 * 30)  # 30 days

        except Exception as e:
            logger.error("Failed to update aggregated metrics", error=str(e))

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value from sorted data."""
        if not data:
            return 0.0

        index = (percentile / 100) * (len(data) - 1)

        if index.is_integer():
            return data[int(index)]
        else:
            lower = data[int(index)]
            upper = data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class SystemHealthChecker:
    """
    Comprehensive system health checker for the async processing pipeline.

    Performs health checks on all system components including database,
    Redis, Celery workers, and external dependencies.
    """

    def __init__(self) -> None:
        """Initialize health checker."""
        self.redis_client = get_redis_client()

    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.

        Returns:
            Dict[str, Any]: Complete health status of all components
        """
        health_status: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {},
        }

        # Check individual components
        components = [
            ("database", self._check_database_health),
            ("redis", self._check_redis_health),
            ("celery_workers", self._check_celery_health),
            ("queues", self._check_queue_health),
            ("disk_space", self._check_disk_space),
            ("memory", self._check_memory_usage),
        ]

        unhealthy_components: List[str] = []

        for component_name, check_func in components:
            try:
                component_health = check_func()
                health_status["components"][component_name] = component_health

                if component_health.get("status") != "healthy":
                    unhealthy_components.append(component_name)

            except Exception as e:
                logger.error(f"Health check failed for {component_name}", error=str(e))
                health_status["components"][component_name] = {
                    "status": "error",
                    "error": str(e),
                }
                unhealthy_components.append(component_name)

        # Determine overall status
        if unhealthy_components:
            health_status["overall_status"] = "unhealthy"
            health_status["unhealthy_components"] = unhealthy_components

        return health_status

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()

            with get_db_session() as db:
                # Test basic connectivity
                db.execute(text("SELECT 1"))

                # Test table access
                stream_count = db.query(func.count(Stream.id)).scalar()

                # Check recent activity
                recent_streams = (
                    db.query(func.count(Stream.id))
                    .filter(
                        Stream.created_at >= datetime.utcnow() - timedelta(minutes=30)
                    )
                    .scalar()
                )

            response_time = (time.time() - start_time) * 1000  # milliseconds

            status = "healthy"
            if response_time > 1000:  # 1 second
                status = "slow"
            elif response_time > 5000:  # 5 seconds
                status = "unhealthy"

            return {
                "status": status,
                "response_time_ms": response_time,
                "total_streams": stream_count,
                "recent_activity": recent_streams,
                "connection_pool": "healthy",  # Would check actual pool status
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "response_time_ms": None}

    def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()

            # Test connectivity
            ping_result = self.redis_client.ping()

            # Test read/write
            test_key = f"health_check:{int(time.time())}"
            self.redis_client.set(test_key, "test", ex=60)
            test_value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)

            response_time = (time.time() - start_time) * 1000  # milliseconds

            # Get Redis info
            redis_info = self.redis_client.info()

            status = "healthy"
            if response_time > 100:  # 100ms
                status = "slow"
            elif response_time > 1000:  # 1 second
                status = "unhealthy"

            return {
                "status": status,
                "ping": ping_result,
                "read_write_test": test_value == "test",
                "response_time_ms": response_time,
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "keyspace_hits": redis_info.get("keyspace_hits", 0),
                "keyspace_misses": redis_info.get("keyspace_misses", 0),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "response_time_ms": None}

    def _check_celery_health(self) -> Dict[str, Any]:
        """Check Celery worker health."""
        try:
            inspect = celery_app.control.inspect()

            # Ping workers
            ping_results = inspect.ping() or {}

            # Get worker stats
            stats = inspect.stats() or {}

            online_workers = len(
                [w for w in ping_results.values() if w.get("ok") == "pong"]
            )
            total_workers = len(ping_results)

            status = "healthy"
            if online_workers == 0:
                status = "unhealthy"
            elif online_workers < total_workers:
                status = "degraded"

            return {
                "status": status,
                "online_workers": online_workers,
                "total_workers": total_workers,
                "worker_details": {
                    worker: {
                        "ping": ping_results.get(worker, {}).get("ok", "no_response"),
                        "total_tasks": stats.get(worker, {}).get("total", 0),
                    }
                    for worker in ping_results.keys()
                },
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_queue_health(self) -> Dict[str, Any]:
        """Check queue health and backlog."""
        try:
            inspect = celery_app.control.inspect()

            # Get queue information
            active_queues = inspect.active_queues() or {}

            total_active = 0
            queue_info = {}

            for worker, queues in active_queues.items():
                for queue in queues:
                    queue_name = queue.get("name", "unknown")
                    if queue_name not in queue_info:
                        queue_info[queue_name] = {"workers": 0, "messages": 0}

                    queue_info[queue_name]["workers"] += 1
                    queue_info[queue_name]["messages"] += len(queue.get("messages", []))
                    total_active += len(queue.get("messages", []))

            # Determine status based on queue backlog
            status = "healthy"
            if total_active > 1000:  # High backlog
                status = "overloaded"
            elif total_active > 100:  # Medium backlog
                status = "busy"

            return {
                "status": status,
                "total_active_messages": total_active,
                "queue_details": queue_info,
                "worker_count": len(active_queues),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil

            # Check disk space for current directory
            total, used, free = shutil.disk_usage("/")

            # Convert to GB
            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)

            usage_percentage = (used / total) * 100

            status = "healthy"
            if usage_percentage > 90:
                status = "critical"
            elif usage_percentage > 80:
                status = "warning"

            return {
                "status": status,
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "usage_percentage": round(usage_percentage, 2),
            }

        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"

            return {
                "status": status,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percentage": round(memory.percent, 2),
            }

        except ImportError:
            # psutil not available
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "unknown", "error": str(e)}


def cleanup_expired_data(max_age_hours: int = 168) -> Dict[str, Any]:
    """
    Clean up expired data across all components.

    Args:
        max_age_hours: Maximum age in hours for data to keep (default: 7 days)

    Returns:
        Dict[str, Any]: Cleanup statistics
    """
    logger.info("Starting comprehensive data cleanup", max_age_hours=max_age_hours)

    try:
        from src.services.async_processing.job_manager import JobManager
        from src.services.async_processing.progress_tracker import ProgressTracker
        from src.services.async_processing.webhook_dispatcher import WebhookDispatcher
        from src.services.async_processing.workflow import StreamProcessingWorkflow

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleanup_stats = {}

        # Job manager cleanup
        job_manager = JobManager()
        cleanup_stats["jobs"] = job_manager.cleanup_expired_jobs(max_age_hours)

        # Progress tracker cleanup
        progress_tracker = ProgressTracker()
        cleanup_stats["progress"] = progress_tracker.cleanup_old_progress(cutoff_time)

        # Webhook dispatcher cleanup
        webhook_dispatcher = WebhookDispatcher()
        cleanup_stats["webhooks"] = webhook_dispatcher.cleanup_old_attempts(cutoff_time)

        # Workflow cleanup
        workflow = StreamProcessingWorkflow()
        cleanup_stats["workflows"] = workflow.cleanup_completed_workflows(max_age_hours)

        # Metrics cleanup
        redis_client = get_redis_client()
        metrics_cleaned = 0

        # Clean up old metrics
        metrics_pattern = "metrics:*"
        metrics_keys = redis_client.keys(metrics_pattern)

        for key in metrics_keys:
            try:
                # Check if key has TTL set, if not it might be old
                ttl = redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    redis_client.delete(key)
                    metrics_cleaned += 1
            except Exception:
                continue

        cleanup_stats["metrics"] = metrics_cleaned

        # Database cleanup of old streams
        with get_db_session() as db:
            old_streams = (
                db.query(Stream)
                .filter(
                    Stream.status.in_([StreamStatus.COMPLETED, StreamStatus.FAILED]),
                    Stream.completed_at < cutoff_time,
                )
                .count()
            )

            # In production, you might want to archive rather than delete
            # For now, we'll just count them
            cleanup_stats["old_streams"] = old_streams

        cleanup_stats.update(
            {
                "cutoff_time": cutoff_time.isoformat(),
                "max_age_hours": max_age_hours,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info("Comprehensive cleanup completed", **cleanup_stats)
        return cleanup_stats

    except Exception as e:
        logger.error("Failed to perform comprehensive cleanup", error=str(e))
        return {"error": str(e)}


def generate_system_report() -> Dict[str, Any]:
    """
    Generate a comprehensive system report.

    Returns:
        Dict[str, Any]: Complete system status report
    """
    logger.info("Generating system report")

    try:
        monitor = JobMonitor()
        health_checker = SystemHealthChecker()

        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health_checker.check_system_health(),
            "queue_metrics": monitor.get_queue_metrics(),
            "worker_health": monitor.get_worker_health(),
            "system_metrics": monitor.get_system_metrics(),
            "task_performance": monitor.get_task_performance(hours=24),
        }

        # Add summary
        overall_health = report["system_health"]["overall_status"]
        total_workers = report["worker_health"]["total_workers"]
        online_workers = report["worker_health"]["online_workers"]
        total_active_tasks = report["queue_metrics"]["total_active_tasks"]

        report["summary"] = {
            "overall_health": overall_health,
            "worker_utilization": f"{online_workers}/{total_workers}",
            "active_tasks": total_active_tasks,
            "system_load": "normal" if overall_health == "healthy" else "degraded",
        }

        logger.info("System report generated successfully")
        return report

    except Exception as e:
        logger.error("Failed to generate system report", error=str(e))
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
