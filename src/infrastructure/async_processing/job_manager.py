"""
Job Manager for async processing pipeline.

This module provides comprehensive job management functionality including
priority queues, resource allocation, job scheduling, monitoring, and
lifecycle management for stream processing workflows.
"""

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

import structlog
from celery import chain

from src.core.cache import get_redis_client
from src.core.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.infrastructure.async_processing.celery_app import celery_app, get_task_options


logger = structlog.get_logger(__name__)


class JobPriority(str, Enum):
    """Job priority levels for queue management."""

    HIGH = "high"  # Premium customers, urgent jobs
    MEDIUM = "medium"  # Standard customers, normal jobs
    LOW = "low"  # Basic customers, batch jobs


class JobStatus(str, Enum):
    """Job status values for tracking."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ResourceLimits:
    """Resource limits for different customer tiers."""

    LIMITS = {
        JobPriority.HIGH: {
            "max_concurrent_jobs": 10,
            "max_job_duration_hours": 24,
            "max_file_size_gb": 50,
            "cpu_priority": 10,
            "memory_limit_gb": 16,
        },
        JobPriority.MEDIUM: {
            "max_concurrent_jobs": 5,
            "max_job_duration_hours": 12,
            "max_file_size_gb": 25,
            "cpu_priority": 5,
            "memory_limit_gb": 8,
        },
        JobPriority.LOW: {
            "max_concurrent_jobs": 2,
            "max_job_duration_hours": 6,
            "max_file_size_gb": 10,
            "cpu_priority": 1,
            "memory_limit_gb": 4,
        },
    }

    @classmethod
    def get_limits(cls, priority: JobPriority) -> Dict[str, Any]:
        """Get resource limits for a priority level."""
        return cls.LIMITS.get(priority, cls.LIMITS[JobPriority.MEDIUM])


class JobManager:
    """
    Comprehensive job manager for async processing pipeline.

    Handles job creation, scheduling, monitoring, resource allocation,
    and lifecycle management with priority-based queuing.
    """

    def __init__(self):
        """Initialize job manager with Redis client for state management."""
        self.redis_client = get_redis_client()
        self.job_prefix = "job:"
        self.queue_prefix = "queue:"
        self.resource_prefix = "resource:"

    def create_job(
        self,
        stream_id: int,
        user_id: int,
        priority: JobPriority = JobPriority.MEDIUM,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new processing job for a stream.

        Args:
            stream_id: ID of the stream to process
            user_id: ID of the user who owns the stream
            priority: Job priority level
            options: Additional job options and configuration

        Returns:
            str: Unique job ID

        Raises:
            ValueError: If resource limits are exceeded
            RuntimeError: If job creation fails
        """
        logger.info(
            "Creating new job",
            stream_id=stream_id,
            user_id=user_id,
            priority=priority.value,
        )

        try:
            # Generate unique job ID
            job_id = str(uuid4())

            # Check resource limits
            self._check_resource_limits(user_id, priority)

            # Create job metadata
            job_data = {
                "job_id": job_id,
                "stream_id": stream_id,
                "user_id": user_id,
                "priority": priority.value,
                "status": JobStatus.PENDING.value,
                "options": options or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "estimated_duration_minutes": options.get("estimated_duration", 60)
                if options
                else 60,
                "resource_limits": ResourceLimits.get_limits(priority),
                "retry_count": 0,
                "max_retries": 3,
            }

            # Store job data in Redis
            job_key = f"{self.job_prefix}{job_id}"
            self.redis_client.hset(
                job_key,
                mapping={
                    k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in job_data.items()
                },
            )

            # Set job expiration (cleanup after completion)
            self.redis_client.expire(job_key, 86400 * 7)  # 7 days

            # Add to priority queue
            self._add_to_queue(job_id, priority)

            # Track resource allocation
            self._allocate_resources(user_id, priority, job_id)

            logger.info("Job created successfully", job_id=job_id, stream_id=stream_id)
            return job_id

        except Exception as e:
            logger.error("Failed to create job", error=str(e), stream_id=stream_id)
            raise RuntimeError(f"Job creation failed: {str(e)}")

    def start_job(self, job_id: str) -> Dict[str, Any]:
        """
        Start processing a job by dispatching it to Celery workers.

        Args:
            job_id: ID of the job to start

        Returns:
            Dict[str, Any]: Job start result with task information

        Raises:
            ValueError: If job not found or cannot be started
            RuntimeError: If job dispatch fails
        """
        logger.info("Starting job", job_id=job_id)

        try:
            # Get job data
            job_data = self.get_job(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")

            if job_data["status"] != JobStatus.PENDING.value:
                raise ValueError(f"Job {job_id} is not in pending state")

            # Update job status
            self.update_job_status(job_id, JobStatus.QUEUED)

            # Get task options based on priority
            # Get max job duration with fallback
            resource_limits = job_data.get("resource_limits", {})
            max_duration = resource_limits.get(
                "max_job_duration_hours", 12
            )  # Default 12 hours

            task_options = get_task_options(
                priority=job_data["priority"],
                countdown=0,
                expires=max_duration * 3600,
            )

            # Create Celery workflow chain
            workflow = self._create_workflow_chain(job_data, task_options)

            # Dispatch workflow
            result = workflow.apply_async()

            # Update job with task information
            self.redis_client.hset(
                f"{self.job_prefix}{job_id}",
                mapping={
                    "celery_task_id": result.id,
                    "status": JobStatus.RUNNING.value,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Remove from queue (now running)
            self._remove_from_queue(job_id, JobPriority(job_data["priority"]))

            start_result = {
                "job_id": job_id,
                "celery_task_id": result.id,
                "status": JobStatus.RUNNING.value,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "workflow_tasks": len(workflow.tasks)
                if hasattr(workflow, "tasks")
                else 6,
            }

            logger.info("Job started successfully", **start_result)
            return start_result

        except Exception as e:
            logger.error("Failed to start job", job_id=job_id, error=str(e))
            # Update job status to failed
            self.update_job_status(job_id, JobStatus.FAILED)
            raise RuntimeError(f"Job start failed: {str(e)}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data by ID.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            Optional[Dict[str, Any]]: Job data or None if not found
        """
        try:
            job_key = f"{self.job_prefix}{job_id}"
            job_data = self.redis_client.hgetall(job_key)

            if not job_data:
                return None

            # Deserialize JSON fields
            for key in ["options", "resource_limits"]:
                if key in job_data:
                    try:
                        job_data[key] = json.loads(job_data[key])
                    except (json.JSONDecodeError, TypeError):
                        job_data[key] = {}

            return job_data

        except Exception as e:
            logger.error("Failed to get job", job_id=job_id, error=str(e))
            return None

    def update_job_status(
        self, job_id: str, status: JobStatus, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job status and optional details.

        Args:
            job_id: ID of the job to update
            status: New status for the job
            details: Optional additional details to store

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            job_key = f"{self.job_prefix}{job_id}"

            update_data = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add completion timestamp if job is finished
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

                # Release resources
                job_data = self.get_job(job_id)
                if job_data:
                    self._release_resources(
                        job_data["user_id"], JobPriority(job_data["priority"]), job_id
                    )

            # Add details if provided
            if details:
                update_data["details"] = json.dumps(details)

            # Update in Redis
            self.redis_client.hset(job_key, mapping=update_data)

            logger.info("Job status updated", job_id=job_id, status=status.value)
            return True

        except Exception as e:
            logger.error("Failed to update job status", job_id=job_id, error=str(e))
            return False

    def cancel_job(self, job_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancel a running or pending job.

        Args:
            job_id: ID of the job to cancel
            reason: Reason for cancellation

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        logger.info("Cancelling job", job_id=job_id, reason=reason)

        try:
            job_data = self.get_job(job_id)
            if not job_data:
                logger.warning("Cannot cancel job - not found", job_id=job_id)
                return False

            current_status = job_data["status"]

            # Can only cancel pending, queued, or running jobs
            if current_status not in [
                JobStatus.PENDING.value,
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
            ]:
                logger.warning(
                    "Cannot cancel job in current status",
                    job_id=job_id,
                    status=current_status,
                )
                return False

            # Cancel Celery task if running
            if (
                "celery_task_id" in job_data
                and current_status == JobStatus.RUNNING.value
            ):
                celery_app.control.revoke(job_data["celery_task_id"], terminate=True)

            # Remove from queue if not yet running
            if current_status in [JobStatus.PENDING.value, JobStatus.QUEUED.value]:
                self._remove_from_queue(job_id, JobPriority(job_data["priority"]))

            # Update job status
            self.update_job_status(job_id, JobStatus.CANCELLED, {"reason": reason})

            # Update stream status in database
            try:
                with get_db_session() as db:
                    stream = (
                        db.query(Stream)
                        .filter(Stream.id == int(job_data["stream_id"]))
                        .first()
                    )
                    if stream:
                        stream.status = StreamStatus.CANCELLED
                        db.commit()
            except Exception as e:
                logger.error(
                    "Failed to update stream status on cancellation", error=str(e)
                )

            logger.info("Job cancelled successfully", job_id=job_id)
            return True

        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get status of all priority queues.

        Returns:
            Dict[str, Any]: Queue status information
        """
        try:
            queue_status = {}

            for priority in JobPriority:
                queue_key = f"{self.queue_prefix}{priority.value}"
                queue_length = self.redis_client.llen(queue_key)

                # Get job details for jobs in queue
                job_ids = self.redis_client.lrange(queue_key, 0, -1)
                jobs = []
                for job_id in job_ids:
                    job_data = self.get_job(
                        job_id.decode() if isinstance(job_id, bytes) else job_id
                    )
                    if job_data:
                        jobs.append(
                            {
                                "job_id": job_data["job_id"],
                                "stream_id": job_data["stream_id"],
                                "created_at": job_data["created_at"],
                            }
                        )

                queue_status[priority.value] = {"length": queue_length, "jobs": jobs}

            return queue_status

        except Exception as e:
            logger.error("Failed to get queue status", error=str(e))
            return {}

    def get_resource_usage(self, user_id: int) -> Dict[str, Any]:
        """
        Get current resource usage for a user.

        Args:
            user_id: ID of the user

        Returns:
            Dict[str, Any]: Resource usage information
        """
        try:
            usage = {}

            for priority in JobPriority:
                resource_key = f"{self.resource_prefix}{user_id}:{priority.value}"
                current_jobs = self.redis_client.smembers(resource_key)
                limits = ResourceLimits.get_limits(priority)

                usage[priority.value] = {
                    "current_jobs": len(current_jobs),
                    "max_concurrent_jobs": limits["max_concurrent_jobs"],
                    "available_slots": limits["max_concurrent_jobs"]
                    - len(current_jobs),
                    "active_job_ids": list(current_jobs),
                }

            return usage

        except Exception as e:
            logger.error("Failed to get resource usage", user_id=user_id, error=str(e))
            return {}

    def cleanup_expired_jobs(self, max_age_hours: int = 168) -> Dict[str, Any]:
        """
        Clean up expired jobs and their resources.

        Args:
            max_age_hours: Maximum age in hours for jobs to keep (default: 7 days)

        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info("Starting job cleanup", max_age_hours=max_age_hours)

        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            cleaned_jobs = 0
            released_resources = 0

            # Find all job keys
            job_pattern = f"{self.job_prefix}*"
            job_keys = self.redis_client.keys(job_pattern)

            for job_key in job_keys:
                try:
                    job_data = self.redis_client.hgetall(job_key)
                    if not job_data:
                        continue

                    # Check if job is old enough to clean up
                    created_at = datetime.fromisoformat(job_data.get("created_at", ""))
                    if created_at < cutoff_time:
                        job_id = job_data.get("job_id")

                        # Release resources if still allocated
                        if job_data.get("status") not in [
                            JobStatus.COMPLETED.value,
                            JobStatus.FAILED.value,
                            JobStatus.CANCELLED.value,
                        ]:
                            user_id = job_data.get("user_id")
                            priority = JobPriority(
                                job_data.get("priority", JobPriority.MEDIUM.value)
                            )
                            self._release_resources(user_id, priority, job_id)
                            released_resources += 1

                        # Delete job data
                        self.redis_client.delete(job_key)
                        cleaned_jobs += 1

                except Exception as e:
                    logger.error(
                        "Failed to clean up individual job",
                        job_key=job_key,
                        error=str(e),
                    )
                    continue

            cleanup_result = {
                "cleaned_jobs": cleaned_jobs,
                "released_resources": released_resources,
                "cutoff_time": cutoff_time.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info("Job cleanup completed", **cleanup_result)
            return cleanup_result

        except Exception as e:
            logger.error("Failed to cleanup expired jobs", error=str(e))
            return {"error": str(e)}

    def _check_resource_limits(self, user_id: int, priority: JobPriority) -> None:
        """Check if user has available resources for the priority level."""
        resource_key = f"{self.resource_prefix}{user_id}:{priority.value}"
        current_jobs = self.redis_client.smembers(resource_key)
        limits = ResourceLimits.get_limits(priority)

        if len(current_jobs) >= limits["max_concurrent_jobs"]:
            raise ValueError(
                f"Resource limit exceeded: {len(current_jobs)}/{limits['max_concurrent_jobs']} concurrent jobs"
            )

    def _add_to_queue(self, job_id: str, priority: JobPriority) -> None:
        """Add job to the appropriate priority queue."""
        queue_key = f"{self.queue_prefix}{priority.value}"
        self.redis_client.lpush(queue_key, job_id)

    def _remove_from_queue(self, job_id: str, priority: JobPriority) -> None:
        """Remove job from the priority queue."""
        queue_key = f"{self.queue_prefix}{priority.value}"
        self.redis_client.lrem(queue_key, 1, job_id)

    def _allocate_resources(
        self, user_id: int, priority: JobPriority, job_id: str
    ) -> None:
        """Allocate resources for a job."""
        resource_key = f"{self.resource_prefix}{user_id}:{priority.value}"
        self.redis_client.sadd(resource_key, job_id)
        # Set expiration for automatic cleanup
        self.redis_client.expire(resource_key, 86400 * 7)  # 7 days

    def _release_resources(
        self, user_id: int, priority: JobPriority, job_id: str
    ) -> None:
        """Release resources for a completed job."""
        resource_key = f"{self.resource_prefix}{user_id}:{priority.value}"
        self.redis_client.srem(resource_key, job_id)

    def _create_workflow_chain(
        self, job_data: Dict[str, Any], task_options: Dict[str, Any]
    ) -> chain:
        """Create Celery workflow chain for stream processing."""
        from src.infrastructure.async_processing.tasks import (
            start_stream_processing,
            ingest_stream_data,
            process_multimodal_content,
            detect_highlights,
            finalize_highlights,
            notify_completion,
        )

        # Create task chain with proper error handling
        workflow = chain(
            start_stream_processing.s(
                stream_id=int(job_data["stream_id"]), options=job_data["options"]
            ).set(**task_options),
            ingest_stream_data.s(stream_id=int(job_data["stream_id"])).set(
                **task_options
            ),
            process_multimodal_content.s(stream_id=int(job_data["stream_id"])).set(
                **task_options
            ),
            detect_highlights.s(stream_id=int(job_data["stream_id"])).set(
                **task_options
            ),
            finalize_highlights.s(stream_id=int(job_data["stream_id"])).set(
                **task_options
            ),
            notify_completion.s(stream_id=int(job_data["stream_id"])).set(
                **task_options
            ),
        )

        return workflow
