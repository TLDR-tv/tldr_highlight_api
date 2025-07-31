"""
Progress Tracker for async processing pipeline.

This module provides comprehensive progress tracking functionality for stream
processing jobs, including real-time status updates, percentage completion,
event logging, and database synchronization.
"""

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import update

from src.infrastructure.cache import get_redis_client
from src.infrastructure.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream


logger = structlog.get_logger(__name__)


class ProgressEvent(str, Enum):
    """Types of progress events that can be tracked."""

    STARTED = "started"
    PROGRESS_UPDATE = "progress_update"
    COMPLETED = "completed"
    ERROR = "error"
    RETRY = "retry"
    CANCELLED = "cancelled"
    MILESTONE_REACHED = "milestone_reached"


class ProgressTracker:
    """
    Comprehensive progress tracking system for async processing jobs.

    Provides real-time progress updates, event logging, status management,
    and integration with both Redis cache and PostgreSQL database.
    """

    def __init__(self):
        """Initialize progress tracker with Redis client."""
        self.redis_client = get_redis_client()
        self.progress_prefix = "progress:"
        self.events_prefix = "events:"
        self.milestones_prefix = "milestones:"

        # Define processing milestones
        self.milestones = {
            5: "Stream processing started",
            15: "Stream data ingestion begun",
            30: "Stream data ingested successfully",
            45: "Multimodal content processing started",
            60: "Multimodal content processed",
            75: "AI highlight detection started",
            85: "Highlights detected and stored",
            95: "Highlights finalized",
            100: "Processing completed successfully",
        }

    def update_progress(
        self,
        stream_id: int,
        progress_percentage: Optional[int] = None,
        status: str = "processing",
        event_type: ProgressEvent = ProgressEvent.PROGRESS_UPDATE,
        details: Optional[Dict[str, Any]] = None,
        milestone: Optional[str] = None,
    ) -> bool:
        """
        Update progress for a stream processing job.

        Args:
            stream_id: ID of the stream being processed
            progress_percentage: Current progress percentage (0-100)
            status: Current processing status
            event_type: Type of progress event
            details: Additional event details
            milestone: Custom milestone message

        Returns:
            bool: True if update successful, False otherwise
        """
        logger.info(
            "Updating progress",
            stream_id=stream_id,
            progress=progress_percentage,
            status=status,
            event_type=event_type.value,
        )

        try:
            timestamp = datetime.now(timezone.utc)

            # Get current progress data
            current_progress = self.get_progress(stream_id) or {}

            # Update progress data
            progress_data = {
                "stream_id": stream_id,
                "progress_percentage": progress_percentage
                if progress_percentage is not None
                else current_progress.get("progress_percentage", 0),
                "status": status,
                "last_updated": timestamp.isoformat(),
                "details": details or {},
            }

            # Add creation time if this is the first update
            if not current_progress:
                progress_data["created_at"] = timestamp.isoformat()
                progress_data["started_at"] = timestamp.isoformat()
            else:
                progress_data["created_at"] = current_progress.get(
                    "created_at", timestamp.isoformat()
                )
                progress_data["started_at"] = current_progress.get(
                    "started_at", timestamp.isoformat()
                )

            # Add completion time if finished
            if status in ["completed", "failed", "cancelled"]:
                progress_data["completed_at"] = timestamp.isoformat()

                # Calculate processing duration
                if "started_at" in progress_data:
                    try:
                        started = datetime.fromisoformat(progress_data["started_at"])
                        duration = (timestamp - started).total_seconds()
                        progress_data["processing_duration_seconds"] = duration
                    except Exception as e:
                        logger.warning(
                            "Failed to calculate processing duration", error=str(e)
                        )

            # Store progress in Redis
            progress_key = f"{self.progress_prefix}{stream_id}"
            self.redis_client.hset(
                progress_key,
                mapping={
                    k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in progress_data.items()
                },
            )

            # Set expiration for automatic cleanup
            self.redis_client.expire(progress_key, 86400 * 7)  # 7 days

            # Log progress event
            self._log_progress_event(
                stream_id=stream_id,
                event_type=event_type,
                progress_percentage=progress_data["progress_percentage"],
                status=status,
                details=details or {},
                timestamp=timestamp,
            )

            # Check for milestones
            if progress_percentage is not None:
                self._check_milestone(stream_id, progress_percentage, milestone)

            # Update database status
            self._update_database_status(stream_id, status, progress_percentage)

            logger.info("Progress updated successfully", stream_id=stream_id)
            return True

        except Exception as e:
            logger.error("Failed to update progress", stream_id=stream_id, error=str(e))
            return False

    def get_progress(self, stream_id: int) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a stream.

        Args:
            stream_id: ID of the stream

        Returns:
            Optional[Dict[str, Any]]: Progress data or None if not found
        """
        try:
            progress_key = f"{self.progress_prefix}{stream_id}"
            progress_data = self.redis_client.hgetall(progress_key)

            if not progress_data:
                return None

            # Deserialize JSON fields
            for key in ["details"]:
                if key in progress_data:
                    try:
                        progress_data[key] = json.loads(progress_data[key])
                    except (json.JSONDecodeError, TypeError):
                        progress_data[key] = {}

            # Convert numeric fields
            if "progress_percentage" in progress_data:
                try:
                    progress_data["progress_percentage"] = int(
                        progress_data["progress_percentage"]
                    )
                except (ValueError, TypeError):
                    progress_data["progress_percentage"] = 0

            if "processing_duration_seconds" in progress_data:
                try:
                    progress_data["processing_duration_seconds"] = float(
                        progress_data["processing_duration_seconds"]
                    )
                except (ValueError, TypeError):
                    progress_data["processing_duration_seconds"] = None

            return progress_data

        except Exception as e:
            logger.error("Failed to get progress", stream_id=stream_id, error=str(e))
            return None

    def get_progress_events(
        self,
        stream_id: int,
        limit: int = 50,
        event_type: Optional[ProgressEvent] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get progress events for a stream.

        Args:
            stream_id: ID of the stream
            limit: Maximum number of events to return
            event_type: Filter by specific event type

        Returns:
            List[Dict[str, Any]]: List of progress events
        """
        try:
            events_key = f"{self.events_prefix}{stream_id}"

            # Get events from Redis (stored as JSON strings)
            raw_events = self.redis_client.lrange(events_key, 0, limit - 1)

            events = []
            for raw_event in raw_events:
                try:
                    event = json.loads(raw_event)

                    # Filter by event type if specified
                    if event_type and event.get("event_type") != event_type.value:
                        continue

                    events.append(event)
                except json.JSONDecodeError:
                    continue

            # Sort by timestamp (most recent first)
            events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return events

        except Exception as e:
            logger.error(
                "Failed to get progress events", stream_id=stream_id, error=str(e)
            )
            return []

    def get_processing_statistics(self, stream_id: int) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics for a stream.

        Args:
            stream_id: ID of the stream

        Returns:
            Dict[str, Any]: Processing statistics and metrics
        """
        try:
            progress = self.get_progress(stream_id)
            events = self.get_progress_events(stream_id)

            if not progress:
                return {"error": "No progress data found"}

            # Calculate statistics
            stats = {
                "stream_id": stream_id,
                "current_progress": progress.get("progress_percentage", 0),
                "status": progress.get("status", "unknown"),
                "created_at": progress.get("created_at"),
                "started_at": progress.get("started_at"),
                "last_updated": progress.get("last_updated"),
                "completed_at": progress.get("completed_at"),
                "processing_duration_seconds": progress.get(
                    "processing_duration_seconds"
                ),
                "total_events": len(events),
            }

            # Event type breakdown
            event_types = {}
            error_count = 0
            retry_count = 0

            for event in events:
                event_type = event.get("event_type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

                if event_type == ProgressEvent.ERROR.value:
                    error_count += 1
                elif event_type == ProgressEvent.RETRY.value:
                    retry_count += 1

            stats.update(
                {
                    "event_types": event_types,
                    "error_count": error_count,
                    "retry_count": retry_count,
                }
            )

            # Estimate completion time if still processing
            if (
                progress.get("status") == "processing"
                and progress.get("progress_percentage", 0) > 0
            ):
                try:
                    started = datetime.fromisoformat(progress["started_at"])
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    progress_pct = progress["progress_percentage"]

                    if progress_pct > 0:
                        estimated_total = elapsed * (100 / progress_pct)
                        estimated_remaining = estimated_total - elapsed
                        stats["estimated_completion_seconds"] = max(
                            0, estimated_remaining
                        )
                        stats["estimated_completion_time"] = (
                            datetime.now(timezone.utc)
                            + timedelta(seconds=estimated_remaining)
                        ).isoformat()
                except Exception as e:
                    logger.warning(
                        "Failed to calculate estimated completion", error=str(e)
                    )

            return stats

        except Exception as e:
            logger.error(
                "Failed to get processing statistics", stream_id=stream_id, error=str(e)
            )
            return {"error": str(e)}

    def cleanup_old_progress(self, cutoff_time: datetime) -> int:
        """
        Clean up old progress data and events.

        Args:
            cutoff_time: Remove data older than this time

        Returns:
            int: Number of records cleaned up
        """
        logger.info(
            "Cleaning up old progress data", cutoff_time=cutoff_time.isoformat()
        )

        try:
            cleaned_count = 0

            # Find all progress keys
            progress_pattern = f"{self.progress_prefix}*"
            progress_keys = self.redis_client.keys(progress_pattern)

            for progress_key in progress_keys:
                try:
                    progress_data = self.redis_client.hgetall(progress_key)
                    if not progress_data:
                        continue

                    # Check if data is old enough to clean up
                    created_at = progress_data.get("created_at")
                    if created_at:
                        created_time = datetime.fromisoformat(created_at)
                        if created_time < cutoff_time:
                            # Extract stream_id from key
                            stream_id = progress_key.replace(self.progress_prefix, "")

                            # Delete progress data
                            self.redis_client.delete(progress_key)

                            # Delete associated events
                            events_key = f"{self.events_prefix}{stream_id}"
                            self.redis_client.delete(events_key)

                            # Delete associated milestones
                            milestones_key = f"{self.milestones_prefix}{stream_id}"
                            self.redis_client.delete(milestones_key)

                            cleaned_count += 1

                except Exception as e:
                    logger.error(
                        "Failed to clean up individual progress record",
                        key=progress_key,
                        error=str(e),
                    )
                    continue

            logger.info("Progress cleanup completed", cleaned_count=cleaned_count)
            return cleaned_count

        except Exception as e:
            logger.error("Failed to cleanup old progress", error=str(e))
            return 0

    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all currently active processing jobs.

        Returns:
            List[Dict[str, Any]]: List of active job progress data
        """
        try:
            active_jobs = []

            # Find all progress keys
            progress_pattern = f"{self.progress_prefix}*"
            progress_keys = self.redis_client.keys(progress_pattern)

            for progress_key in progress_keys:
                try:
                    progress_data = self.redis_client.hgetall(progress_key)
                    if not progress_data:
                        continue

                    status = progress_data.get("status", "")
                    if status in ["processing", "pending", "queued"]:
                        # Deserialize and clean up data
                        stream_id = int(progress_key.replace(self.progress_prefix, ""))

                        job_info = {
                            "stream_id": stream_id,
                            "progress_percentage": int(
                                progress_data.get("progress_percentage", 0)
                            ),
                            "status": status,
                            "started_at": progress_data.get("started_at"),
                            "last_updated": progress_data.get("last_updated"),
                        }

                        # Add duration if available
                        if job_info["started_at"]:
                            try:
                                started = datetime.fromisoformat(job_info["started_at"])
                                duration = (
                                    datetime.now(timezone.utc) - started
                                ).total_seconds()
                                job_info["running_duration_seconds"] = duration
                            except Exception:
                                pass

                        active_jobs.append(job_info)

                except Exception as e:
                    logger.error(
                        "Failed to process active job", key=progress_key, error=str(e)
                    )
                    continue

            # Sort by progress percentage (most complete first)
            active_jobs.sort(
                key=lambda x: x.get("progress_percentage", 0), reverse=True
            )

            return active_jobs

        except Exception as e:
            logger.error("Failed to get active jobs", error=str(e))
            return []

    def _log_progress_event(
        self,
        stream_id: int,
        event_type: ProgressEvent,
        progress_percentage: int,
        status: str,
        details: Dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """Log a progress event to Redis."""
        try:
            event_data = {
                "event_type": event_type.value,
                "progress_percentage": progress_percentage,
                "status": status,
                "details": details,
                "timestamp": timestamp.isoformat(),
            }

            events_key = f"{self.events_prefix}{stream_id}"

            # Add event to the beginning of the list (most recent first)
            self.redis_client.lpush(events_key, json.dumps(event_data))

            # Trim list to keep only last 100 events
            self.redis_client.ltrim(events_key, 0, 99)

            # Set expiration
            self.redis_client.expire(events_key, 86400 * 7)  # 7 days

        except Exception as e:
            logger.error(
                "Failed to log progress event", stream_id=stream_id, error=str(e)
            )

    def _check_milestone(
        self,
        stream_id: int,
        progress_percentage: int,
        custom_milestone: Optional[str] = None,
    ) -> None:
        """Check if a milestone has been reached and log it."""
        try:
            milestones_key = f"{self.milestones_prefix}{stream_id}"
            reached_milestones = self.redis_client.smembers(milestones_key)

            # Convert to integers for comparison
            reached_percentages = set()
            for milestone in reached_milestones:
                try:
                    reached_percentages.add(int(milestone))
                except (ValueError, TypeError):
                    continue

            # Check predefined milestones
            for milestone_pct, milestone_msg in self.milestones.items():
                if (
                    progress_percentage >= milestone_pct
                    and milestone_pct not in reached_percentages
                ):
                    # New milestone reached
                    self._log_progress_event(
                        stream_id=stream_id,
                        event_type=ProgressEvent.MILESTONE_REACHED,
                        progress_percentage=progress_percentage,
                        status="processing",
                        details={
                            "milestone_percentage": milestone_pct,
                            "milestone_message": milestone_msg,
                        },
                        timestamp=datetime.now(timezone.utc),
                    )

                    # Mark milestone as reached
                    self.redis_client.sadd(milestones_key, str(milestone_pct))
                    self.redis_client.expire(milestones_key, 86400 * 7)  # 7 days

            # Log custom milestone if provided
            if custom_milestone:
                self._log_progress_event(
                    stream_id=stream_id,
                    event_type=ProgressEvent.MILESTONE_REACHED,
                    progress_percentage=progress_percentage,
                    status="processing",
                    details={
                        "milestone_percentage": progress_percentage,
                        "milestone_message": custom_milestone,
                        "custom": True,
                    },
                    timestamp=datetime.now(timezone.utc),
                )

        except Exception as e:
            logger.error("Failed to check milestone", stream_id=stream_id, error=str(e))

    def _update_database_status(
        self, stream_id: int, status: str, progress_percentage: Optional[int] = None
    ) -> None:
        """Update stream status in the database."""
        try:
            with get_db_session() as db:
                # Map progress status to stream status
                status_mapping = {
                    "pending": "pending",
                    "processing": "processing",
                    "completed": "completed",
                    "failed": "failed",
                    "cancelled": "cancelled",
                }

                stream_status = status_mapping.get(status, "processing")

                # Update stream in database
                update_data = {"status": stream_status}

                if status in ["completed", "failed", "cancelled"]:
                    update_data["completed_at"] = datetime.now(timezone.utc)

                db.execute(
                    update(Stream).where(Stream.id == stream_id).values(**update_data)
                )
                db.commit()

        except Exception as e:
            logger.error(
                "Failed to update database status", stream_id=stream_id, error=str(e)
            )
