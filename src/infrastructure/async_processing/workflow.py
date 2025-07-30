"""
Workflow Orchestration for async processing pipeline.

This module provides comprehensive workflow orchestration using Celery Canvas
for complex task coordination, including task chaining, grouping, parallel
execution, conditional workflows, and sophisticated error handling.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog
from celery import chain, chord, group, signature
from celery.canvas import Signature
from celery.result import AsyncResult

from src.core.cache import get_redis_client
from src.core.database import get_db_session
from src.infrastructure.persistence.models.stream import Stream, StreamStatus
from src.services.async_processing.celery_app import celery_app, get_task_options
from src.services.async_processing.job_manager import JobManager, JobPriority
from src.services.async_processing.progress_tracker import (
    ProgressTracker,
    ProgressEvent,
)
from src.services.async_processing.error_handler import ErrorHandler


logger = structlog.get_logger(__name__)


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(
        self,
        task_name: str,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        options: Dict[str, Any] = None,
        condition: Optional[str] = None,
        parallel: bool = False,
    ):
        """
        Initialize workflow step.

        Args:
            task_name: Name of the Celery task
            args: Task arguments
            kwargs: Task keyword arguments
            options: Task execution options
            condition: Optional condition for step execution
            parallel: Whether this step can run in parallel
        """
        self.task_name = task_name
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.options = options or {}
        self.condition = condition
        self.parallel = parallel

    def to_signature(self) -> Signature:
        """Convert workflow step to Celery signature."""
        return signature(
            self.task_name, args=self.args, kwargs=self.kwargs, **self.options
        )


class WorkflowDefinition:
    """Defines a complete workflow with steps and execution logic."""

    def __init__(
        self,
        name: str,
        steps: List[WorkflowStep],
        error_handling: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        priority: JobPriority = JobPriority.MEDIUM,
    ):
        """
        Initialize workflow definition.

        Args:
            name: Name of the workflow
            steps: List of workflow steps
            error_handling: Error handling configuration
            timeout_seconds: Workflow timeout
            priority: Execution priority
        """
        self.name = name
        self.steps = steps
        self.error_handling = error_handling or {}
        self.timeout_seconds = timeout_seconds
        self.priority = priority


class StreamProcessingWorkflow:
    """
    Orchestrates the complete stream processing workflow.

    Provides high-level workflow management with Celery Canvas patterns,
    error handling, progress tracking, and conditional execution.
    """

    def __init__(self):
        """Initialize workflow orchestrator."""
        self.redis_client = get_redis_client()
        self.job_manager = JobManager()
        self.progress_tracker = ProgressTracker()
        self.error_handler = ErrorHandler()

        # Workflow tracking
        self.workflow_prefix = "workflow:"
        self.execution_prefix = "execution:"

        # Default workflow definitions
        self.workflows = {
            "standard_stream_processing": self._create_standard_workflow(),
            "batch_stream_processing": self._create_batch_workflow(),
            "priority_stream_processing": self._create_priority_workflow(),
            "test_stream_processing": self._create_test_workflow(),
        }

    async def execute_workflow(
        self,
        workflow_name: str,
        stream_id: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a named workflow for stream processing.

        Args:
            workflow_name: Name of the workflow to execute
            stream_id: ID of the stream to process
            options: Workflow execution options

        Returns:
            Dict[str, Any]: Workflow execution result
        """
        logger.info(
            "Executing workflow",
            workflow_name=workflow_name,
            stream_id=stream_id,
            options=options,
        )

        try:
            # Get workflow definition
            workflow_def = self.workflows.get(workflow_name)
            if not workflow_def:
                raise ValueError(f"Unknown workflow: {workflow_name}")

            # Create execution context
            execution_id = (
                f"{workflow_name}_{stream_id}_{int(datetime.now(timezone.utc).timestamp())}"
            )

            execution_context = {
                "execution_id": execution_id,
                "workflow_name": workflow_name,
                "stream_id": stream_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "options": options or {},
                "status": "running",
            }

            # Store execution context
            self._store_execution_context(execution_id, execution_context)

            # Build and execute workflow
            canvas = self._build_workflow_canvas(workflow_def, stream_id, options or {})
            result = canvas.apply_async()

            # Update execution context with result info
            execution_context.update(
                {"canvas_result_id": result.id, "canvas_type": type(canvas).__name__}
            )

            self._store_execution_context(execution_id, execution_context)

            # Start monitoring
            self._start_workflow_monitoring(execution_id, result)

            return {
                "execution_id": execution_id,
                "canvas_result_id": result.id,
                "workflow_name": workflow_name,
                "stream_id": stream_id,
                "status": "started",
                "estimated_duration_minutes": workflow_def.timeout_seconds // 60
                if workflow_def.timeout_seconds
                else None,
            }

        except Exception as e:
            logger.error(
                "Failed to execute workflow",
                workflow_name=workflow_name,
                stream_id=stream_id,
                error=str(e),
            )
            raise

    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow execution.

        Args:
            execution_id: ID of the workflow execution

        Returns:
            Dict[str, Any]: Workflow status information
        """
        try:
            # Get execution context
            execution_context = self._get_execution_context(execution_id)
            if not execution_context:
                return {"error": "Execution not found"}

            # Get Canvas result
            canvas_result_id = execution_context.get("canvas_result_id")
            if not canvas_result_id:
                return {"error": "No result ID found"}

            result = AsyncResult(canvas_result_id, app=celery_app)

            # Build status response
            status = {
                "execution_id": execution_id,
                "workflow_name": execution_context.get("workflow_name"),
                "stream_id": execution_context.get("stream_id"),
                "started_at": execution_context.get("started_at"),
                "canvas_status": result.status,
                "canvas_result": result.result if result.ready() else None,
            }

            # Add completion info if finished
            if result.ready():
                status.update(
                    {
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "successful": result.successful(),
                        "failed": result.failed(),
                    }
                )

                if result.failed():
                    status["error"] = (
                        str(result.result) if result.result else "Unknown error"
                    )

            # Get progress information
            stream_id = execution_context.get("stream_id")
            if stream_id:
                progress = self.progress_tracker.get_progress(stream_id)
                if progress:
                    status["progress"] = progress

            return status

        except Exception as e:
            logger.error(
                "Failed to get workflow status", execution_id=execution_id, error=str(e)
            )
            return {"error": str(e)}

    def cancel_workflow(
        self, execution_id: str, reason: str = "User cancelled"
    ) -> bool:
        """
        Cancel a running workflow execution.

        Args:
            execution_id: ID of the workflow execution to cancel
            reason: Reason for cancellation

        Returns:
            bool: True if cancellation successful
        """
        logger.info("Cancelling workflow", execution_id=execution_id, reason=reason)

        try:
            # Get execution context
            execution_context = self._get_execution_context(execution_id)
            if not execution_context:
                logger.warning(
                    "Cannot cancel - execution not found", execution_id=execution_id
                )
                return False

            # Revoke Canvas tasks
            canvas_result_id = execution_context.get("canvas_result_id")
            if canvas_result_id:
                result = AsyncResult(canvas_result_id, app=celery_app)
                result.revoke(terminate=True)

            # Update execution context
            execution_context.update(
                {
                    "status": "cancelled",
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                    "cancellation_reason": reason,
                }
            )

            self._store_execution_context(execution_id, execution_context)

            # Update stream status
            stream_id = execution_context.get("stream_id")
            if stream_id:
                with get_db_session() as db:
                    stream = db.query(Stream).filter(Stream.id == stream_id).first()
                    if stream:
                        stream.status = StreamStatus.CANCELLED
                        db.commit()

                # Update progress
                self.progress_tracker.update_progress(
                    stream_id=stream_id,
                    progress_percentage=None,
                    status="cancelled",
                    event_type=ProgressEvent.CANCELLED,
                    details={"reason": reason},
                )

            logger.info("Workflow cancelled successfully", execution_id=execution_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to cancel workflow", execution_id=execution_id, error=str(e)
            )
            return False

    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """
        List all currently active workflow executions.

        Returns:
            List[Dict[str, Any]]: List of active workflow executions
        """
        try:
            active_workflows = []

            # Find all execution keys
            execution_pattern = f"{self.execution_prefix}*"
            execution_keys = self.redis_client.keys(execution_pattern)

            for key in execution_keys:
                try:
                    execution_data = self.redis_client.hgetall(key)
                    if not execution_data:
                        continue

                    # Parse execution data
                    execution_context = {}
                    for k, v in execution_data.items():
                        try:
                            execution_context[k] = (
                                json.loads(v)
                                if v.startswith("{") or v.startswith("[")
                                else v
                            )
                        except json.JSONDecodeError:
                            execution_context[k] = v

                    # Check if still active
                    status = execution_context.get("status", "unknown")
                    if status in ["running", "pending"]:
                        # Get current status
                        execution_id = execution_context.get("execution_id")
                        if execution_id:
                            current_status = self.get_workflow_status(execution_id)
                            active_workflows.append(current_status)

                except Exception as e:
                    logger.error(
                        "Failed to process workflow execution", key=key, error=str(e)
                    )
                    continue

            return active_workflows

        except Exception as e:
            logger.error("Failed to list active workflows", error=str(e))
            return []

    def _create_standard_workflow(self) -> WorkflowDefinition:
        """Create the standard stream processing workflow."""
        steps = [
            WorkflowStep(
                "src.services.async_processing.tasks.start_stream_processing",
                kwargs={"options": {}},
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.ingest_stream_data",
                kwargs={"chunk_size": 30},
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.process_multimodal_content"
            ),
            WorkflowStep("src.services.async_processing.tasks.detect_highlights"),
            WorkflowStep("src.services.async_processing.tasks.finalize_highlights"),
            WorkflowStep("src.services.async_processing.tasks.notify_completion"),
        ]

        return WorkflowDefinition(
            name="standard_stream_processing",
            steps=steps,
            timeout_seconds=3600,  # 1 hour
            priority=JobPriority.MEDIUM,
        )

    def _create_batch_workflow(self) -> WorkflowDefinition:
        """Create a batch processing workflow for multiple streams."""
        steps = [
            WorkflowStep(
                "src.services.async_processing.tasks.start_stream_processing",
                kwargs={"options": {"batch_mode": True}},
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.ingest_stream_data",
                kwargs={"chunk_size": 60},  # Larger chunks for batch
                parallel=True,
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.process_multimodal_content",
                parallel=True,
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.detect_highlights", parallel=True
            ),
            WorkflowStep("src.services.async_processing.tasks.finalize_highlights"),
            WorkflowStep("src.services.async_processing.tasks.notify_completion"),
        ]

        return WorkflowDefinition(
            name="batch_stream_processing",
            steps=steps,
            timeout_seconds=7200,  # 2 hours
            priority=JobPriority.LOW,
        )

    def _create_priority_workflow(self) -> WorkflowDefinition:
        """Create a high-priority workflow for premium customers."""
        steps = [
            WorkflowStep(
                "src.services.async_processing.tasks.start_stream_processing",
                kwargs={"options": {"priority_mode": True}},
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.ingest_stream_data",
                kwargs={"chunk_size": 15},  # Smaller chunks for faster processing
            ),
            WorkflowStep(
                "src.services.async_processing.tasks.process_multimodal_content"
            ),
            WorkflowStep("src.services.async_processing.tasks.detect_highlights"),
            WorkflowStep("src.services.async_processing.tasks.finalize_highlights"),
            WorkflowStep("src.services.async_processing.tasks.notify_completion"),
        ]

        return WorkflowDefinition(
            name="priority_stream_processing",
            steps=steps,
            timeout_seconds=1800,  # 30 minutes
            priority=JobPriority.HIGH,
        )

    def _create_test_workflow(self) -> WorkflowDefinition:
        """Create a test workflow with simulated tasks."""
        steps = [
            WorkflowStep(
                "src.services.async_processing.tasks.start_stream_processing",
                kwargs={"options": {"test_mode": True}},
            ),
            WorkflowStep("src.services.async_processing.tasks.notify_completion"),
        ]

        return WorkflowDefinition(
            name="test_stream_processing",
            steps=steps,
            timeout_seconds=300,  # 5 minutes
            priority=JobPriority.MEDIUM,
        )

    def _build_workflow_canvas(
        self, workflow_def: WorkflowDefinition, stream_id: int, options: Dict[str, Any]
    ) -> Union[chain, chord, group]:
        """Build Celery Canvas from workflow definition."""
        # Get task options for the workflow priority
        task_options = get_task_options(
            priority=workflow_def.priority.value, expires=workflow_def.timeout_seconds
        )

        # Build signatures for each step
        signatures = []
        parallel_groups = []
        current_group = []

        for i, step in enumerate(workflow_def.steps):
            # Prepare step arguments
            step_kwargs = step.kwargs.copy()
            if i == 0:
                # First step gets stream_id
                step_kwargs["stream_id"] = stream_id
                step_kwargs.update(step.kwargs)

            # Create signature
            sig = step.to_signature()
            sig.set(**{**task_options, **step.options})

            if step.parallel and current_group:
                # Add to current parallel group
                current_group.append(sig)
            else:
                # Process any pending parallel group
                if current_group:
                    parallel_groups.append(group(*current_group))
                    current_group = []

                if step.parallel:
                    # Start new parallel group
                    current_group.append(sig)
                else:
                    # Sequential step
                    signatures.append(sig)

        # Process final parallel group
        if current_group:
            parallel_groups.append(group(*current_group))

        # Combine sequential and parallel steps
        if parallel_groups:
            # Mix of sequential and parallel
            final_signatures = []

            parallel_index = 0
            for sig in signatures:
                final_signatures.append(sig)

                # Insert parallel groups at appropriate points
                if parallel_index < len(parallel_groups):
                    final_signatures.append(parallel_groups[parallel_index])
                    parallel_index += 1

            # Add remaining parallel groups
            while parallel_index < len(parallel_groups):
                final_signatures.append(parallel_groups[parallel_index])
                parallel_index += 1

            canvas = chain(*final_signatures)
        else:
            # Pure sequential workflow
            canvas = chain(*signatures)

        return canvas

    def _store_execution_context(
        self, execution_id: str, context: Dict[str, Any]
    ) -> None:
        """Store workflow execution context in Redis."""
        try:
            execution_key = f"{self.execution_prefix}{execution_id}"

            # Serialize context data
            serialized_context = {}
            for k, v in context.items():
                if isinstance(v, (dict, list)):
                    serialized_context[k] = json.dumps(v)
                else:
                    serialized_context[k] = str(v)

            self.redis_client.hset(execution_key, mapping=serialized_context)
            self.redis_client.expire(execution_key, 86400 * 7)  # 7 days

        except Exception as e:
            logger.error(
                "Failed to store execution context",
                execution_id=execution_id,
                error=str(e),
            )

    def _get_execution_context(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution context from Redis."""
        try:
            execution_key = f"{self.execution_prefix}{execution_id}"
            context_data = self.redis_client.hgetall(execution_key)

            if not context_data:
                return None

            # Deserialize context data
            context = {}
            for k, v in context_data.items():
                try:
                    # Try to parse as JSON
                    if v.startswith("{") or v.startswith("["):
                        context[k] = json.loads(v)
                    else:
                        context[k] = v
                except json.JSONDecodeError:
                    context[k] = v

            return context

        except Exception as e:
            logger.error(
                "Failed to get execution context",
                execution_id=execution_id,
                error=str(e),
            )
            return None

    def _start_workflow_monitoring(
        self, execution_id: str, result: AsyncResult
    ) -> None:
        """Start monitoring a workflow execution."""
        # This would typically start a background task to monitor the workflow
        # For now, we'll log the start of monitoring
        logger.info(
            "Started workflow monitoring",
            execution_id=execution_id,
            result_id=result.id,
        )

        # Store monitoring info
        monitoring_key = f"monitoring:{execution_id}"
        monitoring_data = {
            "execution_id": execution_id,
            "result_id": result.id,
            "started_monitoring_at": datetime.now(timezone.utc).isoformat(),
            "status": "monitoring",
        }

        self.redis_client.hset(monitoring_key, mapping=monitoring_data)
        self.redis_client.expire(monitoring_key, 86400)  # 24 hours

    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up completed workflow executions.

        Args:
            max_age_hours: Maximum age in hours for completed workflows to keep

        Returns:
            Dict[str, Any]: Cleanup statistics
        """
        logger.info("Cleaning up completed workflows", max_age_hours=max_age_hours)

        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            cleaned_count = 0

            # Find all execution keys
            execution_pattern = f"{self.execution_prefix}*"
            execution_keys = self.redis_client.keys(execution_pattern)

            for key in execution_keys:
                try:
                    execution_data = self.redis_client.hgetall(key)
                    if not execution_data:
                        continue

                    # Check completion status and age
                    status = execution_data.get("status", "unknown")
                    completed_at = execution_data.get("completed_at")
                    cancelled_at = execution_data.get("cancelled_at")

                    if status in ["completed", "failed", "cancelled"]:
                        # Check age
                        completion_time = completed_at or cancelled_at
                        if completion_time:
                            try:
                                completed_time = datetime.fromisoformat(completion_time)
                                if completed_time < cutoff_time:
                                    # Clean up execution data
                                    self.redis_client.delete(key)

                                    # Clean up monitoring data
                                    execution_id = execution_data.get("execution_id")
                                    if execution_id:
                                        monitoring_key = f"monitoring:{execution_id}"
                                        self.redis_client.delete(monitoring_key)

                                    cleaned_count += 1
                            except ValueError:
                                # Invalid timestamp, skip
                                continue

                except Exception as e:
                    logger.error(
                        "Failed to clean up workflow execution", key=key, error=str(e)
                    )
                    continue

            cleanup_result = {
                "cleaned_workflows": cleaned_count,
                "cutoff_time": cutoff_time.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info("Workflow cleanup completed", **cleanup_result)
            return cleanup_result

        except Exception as e:
            logger.error("Failed to cleanup workflows", error=str(e))
            return {"error": str(e)}
