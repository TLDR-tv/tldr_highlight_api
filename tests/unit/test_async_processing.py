"""
Unit tests for async processing pipeline components.

This module contains comprehensive unit tests for all async processing
components including job manager, progress tracker, webhook dispatcher,
error handler, and workflow orchestration.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.services.async_processing.job_manager import (
    JobManager,
    JobPriority,
    JobStatus,
    ResourceLimits,
)
from src.models.stream import Stream
from src.models.user import User
from src.models.webhook import Webhook
from src.services.async_processing.progress_tracker import (
    ProgressTracker,
    ProgressEvent,
)
from src.services.async_processing.webhook_dispatcher import (
    WebhookDispatcher,
    WebhookEvent,
)
from src.services.async_processing.error_handler import (
    ErrorHandler,
    ErrorCategory,
    RetryStrategy,
)
from src.services.async_processing.workflow import (
    StreamProcessingWorkflow,
    WorkflowStep,
    WorkflowDefinition,
)
from src.utils.job_utils import JobMonitor, SystemHealthChecker


class TestJobManager:
    """Test cases for JobManager class."""

    @pytest.fixture
    def job_manager(self):
        """Create a JobManager instance for testing."""
        with patch(
            "src.services.async_processing.job_manager.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value = Mock()
            return JobManager()

    @pytest.fixture
    def mock_stream(self):
        """Create a mock stream for testing."""
        stream = Mock()
        stream.id = 123
        stream.user_id = 456
        stream.platform = "twitch"
        stream.source_url = "https://twitch.tv/example"
        return stream

    def test_create_job_success(self, job_manager, mock_stream):
        """Test successful job creation."""
        # Mock Redis operations
        job_manager.redis_client.hset = Mock()
        job_manager.redis_client.expire = Mock()
        job_manager.redis_client.lpush = Mock()
        job_manager.redis_client.sadd = Mock()
        job_manager.redis_client.smembers = Mock(return_value=set())

        # Create job
        job_id = job_manager.create_job(
            stream_id=mock_stream.id,
            user_id=mock_stream.user_id,
            priority=JobPriority.HIGH,
            options={"test": "value"},
        )

        # Verify job ID is UUID format
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID length with hyphens

        # Verify Redis operations were called
        job_manager.redis_client.hset.assert_called()
        job_manager.redis_client.expire.assert_called()
        job_manager.redis_client.lpush.assert_called()
        job_manager.redis_client.sadd.assert_called()

    def test_create_job_resource_limit_exceeded(self, job_manager):
        """Test job creation when resource limits are exceeded."""
        # Mock resource limit check to return too many jobs
        high_priority_limits = ResourceLimits.get_limits(JobPriority.HIGH)
        current_jobs = set(
            f"job_{i}" for i in range(high_priority_limits["max_concurrent_jobs"])
        )
        job_manager.redis_client.smembers = Mock(return_value=current_jobs)

        # Attempt to create job should raise RuntimeError (wrapping ValueError)
        with pytest.raises(RuntimeError, match="Job creation failed.*Resource limit exceeded"):
            job_manager.create_job(
                stream_id=123, user_id=456, priority=JobPriority.HIGH
            )

    def test_get_job_success(self, job_manager):
        """Test successful job retrieval."""
        job_id = "test-job-id"
        mock_job_data = {
            "job_id": job_id,
            "stream_id": "123",
            "user_id": "456",
            "priority": "high",
            "status": "pending",
            "options": '{"test": "value"}',
            "resource_limits": '{"max_concurrent_jobs": 10}',
            "created_at": datetime.utcnow().isoformat(),
        }

        job_manager.redis_client.hgetall = Mock(return_value=mock_job_data)

        job = job_manager.get_job(job_id)

        assert job is not None
        assert job["job_id"] == job_id
        assert job["options"] == {"test": "value"}
        assert job["resource_limits"] == {"max_concurrent_jobs": 10}

    def test_get_job_not_found(self, job_manager):
        """Test job retrieval when job doesn't exist."""
        job_manager.redis_client.hgetall = Mock(return_value={})

        job = job_manager.get_job("nonexistent-job")

        assert job is None

    def test_update_job_status_success(self, job_manager):
        """Test successful job status update."""
        job_manager.redis_client.hset = Mock()
        job_manager.get_job = Mock(return_value={"user_id": "456", "priority": "high"})
        job_manager._release_resources = Mock()

        result = job_manager.update_job_status(
            "test-job-id", JobStatus.COMPLETED, {"result": "success"}
        )

        assert result is True
        job_manager.redis_client.hset.assert_called()
        job_manager._release_resources.assert_called()

    def test_cancel_job_success(self, job_manager):
        """Test successful job cancellation."""
        job_id = "test-job-id"
        mock_job_data = {
            "job_id": job_id,
            "stream_id": "123",
            "status": "running",
            "priority": "medium",
            "celery_task_id": "celery-task-123",
        }

        job_manager.get_job = Mock(return_value=mock_job_data)
        job_manager.update_job_status = Mock(return_value=True)
        job_manager._remove_from_queue = Mock()

        with patch(
            "src.services.async_processing.job_manager.celery_app"
        ) as mock_celery:
            mock_celery.control.revoke = Mock()

            with patch("src.services.async_processing.job_manager.get_db_session"):
                result = job_manager.cancel_job(job_id, "Test cancellation")

        assert result is True
        mock_celery.control.revoke.assert_called_with("celery-task-123", terminate=True)
        job_manager.update_job_status.assert_called_with(
            job_id, JobStatus.CANCELLED, {"reason": "Test cancellation"}
        )

    def test_get_queue_status(self, job_manager):
        """Test queue status retrieval."""

        # Mock Redis operations for different priority queues
        def mock_llen(key):
            if "high" in key:
                return 5
            elif "medium" in key:
                return 10
            else:
                return 2

        def mock_lrange(key, start, end):
            if "high" in key:
                return [b"job1", b"job2"]
            return []

        job_manager.redis_client.llen = Mock(side_effect=mock_llen)
        job_manager.redis_client.lrange = Mock(side_effect=mock_lrange)
        job_manager.get_job = Mock(
            return_value={
                "job_id": "job1",
                "stream_id": "123",
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        status = job_manager.get_queue_status()

        assert "high" in status
        assert "medium" in status
        assert "low" in status
        assert status["high"]["length"] == 5
        assert len(status["high"]["jobs"]) == 2


class TestProgressTracker:
    """Test cases for ProgressTracker class."""

    @pytest.fixture
    def progress_tracker(self):
        """Create a ProgressTracker instance for testing."""
        with patch(
            "src.services.async_processing.progress_tracker.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value = Mock()
            return ProgressTracker()

    def test_update_progress_success(self, progress_tracker):
        """Test successful progress update."""
        progress_tracker.redis_client.hgetall = Mock(return_value={})
        progress_tracker.redis_client.hset = Mock()
        progress_tracker.redis_client.expire = Mock()
        progress_tracker.redis_client.lpush = Mock()
        progress_tracker.redis_client.ltrim = Mock()
        progress_tracker.redis_client.smembers = Mock(return_value=set())
        progress_tracker.redis_client.sadd = Mock()

        with patch("src.services.async_processing.progress_tracker.get_db_session"):
            result = progress_tracker.update_progress(
                stream_id=123,
                progress_percentage=50,
                status="processing",
                event_type=ProgressEvent.PROGRESS_UPDATE,
                details={"task": "processing"},
            )

        assert result is True
        progress_tracker.redis_client.hset.assert_called()
        progress_tracker.redis_client.expire.assert_called()
        progress_tracker.redis_client.lpush.assert_called()

    def test_get_progress_success(self, progress_tracker):
        """Test successful progress retrieval."""
        mock_progress_data = {
            "stream_id": "123",
            "progress_percentage": "75",
            "status": "processing",
            "last_updated": datetime.utcnow().isoformat(),
            "details": '{"task": "detecting_highlights"}',
        }

        progress_tracker.redis_client.hgetall = Mock(return_value=mock_progress_data)

        progress = progress_tracker.get_progress(123)

        assert progress is not None
        assert progress["progress_percentage"] == 75
        assert progress["status"] == "processing"
        assert progress["details"] == {"task": "detecting_highlights"}

    def test_get_progress_not_found(self, progress_tracker):
        """Test progress retrieval when no progress exists."""
        progress_tracker.redis_client.hgetall = Mock(return_value={})

        progress = progress_tracker.get_progress(123)

        assert progress is None

    def test_get_progress_events(self, progress_tracker):
        """Test progress events retrieval."""
        mock_events = [
            json.dumps(
                {
                    "event_type": "progress_update",
                    "progress_percentage": 50,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            json.dumps(
                {
                    "event_type": "started",
                    "progress_percentage": 0,
                    "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                }
            ),
        ]

        progress_tracker.redis_client.lrange = Mock(return_value=mock_events)

        events = progress_tracker.get_progress_events(123, limit=10)

        assert len(events) == 2
        assert events[0]["event_type"] == "progress_update"
        assert events[1]["event_type"] == "started"

    def test_get_processing_statistics(self, progress_tracker):
        """Test processing statistics calculation."""
        from datetime import datetime, timedelta, timezone
        
        mock_progress = {
            "stream_id": "123",
            "progress_percentage": 75,
            "status": "processing",
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
            "started_at": (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "processing_duration_seconds": 1500,  # 25 minutes
        }

        mock_events = [
            {"event_type": "started"},
            {"event_type": "progress_update"},
            {"event_type": "error"},
            {"event_type": "retry"},
        ]

        progress_tracker.get_progress = Mock(return_value=mock_progress)
        progress_tracker.get_progress_events = Mock(return_value=mock_events)

        stats = progress_tracker.get_processing_statistics(123)

        assert stats["stream_id"] == 123
        assert stats["current_progress"] == 75
        assert stats["status"] == "processing"
        assert stats["total_events"] == 4
        assert stats["error_count"] == 1
        assert stats["retry_count"] == 1
        # Check for processing duration instead of estimated completion
        assert "processing_duration_seconds" in stats


class TestWebhookDispatcher:
    """Test cases for WebhookDispatcher class."""

    @pytest.fixture
    def webhook_dispatcher(self):
        """Create a WebhookDispatcher instance for testing."""
        with patch(
            "src.services.async_processing.webhook_dispatcher.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value = Mock()
            dispatcher = WebhookDispatcher()
            dispatcher.http_client = Mock()
            return dispatcher

    @pytest.fixture
    def mock_webhook(self):
        """Create a mock webhook for testing."""
        webhook = Mock()
        webhook.id = 1
        webhook.url = "https://example.com/webhook"
        webhook.secret = "test-secret"
        webhook.headers = {"Authorization": "Bearer token"}
        webhook.events = ["stream.started", "stream.completed"]
        webhook.is_active = True
        webhook.rate_limit_per_minute = 60
        return webhook

    @pytest.fixture
    def mock_stream(self):
        """Create a mock stream for testing."""
        stream = Mock()
        stream.id = 123
        stream.user_id = 456
        stream.platform = "twitch"
        stream.source_url = "https://twitch.tv/example"
        return stream

    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing."""
        user = Mock()
        user.id = 456
        user.email = "test@example.com"
        return user

    @pytest.mark.asyncio
    async def test_dispatch_webhook_success(
        self, webhook_dispatcher, mock_webhook, mock_stream, mock_user
    ):
        """Test successful webhook dispatch."""
        # Mock helper methods
        webhook_dispatcher._should_send_event = Mock(return_value=True)
        webhook_dispatcher._check_rate_limit = Mock(return_value=True)
        webhook_dispatcher._send_webhook = AsyncMock(
            return_value={"webhook_id": 1, "status": "delivered"}
        )

        # Mock database operations
        with patch(
            "src.services.async_processing.webhook_dispatcher.get_db_session"
        ) as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Create separate query mocks for each entity
            stream_query = Mock()
            stream_query.filter.return_value.first.return_value = mock_stream
            
            user_query = Mock()
            user_query.filter.return_value.first.return_value = mock_user
            
            webhook_query = Mock()
            mock_webhook.active = True  # Ensure webhook is active
            webhook_query.filter.return_value.all.return_value = [mock_webhook]
            
            # Return different query objects based on what's being queried
            def query_side_effect(model):
                if model == Stream:
                    return stream_query
                elif model == User:
                    return user_query
                elif model == Webhook:
                    return webhook_query
                return Mock()
            
            mock_session.query.side_effect = query_side_effect

            result = await webhook_dispatcher.dispatch_webhook(
                stream_id=123, event=WebhookEvent.STREAM_STARTED, data={"test": "data"}
            )

        assert result["dispatched"] == 1
        assert result["successful"] == 1
        assert result["failed"] == 0
        webhook_dispatcher._send_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, webhook_dispatcher, mock_webhook):
        """Test successful webhook sending."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = "OK"

        webhook_dispatcher.http_client.post = AsyncMock(return_value=mock_response)

        # Mock database operations
        with patch(
            "src.services.async_processing.webhook_dispatcher.get_db_session"
        ) as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_attempt = Mock()
            mock_attempt.id = 1
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_session.refresh.return_value = mock_attempt

        webhook_dispatcher._update_rate_limit = Mock()

        payload = {"event": "test", "data": {}}
        result = await webhook_dispatcher._send_webhook(mock_webhook, payload)

        assert result["status"] == "delivered"
        assert result["webhook_id"] == 1
        webhook_dispatcher.http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_failure(self, webhook_dispatcher, mock_webhook):
        """Test webhook sending failure."""
        # Mock HTTP response with error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.text = "Internal Server Error"

        webhook_dispatcher.http_client.post = AsyncMock(return_value=mock_response)

        # Mock database operations
        with patch(
            "src.services.async_processing.webhook_dispatcher.get_db_session"
        ) as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_attempt = Mock()
            mock_attempt.id = 1
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_session.refresh.return_value = mock_attempt

        payload = {"event": "test", "data": {}}
        result = await webhook_dispatcher._send_webhook(mock_webhook, payload)

        assert result["status"] == "failed"
        assert result["status_code"] == 500

    def test_generate_hmac_signature(self, webhook_dispatcher):
        """Test HMAC signature generation."""
        payload = '{"test": "data"}'
        secret = "test-secret"

        signature = webhook_dispatcher._generate_hmac_signature(payload, secret)

        assert signature.startswith("sha256=")
        assert len(signature) == 71  # "sha256=" + 64 character hex digest

    def test_calculate_retry_delay(self, webhook_dispatcher):
        """Test retry delay calculation."""
        # Test exponential backoff
        delay_0 = webhook_dispatcher._calculate_retry_delay(0)
        delay_1 = webhook_dispatcher._calculate_retry_delay(1)
        delay_2 = webhook_dispatcher._calculate_retry_delay(2)

        assert delay_0 < delay_1 < delay_2
        assert all(isinstance(d, int) for d in [delay_0, delay_1, delay_2])


class TestErrorHandler:
    """Test cases for ErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        with patch(
            "src.services.async_processing.error_handler.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value = Mock()
            return ErrorHandler()

    def test_classify_error_network(self, error_handler):
        """Test network error classification."""
        error = ConnectionError("Connection failed")
        category = error_handler.classify_error(error)
        assert category == ErrorCategory.NETWORK

    def test_classify_error_permanent(self, error_handler):
        """Test permanent error classification."""
        error = ValueError("Invalid input")
        category = error_handler.classify_error(error)
        assert category == ErrorCategory.PERMANENT

    def test_classify_error_rate_limit(self, error_handler):
        """Test rate limit error classification."""
        error = Exception("Rate limit exceeded")
        category = error_handler.classify_error(error)
        assert category == ErrorCategory.RATE_LIMIT

    def test_get_retry_strategy(self, error_handler):
        """Test retry strategy selection."""
        strategy = error_handler.get_retry_strategy(ErrorCategory.TRANSIENT)
        assert strategy == RetryStrategy.EXPONENTIAL_BACKOFF

        strategy = error_handler.get_retry_strategy(ErrorCategory.PERMANENT)
        assert strategy == RetryStrategy.NO_RETRY

        strategy = error_handler.get_retry_strategy(ErrorCategory.RATE_LIMIT)
        assert strategy == RetryStrategy.LINEAR_BACKOFF

    def test_should_retry_within_limits(self, error_handler):
        """Test retry decision within limits."""
        error = ConnectionError("Network error")
        should_retry = error_handler.should_retry(error, attempt_count=1, max_retries=3)
        assert should_retry is True

    def test_should_retry_exceeded_limits(self, error_handler):
        """Test retry decision when limits exceeded."""
        error = ConnectionError("Network error")
        should_retry = error_handler.should_retry(error, attempt_count=5, max_retries=3)
        assert should_retry is False

    def test_should_retry_permanent_error(self, error_handler):
        """Test retry decision for permanent errors."""
        error = ValueError("Invalid input")
        should_retry = error_handler.should_retry(error, attempt_count=1, max_retries=3)
        assert should_retry is False

    def test_calculate_delay_exponential(self, error_handler):
        """Test exponential backoff delay calculation."""
        delay_0 = error_handler.calculate_delay(0, RetryStrategy.EXPONENTIAL_BACKOFF)
        delay_1 = error_handler.calculate_delay(1, RetryStrategy.EXPONENTIAL_BACKOFF)
        delay_2 = error_handler.calculate_delay(2, RetryStrategy.EXPONENTIAL_BACKOFF)

        assert delay_0 < delay_1 < delay_2

    def test_calculate_delay_linear(self, error_handler):
        """Test linear backoff delay calculation."""
        delay_0 = error_handler.calculate_delay(
            0, RetryStrategy.LINEAR_BACKOFF, base_delay=1.0, jitter=False
        )
        delay_1 = error_handler.calculate_delay(
            1, RetryStrategy.LINEAR_BACKOFF, base_delay=1.0, jitter=False
        )
        delay_2 = error_handler.calculate_delay(
            2, RetryStrategy.LINEAR_BACKOFF, base_delay=1.0, jitter=False
        )

        # Linear backoff: base_delay * (attempt + 1)
        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 3.0

    def test_calculate_delay_fixed(self, error_handler):
        """Test fixed delay calculation."""
        base_delay = 5.0
        delay_0 = error_handler.calculate_delay(
            0, RetryStrategy.FIXED_DELAY, base_delay=base_delay, jitter=False
        )
        delay_1 = error_handler.calculate_delay(
            1, RetryStrategy.FIXED_DELAY, base_delay=base_delay, jitter=False
        )
        delay_2 = error_handler.calculate_delay(
            2, RetryStrategy.FIXED_DELAY, base_delay=base_delay, jitter=False
        )

        assert delay_0 == delay_1 == delay_2 == base_delay

    @pytest.mark.asyncio
    async def test_handle_error_with_retry(self, error_handler):
        """Test error handling with successful retry."""
        error = ConnectionError("Network error")
        context = {"stream_id": 123, "task_name": "test_task", "attempt_count": 1}

        # Mock circuit breaker and error tracking
        error_handler._is_circuit_breaker_open = Mock(return_value=False)
        error_handler._record_error = Mock()
        error_handler._log_error_stats = Mock()

        # Mock successful retry function
        retry_func = AsyncMock(return_value="success")

        result = await error_handler.handle_error(
            error, context, retry_func, max_retries=3
        )

        assert result["action"] == "retry"
        assert result["should_retry"] is True
        assert result["retry_successful"] is True
        assert result["retry_result"] == "success"

    def test_circuit_breaker_operations(self, error_handler):
        """Test circuit breaker state management."""
        service_name = "test_service"

        # Mock Redis operations
        error_handler.redis_client.hgetall = Mock(return_value={})
        error_handler.redis_client.hset = Mock()
        error_handler.redis_client.expire = Mock()

        # Test recording multiple errors to trigger circuit breaker
        for i in range(error_handler.circuit_breaker_failure_threshold + 1):
            error_handler._record_error(service_name, Exception(f"Error {i}"))

        # Verify circuit breaker operations
        error_handler.redis_client.hset.assert_called()
        error_handler.redis_client.expire.assert_called()


class TestWorkflow:
    """Test cases for StreamProcessingWorkflow class."""

    @pytest.fixture
    def workflow(self):
        """Create a StreamProcessingWorkflow instance for testing."""
        with patch(
            "src.services.async_processing.workflow.get_redis_client"
        ) as mock_redis:
            mock_redis.return_value = Mock()
            return StreamProcessingWorkflow()

    def test_workflow_step_creation(self):
        """Test WorkflowStep creation."""
        step = WorkflowStep(
            task_name="test.task",
            args=(1, 2),
            kwargs={"param": "value"},
            options={"queue": "test"},
            parallel=True,
        )

        assert step.task_name == "test.task"
        assert step.args == (1, 2)
        assert step.kwargs == {"param": "value"}
        assert step.options == {"queue": "test"}
        assert step.parallel is True

    def test_workflow_definition_creation(self):
        """Test WorkflowDefinition creation."""
        steps = [
            WorkflowStep("task1"),
            WorkflowStep("task2", parallel=True),
            WorkflowStep("task3"),
        ]

        workflow_def = WorkflowDefinition(
            name="test_workflow",
            steps=steps,
            timeout_seconds=3600,
            priority=JobPriority.HIGH,
        )

        assert workflow_def.name == "test_workflow"
        assert len(workflow_def.steps) == 3
        assert workflow_def.timeout_seconds == 3600
        assert workflow_def.priority == JobPriority.HIGH

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow):
        """Test successful workflow execution."""
        workflow_name = "standard_stream_processing"
        stream_id = 123

        # Mock workflow execution
        workflow._store_execution_context = Mock()
        workflow._start_workflow_monitoring = Mock()

        # Mock Canvas building and execution
        mock_canvas = Mock()
        mock_result = Mock()
        mock_result.id = "canvas-result-123"
        mock_canvas.apply_async = Mock(return_value=mock_result)

        workflow._build_workflow_canvas = Mock(return_value=mock_canvas)

        result = await workflow.execute_workflow(workflow_name, stream_id)

        assert result["workflow_name"] == workflow_name
        assert result["stream_id"] == stream_id
        assert result["status"] == "started"
        assert result["canvas_result_id"] == "canvas-result-123"

        workflow._store_execution_context.assert_called()
        workflow._start_workflow_monitoring.assert_called()

    def test_get_workflow_status(self, workflow):
        """Test workflow status retrieval."""
        execution_id = "test_execution_123"

        # Mock execution context
        mock_context = {
            "execution_id": execution_id,
            "workflow_name": "test_workflow",
            "stream_id": "123",
            "started_at": datetime.utcnow().isoformat(),
            "canvas_result_id": "canvas-123",
        }

        workflow._get_execution_context = Mock(return_value=mock_context)

        # Mock Celery result
        with patch(
            "src.services.async_processing.workflow.AsyncResult"
        ) as mock_async_result:
            mock_result = Mock()
            mock_result.status = "SUCCESS"
            mock_result.ready.return_value = True
            mock_result.successful.return_value = True
            mock_result.failed.return_value = False
            mock_result.result = {"test": "result"}
            mock_async_result.return_value = mock_result

            # Mock progress tracker
            workflow.progress_tracker.get_progress = Mock(
                return_value={"progress_percentage": 100, "status": "completed"}
            )

            status = workflow.get_workflow_status(execution_id)

        assert status["execution_id"] == execution_id
        assert status["canvas_status"] == "SUCCESS"
        assert status["successful"] is True
        assert status["progress"]["progress_percentage"] == 100

    def test_cancel_workflow(self, workflow):
        """Test workflow cancellation."""
        execution_id = "test_execution_123"

        # Mock execution context
        mock_context = {
            "execution_id": execution_id,
            "stream_id": "123",
            "canvas_result_id": "canvas-123",
        }

        workflow._get_execution_context = Mock(return_value=mock_context)
        workflow._store_execution_context = Mock()

        # Mock Celery result revocation
        with patch(
            "src.services.async_processing.workflow.AsyncResult"
        ) as mock_async_result:
            mock_result = Mock()
            mock_result.revoke = Mock()
            mock_async_result.return_value = mock_result

            # Mock database operations
            with patch("src.services.async_processing.workflow.get_db_session"):
                # Mock progress tracker
                workflow.progress_tracker.update_progress = Mock()

                result = workflow.cancel_workflow(execution_id, "Test cancellation")

        assert result is True
        mock_result.revoke.assert_called_with(terminate=True)
        workflow._store_execution_context.assert_called()
        workflow.progress_tracker.update_progress.assert_called()


class TestJobUtils:
    """Test cases for job utilities."""

    @pytest.fixture
    def job_monitor(self):
        """Create a JobMonitor instance for testing."""
        with patch("src.utils.job_utils.get_redis_client") as mock_redis:
            mock_redis.return_value = Mock()
            return JobMonitor()

    @pytest.fixture
    def health_checker(self):
        """Create a SystemHealthChecker instance for testing."""
        with patch("src.utils.job_utils.get_redis_client") as mock_redis:
            mock_redis.return_value = Mock()
            return SystemHealthChecker()

    def test_get_queue_metrics(self, job_monitor):
        """Test queue metrics retrieval."""
        # Mock Celery inspect
        with patch("src.utils.job_utils.celery_app") as mock_celery:
            mock_inspect = Mock()
            mock_celery.control.inspect.return_value = mock_inspect

            mock_inspect.active_queues.return_value = {
                "worker1": [{"name": "high_priority", "messages": []}],
                "worker2": [{"name": "medium_priority", "messages": []}],
            }

            mock_inspect.active.return_value = {
                "worker1": ["task1", "task2"],
                "worker2": ["task3"],
            }

            mock_inspect.scheduled.return_value = {"worker1": ["scheduled_task1"]}

            mock_inspect.reserved.return_value = {"worker1": ["reserved_task1"]}

            job_monitor._calculate_processing_rates = Mock(
                return_value={"streams_per_hour": 50.0}
            )

            metrics = job_monitor.get_queue_metrics()

        assert "queue_lengths" in metrics
        assert "total_active_tasks" in metrics
        assert "total_scheduled_tasks" in metrics
        assert "processing_rates" in metrics
        assert metrics["total_active_tasks"] == 3
        assert metrics["total_scheduled_tasks"] == 1

    def test_get_worker_health(self, job_monitor):
        """Test worker health checking."""
        with patch("src.utils.job_utils.celery_app") as mock_celery:
            mock_inspect = Mock()
            mock_celery.control.inspect.return_value = mock_inspect

            mock_inspect.stats.return_value = {
                "worker1": {
                    "total": 100,
                    "pool": {"max-concurrency": 4},
                    "rusage": {"utime": 123.45},
                },
                "worker2": {
                    "total": 200,
                    "pool": {"max-concurrency": 8},
                    "rusage": {"utime": 234.56},
                },
            }

            mock_inspect.ping.return_value = {
                "worker1": {"ok": "pong"},
                "worker2": {"ok": "pong"},
            }

            job_monitor._get_worker_load_avg = Mock(return_value=[1.0, 1.2, 1.5])

            health = job_monitor.get_worker_health()

        assert health["total_workers"] == 2
        assert health["online_workers"] == 2
        assert health["health_percentage"] == 100.0
        assert "worker1" in health["workers"]
        assert "worker2" in health["workers"]

    def test_record_task_metrics(self, job_monitor):
        """Test task metrics recording."""
        job_monitor.redis_client.hset = Mock()
        job_monitor.redis_client.expire = Mock()
        job_monitor._update_aggregated_metrics = Mock()

        job_monitor.record_task_metrics(
            task_name="test_task", execution_time=1.5, success=True, stream_id=123
        )

        job_monitor.redis_client.hset.assert_called()
        job_monitor.redis_client.expire.assert_called()
        job_monitor._update_aggregated_metrics.assert_called()

    def test_check_system_health(self, health_checker):
        """Test comprehensive system health check."""
        # Mock individual health check methods
        health_checker._check_database_health = Mock(
            return_value={"status": "healthy", "response_time_ms": 50.0}
        )

        health_checker._check_redis_health = Mock(
            return_value={"status": "healthy", "response_time_ms": 10.0}
        )

        health_checker._check_celery_health = Mock(
            return_value={"status": "healthy", "online_workers": 2, "total_workers": 2}
        )

        health_checker._check_queue_health = Mock(
            return_value={"status": "healthy", "total_active_messages": 5}
        )

        health_checker._check_disk_space = Mock(
            return_value={"status": "healthy", "usage_percentage": 45.0}
        )

        health_checker._check_memory_usage = Mock(
            return_value={"status": "healthy", "usage_percentage": 60.0}
        )

        health = health_checker.check_system_health()

        assert health["overall_status"] == "healthy"
        assert "components" in health
        assert "database" in health["components"]
        assert "redis" in health["components"]
        assert "celery_workers" in health["components"]
        assert len(health["components"]) == 6  # All components checked

    def test_check_database_health(self, health_checker):
        """Test database health check."""
        with patch("src.utils.job_utils.get_db_session") as mock_db_session:
            mock_session = Mock()
            mock_db_session.return_value.__enter__.return_value = mock_session

            # Mock database operations
            mock_session.execute = Mock()
            mock_session.query.return_value.scalar.return_value = 1000

            health = health_checker._check_database_health()

        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        assert "total_streams" in health

    def test_check_redis_health(self, health_checker):
        """Test Redis health check."""
        # Mock Redis operations
        health_checker.redis_client.ping = Mock(return_value=True)
        health_checker.redis_client.set = Mock()
        health_checker.redis_client.get = Mock(return_value="test")
        health_checker.redis_client.delete = Mock()
        health_checker.redis_client.info = Mock(
            return_value={
                "connected_clients": 10,
                "used_memory_human": "100M",
                "keyspace_hits": 1000,
                "keyspace_misses": 100,
            }
        )

        health = health_checker._check_redis_health()

        assert health["status"] == "healthy"
        assert health["ping"] is True
        assert health["read_write_test"] is True
        assert "response_time_ms" in health
        assert "connected_clients" in health


if __name__ == "__main__":
    pytest.main([__file__])
