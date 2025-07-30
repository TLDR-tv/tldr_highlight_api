"""Batch processing use cases."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.batch import Batch, BatchStatus, BatchItem
from src.domain.entities.webhook import WebhookEvent
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.batch_repository import BatchRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.services.organization_management_service import (
    OrganizationManagementService,
)
from src.domain.services.webhook_delivery_service import WebhookDeliveryService
from src.domain.services.usage_tracking_service import UsageTrackingService
from src.domain.exceptions import (
    QuotaExceededError,
)


@dataclass
class BatchCreateRequest:
    """Request to create a batch processing job."""

    user_id: int
    name: str
    items: List[Dict[str, Any]]  # List of {"url": str, "title": str, "metadata": dict}
    priority: Optional[int] = None
    processing_options: Optional[Dict[str, Any]] = None


@dataclass
class BatchCreateResult(UseCaseResult):
    """Result of batch creation."""

    batch_id: Optional[int] = None
    total_items: Optional[int] = None
    estimated_completion_time: Optional[str] = None


@dataclass
class BatchStatusRequest:
    """Request to get batch status."""

    user_id: int
    batch_id: int
    include_items: bool = False


@dataclass
class BatchStatusResult(UseCaseResult):
    """Result of batch status check."""

    batch_id: Optional[int] = None
    batch_status: Optional[str] = None
    progress_percentage: Optional[float] = None
    successful_items: Optional[int] = None
    failed_items: Optional[int] = None
    items: Optional[List[Dict[str, Any]]] = None


@dataclass
class BatchCancelRequest:
    """Request to cancel a batch."""

    user_id: int
    batch_id: int
    reason: Optional[str] = None


@dataclass
class BatchCancelResult(UseCaseResult):
    """Result of batch cancellation."""

    batch_id: Optional[int] = None
    cancelled_items: Optional[int] = None


@dataclass
class BatchRetryRequest:
    """Request to retry failed items in a batch."""

    user_id: int
    batch_id: int
    item_ids: Optional[List[int]] = None  # If None, retry all failed items


@dataclass
class BatchRetryResult(UseCaseResult):
    """Result of batch retry."""

    batch_id: Optional[int] = None
    retried_items: Optional[int] = None


class BatchProcessingUseCase(UseCase[BatchCreateRequest, BatchCreateResult]):
    """Use case for batch processing operations."""

    def __init__(
        self,
        user_repo: UserRepository,
        batch_repo: BatchRepository,
        org_repo: OrganizationRepository,
        org_service: OrganizationManagementService,
        webhook_service: WebhookDeliveryService,
        usage_service: UsageTrackingService,
    ):
        """Initialize batch processing use case.

        Args:
            user_repo: Repository for user operations
            batch_repo: Repository for batch operations
            org_repo: Repository for organization operations
            org_service: Service for organization management
            webhook_service: Service for webhook delivery
            usage_service: Service for usage tracking
        """
        self.user_repo = user_repo
        self.batch_repo = batch_repo
        self.org_repo = org_repo
        self.org_service = org_service
        self.webhook_service = webhook_service
        self.usage_service = usage_service

    async def create_batch(self, request: BatchCreateRequest) -> BatchCreateResult:
        """Create a new batch processing job.

        Args:
            request: Batch creation request

        Returns:
            Batch creation result
        """
        try:
            # Validate user exists
            user = await self.user_repo.get(request.user_id)
            if not user:
                return BatchCreateResult(
                    status=ResultStatus.NOT_FOUND, errors=["User not found"]
                )

            # Check organization and quotas
            orgs = await self.org_repo.get_by_owner(request.user_id)
            org = orgs[0] if orgs else None

            # Validate batch size limits
            max_batch_size = 100  # Default
            if org:
                max_batch_size = org.plan_limits.max_batch_size

            if len(request.items) > max_batch_size:
                return BatchCreateResult(
                    status=ResultStatus.QUOTA_EXCEEDED,
                    errors=[
                        f"Batch size {len(request.items)} exceeds limit of {max_batch_size}"
                    ],
                )

            # Create batch items
            batch_items = []
            for idx, item_data in enumerate(request.items):
                batch_item = BatchItem(
                    id=None,
                    batch_id=None,  # Will be set when batch is saved
                    url=item_data["url"],
                    title=item_data.get("title", f"Item {idx + 1}"),
                    status=BatchItemStatus.PENDING,
                    metadata=item_data.get("metadata", {}),
                    created_at=Timestamp.now(),
                    updated_at=Timestamp.now(),
                )
                batch_items.append(batch_item)

            # Determine priority
            priority = request.priority
            if priority is None:
                # Higher priority for smaller batches
                if len(batch_items) <= 10:
                    priority = 1
                elif len(batch_items) <= 50:
                    priority = 2
                else:
                    priority = 3

            # Create batch
            batch = Batch(
                id=None,
                user_id=request.user_id,
                organization_id=org.id if org else None,
                name=request.name,
                status=BatchStatus.CREATED,
                priority=priority,
                processing_options=request.processing_options or {},
                created_at=Timestamp.now(),
                updated_at=Timestamp.now(),
            )

            # Save batch with items
            saved_batch = await self.batch_repo.save(batch)

            # Save batch items
            for item in batch_items:
                item.batch_id = saved_batch.id
            await self.batch_repo.save_batch_items(batch_items)

            # Update batch with items
            saved_batch.items = batch_items

            # Queue batch for processing
            saved_batch = await self.batch_repo.queue_for_processing(saved_batch.id)

            # Track API usage
            await self.usage_service.track_api_call(
                user_id=request.user_id,
                api_key_id=1,  # Would come from auth context
                endpoint="/batches",
                method="POST",
                response_time_ms=200,
                status_code=201,
            )

            # Estimate completion time (simplified)
            # Assume 2 minutes per item + queue time
            estimated_minutes = len(batch_items) * 2 + (priority * 10)
            estimated_completion = Timestamp.now().add_minutes(estimated_minutes)

            return BatchCreateResult(
                status=ResultStatus.SUCCESS,
                batch_id=saved_batch.id,
                total_items=len(batch_items),
                estimated_completion_time=estimated_completion.iso_string,
                message=f"Batch created with {len(batch_items)} items",
            )

        except QuotaExceededError as e:
            return BatchCreateResult(
                status=ResultStatus.QUOTA_EXCEEDED, errors=[str(e)]
            )
        except Exception as e:
            return BatchCreateResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to create batch: {str(e)}"],
            )

    async def get_batch_status(self, request: BatchStatusRequest) -> BatchStatusResult:
        """Get status of a batch processing job.

        Args:
            request: Batch status request

        Returns:
            Batch status result
        """
        try:
            # Get batch
            batch = await self.batch_repo.get(request.batch_id)
            if not batch:
                return BatchStatusResult(
                    status=ResultStatus.NOT_FOUND, errors=["Batch not found"]
                )

            # Verify ownership
            if batch.user_id != request.user_id:
                return BatchStatusResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to view this batch"],
                )

            # Calculate progress
            progress = 0.0
            if batch.total_items > 0:
                completed = batch.successful_items + batch.failed_items
                progress = (completed / batch.total_items) * 100

            # Prepare response
            result = BatchStatusResult(
                status=ResultStatus.SUCCESS,
                batch_id=batch.id,
                batch_status=batch.status.value,
                progress_percentage=progress,
                successful_items=batch.successful_items,
                failed_items=batch.failed_items,
                message="Batch status retrieved successfully",
            )

            # Include items if requested
            if request.include_items:
                items = await self.batch_repo.get_batch_items(
                    batch.id,
                    limit=100,  # Reasonable limit
                )
                result.items = [
                    {
                        "id": item.id,
                        "url": item.url,
                        "title": item.title,
                        "status": item.status.value,
                        "error_message": item.error_message,
                        "stream_id": item.stream_id,
                        "processed_at": item.processed_at.iso_string
                        if item.processed_at
                        else None,
                    }
                    for item in items
                ]

            return result

        except Exception as e:
            return BatchStatusResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get batch status: {str(e)}"],
            )

    async def cancel_batch(self, request: BatchCancelRequest) -> BatchCancelResult:
        """Cancel a batch processing job.

        Args:
            request: Batch cancellation request

        Returns:
            Cancellation result
        """
        try:
            # Get batch
            batch = await self.batch_repo.get(request.batch_id)
            if not batch:
                return BatchCancelResult(
                    status=ResultStatus.NOT_FOUND, errors=["Batch not found"]
                )

            # Verify ownership
            if batch.user_id != request.user_id:
                return BatchCancelResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to cancel this batch"],
                )

            # Check if batch can be cancelled
            if batch.status in [BatchStatus.COMPLETED, BatchStatus.CANCELLED]:
                return BatchCancelResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[f"Cannot cancel batch in {batch.status.value} status"],
                )

            # Cancel batch
            cancelled_batch = batch.cancel(reason=request.reason)
            await self.batch_repo.save(cancelled_batch)

            # Count cancelled items
            items = await self.batch_repo.get_batch_items(
                batch.id, status=BatchItemStatus.PENDING
            )
            cancelled_count = len(items)

            # Update items to cancelled
            for item in items:
                item.status = BatchItemStatus.FAILED
                item.error_message = "Batch cancelled by user"
                item.processed_at = Timestamp.now()

            if items:
                await self.batch_repo.save_batch_items(items)

            return BatchCancelResult(
                status=ResultStatus.SUCCESS,
                batch_id=batch.id,
                cancelled_items=cancelled_count,
                message=f"Batch cancelled, {cancelled_count} pending items cancelled",
            )

        except Exception as e:
            return BatchCancelResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to cancel batch: {str(e)}"],
            )

    async def retry_failed_items(self, request: BatchRetryRequest) -> BatchRetryResult:
        """Retry failed items in a batch.

        Args:
            request: Batch retry request

        Returns:
            Retry result
        """
        try:
            # Get batch
            batch = await self.batch_repo.get(request.batch_id)
            if not batch:
                return BatchRetryResult(
                    status=ResultStatus.NOT_FOUND, errors=["Batch not found"]
                )

            # Verify ownership
            if batch.user_id != request.user_id:
                return BatchRetryResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You don't have permission to retry this batch"],
                )

            # Get failed items
            if request.item_ids:
                # Get specific items
                items = []
                for item_id in request.item_ids:
                    item = await self.batch_repo.get_batch_item(item_id)
                    if (
                        item
                        and item.batch_id == batch.id
                        and item.status == BatchItemStatus.FAILED
                    ):
                        items.append(item)
            else:
                # Get all failed items
                items = await self.batch_repo.get_batch_items(
                    batch.id, status=BatchItemStatus.FAILED
                )

            if not items:
                return BatchRetryResult(
                    status=ResultStatus.SUCCESS,
                    batch_id=batch.id,
                    retried_items=0,
                    message="No failed items to retry",
                )

            # Reset items to pending
            for item in items:
                item.status = BatchItemStatus.PENDING
                item.error_message = None
                item.retry_count = (
                    item.retry_count + 1 if hasattr(item, "retry_count") else 1
                )
                item.updated_at = Timestamp.now()

            await self.batch_repo.save_batch_items(items)

            # Update batch status if needed
            if batch.status == BatchStatus.COMPLETED:
                batch.status = BatchStatus.PROCESSING
                batch.updated_at = Timestamp.now()
                await self.batch_repo.save(batch)

            # Re-queue batch for processing
            await self.batch_repo.queue_for_processing(batch.id)

            return BatchRetryResult(
                status=ResultStatus.SUCCESS,
                batch_id=batch.id,
                retried_items=len(items),
                message=f"Retrying {len(items)} failed items",
            )

        except Exception as e:
            return BatchRetryResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to retry batch items: {str(e)}"],
            )

    async def execute(self, request: BatchCreateRequest) -> BatchCreateResult:
        """Execute batch creation (default use case method).

        Args:
            request: Batch creation request

        Returns:
            Batch creation result
        """
        return await self.create_batch(request)

    async def process_batch_completion(self, batch_id: int):
        """Handle batch completion (called by background workers).

        Args:
            batch_id: ID of completed batch
        """
        try:
            batch = await self.batch_repo.get(batch_id)
            if not batch:
                return

            # Update batch status
            if batch.failed_items == 0:
                batch.status = BatchStatus.COMPLETED
            else:
                batch.status = BatchStatus.COMPLETED_WITH_ERRORS

            batch.completed_at = Timestamp.now()
            await self.batch_repo.save(batch)

            # Calculate processing time
            processing_minutes = 0
            if batch.started_at and batch.completed_at:
                duration = batch.completed_at.value - batch.started_at.value
                processing_minutes = duration.total_seconds() / 60

            # Track usage
            await self.usage_service.track_batch_processing(
                user_id=batch.user_id,
                batch_id=batch.id,
                total_items=batch.total_items,
                successful_items=batch.successful_items,
                processing_minutes=processing_minutes,
            )

            # Send webhook
            await self.webhook_service.trigger_event(
                event=WebhookEvent.BATCH_COMPLETED,
                user_id=batch.user_id,
                resource_id=batch.id,
                metadata={
                    "total_items": batch.total_items,
                    "successful_items": batch.successful_items,
                    "failed_items": batch.failed_items,
                    "processing_minutes": processing_minutes,
                },
            )

        except Exception as e:
            # Log error but don't fail - this is background processing
            print(f"Error handling batch completion: {e}")
