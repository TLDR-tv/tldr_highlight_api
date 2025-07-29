"""Batch repository implementation."""

import json
from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from src.domain.repositories.batch_repository import BatchRepository as IBatchRepository
from src.domain.entities.batch import Batch, BatchStatus
from src.domain.value_objects.timestamp import Timestamp
from src.domain.exceptions import EntityNotFoundError
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.batch import Batch as BatchModel
from src.infrastructure.persistence.models.user import User as UserModel
from src.infrastructure.persistence.mappers.batch_mapper import BatchMapper


class BatchRepository(BaseRepository[Batch, BatchModel, int], IBatchRepository):
    """Concrete implementation of BatchRepository using SQLAlchemy."""
    
    def __init__(self, session):
        """Initialize BatchRepository with session."""
        super().__init__(
            session=session,
            model_class=BatchModel,
            mapper=BatchMapper()
        )
    
    async def get_by_user(self, user_id: int,
                        status: Optional[BatchStatus] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Batch]:
        """Get batches for a user, optionally filtered by status.
        
        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of batches for the user
        """
        stmt = select(BatchModel).where(
            BatchModel.user_id == user_id
        )
        
        if status:
            stmt = stmt.where(BatchModel.status == status.value)
        
        stmt = stmt.order_by(BatchModel.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_active_batches(self) -> List[Batch]:
        """Get all batches currently being processed.
        
        Returns:
            List of active batches
        """
        stmt = select(BatchModel).where(
            BatchModel.status.in_([
                BatchStatus.PENDING.value,
                BatchStatus.PROCESSING.value
            ])
        ).order_by(BatchModel.created_at.asc())  # FIFO processing
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_by_date_range(self, start: Timestamp, end: Timestamp,
                              user_id: Optional[int] = None) -> List[Batch]:
        """Get batches within a date range.
        
        Args:
            start: Start timestamp
            end: End timestamp
            user_id: Optional user ID filter
            
        Returns:
            List of batches in the date range
        """
        stmt = select(BatchModel).where(
            and_(
                BatchModel.created_at >= start.value,
                BatchModel.created_at <= end.value
            )
        )
        
        if user_id:
            stmt = stmt.where(BatchModel.user_id == user_id)
        
        stmt = stmt.order_by(BatchModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_with_items(self, batch_id: int) -> Optional[Batch]:
        """Get batch with all its items loaded.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            Batch with items if found, None otherwise
        """
        stmt = select(BatchModel).where(
            BatchModel.id == batch_id
        ).options(
            # In a real implementation, you might have a separate BatchItem table
            # selectinload(BatchModel.batch_items)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        return self.mapper.to_domain(model) if model else None
    
    async def count_by_status(self, status: BatchStatus,
                            user_id: Optional[int] = None) -> int:
        """Count batches by status.
        
        Args:
            status: Batch status
            user_id: Optional user ID filter
            
        Returns:
            Count of batches with the status
        """
        stmt = select(func.count()).select_from(BatchModel).where(
            BatchModel.status == status.value
        )
        
        if user_id:
            stmt = stmt.where(BatchModel.user_id == user_id)
        
        result = await self.session.execute(stmt)
        return result.scalar() or 0
    
    async def get_processing_stats(self, user_id: int) -> Dict[str, Any]:
        """Get batch processing statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with processing statistics
        """
        # Count by status
        status_counts = {}
        for status in BatchStatus:
            count = await self.count_by_status(status, user_id)
            status_counts[status.value] = count
        
        # Get average processing time for completed batches
        avg_processing_stmt = select(
            func.avg(
                func.extract('epoch', BatchModel.completed_at - BatchModel.started_at)
            )
        ).where(
            and_(
                BatchModel.user_id == user_id,
                BatchModel.status.in_([
                    BatchStatus.COMPLETED.value,
                    BatchStatus.PARTIALLY_FAILED.value
                ]),
                BatchModel.started_at.isnot(None),
                BatchModel.completed_at.isnot(None)
            )
        )
        
        result = await self.session.execute(avg_processing_stmt)
        avg_processing_seconds = result.scalar() or 0
        
        # Get success rates
        success_stmt = select(
            func.avg(BatchModel.successful_items / func.nullif(BatchModel.total_items, 0)).label('avg_success_rate'),
            func.sum(BatchModel.total_items).label('total_items'),
            func.sum(BatchModel.successful_items).label('successful_items'),
            func.sum(BatchModel.failed_items).label('failed_items')
        ).where(
            and_(
                BatchModel.user_id == user_id,
                BatchModel.total_items > 0
            )
        )
        
        result = await self.session.execute(success_stmt)
        row = result.one()
        
        avg_success_rate = row.avg_success_rate or 0.0
        total_items = row.total_items or 0
        successful_items = row.successful_items or 0
        failed_items = row.failed_items or 0
        
        return {
            'by_status': status_counts,
            'total_batches': sum(status_counts.values()),
            'avg_processing_time_seconds': float(avg_processing_seconds),
            'avg_success_rate': float(avg_success_rate),
            'total_items_processed': total_items,
            'successful_items': successful_items,
            'failed_items': failed_items
        }
    
    async def get_failed_items_summary(self, user_id: int,
                                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get summary of failed items across batches.
        
        Args:
            user_id: User ID
            limit: Maximum number of failed items to return
            
        Returns:
            List of failed item summaries
        """
        # This would typically query a separate BatchItem table
        # For now, we'll extract from the JSON items field
        stmt = select(
            BatchModel.id,
            BatchModel.name,
            BatchModel.items,
            BatchModel.created_at
        ).where(
            and_(
                BatchModel.user_id == user_id,
                BatchModel.failed_items > 0
            )
        ).order_by(BatchModel.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        
        failed_summaries = []
        for row in result:
            batch_id, batch_name, items_json, created_at = row
            
            # Parse items JSON and extract failed items
            if items_json:
                items = json.loads(items_json) if isinstance(items_json, str) else items_json
                for item in items:
                    if item.get('status') == BatchStatus.FAILED.value:
                        failed_summaries.append({
                            'batch_id': batch_id,
                            'batch_name': batch_name,
                            'item_id': item.get('item_id'),
                            'url': item.get('url'),
                            'error_message': item.get('error_message'),
                            'batch_created_at': created_at.isoformat() if created_at else None
                        })
        
        return failed_summaries[:limit]
    
    async def cleanup_old_batches(self, older_than: Timestamp) -> int:
        """Clean up old completed/failed batches.
        
        Args:
            older_than: Timestamp before which to clean up
            
        Returns:
            Number of batches cleaned up
        """
        # Find old completed/failed batches
        stmt = select(BatchModel).where(
            and_(
                BatchModel.created_at < older_than.value,
                BatchModel.status.in_([
                    BatchStatus.COMPLETED.value,
                    BatchStatus.PARTIALLY_FAILED.value,
                    BatchStatus.FAILED.value,
                    BatchStatus.CANCELLED.value
                ])
            )
        )
        
        result = await self.session.execute(stmt)
        old_batches = list(result.scalars().unique())
        
        # Delete them (cascading will handle related data)
        for batch in old_batches:
            await self.session.delete(batch)
        
        await self.session.flush()
        return len(old_batches)
    
    async def get_next_pending_batch(self) -> Optional[Batch]:
        """Get the next batch ready for processing.
        
        Returns:
            Next pending batch if available, None otherwise
        """
        stmt = select(BatchModel).where(
            BatchModel.status == BatchStatus.PENDING.value
        ).order_by(BatchModel.created_at.asc()).limit(1)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        return self.mapper.to_domain(model) if model else None
    
    async def get_batch_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the batch processing queue.
        
        Returns:
            Dictionary with queue statistics
        """
        # Count pending batches
        pending_count = await self.count_by_status(BatchStatus.PENDING)
        processing_count = await self.count_by_status(BatchStatus.PROCESSING)
        
        # Get oldest pending batch
        oldest_pending_stmt = select(
            func.min(BatchModel.created_at)
        ).where(BatchModel.status == BatchStatus.PENDING.value)
        
        result = await self.session.execute(oldest_pending_stmt)
        oldest_pending_at = result.scalar()
        
        # Get average queue time (time from creation to processing start)
        avg_queue_time_stmt = select(
            func.avg(
                func.extract('epoch', BatchModel.started_at - BatchModel.created_at)
            )
        ).where(
            and_(
                BatchModel.started_at.isnot(None),
                BatchModel.created_at.isnot(None)
            )
        )
        
        result = await self.session.execute(avg_queue_time_stmt)
        avg_queue_time_seconds = result.scalar() or 0
        
        return {
            'pending_batches': pending_count,
            'processing_batches': processing_count,
            'oldest_pending_at': oldest_pending_at.isoformat() if oldest_pending_at else None,
            'avg_queue_time_seconds': float(avg_queue_time_seconds)
        }
    
    async def get_batches_by_success_rate(self, min_success_rate: float = 0.8,
                                        user_id: Optional[int] = None) -> List[Batch]:
        """Get batches with success rate above threshold.
        
        Args:
            min_success_rate: Minimum success rate (0.0 to 1.0)
            user_id: Optional user ID filter
            
        Returns:
            List of high-success batches
        """
        stmt = select(BatchModel).where(
            and_(
                BatchModel.total_items > 0,
                (BatchModel.successful_items / BatchModel.total_items) >= min_success_rate
            )
        )
        
        if user_id:
            stmt = stmt.where(BatchModel.user_id == user_id)
        
        stmt = stmt.order_by(
            (BatchModel.successful_items / BatchModel.total_items).desc()
        )
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def retry_failed_items(self, batch_id: int) -> Optional[Batch]:
        """Reset failed items in a batch for retry.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            Updated batch if found, None otherwise
        """
        batch = await self.get_with_items(batch_id)
        if not batch:
            return None
        
        # Only allow retry for completed/partially failed batches
        if batch.status not in [BatchStatus.COMPLETED, BatchStatus.PARTIALLY_FAILED, BatchStatus.FAILED]:
            raise ValueError(f"Cannot retry batch in {batch.status} status")
        
        # Reset failed items to pending
        updated_items = []
        for item in batch.items:
            if item.status == BatchStatus.FAILED:
                updated_item = item.__class__(
                    url=item.url,
                    item_id=item.item_id,
                    status=BatchStatus.PENDING,
                    stream_id=None,  # Clear stream_id for retry
                    error_message=None,  # Clear error message
                    metadata=item.metadata.copy()
                )
                updated_items.append(updated_item)
            else:
                updated_items.append(item)
        
        # Create updated batch
        updated_batch = Batch(
            id=batch.id,
            name=batch.name,
            user_id=batch.user_id,
            status=BatchStatus.PENDING,  # Reset to pending
            processing_options=batch.processing_options,
            items=updated_items,
            started_at=None,  # Reset processing timestamps
            completed_at=None,
            total_items=batch.total_items,
            processed_items=batch.successful_items,  # Keep successful count
            successful_items=batch.successful_items,
            failed_items=0,  # Reset failed count
            webhook_url=batch.webhook_url,
            created_at=batch.created_at,
            updated_at=Timestamp.now()
        )
        
        return await self.save(updated_batch)
    
    async def get_user_batch_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get batch analytics for a user over a time period.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with batch analytics
        """
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get batches in the time period
        stmt = select(BatchModel).where(
            and_(
                BatchModel.user_id == user_id,
                BatchModel.created_at >= start_date
            )
        )
        
        result = await self.session.execute(stmt)
        batches = list(result.scalars().unique())
        
        if not batches:
            return {
                'period_days': days,
                'total_batches': 0,
                'by_status': {},
                'total_items': 0,
                'success_rate': 0.0,
                'avg_batch_size': 0.0,
                'avg_processing_time_seconds': 0.0
            }
        
        # Calculate statistics
        by_status = {}
        total_items = 0
        successful_items = 0
        processing_times = []
        
        for batch_model in batches:
            status = batch_model.status
            by_status[status] = by_status.get(status, 0) + 1
            
            total_items += batch_model.total_items or 0
            successful_items += batch_model.successful_items or 0
            
            if batch_model.started_at and batch_model.completed_at:
                processing_time = (batch_model.completed_at - batch_model.started_at).total_seconds()
                processing_times.append(processing_time)
        
        success_rate = successful_items / total_items if total_items > 0 else 0.0
        avg_batch_size = total_items / len(batches) if batches else 0.0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            'period_days': days,
            'total_batches': len(batches),
            'by_status': by_status,
            'total_items': total_items,
            'successful_items': successful_items,
            'failed_items': total_items - successful_items,
            'success_rate': success_rate,
            'avg_batch_size': avg_batch_size,
            'avg_processing_time_seconds': avg_processing_time
        }