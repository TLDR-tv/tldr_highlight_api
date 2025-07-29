"""Usage record repository protocol."""

from typing import Protocol, List, Optional, Dict
from abc import abstractmethod

from src.domain.repositories.base import Repository
from src.domain.entities.usage_record import UsageRecord, UsageType
from src.domain.value_objects.timestamp import Timestamp


class UsageRecordRepository(Repository[UsageRecord, int], Protocol):
    """Repository protocol for UsageRecord entities.
    
    Extends the base repository with usage record-specific operations.
    """
    
    @abstractmethod
    async def get_by_user(self, user_id: int,
                        usage_type: Optional[UsageType] = None,
                        start: Optional[Timestamp] = None,
                        end: Optional[Timestamp] = None) -> List[UsageRecord]:
        """Get usage records for a user with optional filters."""
        ...
    
    @abstractmethod
    async def get_by_organization(self, organization_id: int,
                                usage_type: Optional[UsageType] = None,
                                start: Optional[Timestamp] = None,
                                end: Optional[Timestamp] = None) -> List[UsageRecord]:
        """Get usage records for an organization."""
        ...
    
    @abstractmethod
    async def get_by_resource(self, resource_id: int,
                            resource_type: str) -> List[UsageRecord]:
        """Get usage records for a specific resource."""
        ...
    
    @abstractmethod
    async def get_billable_usage(self, user_id: int,
                               billing_period_start: Timestamp,
                               billing_period_end: Timestamp) -> List[UsageRecord]:
        """Get billable usage records for a billing period."""
        ...
    
    @abstractmethod
    async def calculate_usage_totals(self, user_id: int,
                                   usage_type: UsageType,
                                   start: Timestamp,
                                   end: Timestamp) -> Dict[str, float]:
        """Calculate usage totals by type for a period."""
        ...
    
    @abstractmethod
    async def get_usage_by_api_key(self, api_key_id: int,
                                 start: Optional[Timestamp] = None,
                                 end: Optional[Timestamp] = None) -> List[UsageRecord]:
        """Get usage records for an API key."""
        ...
    
    @abstractmethod
    async def aggregate_by_type(self, user_id: int,
                              start: Timestamp,
                              end: Timestamp) -> Dict[UsageType, Dict[str, float]]:
        """Aggregate usage by type for a period."""
        ...
    
    @abstractmethod
    async def get_incomplete_records(self, older_than: Timestamp) -> List[UsageRecord]:
        """Get incomplete usage records older than specified time."""
        ...
    
    @abstractmethod
    async def bulk_create(self, records: List[UsageRecord]) -> List[UsageRecord]:
        """Create multiple usage records at once."""
        ...