"""Organization repository protocol."""

from typing import Protocol, Optional, List, Dict
from abc import abstractmethod

from src.domain.repositories.base import Repository
from src.domain.entities.organization import Organization, PlanType
from src.domain.value_objects.company_name import CompanyName


class OrganizationRepository(Repository[Organization, int], Protocol):
    """Repository protocol for Organization entities.
    
    Extends the base repository with organization-specific operations.
    """
    
    @abstractmethod
    async def get_by_owner(self, owner_id: int) -> List[Organization]:
        """Get organizations owned by a user."""
        ...
    
    @abstractmethod
    async def get_by_member(self, user_id: int) -> List[Organization]:
        """Get organizations where user is a member."""
        ...
    
    @abstractmethod
    async def get_by_name(self, name: CompanyName) -> Optional[Organization]:
        """Get organization by exact name."""
        ...
    
    @abstractmethod
    async def search_by_name(self, query: str) -> List[Organization]:
        """Search organizations by name (partial match)."""
        ...
    
    @abstractmethod
    async def get_by_plan_type(self, plan_type: PlanType) -> List[Organization]:
        """Get all organizations with specific plan type."""
        ...
    
    @abstractmethod
    async def get_active_organizations(self) -> List[Organization]:
        """Get all active organizations."""
        ...
    
    @abstractmethod
    async def count_by_plan_type(self) -> Dict[PlanType, int]:
        """Get count of organizations by plan type."""
        ...
    
    @abstractmethod
    async def get_expiring_soon(self, days: int = 30) -> List[Organization]:
        """Get organizations with subscriptions expiring soon."""
        ...
    
    @abstractmethod
    async def get_with_usage_stats(self, organization_id: int) -> Optional[Organization]:
        """Get organization with usage statistics."""
        ...