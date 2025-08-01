"""Organization management workflow - clean Pythonic implementation.

Handles organization operations for the B2B platform.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.domain.entities.organization import Organization
from src.domain.entities.user import User
from src.domain.value_objects import CompanyName, Email
from src.domain.repositories import (
    OrganizationRepository,
    UserRepository,
    APIKeyRepository,
)


@dataclass
class OrganizationManager:
    """Manages organizations for B2B customers.
    
    Simple workflow that handles organization creation,
    updates, and member management.
    """
    
    organization_repo: OrganizationRepository
    user_repo: UserRepository
    api_key_repo: APIKeyRepository
    
    async def create_organization(
        self,
        name: str,
        owner_id: int,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Create a new organization."""
        # Validate owner exists
        owner = await self.user_repo.get(owner_id)
        if not owner:
            raise ValueError(f"User {owner_id} not found")
        
        # Create organization with domain factory
        org = Organization.create(
            name=CompanyName(name),
            owner_id=owner_id,
            settings=settings or {},
        )
        
        # Save and return
        saved_org = await self.organization_repo.save(org)
        
        # Add owner as admin member
        await self.add_member(
            organization_id=saved_org.id,
            user_id=owner_id,
            role="admin",
            added_by=owner_id,
        )
        
        return saved_org
    
    async def update_organization(
        self,
        organization_id: int,
        requester_id: int,
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Update organization details."""
        # Get organization
        org = await self.organization_repo.get(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check permissions
        if not await self._user_can_manage_org(requester_id, organization_id):
            raise PermissionError("User cannot manage this organization")
        
        # Update fields
        if name:
            org.update_name(CompanyName(name))
        
        if settings is not None:
            org.update_settings(settings)
        
        # Save and return
        return await self.organization_repo.save(org)
    
    async def add_member(
        self,
        organization_id: int,
        user_id: int,
        role: str,
        added_by: int,
    ) -> None:
        """Add a member to an organization."""
        # Validate organization
        org = await self.organization_repo.get(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check permissions
        if not await self._user_can_manage_org(added_by, organization_id):
            raise PermissionError("User cannot manage this organization")
        
        # Validate user
        user = await self.user_repo.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Add member using domain method
        org.add_member(user_id, role)
        
        # Save
        await self.organization_repo.save(org)
    
    async def remove_member(
        self,
        organization_id: int,
        user_id: int,
        removed_by: int,
    ) -> None:
        """Remove a member from an organization."""
        # Get organization
        org = await self.organization_repo.get(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check permissions
        if not await self._user_can_manage_org(removed_by, organization_id):
            raise PermissionError("User cannot manage this organization")
        
        # Remove member using domain method
        org.remove_member(user_id)
        
        # Save
        await self.organization_repo.save(org)
    
    async def get_organization(
        self,
        organization_id: int,
        requester_id: int,
    ) -> Organization:
        """Get organization details."""
        # Get organization
        org = await self.organization_repo.get(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check if user has access
        if not await self._user_has_org_access(requester_id, organization_id):
            raise PermissionError("User cannot access this organization")
        
        return org
    
    async def list_user_organizations(
        self,
        user_id: int,
    ) -> List[Organization]:
        """List all organizations a user belongs to."""
        # Get user to validate
        user = await self.user_repo.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Get organizations
        orgs = []
        for org_id in user.organization_ids:
            org = await self.organization_repo.get(org_id)
            if org:
                orgs.append(org)
        
        return orgs
    
    async def create_api_key(
        self,
        organization_id: int,
        name: str,
        scopes: List[str],
        created_by: int,
    ) -> str:
        """Create an API key for the organization."""
        # Validate organization
        org = await self.organization_repo.get(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check permissions
        if not await self._user_can_manage_org(created_by, organization_id):
            raise PermissionError("User cannot manage this organization")
        
        # Create API key using domain service
        from src.domain.entities.api_key import APIKey
        api_key = APIKey.create(
            name=name,
            organization_id=organization_id,
            scopes=scopes,
            created_by=created_by,
        )
        
        # Save and return the key value
        saved_key = await self.api_key_repo.save(api_key)
        return saved_key.key  # Return the actual key string
    
    # Private helper methods
    
    async def _user_can_manage_org(
        self,
        user_id: int,
        organization_id: int,
    ) -> bool:
        """Check if user can manage organization."""
        org = await self.organization_repo.get(organization_id)
        if not org:
            return False
        
        # Owner can always manage
        if org.owner_id == user_id:
            return True
        
        # Check if user is admin member
        member = org.get_member(user_id)
        return member and member.role == "admin"
    
    async def _user_has_org_access(
        self,
        user_id: int,
        organization_id: int,
    ) -> bool:
        """Check if user has access to organization."""
        org = await self.organization_repo.get(organization_id)
        if not org:
            return False
        
        # Check if user is member
        return org.has_member(user_id)