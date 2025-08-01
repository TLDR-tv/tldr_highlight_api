"""Organization management service."""

import secrets
from datetime import datetime
from typing import Optional
from uuid import UUID
import structlog

from ...domain.models.organization import Organization
from ...domain.models.user import UserRole
from ...infrastructure.storage.repositories import OrganizationRepository
from .user_service import UserService

logger = structlog.get_logger()


class OrganizationService:
    """Service for organization management."""
    
    def __init__(
        self,
        organization_repository: OrganizationRepository,
        user_service: UserService,
    ):
        """Initialize with dependencies."""
        self.organization_repository = organization_repository
        self.user_service = user_service
    
    async def create_organization(
        self,
        name: str,
        owner_email: str,
        owner_name: str,
        owner_password: str,
        webhook_url: Optional[str] = None,
    ) -> tuple[Organization, object]:  # Returns (org, user)
        """Create a new organization with owner user.
        
        Args:
            name: Organization name
            owner_email: Owner's email
            owner_name: Owner's full name
            owner_password: Owner's password
            webhook_url: Optional webhook URL
            
        Returns:
            Tuple of (organization, owner_user)
            
        Raises:
            ValueError: If validation fails
        """
        # Check if organization name already exists
        existing_org = await self.organization_repository.get_by_slug(
            Organization._generate_slug(name)
        )
        if existing_org:
            raise ValueError("Organization with similar name already exists")
        
        # Create organization
        org = Organization(
            name=name.strip(),
            webhook_url=webhook_url,
            webhook_secret=secrets.token_urlsafe(32) if webhook_url else None,
        )
        
        saved_org = await self.organization_repository.add(org)
        
        # Create owner user
        try:
            owner_user = await self.user_service.create_user(
                organization_id=saved_org.id,
                email=owner_email,
                name=owner_name,
                password=owner_password,
                role=UserRole.ADMIN,
            )
        except Exception as e:
            # Rollback organization creation
            await self.organization_repository.delete(saved_org.id)
            raise
        
        logger.info(
            "Organization created",
            organization_id=str(saved_org.id),
            name=name,
            owner_email=owner_email,
        )
        
        return saved_org, owner_user
    
    async def update_organization(
        self,
        organization_id: UUID,
        name: Optional[str] = None,
        webhook_url: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Organization:
        """Update organization details.
        
        Args:
            organization_id: Organization to update
            name: New name (optional)
            webhook_url: New webhook URL (optional)
            is_active: Active status (optional)
            
        Returns:
            Updated organization
            
        Raises:
            ValueError: If organization not found
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            raise ValueError("Organization not found")
        
        # Update name if provided
        if name is not None:
            name = name.strip()
            # Check if new name conflicts
            new_slug = Organization._generate_slug(name)
            if new_slug != org.slug:
                existing_org = await self.organization_repository.get_by_slug(new_slug)
                if existing_org:
                    raise ValueError("Organization with similar name already exists")
            org.name = name
            org.slug = new_slug
        
        # Update webhook URL if provided
        if webhook_url is not None:
            org.webhook_url = webhook_url or None
            # Generate new secret if webhook URL is set and didn't have one
            if webhook_url and not org.webhook_secret:
                org.webhook_secret = secrets.token_urlsafe(32)
        
        # Update active status if provided
        if is_active is not None:
            org.is_active = is_active
        
        org.updated_at = datetime.utcnow()
        updated_org = await self.organization_repository.update(org)
        
        logger.info(
            "Organization updated",
            organization_id=str(organization_id),
            name=org.name,
        )
        
        return updated_org
    
    async def get_usage_stats(self, organization_id: UUID) -> dict:
        """Get detailed usage statistics for an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            Usage statistics dictionary
            
        Raises:
            ValueError: If organization not found
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            raise ValueError("Organization not found")
        
        # Calculate additional metrics
        avg_highlights_per_stream = (
            org.total_highlights_generated / org.total_streams_processed
            if org.total_streams_processed > 0
            else 0
        )
        
        avg_processing_seconds_per_stream = (
            org.total_processing_seconds / org.total_streams_processed
            if org.total_streams_processed > 0
            else 0
        )
        
        return {
            "organization_id": str(org.id),
            "name": org.name,
            "total_streams_processed": org.total_streams_processed,
            "total_highlights_generated": org.total_highlights_generated,
            "total_processing_seconds": org.total_processing_seconds,
            "total_processing_hours": org.total_processing_seconds / 3600,
            "avg_highlights_per_stream": round(avg_highlights_per_stream, 2),
            "avg_processing_seconds_per_stream": round(avg_processing_seconds_per_stream, 2),
            "created_at": org.created_at.isoformat(),
            "is_active": org.is_active,
        }
    
    async def regenerate_webhook_secret(self, organization_id: UUID) -> str:
        """Regenerate webhook secret for an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            New webhook secret
            
        Raises:
            ValueError: If organization not found or no webhook configured
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            raise ValueError("Organization not found")
        
        if not org.webhook_url:
            raise ValueError("No webhook URL configured")
        
        # Generate new secret
        new_secret = secrets.token_urlsafe(32)
        org.webhook_secret = new_secret
        org.updated_at = datetime.utcnow()
        
        await self.organization_repository.update(org)
        
        logger.info(
            "Webhook secret regenerated",
            organization_id=str(organization_id),
        )
        
        return new_secret
    
    async def add_wake_word(self, organization_id: UUID, wake_word: str) -> Organization:
        """Add a custom wake word to the organization.
        
        Args:
            organization_id: Organization ID
            wake_word: Wake word to add
            
        Returns:
            Updated organization
            
        Raises:
            ValueError: If organization not found
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            raise ValueError("Organization not found")
        
        org.add_wake_word(wake_word)
        org.updated_at = datetime.utcnow()
        
        updated_org = await self.organization_repository.update(org)
        
        logger.info(
            "Wake word added",
            organization_id=str(organization_id),
            wake_word=wake_word,
        )
        
        return updated_org
    
    async def remove_wake_word(self, organization_id: UUID, wake_word: str) -> Organization:
        """Remove a custom wake word from the organization.
        
        Args:
            organization_id: Organization ID
            wake_word: Wake word to remove
            
        Returns:
            Updated organization
            
        Raises:
            ValueError: If organization not found
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            raise ValueError("Organization not found")
        
        org.remove_wake_word(wake_word)
        org.updated_at = datetime.utcnow()
        
        updated_org = await self.organization_repository.update(org)
        
        logger.info(
            "Wake word removed",
            organization_id=str(organization_id),
            wake_word=wake_word,
        )
        
        return updated_org
    
    async def record_stream_usage(
        self,
        organization_id: UUID,
        processing_seconds: float,
        highlights_count: int = 0,
    ) -> None:
        """Record stream processing usage for billing.
        
        Args:
            organization_id: Organization ID
            processing_seconds: Seconds of processing time
            highlights_count: Number of highlights generated
        """
        org = await self.organization_repository.get(organization_id)
        if not org:
            logger.error("Failed to record usage - organization not found", organization_id=str(organization_id))
            return
        
        org.record_usage(
            streams=1,
            highlights=highlights_count,
            seconds=processing_seconds,
        )
        org.updated_at = datetime.utcnow()
        
        await self.organization_repository.update(org)
        
        logger.info(
            "Stream usage recorded",
            organization_id=str(organization_id),
            processing_seconds=processing_seconds,
            highlights_count=highlights_count,
        )