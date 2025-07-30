"""User domain entity."""

from dataclasses import dataclass, field
from typing import Optional, List

from src.domain.entities.base import Entity
from src.domain.value_objects.email import Email
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp


@dataclass
class User(Entity[int]):
    """Domain entity representing a user.
    
    Users are enterprise customers who use the TL;DR Highlight API
    to extract highlights from their content.
    """
    
    email: Email
    company_name: CompanyName
    password_hash: str
    
    # Status
    is_active: bool = True
    
    # Related entity IDs (not full objects to avoid circular deps)
    api_key_ids: List[int] = field(default_factory=list)
    organization_ids: List[int] = field(default_factory=list)
    stream_ids: List[int] = field(default_factory=list)
    batch_ids: List[int] = field(default_factory=list)
    webhook_ids: List[int] = field(default_factory=list)
    
    def change_email(self, new_email: Email) -> "User":
        """Create new user instance with changed email."""
        return User(
            id=self.id,
            email=new_email,
            company_name=self.company_name,
            password_hash=self.password_hash,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            api_key_ids=self.api_key_ids.copy(),
            organization_ids=self.organization_ids.copy(),
            stream_ids=self.stream_ids.copy(),
            batch_ids=self.batch_ids.copy(),
            webhook_ids=self.webhook_ids.copy()
        )
    
    def change_company_name(self, new_company_name: CompanyName) -> "User":
        """Create new user instance with changed company name."""
        return User(
            id=self.id,
            email=self.email,
            company_name=new_company_name,
            password_hash=self.password_hash,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            api_key_ids=self.api_key_ids.copy(),
            organization_ids=self.organization_ids.copy(),
            stream_ids=self.stream_ids.copy(),
            batch_ids=self.batch_ids.copy(),
            webhook_ids=self.webhook_ids.copy()
        )
    
    def update_password_hash(self, new_hash: str) -> "User":
        """Create new user instance with updated password hash."""
        return User(
            id=self.id,
            email=self.email,
            company_name=self.company_name,
            password_hash=new_hash,
            is_active=self.is_active,
            created_at=self.created_at,
            updated_at=Timestamp.now(),
            api_key_ids=self.api_key_ids.copy(),
            organization_ids=self.organization_ids.copy(),
            stream_ids=self.stream_ids.copy(),
            batch_ids=self.batch_ids.copy(),
            webhook_ids=self.webhook_ids.copy()
        )
    
    @property
    def has_organization(self) -> bool:
        """Check if user belongs to any organization."""
        return len(self.organization_ids) > 0
    
    @property
    def has_active_content(self) -> bool:
        """Check if user has any streams or batches."""
        return len(self.stream_ids) > 0 or len(self.batch_ids) > 0