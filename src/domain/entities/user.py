"""User domain entity."""

from dataclasses import dataclass, field, replace
from typing import List

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
    webhook_ids: List[int] = field(default_factory=list)

    def change_email(self, new_email: Email) -> "User":
        """Create new user instance with changed email."""
        return replace(self, email=new_email, updated_at=Timestamp.now())

    def change_company_name(self, new_company_name: CompanyName) -> "User":
        """Create new user instance with changed company name."""
        return replace(self, company_name=new_company_name, updated_at=Timestamp.now())

    def update_password_hash(self, new_hash: str) -> "User":
        """Create new user instance with updated password hash."""
        return replace(self, password_hash=new_hash, updated_at=Timestamp.now())

    @property
    def has_organization(self) -> bool:
        """Check if user belongs to any organization."""
        return len(self.organization_ids) > 0

    @property
    def has_active_content(self) -> bool:
        """Check if user has any streams."""
        return len(self.stream_ids) > 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"User({self.email.value} - {self.company_name.value})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"User(id={self.id}, email={self.email.value!r}, "
            f"company={self.company_name.value!r}, active={self.is_active})"
        )
