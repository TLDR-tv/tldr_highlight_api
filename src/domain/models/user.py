"""User domain model - represents platform employees."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class UserRole(Enum):
    """User roles within an organization."""

    MEMBER = "member"
    ADMIN = "admin"


@dataclass
class User:
    """Platform employee user."""

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    email: str = ""
    name: str = ""
    role: UserRole = UserRole.MEMBER
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None

    # Security
    hashed_password: str = ""

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN

    @property
    def can_manage_organization(self) -> bool:
        """Check if user can manage organization settings."""
        return self.is_admin

    @property
    def can_view_all_highlights(self) -> bool:
        """All active users can view all highlights for their platform."""
        return self.is_active

    def record_login(self) -> None:
        """Update last login timestamp."""
        self.last_login_at = datetime.utcnow()

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.email})"
