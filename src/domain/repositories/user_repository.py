"""User repository protocol."""

from typing import Protocol, Optional, List

from src.domain.repositories.base import Repository
from src.domain.entities.user import User
from src.domain.value_objects.email import Email


class UserRepository(Repository[User, int], Protocol):
    """Repository protocol for User entities.

    Extends the base repository with user-specific operations.
    """

    async def get_by_email(self, email: Email) -> Optional[User]:
        """Get user by email address."""
        ...

    async def exists_by_email(self, email: Email) -> bool:
        """Check if user exists with given email."""
        ...

    async def get_by_organization(self, organization_id: int) -> List[User]:
        """Get all users in an organization."""
        ...

    async def search_by_company(self, company_name: str) -> List[User]:
        """Search users by company name (partial match)."""
        ...

    async def get_with_api_keys(self, user_id: int) -> Optional[User]:
        """Get user with their API keys loaded."""
        ...

    async def count_active_users(self) -> int:
        """Count total active users."""
        ...
