"""User repository implementation."""

from typing import Optional, List
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from src.domain.repositories.user_repository import UserRepository as IUserRepository
from src.domain.entities.user import User
from src.domain.value_objects.email import Email
from src.infrastructure.persistence.repositories.base_repository import BaseRepository
from src.infrastructure.persistence.models.user import User as UserModel
from src.infrastructure.persistence.mappers.user_mapper import UserMapper


class UserRepository(BaseRepository[User, UserModel, int], IUserRepository):
    """Concrete implementation of UserRepository using SQLAlchemy."""
    
    def __init__(self, session):
        """Initialize UserRepository with session."""
        super().__init__(
            session=session,
            model_class=UserModel,
            mapper=UserMapper()
        )
    
    async def get_by_email(self, email: Email) -> Optional[User]:
        """Get user by email address.
        
        Args:
            email: Email value object
            
        Returns:
            User domain entity if found, None otherwise
        """
        stmt = select(UserModel).where(
            UserModel.email == email.value
        ).options(
            selectinload(UserModel.api_keys),
            selectinload(UserModel.owned_organizations),
            selectinload(UserModel.streams),
            selectinload(UserModel.batches),
            selectinload(UserModel.webhooks)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
            
        return self.mapper.to_domain(model)
    
    async def exists_by_email(self, email: Email) -> bool:
        """Check if user exists with given email.
        
        Args:
            email: Email value object
            
        Returns:
            True if user exists, False otherwise
        """
        from sqlalchemy import exists
        
        stmt = select(exists().where(UserModel.email == email.value))
        result = await self.session.execute(stmt)
        return result.scalar()
    
    async def get_by_organization(self, organization_id: int) -> List[User]:
        """Get all users in an organization.
        
        Args:
            organization_id: Organization ID
            
        Returns:
            List of users in the organization
        """
        # First, get users who own the organization
        owner_stmt = select(UserModel).join(
            UserModel.owned_organizations
        ).where(
            UserModel.owned_organizations.any(id=organization_id)
        )
        
        # Execute query
        result = await self.session.execute(owner_stmt)
        models = list(result.scalars().unique())
        
        # TODO: When we add organization membership, also query members
        # For now, just return owners
        
        return self.mapper.to_domain_list(models)
    
    async def search_by_company(self, company_name: str) -> List[User]:
        """Search users by company name (partial match).
        
        Args:
            company_name: Company name to search for
            
        Returns:
            List of users matching the company name
        """
        stmt = select(UserModel).where(
            UserModel.company_name.ilike(f"%{company_name}%")
        ).options(
            selectinload(UserModel.api_keys),
            selectinload(UserModel.owned_organizations)
        )
        
        result = await self.session.execute(stmt)
        models = list(result.scalars().unique())
        
        return self.mapper.to_domain_list(models)
    
    async def get_with_api_keys(self, user_id: int) -> Optional[User]:
        """Get user with their API keys loaded.
        
        Args:
            user_id: User ID
            
        Returns:
            User with API keys if found, None otherwise
        """
        stmt = select(UserModel).where(
            UserModel.id == user_id
        ).options(
            selectinload(UserModel.api_keys)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
            
        return self.mapper.to_domain(model)
    
    async def count_active_users(self) -> int:
        """Count total active users.
        
        Returns:
            Count of active users
        """
        # For now, all users are considered active
        # In the future, we might add an 'is_active' field
        stmt = select(func.count()).select_from(UserModel)
        result = await self.session.execute(stmt)
        return result.scalar() or 0