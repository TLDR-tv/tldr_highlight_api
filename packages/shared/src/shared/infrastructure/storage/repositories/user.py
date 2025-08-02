"""User repository implementation."""

from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....domain.models.user import User
from ...database.models import UserModel


class UserRepository:
    """SQLAlchemy implementation of user repository."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def add(self, entity: User) -> User:
        """Add user to repository."""
        model = UserModel(
            id=entity.id,
            organization_id=entity.organization_id,
            email=entity.email,
            name=entity.name,
            role=entity.role,
            is_active=entity.is_active,
            hashed_password=entity.hashed_password,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_login_at=entity.last_login_at,
        )

        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    # Alias for compatibility with tests
    async def create(self, entity: User) -> User:
        """Create new user (alias for add)."""
        return await self.add(entity)

    async def get(self, id: UUID) -> Optional[User]:
        """Get user by ID."""
        model = await self.session.get(UserModel, id)
        return self._to_entity(model) if model else None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_by_organization(self, org_id: UUID) -> list[User]:
        """List all users in an organization."""
        result = await self.session.execute(
            select(UserModel)
            .where(UserModel.organization_id == org_id)
            .order_by(UserModel.name)
        )
        models = result.scalars().all()
        return [self._to_entity(model) for model in models]

    async def update(self, entity: User) -> User:
        """Update existing user."""
        model = await self.session.get(UserModel, entity.id)
        if not model:
            raise ValueError(f"User {entity.id} not found")

        model.email = entity.email
        model.name = entity.name
        model.role = entity.role
        model.is_active = entity.is_active
        model.hashed_password = entity.hashed_password
        model.updated_at = entity.updated_at
        model.last_login_at = entity.last_login_at

        await self.session.commit()
        await self.session.refresh(model)

        return self._to_entity(model)

    async def delete(self, id: UUID) -> None:
        """Delete user by ID."""
        model = await self.session.get(UserModel, id)
        if model:
            await self.session.delete(model)
            await self.session.commit()

    async def list(self, **filters) -> list[User]:
        """List users with optional filters."""
        query = select(UserModel)

        if "organization_id" in filters:
            query = query.where(UserModel.organization_id == filters["organization_id"])
        if "is_active" in filters:
            query = query.where(UserModel.is_active == filters["is_active"])

        result = await self.session.execute(query.order_by(UserModel.name))
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    def _to_entity(self, model: UserModel) -> User:
        """Convert model to entity."""
        return User(
            id=model.id,
            organization_id=model.organization_id,
            email=model.email,
            name=model.name,
            role=model.role,
            is_active=model.is_active,
            hashed_password=model.hashed_password,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_login_at=model.last_login_at,
        )
