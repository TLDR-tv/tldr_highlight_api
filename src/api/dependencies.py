"""FastAPI dependency injection."""
from typing import AsyncGenerator, Optional, Annotated
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from contextvars import ContextVar
from uuid import UUID

from ..infrastructure.storage.database import Database
from ..infrastructure.config import get_settings, Settings
from ..infrastructure.security.api_key_service import APIKeyService
from ..infrastructure.security.url_signer import JWTURLSigner, SecureContentDelivery
from ..infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
    StreamRepository,
    HighlightRepository,
    APIKeyRepository,
    WakeWordRepository
)
from ..domain.models.api_key import APIKey
from ..domain.models.organization import Organization
from ..domain.models.user import User

# Context variables for request-scoped data
current_organization: ContextVar[Optional[Organization]] = ContextVar('current_organization', default=None)
current_user: ContextVar[Optional[User]] = ContextVar('current_user', default=None)
current_api_key: ContextVar[Optional[APIKey]] = ContextVar('current_api_key', default=None)


# Settings dependency
def get_settings_dep() -> Settings:
    """Get application settings."""
    return get_settings()


# Database dependencies
_db_instance: Optional[Database] = None


async def get_database() -> Database:
    """Get database instance (singleton)."""
    global _db_instance
    if _db_instance is None:
        settings = get_settings()
        _db_instance = Database(settings.database_url)
        await _db_instance.create_tables()
    return _db_instance


async def get_session(
    db: Database = Depends(get_database)
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with db.session() as session:
        yield session


# Repository dependencies
def get_organization_repository(
    session: AsyncSession = Depends(get_session)
) -> OrganizationRepository:
    """Get organization repository."""
    return OrganizationRepository(session)


def get_user_repository(
    session: AsyncSession = Depends(get_session)
) -> UserRepository:
    """Get user repository."""
    return UserRepository(session)


def get_stream_repository(
    session: AsyncSession = Depends(get_session)
) -> StreamRepository:
    """Get stream repository."""
    return StreamRepository(session)


def get_highlight_repository(
    session: AsyncSession = Depends(get_session)
) -> HighlightRepository:
    """Get highlight repository."""
    return HighlightRepository(session)


def get_api_key_repository(
    session: AsyncSession = Depends(get_session)
) -> APIKeyRepository:
    """Get API key repository."""
    return APIKeyRepository(session)


def get_wake_word_repository(
    session: AsyncSession = Depends(get_session)
) -> WakeWordRepository:
    """Get wake word repository."""
    return WakeWordRepository(session)


# Service dependencies
def get_api_key_service(
    repository: APIKeyRepository = Depends(get_api_key_repository)
) -> APIKeyService:
    """Get API key service."""
    return APIKeyService(repository)


def get_jwt_signer(
    settings: Settings = Depends(get_settings_dep)
) -> JWTURLSigner:
    """Get JWT URL signer."""
    return JWTURLSigner(settings.jwt_secret_key)


def get_secure_content_delivery(
    jwt_signer: JWTURLSigner = Depends(get_jwt_signer),
    settings: Settings = Depends(get_settings_dep)
) -> SecureContentDelivery:
    """Get secure content delivery service."""
    # This would be initialized with actual storage service
    # For now, returning None as storage isn't implemented yet
    return None  # TODO: Implement with S3 storage


# Authentication dependencies
async def get_api_key(
    api_key: Annotated[str, Header(alias="X-API-Key")],
    api_key_service: APIKeyService = Depends(get_api_key_service)
) -> APIKey:
    """Validate API key from header."""
    key_entity = await api_key_service.validate_api_key(api_key)
    if not key_entity:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Set in context
    current_api_key.set(key_entity)
    return key_entity


async def get_current_organization(
    api_key: APIKey = Depends(get_api_key),
    org_repository: OrganizationRepository = Depends(get_organization_repository)
) -> Organization:
    """Get current organization from API key."""
    org = await org_repository.get(api_key.organization_id)
    if not org or not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization not found or inactive"
        )
    
    # Set in context
    current_organization.set(org)
    return org


async def require_scope(required_scope: str):
    """Create a dependency that requires a specific API scope."""
    async def check_scope(api_key: APIKey = Depends(get_api_key)):
        if not api_key.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required scope: {required_scope}"
            )
        return api_key
    return check_scope


# User authentication (for web interface)
async def get_current_user(
    # This would typically check JWT token from cookie or header
    # For now, returning None
    user_repository: UserRepository = Depends(get_user_repository)
) -> Optional[User]:
    """Get current authenticated user."""
    # TODO: Implement JWT-based user authentication
    return None


async def require_admin_user(
    user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require authenticated admin user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user