"""FastAPI dependency injection."""

from typing import AsyncGenerator, Optional, Annotated
from uuid import UUID
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from contextvars import ContextVar

from shared.infrastructure.database.database import Database
from shared.infrastructure.config.config import get_settings, Settings
from shared.infrastructure.security.api_key_service import APIKeyService
from shared.infrastructure.security.url_signer import JWTURLSigner, SecureContentDelivery
from shared.infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
    StreamRepository,
    HighlightRepository,
    APIKeyRepository,
    WakeWordRepository,
)
from shared.domain.models.api_key import APIKey
from shared.domain.models.organization import Organization
from shared.domain.models.user import User

# Context variables for request-scoped data
current_organization: ContextVar[Optional[Organization]] = ContextVar(
    "current_organization", default=None
)
current_user: ContextVar[Optional[User]] = ContextVar("current_user", default=None)
current_api_key: ContextVar[Optional[APIKey]] = ContextVar(
    "current_api_key", default=None
)


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
    db: Database = Depends(get_database),
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with db.session() as session:
        yield session


# Repository dependencies
def get_organization_repository(
    session: AsyncSession = Depends(get_session),
) -> OrganizationRepository:
    """Get organization repository."""
    return OrganizationRepository(session)


def get_user_repository(session: AsyncSession = Depends(get_session)) -> UserRepository:
    """Get user repository."""
    return UserRepository(session)


def get_stream_repository(
    session: AsyncSession = Depends(get_session),
) -> StreamRepository:
    """Get stream repository."""
    return StreamRepository(session)


def get_highlight_repository(
    session: AsyncSession = Depends(get_session),
) -> HighlightRepository:
    """Get highlight repository."""
    return HighlightRepository(session)


def get_api_key_repository(
    session: AsyncSession = Depends(get_session),
) -> APIKeyRepository:
    """Get API key repository."""
    return APIKeyRepository(session)


def get_wake_word_repository(
    session: AsyncSession = Depends(get_session),
) -> WakeWordRepository:
    """Get wake word repository."""
    return WakeWordRepository(session)


# Service dependencies
def get_api_key_service(
    repository: APIKeyRepository = Depends(get_api_key_repository),
) -> APIKeyService:
    """Get API key service."""
    return APIKeyService(repository)


def get_highlight_service(
    highlight_repository: HighlightRepository = Depends(get_highlight_repository),
) -> "HighlightService":
    """Get highlight service."""
    from shared.application.services.highlight_service import HighlightService

    return HighlightService(highlight_repository)


def get_user_service(
    user_repository: UserRepository = Depends(get_user_repository),
    settings: Settings = Depends(get_settings_dep),
) -> "UserService":
    """Get user service."""
    from shared.application.services.user_service import UserService
    from shared.infrastructure.security.password_service import PasswordService
    from shared.infrastructure.security.jwt_service import JWTService

    password_service = PasswordService()
    jwt_service = JWTService(settings)
    return UserService(user_repository, password_service, jwt_service)


def get_organization_service(
    organization_repository: OrganizationRepository = Depends(
        get_organization_repository
    ),
    user_service: "UserService" = Depends(get_user_service),
) -> "OrganizationService":
    """Get organization service."""
    from shared.application.services.organization_service import OrganizationService

    return OrganizationService(organization_repository, user_service)


def get_jwt_signer(settings: Settings = Depends(get_settings_dep)) -> JWTURLSigner:
    """Get JWT URL signer."""
    return JWTURLSigner(settings.jwt_secret_key)


def get_secure_content_delivery(
    jwt_signer: JWTURLSigner = Depends(get_jwt_signer),
    settings: Settings = Depends(get_settings_dep),
) -> SecureContentDelivery:
    """Get secure content delivery service."""
    # This would be initialized with actual storage service
    # For now, returning None as storage isn't implemented yet
    return None  # TODO: Implement with S3 storage


# Authentication dependencies
async def get_api_key(
    api_key: Annotated[str, Header(alias="X-API-Key")],
    api_key_service: APIKeyService = Depends(get_api_key_service),
) -> APIKey:
    """Validate API key from header."""
    key_entity = await api_key_service.validate_api_key(api_key)
    if not key_entity:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    # Set in context
    current_api_key.set(key_entity)
    return key_entity


async def get_current_organization(
    api_key: APIKey = Depends(get_api_key),
    org_repository: OrganizationRepository = Depends(get_organization_repository),
) -> Organization:
    """Get current organization from API key."""
    org = await org_repository.get(api_key.organization_id)
    if not org or not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization not found or inactive",
        )

    # Set in context
    current_organization.set(org)
    return org


def require_scope(required_scope: str):
    """Create a dependency that requires a specific API scope."""

    async def check_scope(api_key: APIKey = Depends(get_api_key)):
        if not api_key.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required scope: {required_scope}",
            )
        return api_key

    return check_scope


# JWT authentication dependencies
async def get_jwt_service(
    settings: Settings = Depends(get_settings_dep),
) -> JWTURLSigner:
    """Get JWT service for user authentication."""
    from shared.infrastructure.security.jwt_service import JWTService

    return JWTService(settings)


# User authentication (for web interface)
async def get_current_user(
    authorization: Annotated[Optional[str], Header()] = None,
    jwt_service=Depends(get_jwt_service),
    user_repository: UserRepository = Depends(get_user_repository),
) -> Optional[User]:
    """Get current authenticated user from JWT token."""
    if not authorization:
        return None

    # Extract token from "Bearer <token>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None

    # Verify token
    token_payload = jwt_service.verify_access_token(token)
    if not token_payload:
        return None

    # Get user
    user_id = UUID(token_payload.sub)
    user = await user_repository.get(user_id)

    # Set in context
    if user and user.is_active:
        current_user.set(user)
        return user

    return None


async def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    """Require authenticated user."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin_user(user: User = Depends(require_user)) -> User:
    """Require authenticated admin user."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return user


async def require_org_member(
    org_id: UUID,
    user: User = Depends(require_user),
) -> User:
    """Require user to be member of specific organization."""
    if user.organization_id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    return user
