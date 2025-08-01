"""API key management use cases."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from src.application.use_cases.base import UseCaseResult, ResultStatus
from src.domain.entities.api_key import APIKey, APIKeyScope
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.services.security_services import APIKeyHashingService


@dataclass
class CreateAPIKeyRequest:
    """Request for creating a new API key."""

    user_id: int
    name: str
    scopes: List[APIKeyScope]
    expires_at: Optional[datetime] = None


@dataclass
class CreateAPIKeyResult(UseCaseResult):
    """Result of API key creation."""

    api_key: Optional[APIKey] = None
    key: Optional[str] = None  # The actual key string (shown only once)


@dataclass
class ListAPIKeysResult(UseCaseResult):
    """Result of listing API keys."""

    api_keys: List[APIKey] = field(default_factory=list)


@dataclass
class RevokeAPIKeyResult(UseCaseResult):
    """Result of API key revocation."""

    pass


@dataclass
class RotateAPIKeyResult(UseCaseResult):
    """Result of API key rotation."""

    api_key: Optional[APIKey] = None
    new_key: Optional[str] = None  # The new key string (shown only once)


class APIKeyManagementUseCase:
    """Handle API key CRUD operations."""

    def __init__(
        self,
        api_key_repo: APIKeyRepository,
        user_repo: UserRepository,
        org_repo: OrganizationRepository,
        api_key_service: APIKeyHashingService,
    ) -> None:
        self.api_key_repo = api_key_repo
        self.user_repo = user_repo
        self.org_repo = org_repo
        self.api_key_service = api_key_service

    async def create_api_key(self, request: CreateAPIKeyRequest) -> CreateAPIKeyResult:
        """Create a new API key for a user."""
        # Verify user exists
        user = await self.user_repo.get_by_id(request.user_id)
        if not user:
            return CreateAPIKeyResult(
                status=ResultStatus.NOT_FOUND,
                errors=["User not found"],
                message="Cannot create API key",
            )

        # Get user's organization
        orgs = await self.org_repo.get_by_owner(request.user_id)
        if not orgs:
            return CreateAPIKeyResult(
                status=ResultStatus.FAILURE,
                errors=["User has no organization"],
                message="Cannot create API key without organization",
            )

        organization = orgs[0]  # Use first organization

        # Generate new API key
        api_key_value = self.api_key_service.generate_key()
        api_key_hash = self.api_key_service.hash_key(api_key_value)

        # Create API key entity
        api_key = APIKey(
            user_id=request.user_id,
            organization_id=organization.id,
            name=request.name,
            key_hash=api_key_hash,
            scopes=request.scopes,
            created_at=Timestamp.now(),
            last_used_at=None,
            expires_at=Timestamp(request.expires_at) if request.expires_at else None,
            is_active=True,
        )

        saved_api_key = await self.api_key_repo.save(api_key)

        return CreateAPIKeyResult(
            status=ResultStatus.SUCCESS,
            api_key=saved_api_key,
            key=api_key_value,  # Return the actual key
            message="API key created successfully",
        )

    async def list_api_keys(self, user_id: int) -> ListAPIKeysResult:
        """List all API keys for a user."""
        # Verify user exists
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return ListAPIKeysResult(
                status=ResultStatus.NOT_FOUND,
                errors=["User not found"],
                message="Cannot list API keys",
            )

        # Get all API keys for user
        api_keys = await self.api_key_repo.get_by_user(user_id)

        # Sort by creation date (newest first)
        api_keys.sort(key=lambda k: k.created_at.value, reverse=True)

        return ListAPIKeysResult(
            status=ResultStatus.SUCCESS,
            api_keys=api_keys,
            message=f"Found {len(api_keys)} API keys",
        )

    async def revoke_api_key(self, user_id: int, api_key_id: int) -> RevokeAPIKeyResult:
        """Revoke an API key."""
        # Get the API key
        api_key = await self.api_key_repo.get_by_id(api_key_id)
        if not api_key:
            return RevokeAPIKeyResult(
                status=ResultStatus.NOT_FOUND,
                errors=["API key not found"],
                message="Cannot revoke API key",
            )

        # Verify ownership
        if api_key.user_id != user_id:
            return RevokeAPIKeyResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["You do not own this API key"],
                message="Cannot revoke API key",
            )

        # Revoke the key
        revoked_key = api_key.revoke()
        await self.api_key_repo.save(revoked_key)

        return RevokeAPIKeyResult(
            status=ResultStatus.SUCCESS,
            message="API key revoked successfully",
        )

    async def rotate_api_key(self, user_id: int, api_key_id: int) -> RotateAPIKeyResult:
        """Rotate an API key (revoke old, create new with same settings)."""
        # Get the existing API key
        old_key = await self.api_key_repo.get_by_id(api_key_id)
        if not old_key:
            return RotateAPIKeyResult(
                status=ResultStatus.NOT_FOUND,
                errors=["API key not found"],
                message="Cannot rotate API key",
            )

        # Verify ownership
        if old_key.user_id != user_id:
            return RotateAPIKeyResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["You do not own this API key"],
                message="Cannot rotate API key",
            )

        # Generate new key value
        new_key_value = self.api_key_service.generate_key()
        new_key_hash = self.api_key_service.hash_key(new_key_value)

        # Create new API key with same settings
        new_api_key = APIKey(
            user_id=old_key.user_id,
            organization_id=old_key.organization_id,
            name=f"{old_key.name} (rotated)",
            key_hash=new_key_hash,
            scopes=old_key.scopes,
            created_at=Timestamp.now(),
            last_used_at=None,
            expires_at=old_key.expires_at,
            is_active=True,
        )

        # Save new key
        saved_new_key = await self.api_key_repo.save(new_api_key)

        # Revoke old key
        revoked_old_key = old_key.revoke()
        await self.api_key_repo.save(revoked_old_key)

        return RotateAPIKeyResult(
            status=ResultStatus.SUCCESS,
            api_key=saved_new_key,
            new_key=new_key_value,
            message="API key rotated successfully",
        )
