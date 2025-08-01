"""API key validation use case."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

from src.application.use_cases.base import UseCaseResult, ResultStatus
from src.domain.entities.api_key import APIKeyScope
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.services.security_services import APIKeyHashingService


@dataclass
class ValidateAPIKeyRequest:
    """Request for API key validation."""

    api_key: str
    required_scopes: Optional[List[APIKeyScope]] = None


@dataclass
class ValidateAPIKeyResult(UseCaseResult):
    """Result of API key validation."""

    user_id: Optional[int] = None
    organization_id: Optional[int] = None
    scopes: Optional[List[str]] = None
    rate_limit: Optional[int] = None


class APIKeyValidationUseCase:
    """Handle API key validation and usage tracking."""

    def __init__(
        self,
        api_key_repo: APIKeyRepository,
        org_repo: OrganizationRepository,
        api_key_service: APIKeyHashingService,
    ) -> None:
        self.api_key_repo = api_key_repo
        self.org_repo = org_repo
        self.api_key_service = api_key_service

    async def execute(self, request: ValidateAPIKeyRequest) -> ValidateAPIKeyResult:
        """Validate API key and check required scopes."""
        # Hash the provided key
        key_hash = self.api_key_service.hash_key(request.api_key)

        # Find API key by hash
        api_key = await self.api_key_repo.get_by_key_hash(key_hash)
        if not api_key:
            return ValidateAPIKeyResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["Invalid API key"],
                message="Authentication failed",
            )

        # Check if key is valid
        if not api_key.is_valid:
            reasons = []
            if not api_key.is_active:
                reasons.append("API key is revoked")
            if api_key.expires_at and api_key.expires_at.value < datetime.now(
                timezone.utc
            ):
                reasons.append("API key is expired")

            return ValidateAPIKeyResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=reasons,
                message="API key is not valid",
            )

        # Check required scopes
        if request.required_scopes:
            missing_scopes = [
                scope.value
                for scope in request.required_scopes
                if scope not in api_key.scopes
            ]
            if missing_scopes:
                return ValidateAPIKeyResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=[f"Missing required scopes: {', '.join(missing_scopes)}"],
                    message="Insufficient permissions",
                )

        # Get organization for rate limit
        rate_limit = 60  # Default rate limit
        if api_key.organization_id:
            orgs = await self.org_repo.get_by_id(api_key.organization_id)
            if orgs:
                org = orgs[0]
                rate_limit = org.plan_limits.api_rate_limit_per_minute

        # Update last used timestamp
        updated_api_key = api_key.record_usage()
        await self.api_key_repo.save(updated_api_key)

        return ValidateAPIKeyResult(
            status=ResultStatus.SUCCESS,
            user_id=api_key.user_id,
            organization_id=api_key.organization_id,
            scopes=[scope.value for scope in api_key.scopes],
            rate_limit=rate_limit,
            message="API key is valid",
        )
