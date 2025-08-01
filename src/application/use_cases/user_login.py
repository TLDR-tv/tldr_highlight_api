"""User login use case."""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.application.use_cases.base import UseCaseResult, ResultStatus
from src.domain.entities.api_key import APIKey, APIKeyScope
from src.domain.value_objects.email import Email
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.services.security_services import (
    PasswordHashingService,
    APIKeyHashingService,
)


@dataclass
class LoginRequest:
    """Request for user login."""

    email: str
    password: str
    create_api_key: bool = False
    api_key_name: Optional[str] = None


@dataclass
class LoginResult(UseCaseResult):
    """Result of user login."""

    user_id: Optional[int] = None
    api_key: Optional[str] = None
    expires_at: Optional[datetime] = None


class UserLoginUseCase:
    """Handle user login and optional API key creation."""

    def __init__(
        self,
        user_repo: UserRepository,
        api_key_repo: APIKeyRepository,
        org_repo: OrganizationRepository,
        password_service: PasswordHashingService,
        api_key_service: APIKeyHashingService,
    ) -> None:
        self.user_repo = user_repo
        self.api_key_repo = api_key_repo
        self.org_repo = org_repo
        self.password_service = password_service
        self.api_key_service = api_key_service

    async def execute(self, request: LoginRequest) -> LoginResult:
        """Authenticate user and optionally create API key."""
        # Validate email
        try:
            email = Email(request.email)
        except ValueError as e:
            return LoginResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)],
                message="Invalid email format",
            )

        # Find user
        user = await self.user_repo.get_by_email(email)
        if not user:
            return LoginResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["Invalid credentials"],
                message="Login failed",
            )

        # Verify password
        if not self.password_service.verify_password(
            request.password, user.password_hash
        ):
            return LoginResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["Invalid credentials"],
                message="Login failed",
            )

        # Check if user is active
        if not user.is_active:
            return LoginResult(
                status=ResultStatus.UNAUTHORIZED,
                errors=["Account is disabled"],
                message="Login failed",
            )

        # Update last login
        user = user.update_last_login(Timestamp.now())
        await self.user_repo.save(user)

        # Create API key if requested
        api_key_value = None
        expires_at = None

        if request.create_api_key:
            # Get user's organization
            orgs = await self.org_repo.get_by_owner(user.id)
            if not orgs:
                return LoginResult(
                    status=ResultStatus.FAILURE,
                    errors=["User has no organization"],
                    message="Cannot create API key without organization",
                )

            organization = orgs[0]  # Use first organization

            # Generate new API key
            api_key_value = self.api_key_service.generate_key()
            api_key_hash = self.api_key_service.hash_key(api_key_value)

            # Create API key entity
            key_name = (
                request.api_key_name
                or f"Session key {datetime.now(timezone.utc).isoformat()}"
            )
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)

            api_key = APIKey(
                user_id=user.id,
                organization_id=organization.id,
                name=key_name,
                key_hash=api_key_hash,
                scopes=[APIKeyScope.STREAM_READ, APIKeyScope.STREAM_WRITE],
                created_at=Timestamp.now(),
                last_used_at=None,
                expires_at=Timestamp(expires_at),
                is_active=True,
            )

            await self.api_key_repo.save(api_key)

        return LoginResult(
            status=ResultStatus.SUCCESS,
            user_id=user.id,
            api_key=api_key_value,
            expires_at=expires_at,
            message="Login successful",
        )
