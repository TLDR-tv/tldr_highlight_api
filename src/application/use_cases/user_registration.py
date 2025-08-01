"""User registration use case."""

from dataclasses import dataclass
from typing import Optional

from src.application.use_cases.base import UseCaseResult, ResultStatus
from src.domain.entities.user import User
from src.domain.entities.api_key import APIKey, APIKeyScope
from src.domain.value_objects.email import Email
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.application.workflows import OrganizationManager
from src.domain.services.security_services import (
    PasswordHashingService,
    APIKeyHashingService,
)
from src.domain.exceptions import DuplicateEntityError


@dataclass
class RegisterRequest:
    """Request for user registration."""

    email: str
    password: str
    company_name: str
    organization_name: Optional[str] = None


@dataclass
class RegisterResult(UseCaseResult):
    """Result of user registration."""

    user_id: Optional[int] = None
    api_key: Optional[str] = None
    organization_id: Optional[int] = None


class UserRegistrationUseCase:
    """Handle user registration with organization creation."""

    def __init__(
        self,
        user_repo: UserRepository,
        api_key_repo: APIKeyRepository,
        org_repo: OrganizationRepository,
        org_service: OrganizationManager,
        password_service: PasswordHashingService,
        api_key_service: APIKeyHashingService,
    ) -> None:
        self.user_repo = user_repo
        self.api_key_repo = api_key_repo
        self.org_repo = org_repo
        self.org_service = org_service
        self.password_service = password_service
        self.api_key_service = api_key_service

    async def execute(self, request: RegisterRequest) -> RegisterResult:
        """Register a new user with organization and API key."""
        # Validate input
        try:
            email = Email(request.email)
            company_name = CompanyName(request.company_name)
        except ValueError as e:
            return RegisterResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)],
                message="Invalid input data",
            )

        # Check if user already exists
        existing = await self.user_repo.get_by_email(email)
        if existing:
            return RegisterResult(
                status=ResultStatus.FAILURE,
                errors=["User with this email already exists"],
                message="Registration failed",
            )

        # Create password hash using domain service
        password_hash = self.password_service.hash_password(request.password)

        # Create user
        user = User(
            email=email,
            password_hash=password_hash,
            company_name=company_name,
            created_at=Timestamp.now(),
            is_active=True,
        )

        try:
            saved_user = await self.user_repo.save(user)
        except DuplicateEntityError:
            return RegisterResult(
                status=ResultStatus.FAILURE,
                errors=["User with this email already exists"],
                message="Registration failed",
            )

        # Create organization
        org_name = request.organization_name or f"{request.company_name} Organization"
        org_result = await self.org_service.create_organization(
            owner_id=saved_user.id, name=org_name
        )

        if not org_result.is_success:
            # Rollback user creation
            await self.user_repo.delete(saved_user.id)
            return RegisterResult(
                status=ResultStatus.FAILURE,
                errors=org_result.errors,
                message="Failed to create organization",
            )

        # Create initial API key
        api_key_value = self.api_key_service.generate_key()
        api_key_hash = self.api_key_service.hash_key(api_key_value)

        api_key = APIKey(
            user_id=saved_user.id,
            organization_id=org_result.organization.id,
            name=f"Initial key for {request.email}",
            key_hash=api_key_hash,
            scopes=[APIKeyScope.STREAM_READ, APIKeyScope.STREAM_WRITE],
            created_at=Timestamp.now(),
            last_used_at=None,
            expires_at=None,
            is_active=True,
        )

        await self.api_key_repo.save(api_key)

        return RegisterResult(
            status=ResultStatus.SUCCESS,
            user_id=saved_user.id,
            api_key=api_key_value,  # Return the actual key, not the hash
            organization_id=org_result.organization.id,
            message="User registered successfully",
        )
