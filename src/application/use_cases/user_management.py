"""User management use cases."""

from dataclasses import dataclass
from typing import Optional
import bcrypt

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.user import User
from src.domain.value_objects.email import Email
from src.domain.value_objects.company_name import CompanyName
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.organization_repository import OrganizationRepository


@dataclass
class UpdateProfileRequest:
    """Request to update user profile."""

    user_id: int
    email: Optional[str] = None
    company_name: Optional[str] = None
    password: Optional[str] = None


@dataclass
class UpdateProfileResult(UseCaseResult):
    """Result of profile update."""

    user: Optional[User] = None


@dataclass
class ChangePasswordRequest:
    """Request to change user password."""

    user_id: int
    current_password: str
    new_password: str


@dataclass
class ChangePasswordResult(UseCaseResult):
    """Result of password change."""

    pass


@dataclass
class GetProfileRequest:
    """Request to get user profile."""

    user_id: int


@dataclass
class GetProfileResult(UseCaseResult):
    """Result of getting user profile."""

    user: Optional[User] = None
    api_keys_count: Optional[int] = None
    organizations_count: Optional[int] = None


class UserManagementUseCase(UseCase[UpdateProfileRequest, UpdateProfileResult]):
    """Use case for user management operations."""

    def __init__(
        self,
        user_repo: UserRepository,
        api_key_repo: APIKeyRepository,
        org_repo: OrganizationRepository,
    ):
        """Initialize user management use case.

        Args:
            user_repo: Repository for user operations
            api_key_repo: Repository for API key operations
            org_repo: Repository for organization operations
        """
        self.user_repo = user_repo
        self.api_key_repo = api_key_repo
        self.org_repo = org_repo

    async def update_profile(
        self, request: UpdateProfileRequest
    ) -> UpdateProfileResult:
        """Update user profile.

        Args:
            request: Profile update request

        Returns:
            Profile update result
        """
        try:
            # Get existing user
            user = await self.user_repo.get(request.user_id)
            if not user:
                return UpdateProfileResult(
                    status=ResultStatus.NOT_FOUND, errors=["User not found"]
                )

            # Update email if provided
            if request.email and request.email != user.email.value:
                try:
                    new_email = Email(request.email)
                except ValueError as e:
                    return UpdateProfileResult(
                        status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
                    )

                # Check if email is already taken
                existing = await self.user_repo.get_by_email(new_email)
                if existing:
                    return UpdateProfileResult(
                        status=ResultStatus.VALIDATION_ERROR,
                        errors=["Email address is already in use"],
                    )

                user = user.change_email(new_email)

            # Update company name if provided
            if request.company_name:
                try:
                    new_company = CompanyName(request.company_name)
                except ValueError as e:
                    return UpdateProfileResult(
                        status=ResultStatus.VALIDATION_ERROR, errors=[str(e)]
                    )

                user = user.change_company_name(new_company)

            # Update password if provided
            if request.password:
                password_hash = bcrypt.hashpw(
                    request.password.encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")

                user = user.update_password_hash(password_hash)

            # Save updated user
            saved_user = await self.user_repo.save(user)

            return UpdateProfileResult(
                status=ResultStatus.SUCCESS,
                user=saved_user,
                message="Profile updated successfully",
            )

        except Exception as e:
            return UpdateProfileResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to update profile: {str(e)}"],
            )

    async def change_password(
        self, request: ChangePasswordRequest
    ) -> ChangePasswordResult:
        """Change user password.

        Args:
            request: Password change request

        Returns:
            Password change result
        """
        try:
            # Get user
            user = await self.user_repo.get(request.user_id)
            if not user:
                return ChangePasswordResult(
                    status=ResultStatus.NOT_FOUND, errors=["User not found"]
                )

            # Verify current password
            if not bcrypt.checkpw(
                request.current_password.encode("utf-8"),
                user.password_hash.encode("utf-8"),
            ):
                return ChangePasswordResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=["Current password is incorrect"],
                )

            # Check that new password is different
            if request.current_password == request.new_password:
                return ChangePasswordResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=["New password must be different from current password"],
                )

            # Hash new password
            new_password_hash = bcrypt.hashpw(
                request.new_password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            # Update user
            user = user.update_password_hash(new_password_hash)
            await self.user_repo.save(user)

            return ChangePasswordResult(
                status=ResultStatus.SUCCESS, message="Password changed successfully"
            )

        except Exception as e:
            return ChangePasswordResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to change password: {str(e)}"],
            )

    async def get_profile(self, request: GetProfileRequest) -> GetProfileResult:
        """Get user profile with additional stats.

        Args:
            request: Get profile request

        Returns:
            Profile result with stats
        """
        try:
            # Get user
            user = await self.user_repo.get(request.user_id)
            if not user:
                return GetProfileResult(
                    status=ResultStatus.NOT_FOUND, errors=["User not found"]
                )

            # Get API keys count
            api_keys = await self.api_key_repo.get_by_user(request.user_id)
            active_keys_count = len([k for k in api_keys if k.is_valid])

            # Get organizations count
            orgs = await self.org_repo.get_by_owner(request.user_id)
            orgs_count = len(orgs)

            return GetProfileResult(
                status=ResultStatus.SUCCESS,
                user=user,
                api_keys_count=active_keys_count,
                organizations_count=orgs_count,
                message="Profile retrieved successfully",
            )

        except Exception as e:
            return GetProfileResult(
                status=ResultStatus.FAILURE, errors=[f"Failed to get profile: {str(e)}"]
            )

    async def execute(self, request: UpdateProfileRequest) -> UpdateProfileResult:
        """Execute profile update (default use case method).

        Args:
            request: Profile update request

        Returns:
            Profile update result
        """
        return await self.update_profile(request)
