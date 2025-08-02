"""User management service."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
import structlog

from ...domain.models.user import User, UserRole
from ...infrastructure.storage.repositories import UserRepository
from ...infrastructure.security.password_service import PasswordService
from ...infrastructure.security.jwt_service import JWTService

logger = structlog.get_logger()


class UserService:
    """Service for user management and authentication."""

    def __init__(
        self,
        user_repository: UserRepository,
        password_service: PasswordService,
        jwt_service: JWTService,
    ):
        """Initialize with dependencies."""
        self.user_repository = user_repository
        self.password_service = password_service
        self.jwt_service = jwt_service

    async def create_user(
        self,
        organization_id: UUID,
        email: str,
        name: str,
        password: str,
        role: UserRole = UserRole.MEMBER,
    ) -> User:
        """Create a new user.

        Args:
            organization_id: Organization the user belongs to
            email: User's email address
            name: User's full name
            password: Plain text password
            role: User's role in the organization

        Returns:
            Created user

        Raises:
            ValueError: If email already exists or password is invalid

        """
        # Check if email already exists
        existing_user = await self.user_repository.get_by_email(email)
        if existing_user:
            raise ValueError("User with this email already exists")

        # Validate password strength
        is_valid, errors = self.password_service.validate_password_strength(password)
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Hash password
        hashed_password = self.password_service.hash_password(password)

        # Create user
        user = User(
            organization_id=organization_id,
            email=email.lower().strip(),
            name=name.strip(),
            role=role,
            hashed_password=hashed_password,
        )

        saved_user = await self.user_repository.add(user)

        logger.info(
            "User created",
            user_id=str(saved_user.id),
            organization_id=str(organization_id),
            email=email,
            role=role.value,
        )

        return saved_user

    async def authenticate(
        self, email: str, password: str
    ) -> tuple[Optional[User], Optional[str], Optional[str]]:
        """Authenticate user and return tokens.

        Args:
            email: User's email
            password: Plain text password

        Returns:
            Tuple of (user, access_token, refresh_token) or (None, None, None) if auth fails

        """
        # Get user by email
        user = await self.user_repository.get_by_email(email.lower().strip())
        if not user:
            logger.warning("Authentication failed - user not found", email=email)
            return None, None, None

        # Check if user is active
        if not user.is_active:
            logger.warning(
                "Authentication failed - user inactive", user_id=str(user.id)
            )
            return None, None, None

        # Verify password
        if not self.password_service.verify_password(password, user.hashed_password):
            logger.warning(
                "Authentication failed - invalid password", user_id=str(user.id)
            )
            return None, None, None

        # Generate tokens
        access_token = self.jwt_service.create_access_token(
            user_id=user.id,
            organization_id=user.organization_id,
            email=user.email,
            role=user.role.value,
        )

        refresh_token = self.jwt_service.create_refresh_token(user_id=user.id)

        # Update last login
        user.record_login()
        await self.user_repository.update(user)

        logger.info("User authenticated", user_id=str(user.id), email=email)

        return user, access_token, refresh_token

    async def refresh_tokens(
        self, refresh_token: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token) or (None, None) if invalid

        """
        # Verify refresh token
        payload = self.jwt_service.verify_refresh_token(refresh_token)
        if not payload:
            logger.warning("Token refresh failed - invalid token")
            return None, None

        # Get user
        user_id = UUID(payload["sub"])
        user = await self.user_repository.get(user_id)
        if not user or not user.is_active:
            logger.warning(
                "Token refresh failed - user not found or inactive",
                user_id=str(user_id),
            )
            return None, None

        # Generate new tokens
        new_access_token = self.jwt_service.create_access_token(
            user_id=user.id,
            organization_id=user.organization_id,
            email=user.email,
            role=user.role.value,
        )

        new_refresh_token = self.jwt_service.create_refresh_token(user_id=user.id)

        logger.info("Tokens refreshed", user_id=str(user.id))

        return new_access_token, new_refresh_token

    async def update_profile(
        self, user_id: UUID, name: Optional[str] = None, email: Optional[str] = None
    ) -> User:
        """Update user profile.

        Args:
            user_id: User to update
            name: New name (optional)
            email: New email (optional)

        Returns:
            Updated user

        Raises:
            ValueError: If user not found or email already exists

        """
        user = await self.user_repository.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Update name if provided
        if name is not None:
            user.name = name.strip()

        # Update email if provided
        if email is not None:
            email = email.lower().strip()
            # Check if new email already exists
            if email != user.email:
                existing_user = await self.user_repository.get_by_email(email)
                if existing_user:
                    raise ValueError("Email already in use")
                user.email = email

        user.updated_at = datetime.now(timezone.utc)
        updated_user = await self.user_repository.update(user)

        logger.info("User profile updated", user_id=str(user_id))

        return updated_user

    async def change_password(
        self, user_id: UUID, old_password: str, new_password: str
    ) -> bool:
        """Change user's password.

        Args:
            user_id: User changing password
            old_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            ValueError: If validation fails

        """
        user = await self.user_repository.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Verify old password
        if not self.password_service.verify_password(
            old_password, user.hashed_password
        ):
            raise ValueError("Current password is incorrect")

        # Validate new password
        is_valid, errors = self.password_service.validate_password_strength(
            new_password
        )
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Hash and update password
        user.hashed_password = self.password_service.hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        await self.user_repository.update(user)

        logger.info("User password changed", user_id=str(user_id))

        return True

    async def request_password_reset(self, email: str) -> Optional[str]:
        """Request password reset for user.

        Args:
            email: User's email

        Returns:
            Reset token if user exists, None otherwise

        """
        user = await self.user_repository.get_by_email(email.lower().strip())
        if not user or not user.is_active:
            # Don't reveal if user exists
            logger.warning("Password reset requested for unknown email", email=email)
            return None

        # Generate reset token
        reset_token = self.jwt_service.create_password_reset_token(
            user_id=user.id,
            email=user.email,
        )

        logger.info("Password reset requested", user_id=str(user.id))

        # In a real system, you would send this token via email
        # For now, we'll return it
        return reset_token

    async def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using reset token.

        Args:
            token: Password reset token
            new_password: New password

        Returns:
            True if successful

        Raises:
            ValueError: If token invalid or password weak

        """
        # Verify reset token
        payload = self.jwt_service.verify_password_reset_token(token)
        if not payload:
            raise ValueError("Invalid or expired reset token")

        # Get user
        user_id = UUID(payload["sub"])
        user = await self.user_repository.get(user_id)
        if not user or user.email != payload["email"]:
            raise ValueError("Invalid reset token")

        # Validate new password
        is_valid, errors = self.password_service.validate_password_strength(
            new_password
        )
        if not is_valid:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        # Update password
        user.hashed_password = self.password_service.hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        await self.user_repository.update(user)

        logger.info("Password reset completed", user_id=str(user_id))

        return True

    async def list_organization_users(self, organization_id: UUID) -> list[User]:
        """List all users in an organization.

        Args:
            organization_id: Organization ID

        Returns:
            List of users

        """
        return await self.user_repository.list_by_organization(organization_id)

    async def update_user_role(
        self, user_id: UUID, role: UserRole, admin_user_id: UUID
    ) -> User:
        """Update user's role (admin only).

        Args:
            user_id: User to update
            role: New role
            admin_user_id: Admin performing the update

        Returns:
            Updated user

        Raises:
            ValueError: If validation fails

        """
        # Get admin user
        admin_user = await self.user_repository.get(admin_user_id)
        if not admin_user or not admin_user.is_admin:
            raise ValueError("Only admins can change user roles")

        # Get target user
        user = await self.user_repository.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Ensure same organization
        if user.organization_id != admin_user.organization_id:
            raise ValueError("Cannot modify users from other organizations")

        # Prevent removing last admin
        if user.role == UserRole.ADMIN and role != UserRole.ADMIN:
            admin_count = len(
                [
                    u
                    for u in await self.user_repository.list_by_organization(
                        user.organization_id
                    )
                    if u.role == UserRole.ADMIN and u.id != user_id
                ]
            )
            if admin_count == 0:
                raise ValueError("Cannot remove last admin from organization")

        # Update role
        user.role = role
        user.updated_at = datetime.now(timezone.utc)
        updated_user = await self.user_repository.update(user)

        logger.info(
            "User role updated",
            user_id=str(user_id),
            new_role=role.value,
            admin_id=str(admin_user_id),
        )

        return updated_user

    async def deactivate_user(self, user_id: UUID, admin_user_id: UUID) -> bool:
        """Deactivate a user (admin only).

        Args:
            user_id: User to deactivate
            admin_user_id: Admin performing the deactivation

        Returns:
            True if successful

        Raises:
            ValueError: If validation fails

        """
        # Get admin user
        admin_user = await self.user_repository.get(admin_user_id)
        if not admin_user or not admin_user.is_admin:
            raise ValueError("Only admins can deactivate users")

        # Get target user
        user = await self.user_repository.get(user_id)
        if not user:
            raise ValueError("User not found")

        # Ensure same organization
        if user.organization_id != admin_user.organization_id:
            raise ValueError("Cannot modify users from other organizations")

        # Prevent deactivating self
        if user_id == admin_user_id:
            raise ValueError("Cannot deactivate yourself")

        # Deactivate user
        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        await self.user_repository.update(user)

        logger.info(
            "User deactivated",
            user_id=str(user_id),
            admin_id=str(admin_user_id),
        )

        return True
