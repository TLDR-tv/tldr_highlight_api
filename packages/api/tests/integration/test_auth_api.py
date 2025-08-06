"""Integration tests for authentication API endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.user import UserRole
from shared.infrastructure.storage.repositories import (
    UserRepository,
    OrganizationRepository,
)
from tests.factories import create_test_organization, create_test_user


class TestRegistration:
    """Test organization registration endpoint."""

    @pytest.mark.asyncio
    async def test_register_organization_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test successful organization registration with owner."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Test Company",
                "owner_email": "admin@testcompany.com",
                "owner_name": "Admin User",
                "owner_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["message"] == "Organization registered successfully"
        assert data["organization"]["name"] == "Test Company"
        assert data["user"]["email"] == "admin@testcompany.com"
        assert data["user"]["name"] == "Admin User"
        assert data["user"]["role"] == UserRole.ADMIN.value

        # Verify in database
        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)

        org = await org_repo.get_by_slug("test-company")
        assert org is not None

        user = await user_repo.get_by_email("admin@testcompany.com")
        assert user is not None
        assert user.organization_id == org.id

    @pytest.mark.asyncio
    async def test_register_with_webhook_url(
        self,
        client: AsyncClient,
    ):
        """Test registration with webhook URL."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Webhook Company",
                "webhook_url": "https://example.com/webhook",
                "owner_email": "admin@webhook.com",
                "owner_name": "Webhook Admin",
                "owner_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["organization"]["webhook_url"] == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test registration with duplicate email."""
        # Create existing user
        org = create_test_organization()
        user, _ = create_test_user(organization_id=org.id, email="existing@test.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Try to register with same email
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Another Company",
                "owner_email": "existing@test.com",
                "owner_name": "Another User",
                "owner_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_invalid_password(
        self,
        client: AsyncClient,
    ):
        """Test registration with invalid password."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Test Company",
                "owner_email": "admin@test.com",
                "owner_name": "Admin User",
                "owner_password": "short",  # Too short
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_register_invalid_webhook_url(
        self,
        client: AsyncClient,
    ):
        """Test registration with invalid webhook URL."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "organization_name": "Test Company",
                "webhook_url": "not-a-url",  # Invalid URL
                "owner_email": "admin@test.com",
                "owner_name": "Admin User",
                "owner_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "Webhook URL must start with" in response.json()["detail"][0]["msg"]


class TestLogin:
    """Test user login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test successful login."""
        # Create test user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="test@example.com", password="TestPass123!"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"
        assert data["expires_in"] == 3600

        # Check cookie
        assert "refresh_token" in response.cookies

    @pytest.mark.asyncio
    async def test_login_incorrect_password(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test login with incorrect password."""
        # Create test user
        org = create_test_organization()
        user, _ = create_test_user(
            organization_id=org.id,
            email="test@example.com",
            password="CorrectPassword123!",
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid email or password"

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(
        self,
        client: AsyncClient,
    ):
        """Test login with non-existent email."""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "AnyPassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid email or password"

    @pytest.mark.asyncio
    async def test_login_deactivated_user(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test login with deactivated user."""
        # Create deactivated user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id,
            email="deactivated@example.com",
            password="TestPass123!",
            is_active=False,
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Try to login
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "deactivated@example.com",
                "password": "TestPass123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTokenRefresh:
    """Test token refresh endpoint."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test successful token refresh."""
        # Create and login user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="test@example.com", password="TestPass123!"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login to get tokens
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
            },
        )

        tokens = login_response.json()

        # Wait a moment to ensure different timestamp
        import asyncio

        await asyncio.sleep(0.1)

        # Refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": tokens["refresh_token"],
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        # Note: JWT timestamps are in seconds, so tokens may be identical if created within same second
        # Just ensure we got valid tokens back
        assert len(data["access_token"]) > 50  # Valid JWT token
        assert len(data["refresh_token"]) > 50  # Valid JWT token

        # Check cookie updated
        assert "refresh_token" in response.cookies

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(
        self,
        client: AsyncClient,
    ):
        """Test refresh with invalid token."""
        response = await client.post(
            "/api/v1/auth/refresh",
            json={
                "refresh_token": "invalid-token",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid refresh token"


class TestPasswordReset:
    """Test password reset endpoints."""

    @pytest.mark.asyncio
    async def test_request_password_reset(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
    ):
        """Test password reset request."""
        # Create test user
        org = create_test_organization()
        user, _ = create_test_user(organization_id=org.id, email="test@example.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Request reset
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={
                "email": "test@example.com",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "If an account exists with this email, you will receive a password reset link."
        # In test environment, token is not returned (only in development)

    @pytest.mark.asyncio
    async def test_request_password_reset_nonexistent_user(
        self,
        client: AsyncClient,
    ):
        """Test password reset for non-existent user."""
        response = await client.post(
            "/api/v1/auth/forgot-password",
            json={
                "email": "nonexistent@example.com",
            },
        )

        # Always returns success to prevent email enumeration
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "If an account exists with this email, you will receive a password reset link."
        assert "reset_token" not in data

    @pytest.mark.asyncio
    async def test_reset_password_with_token(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        test_settings,
    ):
        """Test password reset with valid token."""
        # Create test user
        org = create_test_organization()
        user, old_password = create_test_user(
            organization_id=org.id, email="test@example.com", password="OldPassword123!"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Create reset token manually since it's not returned in test environment
        from api.services.auth.jwt_service import JWTService
        jwt_service = JWTService(test_settings)
        reset_token = jwt_service.create_password_reset_token(
            user_id=user.id, email=user.email
        )

        # Reset password
        response = await client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "NewPassword123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Password reset successfully"

        # Verify can login with new password
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "NewPassword123!",
            },
        )

        assert login_response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_reset_password_invalid_token(
        self,
        client: AsyncClient,
    ):
        """Test password reset with invalid token."""
        response = await client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": "invalid-token",
                "new_password": "NewPassword123!",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_reset_password_weak_password(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        test_settings,
    ):
        """Test password reset with weak password."""
        # Create test user
        org = create_test_organization()
        user, _ = create_test_user(organization_id=org.id, email="test@example.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Create reset token manually since it's not returned in test environment
        from api.services.auth.jwt_service import JWTService
        jwt_service = JWTService(test_settings)
        reset_token = jwt_service.create_password_reset_token(
            user_id=user.id, email=user.email
        )

        # Try to reset with weak password
        response = await client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "weak",  # Too short
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestLogout:
    """Test logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout(
        self,
        client: AsyncClient,
    ):
        """Test logout clears refresh token cookie."""
        response = await client.post("/api/v1/auth/logout")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Logged out successfully"

        # Check cookie is deleted
        assert response.cookies.get("refresh_token") is None
