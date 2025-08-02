"""Integration tests for user management API endpoints."""

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


class TestCurrentUserOperations:
    """Test operations on current user profile."""

    @pytest.mark.asyncio
    async def test_get_current_user_profile(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test getting current user profile."""
        # Create and login user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id,
            email="test@example.com",
            name="Test User",
            role=UserRole.MEMBER,
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Get profile
        response = await client.get(
            "/api/v1/users/me",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"
        assert data["role"] == "member"
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_update_current_user_profile(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test updating current user profile."""
        # Create and login user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="old@example.com", name="Old Name"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "old@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Update profile
        response = await client.put(
            "/api/v1/users/me",
            headers=auth_headers(token),
            json={
                "name": "New Name",
                "email": "new@example.com",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["email"] == "new@example.com"
        assert data["name"] == "New Name"

    @pytest.mark.asyncio
    async def test_update_profile_duplicate_email(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test updating profile with duplicate email."""
        org = create_test_organization()

        # Create two users
        user1, password1 = create_test_user(
            organization_id=org.id, email="user1@example.com"
        )
        user2, _ = create_test_user(organization_id=org.id, email="user2@example.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user1)
        await user_repo.create(user2)
        await test_session.commit()

        # Login as user1
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "user1@example.com", "password": password1},
        )
        token = login_response.json()["access_token"]

        # Try to update to user2's email
        response = await client.put(
            "/api/v1/users/me",
            headers=auth_headers(token),
            json={
                "email": "user2@example.com",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already in use" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_change_password(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test changing current user password."""
        # Create and login user
        org = create_test_organization()
        user, old_password = create_test_user(
            organization_id=org.id, email="test@example.com", password="OldPassword123!"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "OldPassword123!"},
        )
        token = login_response.json()["access_token"]

        # Change password
        response = await client.put(
            "/api/v1/users/me/password",
            headers=auth_headers(token),
            json={
                "current_password": "OldPassword123!",
                "new_password": "NewPassword123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "Password changed successfully"

        # Verify can login with new password
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "NewPassword123!"},
        )
        assert login_response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_change_password_incorrect_current(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test changing password with incorrect current password."""
        # Create and login user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="test@example.com"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Try to change password with wrong current password
        response = await client.put(
            "/api/v1/users/me/password",
            headers=auth_headers(token),
            json={
                "current_password": "WrongPassword123!",
                "new_password": "NewPassword123!",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "incorrect" in response.json()["detail"].lower()


class TestAdminUserOperations:
    """Test admin-only user management operations."""

    @pytest.mark.asyncio
    async def test_list_organization_users(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test listing all organization users."""
        org = create_test_organization()

        # Create admin and regular users
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )
        user1, _ = create_test_user(
            organization_id=org.id, email="user1@example.com", role=UserRole.MEMBER
        )
        user2, _ = create_test_user(
            organization_id=org.id, email="user2@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await user_repo.create(user1)
        await user_repo.create(user2)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # List users
        response = await client.get(
            "/api/v1/users",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total"] == 3
        emails = [u["email"] for u in data["users"]]
        assert "admin@example.com" in emails
        assert "user1@example.com" in emails
        assert "user2@example.com" in emails

    @pytest.mark.asyncio
    async def test_list_users_non_admin_forbidden(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test non-admin cannot list users."""
        # Create regular user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="member@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login as member
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "member@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Try to list users
        response = await client.get(
            "/api/v1/users",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_create_user(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test creating a new user."""
        # Create admin
        org = create_test_organization()
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Create new user
        response = await client.post(
            "/api/v1/users",
            headers=auth_headers(token),
            json={
                "email": "newuser@example.com",
                "name": "New User",
                "password": "NewUserPass123!",
                "role": "member",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["email"] == "newuser@example.com"
        assert data["name"] == "New User"
        assert data["role"] == "member"

        # Verify user can login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "newuser@example.com", "password": "NewUserPass123!"},
        )
        assert login_response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_get_user_details(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test getting specific user details."""
        org = create_test_organization()

        # Create admin and target user
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )
        target_user, _ = create_test_user(
            organization_id=org.id, email="target@example.com", name="Target User"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await user_repo.create(target_user)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Get user details
        response = await client.get(
            f"/api/v1/users/{target_user.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "target@example.com"
        assert data["name"] == "Target User"

    @pytest.mark.asyncio
    async def test_update_user_role(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test updating user role."""
        org = create_test_organization()

        # Create admin and target user
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )
        target_user, _ = create_test_user(
            organization_id=org.id, email="target@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await user_repo.create(target_user)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Update role
        response = await client.put(
            f"/api/v1/users/{target_user.id}/role",
            headers=auth_headers(token),
            json={"role": "admin"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_admin_cannot_change_own_role(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test admin cannot change their own role."""
        # Create admin
        org = create_test_organization()
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Try to change own role
        response = await client.put(
            f"/api/v1/users/{admin.id}/role",
            headers=auth_headers(token),
            json={"role": "member"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot remove last admin" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_deactivate_user(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test deactivating a user."""
        org = create_test_organization()

        # Create admin and target user
        admin, admin_password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )
        target_user, target_password = create_test_user(
            organization_id=org.id, email="target@example.com"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await user_repo.create(target_user)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Deactivate user
        response = await client.delete(
            f"/api/v1/users/{target_user.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["message"] == "User deactivated successfully"

        # Verify user cannot login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "target@example.com", "password": target_password},
        )
        assert login_response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAuthorizationScenarios:
    """Test various authorization scenarios."""

    @pytest.mark.asyncio
    async def test_user_cannot_view_other_user(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test regular user cannot view other users."""
        org = create_test_organization()

        # Create two users
        user1, password1 = create_test_user(
            organization_id=org.id, email="user1@example.com", role=UserRole.MEMBER
        )
        user2, _ = create_test_user(
            organization_id=org.id, email="user2@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user1)
        await user_repo.create(user2)
        await test_session.commit()

        # Login as user1
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "user1@example.com", "password": password1},
        )
        token = login_response.json()["access_token"]

        # Try to view user2
        response = await client.get(
            f"/api/v1/users/{user2.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_cross_organization_access_denied(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test users cannot access other organization's users."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")

        # Create admin in org1 and user in org2
        admin, admin_password = create_test_user(
            organization_id=org1.id, email="admin@org1.com", role=UserRole.ADMIN
        )
        other_user, _ = create_test_user(organization_id=org2.id, email="user@org2.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org1)
        await org_repo.create(org2)
        await user_repo.create(admin)
        await user_repo.create(other_user)
        await test_session.commit()

        # Login as admin from org1
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@org1.com", "password": admin_password},
        )
        token = login_response.json()["access_token"]

        # Try to access user from org2
        response = await client.get(
            f"/api/v1/users/{other_user.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "User not found" in response.json()["detail"]  # Generic message

    @pytest.mark.asyncio
    async def test_unauthenticated_access_denied(
        self,
        client: AsyncClient,
    ):
        """Test unauthenticated access is denied."""
        # Try to access protected endpoints without token
        response = await client.get("/api/v1/users/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        response = await client.get("/api/v1/users")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        response = await client.post(
            "/api/v1/users",
            json={"email": "test@example.com", "password": "Test123!"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
