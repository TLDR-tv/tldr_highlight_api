"""Tests for user management router."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.models.api_key import APIKey
from src.models.organization import Organization
from src.api.dependencies.auth import hash_password


@pytest.fixture
def mock_current_user():
    """Create a mock current user."""
    user = User(
        id=1,
        email="test@example.com",
        company_name="Test Company",
        password_hash=hash_password("current_password"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    user = User(
        id=2,
        email="admin@example.com",
        company_name="Admin Company",
        password_hash=hash_password("admin_password"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    return user


@pytest.mark.asyncio
class TestUserProfileEndpoints:
    """Test cases for user profile endpoints."""

    async def test_get_current_user_profile(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test getting current user's profile."""
        # Mock API keys and organizations queries
        mock_api_keys = [APIKey(id=1, user_id=1, active=True), APIKey(id=2, user_id=1, active=True)]
        mock_orgs = [Organization(id=1, owner_id=1)]
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock the results
                mock_api_result = AsyncMock()
                mock_api_result.scalars.return_value.all.return_value = mock_api_keys
                
                mock_org_result = AsyncMock()
                mock_org_result.scalars.return_value.all.return_value = mock_orgs
                
                mock_execute.side_effect = [mock_api_result, mock_org_result]
                
                response = await async_client.get("/api/v1/users/me")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["id"] == 1
                assert data["email"] == "test@example.com"
                assert data["company_name"] == "Test Company"
                assert data["api_keys_count"] == 2
                assert data["organizations_count"] == 1

    async def test_update_current_user_profile(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test updating current user's profile."""
        update_data = {
            "email": "newemail@example.com",
            "company_name": "Updated Company"
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock email check - no existing user
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_execute.return_value = mock_result
                
                db_session.commit = AsyncMock()
                db_session.refresh = AsyncMock()
                
                response = await async_client.put("/api/v1/users/me", json=update_data)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["email"] == "newemail@example.com"
                assert data["company_name"] == "Updated Company"

    async def test_update_current_user_email_already_exists(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test updating user with email that already exists."""
        update_data = {"email": "existing@example.com"}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock email check - existing user found
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = User(id=99, email="existing@example.com")
                mock_execute.return_value = mock_result
                
                response = await async_client.put("/api/v1/users/me", json=update_data)
                
                assert response.status_code == status.HTTP_400_BAD_REQUEST
                assert "Email address is already in use" in response.json()["detail"]

    async def test_change_password_success(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test successful password change."""
        password_data = {
            "current_password": "current_password",
            "new_password": "new_secure_password123"
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            db_session.commit = AsyncMock()
            
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Password changed successfully"
            assert "changed_at" in data

    async def test_change_password_wrong_current(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test password change with wrong current password."""
        password_data = {
            "current_password": "wrong_password",
            "new_password": "new_password123"
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Current password is incorrect" in response.json()["detail"]

    async def test_change_password_same_as_current(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test password change with same password."""
        password_data = {
            "current_password": "current_password",
            "new_password": "current_password"
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "New password must be different" in response.json()["detail"]


@pytest.mark.asyncio
class TestAdminUserEndpoints:
    """Test cases for admin user management endpoints."""

    async def test_get_user_by_id_admin(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test getting user by ID as admin."""
        target_user = User(
            id=10,
            email="target@example.com",
            company_name="Target Company",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = target_user
                mock_execute.return_value = mock_result
                
                response = await async_client.get("/api/v1/users/10")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["id"] == 10
                assert data["email"] == "target@example.com"

    async def test_get_user_by_id_not_found(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test getting non-existent user by ID."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_execute.return_value = mock_result
                
                response = await async_client.get("/api/v1/users/999")
                
                assert response.status_code == status.HTTP_404_NOT_FOUND
                assert "User not found" in response.json()["detail"]

    async def test_update_user_by_id_admin(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test updating user by ID as admin."""
        target_user = User(
            id=10,
            email="target@example.com",
            company_name="Target Company",
            password_hash="old_hash",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        update_data = {
            "email": "updated@example.com",
            "company_name": "Updated Target Company"
        }
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                # First call: get target user
                mock_user_result = AsyncMock()
                mock_user_result.scalar_one_or_none.return_value = target_user
                
                # Second call: check email availability
                mock_email_result = AsyncMock()
                mock_email_result.scalar_one_or_none.return_value = None
                
                mock_execute.side_effect = [mock_user_result, mock_email_result]
                
                db_session.commit = AsyncMock()
                db_session.refresh = AsyncMock()
                
                response = await async_client.put("/api/v1/users/10", json=update_data)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["email"] == "updated@example.com"
                assert data["company_name"] == "Updated Target Company"

    async def test_list_users_admin(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test listing users as admin."""
        users = [
            User(id=i, email=f"user{i}@example.com", company_name=f"Company {i}")
            for i in range(1, 4)
        ]
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                # First call: count query
                mock_count_result = AsyncMock()
                mock_count_result.scalar.return_value = 3
                
                # Second call: users query
                mock_users_result = AsyncMock()
                mock_users_result.scalars.return_value.all.return_value = users
                
                mock_execute.side_effect = [mock_count_result, mock_users_result]
                
                response = await async_client.get("/api/v1/users?page=1&per_page=10")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 3
                assert data["page"] == 1
                assert data["per_page"] == 10
                assert len(data["users"]) == 3
                assert data["users"][0]["email"] == "user1@example.com"

    async def test_list_users_pagination(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test users list pagination."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock count and empty page
                mock_count_result = AsyncMock()
                mock_count_result.scalar.return_value = 50
                
                mock_users_result = AsyncMock()
                mock_users_result.scalars.return_value.all.return_value = []
                
                mock_execute.side_effect = [mock_count_result, mock_users_result]
                
                response = await async_client.get("/api/v1/users?page=10&per_page=10")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 50
                assert data["page"] == 10
                assert len(data["users"]) == 0

    async def test_delete_user_admin(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test deleting user as admin."""
        target_user = User(id=10, email="delete@example.com")
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = target_user
                mock_execute.return_value = mock_result
                
                db_session.delete = AsyncMock()
                db_session.commit = AsyncMock()
                
                response = await async_client.delete("/api/v1/users/10")
                
                assert response.status_code == status.HTTP_200_OK
                assert response.json()["message"] == "User deleted successfully"
                db_session.delete.assert_called_once_with(target_user)

    async def test_delete_user_self_prevention(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test that admin cannot delete themselves."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            response = await async_client.delete("/api/v1/users/2")  # Admin's own ID
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Cannot delete your own account" in response.json()["detail"]

    async def test_delete_user_not_found(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test deleting non-existent user."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_execute.return_value = mock_result
                
                response = await async_client.delete("/api/v1/users/999")
                
                assert response.status_code == status.HTTP_404_NOT_FOUND
                assert "User not found" in response.json()["detail"]


@pytest.mark.asyncio
class TestUserEndpointValidation:
    """Test cases for user endpoint validation."""

    async def test_update_profile_invalid_data(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test updating profile with invalid data."""
        invalid_data = {
            "email": "not-an-email",  # Invalid email format
            "company_name": ""  # Empty company name
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.put("/api/v1/users/me", json=invalid_data)
            
            # The actual validation would depend on schema validation
            # This test assumes pydantic validation is in place
            assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]

    async def test_list_users_invalid_pagination(
        self, async_client: AsyncClient, mock_admin_user
    ):
        """Test listing users with invalid pagination parameters."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            # Test negative page
            response = await async_client.get("/api/v1/users?page=-1")
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            
            # Test per_page over limit
            response = await async_client.get("/api/v1/users?per_page=101")
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_update_user_partial_update(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test partial profile updates."""
        # Update only email
        update_data = {"email": "newemail@example.com"}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_execute.return_value = mock_result
                
                db_session.commit = AsyncMock()
                db_session.refresh = AsyncMock()
                
                response = await async_client.put("/api/v1/users/me", json=update_data)
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["email"] == "newemail@example.com"
                # Company name should remain unchanged
                assert data["company_name"] == "Test Company"

    async def test_update_user_empty_update(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test updating user with no changes."""
        update_data = {}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.put("/api/v1/users/me", json=update_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Should return current user data unchanged
            assert data["email"] == "test@example.com"
            assert data["company_name"] == "Test Company"

    async def test_change_password_validation(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test password change with various validation scenarios."""
        
        # Test missing current password
        password_data = {"new_password": "new_password123"}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test missing new password
        password_data = {"current_password": "current_password"}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_change_password_weak_password(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test password change with weak new password."""
        password_data = {
            "current_password": "current_password",
            "new_password": "123"  # Too short
        }
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            response = await async_client.post("/api/v1/users/me/change-password", json=password_data)
            # Depending on validation, this might be 422 or 400
            assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]

    async def test_get_user_profile_with_counts(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test user profile includes accurate counts."""
        # Create various API keys with different statuses
        mock_api_keys = [
            APIKey(id=1, user_id=1, active=True, name="Active Key 1"),
            APIKey(id=2, user_id=1, active=True, name="Active Key 2"),
            APIKey(id=3, user_id=1, active=False, name="Inactive Key"),
        ]
        mock_orgs = [
            Organization(id=1, owner_id=1, name="Org 1"),
            Organization(id=2, owner_id=1, name="Org 2"),
        ]
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_api_result = AsyncMock()
                mock_api_result.scalars.return_value.all.return_value = mock_api_keys
                
                mock_org_result = AsyncMock()
                mock_org_result.scalars.return_value.all.return_value = mock_orgs
                
                mock_execute.side_effect = [mock_api_result, mock_org_result]
                
                response = await async_client.get("/api/v1/users/me")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["api_keys_count"] == 3  # Total including inactive
                assert data["organizations_count"] == 2

    async def test_admin_update_user_validation(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test admin user update with validation."""
        target_user = User(
            id=10,
            email="target@example.com",
            company_name="Target Company",
            password_hash="old_hash",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Test invalid email format
        invalid_update = {"email": "not-an-email"}
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            response = await async_client.put("/api/v1/users/10", json=invalid_update)
            assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST]

    async def test_admin_delete_user_cascading(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test that deleting user handles cascading properly."""
        target_user = User(id=10, email="delete@example.com")
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = target_user
                mock_execute.return_value = mock_result
                
                db_session.delete = AsyncMock()
                db_session.commit = AsyncMock()
                
                response = await async_client.delete("/api/v1/users/10")
                
                assert response.status_code == status.HTTP_200_OK
                
                # Verify delete was called with correct user
                db_session.delete.assert_called_once_with(target_user)
                db_session.commit.assert_called_once()

    async def test_list_users_search_filtering(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test user listing with search/filtering capabilities."""
        users = [
            User(id=1, email="alice@example.com", company_name="Alpha Corp"),
            User(id=2, email="bob@example.com", company_name="Beta Inc"),
            User(id=3, email="charlie@alpha.com", company_name="Gamma Ltd"),
        ]
        
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock search results
                mock_count_result = AsyncMock()
                mock_count_result.scalar.return_value = 2
                
                mock_users_result = AsyncMock()
                mock_users_result.scalars.return_value.all.return_value = [users[0], users[2]]
                
                mock_execute.side_effect = [mock_count_result, mock_users_result]
                
                # Test search by email domain
                response = await async_client.get("/api/v1/users?search=alpha")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 2
                assert len(data["users"]) == 2

    async def test_user_profile_security(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test that sensitive fields are not exposed in user profile."""
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch("sqlalchemy.ext.asyncio.AsyncSession.execute") as mock_execute:
                # Mock empty API keys and orgs
                mock_result = AsyncMock()
                mock_result.scalars.return_value.all.return_value = []
                mock_execute.return_value = mock_result
                
                response = await async_client.get("/api/v1/users/me")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                
                # Ensure sensitive fields are not exposed
                assert "password_hash" not in data
                assert "password" not in data
                
                # Ensure expected fields are present
                assert "id" in data
                assert "email" in data
                assert "company_name" in data
                assert "created_at" in data

    async def test_concurrent_user_updates(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_current_user
    ):
        """Test handling of concurrent user updates."""
        update_data = {"company_name": "Updated Company"}
        
        with patch("src.api.routers.users.get_current_user", return_value=mock_current_user):
            with patch.object(db_session, "execute") as mock_execute:
                mock_result = AsyncMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_execute.return_value = mock_result
                
                # Simulate database commit error (optimistic locking)
                db_session.commit = AsyncMock(side_effect=Exception("Concurrent modification"))
                db_session.refresh = AsyncMock()
                
                response = await async_client.put("/api/v1/users/me", json=update_data)
                
                # Should handle the error gracefully
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    async def test_admin_list_users_large_dataset(
        self, async_client: AsyncClient, db_session: AsyncSession, mock_admin_user
    ):
        """Test listing users with large dataset pagination."""
        with patch("src.api.routers.users.require_admin", return_value=lambda: mock_admin_user):
            with patch.object(db_session, "execute") as mock_execute:
                # Mock large dataset
                mock_count_result = AsyncMock()
                mock_count_result.scalar.return_value = 10000
                
                mock_users_result = AsyncMock()
                mock_users_result.scalars.return_value.all.return_value = []
                
                mock_execute.side_effect = [mock_count_result, mock_users_result]
                
                # Test last page
                response = await async_client.get("/api/v1/users?page=1000&per_page=10")
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["total"] == 10000
                assert data["page"] == 1000
                assert len(data["users"]) == 0

    async def test_user_permissions_enforcement(
        self, async_client: AsyncClient, mock_current_user
    ):
        """Test that non-admin users cannot access admin endpoints."""
        # Regular user trying to access admin endpoints should be blocked
        # by the require_admin dependency, but test the expectation
        
        # These would normally be blocked by FastAPI dependencies
        # but we can test the expected behavior
        user_id = 999
        
        response = await async_client.get(f"/api/v1/users/{user_id}")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        response = await async_client.put(f"/api/v1/users/{user_id}", json={"email": "new@test.com"})
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        response = await async_client.delete(f"/api/v1/users/{user_id}")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        response = await async_client.get("/api/v1/users")
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]