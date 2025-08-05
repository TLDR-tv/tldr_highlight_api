"""Integration tests for organization management API endpoints."""

import pytest
from uuid import uuid4
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.user import UserRole
from shared.domain.models.api_key import APIScopes
from shared.infrastructure.storage.repositories import (
    UserRepository,
    OrganizationRepository,
    APIKeyRepository,
)
from tests.factories import (
    create_test_organization,
    create_test_user,
    create_test_api_key,
)


class TestOrganizationProfile:
    """Test organization profile endpoints."""

    @pytest.mark.asyncio
    async def test_get_organization_user_auth(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test getting organization details with user auth."""
        # Create organization and user
        org = create_test_organization(name="Test Corp")
        user, password = create_test_user(
            organization_id=org.id, email="user@testcorp.com"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "user@testcorp.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Get organization
        response = await client.get(
            "/api/v1/organizations/current",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Test Corp"
        assert data["id"] == str(org.id)

    @pytest.mark.asyncio
    async def test_get_organization_api_key_auth(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test getting organization details with API key auth."""
        # Create organization and API key
        org = create_test_organization(name="API Corp")
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.ORG_READ}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Get organization with API key
        response = await client.get(
            "/api/v1/organizations/me",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "API Corp"
        assert data["id"] == str(org.id)

    @pytest.mark.asyncio
    async def test_update_organization(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test updating organization details."""
        # Create organization and admin
        org = create_test_organization(
            name="Old Name", webhook_url="https://old.webhook.com"
        )
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Update organization
        response = await client.put(
            "/api/v1/organizations/current",
            headers=auth_headers(token),
            json={
                "name": "New Name",
                "webhook_url": "https://new.webhook.com",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "New Name"
        assert data["webhook_url"] == "https://new.webhook.com"

    @pytest.mark.asyncio
    async def test_update_organization_non_admin_forbidden(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test non-admin cannot update organization."""
        # Create organization and member
        org = create_test_organization()
        member, password = create_test_user(
            organization_id=org.id, email="member@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(member)
        await test_session.commit()

        # Login as member
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "member@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Try to update organization
        response = await client.put(
            "/api/v1/organizations/current",
            headers=auth_headers(token),
            json={"name": "New Name"},
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestWebhookManagement:
    """Test webhook configuration endpoints."""

    @pytest.mark.asyncio
    async def test_configure_webhook(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test configuring webhook URL."""
        # Create organization and admin
        org = create_test_organization(webhook_url=None)
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Configure webhook
        response = await client.put(
            "/api/v1/organizations/current/webhook",
            headers=auth_headers(token),
            params={"webhook_url": "https://example.com/webhook"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["webhook_url"] == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_regenerate_webhook_secret(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test regenerating webhook secret."""
        # Create organization with webhook
        org = create_test_organization(
            webhook_url="https://example.com/webhook", webhook_secret="old-secret"
        )
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Regenerate secret
        response = await client.post(
            "/api/v1/organizations/current/webhook/secret",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "webhook_secret" in data
        assert data["webhook_secret"] != "old-secret"
        assert len(data["webhook_secret"]) > 32  # Should be a secure secret


class TestWakeWordManagement:
    """Test wake word management endpoints."""

    @pytest.mark.asyncio
    async def test_add_wake_word(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test adding custom wake word."""
        # Create organization and admin
        org = create_test_organization(wake_words=set())
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Add wake word
        response = await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "custom_trigger"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "custom_trigger" in data["wake_words"]

    @pytest.mark.asyncio
    async def test_add_duplicate_wake_word(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test adding duplicate wake word."""
        # Create organization without wake words first
        org = create_test_organization(wake_words=set())
        admin, password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Add a wake word first
        await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "existing_word"},
        )

        # Try to add duplicate
        response = await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "existing_word"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_remove_wake_word(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test removing wake word."""
        # Create organization without wake words
        org = create_test_organization(wake_words=set())
        admin, password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Add wake words first
        await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "word1"},
        )
        await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "word2"},
        )

        # Remove wake word
        response = await client.delete(
            "/api/v1/organizations/current/wake-words/word1",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "word1" not in data["wake_words"]
        assert "word2" in data["wake_words"]


class TestUsageAndAPIKeys:
    """Test usage statistics and API key management."""

    @pytest.mark.asyncio
    async def test_get_usage_statistics(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test getting organization usage statistics."""
        # Create organization and user
        org = create_test_organization()
        user, password = create_test_user(
            organization_id=org.id, email="user@example.com"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Get usage stats
        response = await client.get(
            "/api/v1/organizations/current/usage",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check expected fields
        assert "total_streams_processed" in data
        assert "total_highlights_generated" in data
        assert "total_processing_seconds" in data
        assert "total_processing_hours" in data
        assert "avg_highlights_per_stream" in data
        assert "avg_processing_seconds_per_stream" in data

    @pytest.mark.asyncio
    async def test_list_api_keys(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test listing organization API keys."""
        # Create organization with API keys
        org = create_test_organization()
        admin, password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )

        # Create multiple API keys
        api_key1, raw_key1 = create_test_api_key(
            organization_id=org.id, name="Production Key"
        )
        api_key2, raw_key2 = create_test_api_key(
            organization_id=org.id, name="Development Key"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)

        await org_repo.create(org)
        await user_repo.create(admin)
        await api_key_repo.create(api_key1)
        await api_key_repo.create(api_key2)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # List API keys
        response = await client.get(
            "/api/v1/organizations/current/api-keys",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total"] == 2
        assert len(data["api_keys"]) == 2
        names = [key["name"] for key in data["api_keys"]]
        assert "Production Key" in names
        assert "Development Key" in names

        # Check API key structure
        for key in data["api_keys"]:
            assert "id" in key
            assert "name" in key
            assert "prefix" in key
            assert "scopes" in key
            assert "created_at" in key
            assert "is_active" in key
            assert "key" not in key  # Full key should not be exposed

    @pytest.mark.asyncio
    async def test_list_users_in_organization(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test listing users in organization via organization endpoint."""
        org = create_test_organization()

        # Create multiple users
        user1, password = create_test_user(
            organization_id=org.id, email="user1@example.com"
        )
        user2, _ = create_test_user(organization_id=org.id, email="user2@example.com")
        user3, _ = create_test_user(organization_id=org.id, email="user3@example.com")

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(user1)
        await user_repo.create(user2)
        await user_repo.create(user3)
        await test_session.commit()

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "user1@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # List users
        response = await client.get(
            "/api/v1/organizations/current/users",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total"] == 3
        emails = [u["email"] for u in data["users"]]
        assert "user1@example.com" in emails
        assert "user2@example.com" in emails
        assert "user3@example.com" in emails

    @pytest.mark.asyncio
    async def test_create_api_key(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test creating a new API key."""
        # Create organization and admin
        org = create_test_organization()
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Create API key
        response = await client.post(
            "/api/v1/organizations/current/api-keys",
            headers=auth_headers(token),
            json={
                "name": "Production API Key",
                "scopes": ["streams:read", "streams:write", "highlights:read"],
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check response structure
        assert "api_key" in data
        assert "raw_key" in data
        assert "message" in data

        # Check API key details
        api_key = data["api_key"]
        assert api_key["name"] == "Production API Key"
        assert api_key["is_active"] is True
        assert set(api_key["scopes"]) == {"streams:read", "streams:write", "highlights:read"}
        assert api_key["prefix"].startswith("tldr_")

        # Check raw key format
        raw_key = data["raw_key"]
        assert raw_key.startswith("tldr_")
        assert len(raw_key) > 30  # Should be a secure length

    @pytest.mark.asyncio
    async def test_create_api_key_non_admin_forbidden(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test non-admin cannot create API keys."""
        # Create organization and member
        org = create_test_organization()
        member, password = create_test_user(
            organization_id=org.id, email="member@example.com", role=UserRole.MEMBER
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(member)
        await test_session.commit()

        # Login as member
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "member@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Try to create API key
        response = await client.post(
            "/api/v1/organizations/current/api-keys",
            headers=auth_headers(token),
            json={
                "name": "Test Key",
                "scopes": ["streams:read"],
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_delete_api_key(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test deleting an API key."""
        # Create organization with API key
        org = create_test_organization()
        admin, password = create_test_user(
            organization_id=org.id, email="admin@example.com", role=UserRole.ADMIN
        )
        api_key, _ = create_test_api_key(
            organization_id=org.id, name="Test Key to Delete"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await user_repo.create(admin)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Login as admin
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Delete API key
        response = await client.delete(
            f"/api/v1/organizations/current/api-keys/{api_key.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["detail"] == "API key revoked successfully"

        # Verify key is revoked
        revoked_key = await api_key_repo.get(api_key.id)
        assert revoked_key.is_active is False

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test deleting non-existent API key."""
        # Create organization and admin
        org = create_test_organization()
        admin, password = create_test_user(
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
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Try to delete non-existent key
        response = await client.delete(
            f"/api/v1/organizations/current/api-keys/{uuid4()}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_api_key_wrong_organization(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test cannot delete API key from different organization."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")
        
        admin1, password1 = create_test_user(
            organization_id=org1.id, email="admin1@example.com", role=UserRole.ADMIN
        )
        
        # Create API key for org2
        api_key_org2, _ = create_test_api_key(
            organization_id=org2.id, name="Org2 Key"
        )

        org_repo = OrganizationRepository(test_session)
        user_repo = UserRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        
        await org_repo.create(org1)
        await org_repo.create(org2)
        await user_repo.create(admin1)
        await api_key_repo.create(api_key_org2)
        await test_session.commit()

        # Login as admin of org1
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin1@example.com", "password": password1},
        )
        token = login_response.json()["access_token"]

        # Try to delete org2's API key
        response = await client.delete(
            f"/api/v1/organizations/current/api-keys/{api_key_org2.id}",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"]


class TestAPIKeyScopes:
    """Test API key scope requirements."""

    @pytest.mark.asyncio
    async def test_api_key_insufficient_scope(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test API key with insufficient scope."""
        # Create organization and API key with limited scope
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ},  # No ORG_READ scope
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Try to get organization (requires ORG_READ)
        response = await client.get(
            "/api/v1/organizations/me",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "scope" in response.json()["detail"].lower()
