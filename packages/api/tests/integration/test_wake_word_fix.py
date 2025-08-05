"""Test wake word functionality after fixing the model field name."""

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.user import UserRole
from shared.infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
)
from tests.factories import create_test_organization, create_test_user


class TestWakeWordFix:
    """Test wake word management after fixing field name issue."""

    @pytest.mark.asyncio
    async def test_add_wake_word_works(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test adding wake word works after fix."""
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

        # Login
        login_response = await client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": password},
        )
        token = login_response.json()["access_token"]

        # Add wake word - should work now
        response = await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "test_wake_word"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "test_wake_word" in data["wake_words"]

    @pytest.mark.asyncio
    async def test_remove_wake_word_works(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test removing wake word works after fix."""
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

        # Remove wake word - should work now
        response = await client.delete(
            "/api/v1/organizations/current/wake-words/word1",
            headers=auth_headers(token),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "word1" not in data["wake_words"]
        assert "word2" in data["wake_words"]

    @pytest.mark.asyncio
    async def test_wake_words_persist_after_update(
        self,
        client: AsyncClient,
        test_session: AsyncSession,
        auth_headers,
    ):
        """Test wake words persist after organization update."""
        # Create organization
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

        # Add wake words
        await client.post(
            "/api/v1/organizations/current/wake-words",
            headers=auth_headers(token),
            json={"wake_word": "persistent_word"},
        )

        # Update organization
        response = await client.put(
            "/api/v1/organizations/current",
            headers=auth_headers(token),
            json={"name": "Updated Org Name"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "persistent_word" in data["wake_words"]