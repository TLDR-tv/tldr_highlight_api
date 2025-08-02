"""Integration tests for highlight retrieval API endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models.api_key import APIScopes
from src.domain.models.highlight import DimensionScore
from src.infrastructure.storage.repositories import (
    OrganizationRepository,
    APIKeyRepository,
    StreamRepository,
    HighlightRepository,
)
from tests.factories import (
    create_test_organization,
    create_test_api_key,
    create_test_stream,
    create_test_highlight,
)


class TestHighlightListing:
    """Test highlight listing endpoints."""

    @pytest.mark.asyncio
    async def test_list_highlights_with_api_key(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test listing highlights with API key authentication."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        # Create stream and highlights
        stream = create_test_stream(organization_id=org.id)
        highlight1 = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            title="Epic Play 1",
            overall_score=0.9,
        )
        highlight2 = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            title="Epic Play 2",
            overall_score=0.85,
            wake_word_triggered=True,
            wake_word_detected="clip that",
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await highlight_repo.create(highlight1)
        await highlight_repo.create(highlight2)
        await test_session.commit()

        # List highlights
        response = await async_client.get(
            "/api/v1/highlights/",
            headers=api_key_headers(raw_key),
            follow_redirects=False,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total"] >= 2
        assert len(data["highlights"]) == 2
        assert data["limit"] == 100
        assert data["offset"] == 0
        assert data["has_more"] is False

        # Check highlight data
        highlights = data["highlights"]
        assert any(h["title"] == "Epic Play 1" for h in highlights)
        assert any(h["title"] == "Epic Play 2" for h in highlights)

        # Check wake word highlight
        wake_word_highlight = next(h for h in highlights if h["wake_word_triggered"])
        assert wake_word_highlight["wake_word_detected"] == "clip that"

    @pytest.mark.asyncio
    async def test_list_highlights_with_filters(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test listing highlights with various filters."""
        # Create test data
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        stream1 = create_test_stream(organization_id=org.id)
        stream2 = create_test_stream(organization_id=org.id)

        # Create highlights with different properties
        highlight1 = create_test_highlight(
            stream_id=stream1.id,
            organization_id=org.id,
            overall_score=0.95,
            wake_word_triggered=True,
        )
        highlight2 = create_test_highlight(
            stream_id=stream1.id,
            organization_id=org.id,
            overall_score=0.6,
            wake_word_triggered=False,
        )
        highlight3 = create_test_highlight(
            stream_id=stream2.id,
            organization_id=org.id,
            overall_score=0.8,
            wake_word_triggered=True,
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream1)
        await stream_repo.create(stream2)
        await highlight_repo.create(highlight1)
        await highlight_repo.create(highlight2)
        await highlight_repo.create(highlight3)
        await test_session.commit()

        # Test filter by stream
        response = await async_client.get(
            f"/api/v1/highlights/?stream_id={stream1.id}",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 2
        assert all(h["stream_id"] == str(stream1.id) for h in data["highlights"])

        # Test filter by wake word
        response = await async_client.get(
            "/api/v1/highlights/?wake_word_triggered=true",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 2
        assert all(h["wake_word_triggered"] for h in data["highlights"])

        # Test filter by minimum score
        response = await async_client.get(
            "/api/v1/highlights/?min_score=0.8",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 2
        assert all(h["overall_score"] >= 0.8 for h in data["highlights"])

        # Test order by score
        response = await async_client.get(
            "/api/v1/highlights/?order_by=score",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        scores = [h["overall_score"] for h in data["highlights"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_list_highlights_pagination(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test highlight listing pagination."""
        # Create test data
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )
        stream = create_test_stream(organization_id=org.id)

        # Create multiple highlights
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)

        for i in range(5):
            highlight = create_test_highlight(
                stream_id=stream.id,
                organization_id=org.id,
                title=f"Highlight {i}",
            )
            await highlight_repo.create(highlight)

        await test_session.commit()

        # Test pagination
        response = await async_client.get(
            "/api/v1/highlights/?limit=2&offset=0",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert data["has_more"] is True

        # Get next page
        response = await async_client.get(
            "/api/v1/highlights/?limit=2&offset=2",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 2
        assert data["offset"] == 2

        # Get last page
        response = await async_client.get(
            "/api/v1/highlights/?limit=2&offset=4",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["highlights"]) == 1
        assert data["has_more"] is False


class TestHighlightRetrieval:
    """Test single highlight retrieval."""

    @pytest.mark.asyncio
    async def test_get_highlight_details(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test getting detailed highlight information."""
        # Create test data
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )
        stream = create_test_stream(organization_id=org.id)

        # Create highlight with custom dimension scores
        dimension_scores = [
            DimensionScore(name="action_intensity", score=0.9, confidence=0.95),
            DimensionScore(name="crowd_reaction", score=0.85, confidence=0.88),
            DimensionScore(name="game_impact", score=0.92, confidence=0.9),
        ]

        highlight = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            title="Game-Winning Goal",
            description="Amazing last-minute goal that won the championship",
            dimension_scores=dimension_scores,
            tags=["goal", "championship", "clutch"],
            transcript="And he scores! Unbelievable!",
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await highlight_repo.create(highlight)
        await test_session.commit()

        # Get highlight details
        response = await async_client.get(
            f"/api/v1/highlights/{highlight.id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check basic fields
        assert data["id"] == str(highlight.id)
        assert data["title"] == "Game-Winning Goal"
        assert (
            data["description"] == "Amazing last-minute goal that won the championship"
        )
        assert data["organization_id"] == str(org.id)
        assert data["stream_id"] == str(stream.id)

        # Check dimension scores
        assert len(data["dimension_scores"]) == 3
        scores_by_name = {ds["name"]: ds for ds in data["dimension_scores"]}
        assert scores_by_name["action_intensity"]["score"] == 0.9
        assert scores_by_name["action_intensity"]["confidence"] == 0.95

        # Check other fields
        assert data["tags"] == ["goal", "championship", "clutch"]
        assert data["transcript"] == "And he scores! Unbelievable!"
        assert data["clip_url"] is not None
        assert data["thumbnail_url"] is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_highlight(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test getting a highlight that doesn't exist."""
        # Create organization and API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Try to get non-existent highlight
        from uuid import uuid4

        fake_id = uuid4()

        response = await async_client.get(
            f"/api/v1/highlights/{fake_id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


class TestStreamHighlights:
    """Test stream-specific highlight endpoints."""

    @pytest.mark.asyncio
    async def test_get_stream_highlights(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test getting all highlights for a stream."""
        # Create test data
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )
        stream = create_test_stream(organization_id=org.id)

        # Create highlights at different times
        highlight1 = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            start_time=60.0,  # 1 minute
            end_time=90.0,
        )
        highlight2 = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            start_time=180.0,  # 3 minutes
            end_time=210.0,
        )
        highlight3 = create_test_highlight(
            stream_id=stream.id,
            organization_id=org.id,
            start_time=120.0,  # 2 minutes
            end_time=150.0,
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await highlight_repo.create(highlight1)
        await highlight_repo.create(highlight2)
        await highlight_repo.create(highlight3)
        await test_session.commit()

        # Get stream highlights
        response = await async_client.get(
            f"/api/v1/highlights/streams/{stream.id}/highlights",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["stream_id"] == str(stream.id)
        assert data["total"] == 3
        assert len(data["highlights"]) == 3

        # Check highlights are in chronological order
        start_times = [h["start_time"] for h in data["highlights"]]
        assert start_times == sorted(start_times)
        assert start_times == [60.0, 120.0, 180.0]


class TestHighlightAuthorization:
    """Test authorization and access control for highlights."""

    @pytest.mark.asyncio
    async def test_insufficient_scope(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test API key without HIGHLIGHTS_READ scope."""
        # Create API key without highlights scope
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id,
            scopes={APIScopes.STREAMS_READ},  # Wrong scope
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Try to list highlights
        response = await async_client.get(
            "/api/v1/highlights/",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "scope" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_cross_organization_access_denied(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test that organizations cannot access other org's highlights."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")

        # Create API key for org1
        api_key, raw_key = create_test_api_key(
            organization_id=org1.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        # Create stream and highlight for org2
        stream = create_test_stream(organization_id=org2.id)
        highlight = create_test_highlight(
            stream_id=stream.id,
            organization_id=org2.id,
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org1)
        await org_repo.create(org2)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream)
        await highlight_repo.create(highlight)
        await test_session.commit()

        # Try to access org2's highlight with org1's API key
        response = await async_client.get(
            f"/api/v1/highlights/{highlight.id}",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_only_own_org_highlights_listed(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test that list only returns organization's own highlights."""
        # Create two organizations
        org1 = create_test_organization(name="Org 1")
        org2 = create_test_organization(name="Org 2")

        # Create API key for org1
        api_key, raw_key = create_test_api_key(
            organization_id=org1.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        # Create streams and highlights for both orgs
        stream1 = create_test_stream(organization_id=org1.id)
        stream2 = create_test_stream(organization_id=org2.id)

        highlight1 = create_test_highlight(
            stream_id=stream1.id,
            organization_id=org1.id,
            title="Org 1 Highlight",
        )
        highlight2 = create_test_highlight(
            stream_id=stream2.id,
            organization_id=org2.id,
            title="Org 2 Highlight",
        )

        # Save to database
        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        stream_repo = StreamRepository(test_session)
        highlight_repo = HighlightRepository(test_session)

        await org_repo.create(org1)
        await org_repo.create(org2)
        await api_key_repo.create(api_key)
        await stream_repo.create(stream1)
        await stream_repo.create(stream2)
        await highlight_repo.create(highlight1)
        await highlight_repo.create(highlight2)
        await test_session.commit()

        # List highlights with org1's API key
        response = await async_client.get(
            "/api/v1/highlights/",
            headers=api_key_headers(raw_key),
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should only see org1's highlight
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["title"] == "Org 1 Highlight"


class TestHighlightValidation:
    """Test input validation for highlight endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_query_parameters(
        self,
        async_client: AsyncClient,
        test_session: AsyncSession,
        api_key_headers,
    ):
        """Test validation of query parameters."""
        # Create API key
        org = create_test_organization()
        api_key, raw_key = create_test_api_key(
            organization_id=org.id, scopes={APIScopes.HIGHLIGHTS_READ}
        )

        org_repo = OrganizationRepository(test_session)
        api_key_repo = APIKeyRepository(test_session)
        await org_repo.create(org)
        await api_key_repo.create(api_key)
        await test_session.commit()

        # Test invalid min_score
        response = await async_client.get(
            "/api/v1/highlights/?min_score=1.5",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid order_by
        response = await async_client.get(
            "/api/v1/highlights/?order_by=invalid",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid limit
        response = await async_client.get(
            "/api/v1/highlights/?limit=0",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test negative offset
        response = await async_client.get(
            "/api/v1/highlights/?offset=-1",
            headers=api_key_headers(raw_key),
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
