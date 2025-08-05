"""Unit tests for URL signer module."""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, Mock
import jwt

from shared.infrastructure.security.url_signer import (
    JWTURLSigner,
    SecureContentDelivery,
    ResourceType,
)
from shared.domain.protocols import StorageService


class TestJWTURLSigner:
    """Test JWTURLSigner class."""

    @pytest.fixture
    def signer(self):
        """Create a test JWT URL signer."""
        return JWTURLSigner(secret_key="test-secret-key")

    @pytest.fixture
    def organization_id(self):
        """Create a test organization ID."""
        return uuid4()

    def test_initialization(self):
        """Test JWT URL signer initialization."""
        signer = JWTURLSigner(secret_key="my-secret", algorithm="HS384")
        assert signer.secret_key == "my-secret"
        assert signer.algorithm == "HS384"

    def test_initialization_default_algorithm(self):
        """Test JWT URL signer with default algorithm."""
        signer = JWTURLSigner(secret_key="my-secret")
        assert signer.algorithm == "HS256"

    def test_create_access_token_single_clip(self, signer, organization_id):
        """Test creating access token for single clip."""
        highlight_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=[highlight_id],
            expiry_seconds=3600
        )
        
        # Decode token to verify claims
        claims = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert claims["sub"] == str(organization_id)
        assert claims["resource_type"] == "clip"
        assert claims["highlight_ids"] == [str(highlight_id)]
        assert claims["type"] == "content_access"
        assert "exp" in claims
        assert "iat" in claims

    def test_create_access_token_stream_clips(self, signer, organization_id):
        """Test creating access token for stream clips."""
        stream_id = uuid4()
        stream_fingerprint = "streamer123"
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="stream_clips",
            stream_id=stream_id,
            stream_fingerprint=stream_fingerprint,
            expiry_seconds=7200
        )
        
        claims = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert claims["sub"] == str(organization_id)
        assert claims["resource_type"] == "stream_clips"
        assert claims["stream_id"] == str(stream_id)
        assert claims["stream_fingerprint"] == stream_fingerprint

    def test_create_access_token_streamer_clips(self, signer, organization_id):
        """Test creating access token for streamer clips."""
        stream_fingerprint = "streamer456"
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="streamer_clips",
            stream_fingerprint=stream_fingerprint
        )
        
        claims = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert claims["resource_type"] == "streamer_clips"
        assert claims["stream_fingerprint"] == stream_fingerprint

    def test_create_access_token_all_clips(self, signer, organization_id):
        """Test creating access token for all clips."""
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="all_clips"
        )
        
        claims = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert claims["resource_type"] == "all_clips"

    def test_create_access_token_with_additional_claims(self, signer, organization_id):
        """Test creating access token with additional claims."""
        additional_claims = {
            "user_id": "user123",
            "permissions": ["read", "download"]
        }
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            additional_claims=additional_claims
        )
        
        claims = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert claims["user_id"] == "user123"
        assert claims["permissions"] == ["read", "download"]

    def test_verify_access_token_valid_single_clip(self, signer, organization_id):
        """Test verifying valid token for single clip."""
        highlight_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=[highlight_id]
        )
        
        # Verify with correct highlight ID
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_highlight_id=highlight_id
        )
        
        assert claims is not None
        assert claims["resource_type"] == "clip"

    def test_verify_access_token_invalid_highlight_id(self, signer, organization_id):
        """Test verifying token with wrong highlight ID."""
        highlight_id = uuid4()
        wrong_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=[highlight_id]
        )
        
        # Verify with wrong highlight ID
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_highlight_id=wrong_id
        )
        
        assert claims is None

    def test_verify_access_token_stream_clips(self, signer, organization_id):
        """Test verifying token for stream clips."""
        stream_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="stream_clips",
            stream_id=stream_id
        )
        
        # Verify with correct stream ID
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_stream_id=stream_id
        )
        
        assert claims is not None
        assert claims["resource_type"] == "stream_clips"

    def test_verify_access_token_wrong_stream_id(self, signer, organization_id):
        """Test verifying token with wrong stream ID."""
        stream_id = uuid4()
        wrong_stream_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="stream_clips",
            stream_id=stream_id
        )
        
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_stream_id=wrong_stream_id
        )
        
        assert claims is None

    def test_verify_access_token_streamer_clips(self, signer, organization_id):
        """Test verifying token for streamer clips."""
        stream_fingerprint = "streamer789"
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="streamer_clips",
            stream_fingerprint=stream_fingerprint
        )
        
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_stream_fingerprint=stream_fingerprint
        )
        
        assert claims is not None

    def test_verify_access_token_all_clips(self, signer, organization_id):
        """Test verifying token for all clips."""
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="all_clips"
        )
        
        # Should work for any requested resource
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id,
            requested_highlight_id=uuid4()
        )
        
        assert claims is not None

    def test_verify_access_token_wrong_organization(self, signer):
        """Test verifying token with wrong organization."""
        org_id = uuid4()
        wrong_org_id = uuid4()
        
        token = signer.create_access_token(
            organization_id=org_id,
            resource_type="all_clips"
        )
        
        claims = signer.verify_access_token(
            token=token,
            organization_id=wrong_org_id
        )
        
        assert claims is None

    def test_verify_access_token_expired(self, signer, organization_id):
        """Test verifying expired token."""
        # Create token that expires immediately
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="all_clips",
            expiry_seconds=-1  # Already expired
        )
        
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id
        )
        
        assert claims is None

    def test_verify_access_token_invalid_signature(self, signer, organization_id):
        """Test verifying token with invalid signature."""
        # Create token with different secret
        wrong_signer = JWTURLSigner(secret_key="wrong-secret")
        token = wrong_signer.create_access_token(
            organization_id=organization_id,
            resource_type="all_clips"
        )
        
        claims = signer.verify_access_token(
            token=token,
            organization_id=organization_id
        )
        
        assert claims is None

    def test_verify_access_token_wrong_type(self, signer, organization_id):
        """Test verifying token with wrong type."""
        # Create a custom token with wrong type
        now = datetime.now(timezone.utc)
        claims = {
            "sub": str(organization_id),
            "resource_type": "all_clips",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "type": "wrong_type"  # Wrong type
        }
        token = jwt.encode(claims, "test-secret-key", algorithm="HS256")
        
        result = signer.verify_access_token(
            token=token,
            organization_id=organization_id
        )
        
        assert result is None

    def test_verify_access_token_unknown_resource_type(self, signer, organization_id):
        """Test verifying token with unknown resource type."""
        now = datetime.now(timezone.utc)
        claims = {
            "sub": str(organization_id),
            "resource_type": "unknown_type",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "type": "content_access"
        }
        token = jwt.encode(claims, "test-secret-key", algorithm="HS256")
        
        result = signer.verify_access_token(
            token=token,
            organization_id=organization_id
        )
        
        assert result is None

    def test_verify_access_token_malformed(self, signer, organization_id):
        """Test verifying malformed token."""
        claims = signer.verify_access_token(
            token="not.a.valid.token",
            organization_id=organization_id
        )
        
        assert claims is None

    def test_create_clip_url_with_query_params(self, signer):
        """Test creating clip URL with existing query parameters."""
        base_url = "https://example.com/clip.mp4?quality=high"
        token = "sample-token"
        
        url = signer.create_clip_url(base_url, token)
        
        assert url == "https://example.com/clip.mp4?quality=high&token=sample-token"

    def test_create_clip_url_without_query_params(self, signer):
        """Test creating clip URL without query parameters."""
        base_url = "https://example.com/clip.mp4"
        token = "sample-token"
        
        url = signer.create_clip_url(base_url, token)
        
        assert url == "https://example.com/clip.mp4?token=sample-token"

    def test_verify_multiple_highlight_ids(self, signer, organization_id):
        """Test verifying token with multiple highlight IDs."""
        highlight_ids = [uuid4(), uuid4(), uuid4()]
        
        token = signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=highlight_ids
        )
        
        # Should work for any of the highlight IDs
        for highlight_id in highlight_ids:
            claims = signer.verify_access_token(
                token=token,
                organization_id=organization_id,
                requested_highlight_id=highlight_id
            )
            assert claims is not None


class TestSecureContentDelivery:
    """Test SecureContentDelivery class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage service."""
        return AsyncMock(spec=StorageService)

    @pytest.fixture
    def jwt_signer(self):
        """Create a JWT signer."""
        return JWTURLSigner(secret_key="test-secret")

    @pytest.fixture
    def delivery_service(self, mock_storage, jwt_signer):
        """Create secure content delivery service."""
        return SecureContentDelivery(storage=mock_storage, jwt_signer=jwt_signer)

    @pytest.fixture
    def organization_id(self):
        """Create a test organization ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_generate_single_clip_url(
        self, delivery_service, mock_storage, organization_id
    ):
        """Test generating URL for single clip."""
        clip_path = "clips/highlight123.mp4"
        highlight_id = uuid4()
        base_url = "https://storage.example.com/clips/highlight123.mp4"
        
        mock_storage.generate_signed_url.return_value = base_url
        
        url = await delivery_service.generate_single_clip_url(
            clip_path=clip_path,
            highlight_id=highlight_id,
            organization_id=organization_id,
            expiry_seconds=3600
        )
        
        # Should call storage service
        mock_storage.generate_signed_url.assert_called_once_with(clip_path, 3600)
        
        # URL should contain token
        assert url.startswith(base_url)
        assert "token=" in url
        
        # Extract and verify token
        token = url.split("token=")[1]
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert claims["resource_type"] == "clip"
        assert claims["highlight_ids"] == [str(highlight_id)]

    @pytest.mark.asyncio
    async def test_generate_stream_clips_token(
        self, delivery_service, organization_id
    ):
        """Test generating token for stream clips."""
        stream_id = uuid4()
        stream_fingerprint = "streamer123"
        
        token = await delivery_service.generate_stream_clips_token(
            stream_id=stream_id,
            organization_id=organization_id,
            stream_fingerprint=stream_fingerprint,
            expiry_seconds=7200
        )
        
        # Verify token
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert claims["resource_type"] == "stream_clips"
        assert claims["stream_id"] == str(stream_id)
        assert claims["stream_fingerprint"] == stream_fingerprint

    @pytest.mark.asyncio
    async def test_generate_multi_clip_token(
        self, delivery_service, organization_id
    ):
        """Test generating token for multiple clips."""
        highlight_ids = [uuid4(), uuid4(), uuid4()]
        
        token = await delivery_service.generate_multi_clip_token(
            highlight_ids=highlight_ids,
            organization_id=organization_id,
            expiry_seconds=3600
        )
        
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert claims["resource_type"] == "clip"
        assert len(claims["highlight_ids"]) == 3
        assert all(str(hid) in claims["highlight_ids"] for hid in highlight_ids)

    @pytest.mark.asyncio
    async def test_generate_streamer_clips_token(
        self, delivery_service, organization_id
    ):
        """Test generating token for streamer clips."""
        stream_fingerprint = "streamer456"
        
        token = await delivery_service.generate_streamer_clips_token(
            stream_fingerprint=stream_fingerprint,
            organization_id=organization_id,
            expiry_seconds=3600
        )
        
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert claims["resource_type"] == "streamer_clips"
        assert claims["stream_fingerprint"] == stream_fingerprint

    @pytest.mark.asyncio
    async def test_generate_organization_token(
        self, delivery_service, organization_id
    ):
        """Test generating token for organization."""
        token = await delivery_service.generate_organization_token(
            organization_id=organization_id,
            expiry_seconds=86400
        )
        
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert claims["resource_type"] == "all_clips"
        assert claims["sub"] == str(organization_id)

    @pytest.mark.asyncio
    async def test_generate_organization_token_default_expiry(
        self, delivery_service, organization_id
    ):
        """Test organization token with default expiry."""
        token = await delivery_service.generate_organization_token(
            organization_id=organization_id
        )
        
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
        # Check expiry is roughly 24 hours from now
        exp_time = datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
        iat_time = datetime.fromtimestamp(claims["iat"], tz=timezone.utc)
        diff = exp_time - iat_time
        assert 86300 < diff.total_seconds() < 86500  # Allow some wiggle room

    @pytest.mark.asyncio
    async def test_build_clip_url_with_token(
        self, delivery_service, mock_storage
    ):
        """Test building clip URL with existing token."""
        clip_path = "clips/video.mp4"
        token = "existing-token-123"
        base_url = "https://storage.example.com/clips/video.mp4"
        
        mock_storage.generate_signed_url.return_value = base_url
        
        url = await delivery_service.build_clip_url_with_token(
            clip_path=clip_path,
            token=token,
            storage_expiry=3600
        )
        
        assert url == f"{base_url}?token={token}"
        mock_storage.generate_signed_url.assert_called_once_with(clip_path, 3600)

    def test_resource_type_literal(self):
        """Test ResourceType literal values."""
        # This test verifies that the type hints work correctly
        valid_types: list[ResourceType] = [
            "clip",
            "thumbnail", 
            "stream_clips",
            "streamer_clips",
            "all_clips"
        ]
        
        # Should not raise any type errors
        for resource_type in valid_types:
            assert isinstance(resource_type, str)