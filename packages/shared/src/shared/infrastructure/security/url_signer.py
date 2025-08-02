"""URL signing service using JWT for secure content delivery."""

from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Literal
from uuid import UUID
import jwt

from ...domain.protocols import StorageService


ResourceType = Literal[
    "clip", "thumbnail", "stream_clips", "streamer_clips", "all_clips"
]


class JWTURLSigner:
    """Service for creating and validating signed URLs using JWT."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """Initialize with secret key."""
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_access_token(
        self,
        organization_id: UUID,
        resource_type: ResourceType,
        expiry_seconds: int = 3600,
        stream_id: Optional[UUID] = None,
        stream_fingerprint: Optional[str] = None,
        highlight_ids: Optional[list[UUID]] = None,
        additional_claims: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a JWT token for resource access with flexible claims.

        Access patterns:
        1. Single clip: resource_type="clip", highlight_ids=[single_id]
        2. Multiple specific clips: resource_type="clip", highlight_ids=[id1, id2, ...]
        3. All clips from a stream: resource_type="stream_clips", stream_id=UUID
        4. All clips from a streamer: resource_type="streamer_clips", stream_fingerprint=str
        5. All clips for organization: resource_type="all_clips"

        Claims:
        - sub: subject (organization_id)
        - resource_type: type of resource access
        - stream_id: specific stream (optional)
        - stream_fingerprint: streamer identifier (optional)
        - highlight_ids: specific highlight IDs (optional)
        - exp: expiration timestamp
        - iat: issued at timestamp
        """
        now = datetime.now(timezone.utc)

        claims = {
            "sub": str(organization_id),
            "resource_type": resource_type,
            "iat": now,
            "exp": now + timedelta(seconds=expiry_seconds),
            "type": "content_access",
        }

        if stream_id:
            claims["stream_id"] = str(stream_id)

        if stream_fingerprint:
            claims["stream_fingerprint"] = stream_fingerprint

        if highlight_ids:
            claims["highlight_ids"] = [str(h_id) for h_id in highlight_ids]

        if additional_claims:
            claims.update(additional_claims)

        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)

    def verify_access_token(
        self,
        token: str,
        organization_id: UUID,
        requested_highlight_id: Optional[UUID] = None,
        requested_stream_id: Optional[UUID] = None,
        requested_stream_fingerprint: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Verify JWT token and check if it grants access to requested resource.
        Returns decoded claims if valid, None otherwise.
        """
        try:
            # Decode and verify token
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["sub", "resource_type", "exp", "iat", "type"]},
            )

            # Verify token type
            if claims.get("type") != "content_access":
                return None

            # Verify organization matches
            if claims.get("sub") != str(organization_id):
                return None

            resource_type = claims.get("resource_type")

            # Check access based on resource type
            if resource_type == "all_clips":
                # Can access any clip in the organization
                return claims

            elif resource_type == "stream_clips":
                # Can access clips from specific stream
                if requested_stream_id and claims.get("stream_id") == str(
                    requested_stream_id
                ):
                    return claims
                return None

            elif resource_type == "streamer_clips":
                # Can access all clips from a specific streamer
                if (
                    requested_stream_fingerprint
                    and claims.get("stream_fingerprint") == requested_stream_fingerprint
                ):
                    return claims
                return None

            elif resource_type == "clip":
                # Can only access specific clips
                if requested_highlight_id:
                    allowed_ids = claims.get("highlight_ids", [])
                    if str(requested_highlight_id) in allowed_ids:
                        return claims
                return None

            else:
                return None

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_clip_url(self, base_url: str, token: str) -> str:
        """Create a URL with JWT token appended."""
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}token={token}"


class SecureContentDelivery:
    """Service for secure content delivery with flexible access control."""

    def __init__(self, storage: StorageService, jwt_signer: JWTURLSigner):
        """Initialize with storage service and JWT signer."""
        self.storage = storage
        self.jwt_signer = jwt_signer

    async def generate_single_clip_url(
        self,
        clip_path: str,
        highlight_id: UUID,
        organization_id: UUID,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate URL for accessing a single clip."""
        # Get base URL from storage
        base_url = await self.storage.generate_signed_url(clip_path, expiry_seconds)

        # Create token for single clip access
        token = self.jwt_signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=[highlight_id],
            expiry_seconds=expiry_seconds,
        )

        return self.jwt_signer.create_clip_url(base_url, token)

    async def generate_stream_clips_token(
        self,
        stream_id: UUID,
        organization_id: UUID,
        stream_fingerprint: Optional[str] = None,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate token for accessing all clips from a stream.
        This token can be used to access any clip from the specified stream.
        """
        return self.jwt_signer.create_access_token(
            organization_id=organization_id,
            resource_type="stream_clips",
            stream_id=stream_id,
            stream_fingerprint=stream_fingerprint,
            expiry_seconds=expiry_seconds,
        )

    async def generate_multi_clip_token(
        self,
        highlight_ids: list[UUID],
        organization_id: UUID,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate token for accessing multiple specific clips.
        This token can be used to access any of the specified clips.
        """
        return self.jwt_signer.create_access_token(
            organization_id=organization_id,
            resource_type="clip",
            highlight_ids=highlight_ids,
            expiry_seconds=expiry_seconds,
        )

    async def generate_streamer_clips_token(
        self, stream_fingerprint: str, organization_id: UUID, expiry_seconds: int = 3600
    ) -> str:
        """Generate token for accessing all clips from a specific streamer.
        This token can be used to access any clip from the specified streamer
        across all their streams.
        """
        return self.jwt_signer.create_access_token(
            organization_id=organization_id,
            resource_type="streamer_clips",
            stream_fingerprint=stream_fingerprint,
            expiry_seconds=expiry_seconds,
        )

    async def generate_organization_token(
        self,
        organization_id: UUID,
        expiry_seconds: int = 86400,  # 24 hours default
    ) -> str:
        """Generate token for accessing all clips in an organization.
        This is typically for admin users who can see everything.
        """
        return self.jwt_signer.create_access_token(
            organization_id=organization_id,
            resource_type="all_clips",
            expiry_seconds=expiry_seconds,
        )

    async def build_clip_url_with_token(
        self, clip_path: str, token: str, storage_expiry: int = 3600
    ) -> str:
        """Build a clip URL using an existing token."""
        base_url = await self.storage.generate_signed_url(clip_path, storage_expiry)
        return self.jwt_signer.create_clip_url(base_url, token)
