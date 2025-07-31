"""URL signer protocol for dependency injection.

This module defines the protocol for URL signing implementations,
enabling dependency injection and supporting multiple signing strategies.
"""

from typing import Dict, Optional, Tuple, Any, List, Protocol
from datetime import datetime
from enum import Enum


class TokenScope(str, Enum):
    """Token access scopes."""

    VIEW = "view"
    DOWNLOAD = "download"
    STREAM = "stream"
    LIST = "list"
    FULL = "full"


class SignedURLType(str, Enum):
    """Types of signed URLs."""

    HIGHLIGHT_ACCESS = "highlight_access"
    STREAM_ACCESS = "stream_access"
    BATCH_ACCESS = "batch_access"
    TEMPORARY_SHARE = "temporary_share"


class URLSignerProtocol(Protocol):
    """Protocol for URL signing implementations."""

    def generate_highlight_url(
        self,
        base_url: str,
        highlight_id: int,
        stream_id: int,
        organization_id: int,
        scope: TokenScope = TokenScope.VIEW,
        expiry_hours: int = 24,
        ip_restriction: Optional[str] = None,
        usage_limit: Optional[int] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for accessing a highlight.

        Args:
            base_url: Base URL for content access
            highlight_id: ID of the highlight
            stream_id: ID of the stream
            organization_id: ID of the organization
            scope: Access scope for the token
            expiry_hours: Hours until expiration
            ip_restriction: Optional IP address restriction
            usage_limit: Optional usage count limit
            additional_claims: Additional JWT claims

        Returns:
            Signed URL with embedded token
        """
        ...

    def generate_stream_url(
        self,
        base_url: str,
        stream_id: int,
        organization_id: int,
        scope: TokenScope = TokenScope.LIST,
        expiry_hours: int = 24,
        ip_restriction: Optional[str] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for accessing stream highlights.

        Args:
            base_url: Base URL for content access
            stream_id: ID of the stream
            organization_id: ID of the organization
            scope: Access scope for the token
            expiry_hours: Hours until expiration
            ip_restriction: Optional IP address restriction
            additional_claims: Additional JWT claims

        Returns:
            Signed URL with embedded token
        """
        ...

    def generate_batch_url(
        self,
        base_url: str,
        highlight_ids: List[int],
        organization_id: int,
        scope: TokenScope = TokenScope.VIEW,
        expiry_hours: int = 24,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for accessing multiple highlights.

        Args:
            base_url: Base URL for content access
            highlight_ids: List of highlight IDs
            organization_id: ID of the organization
            scope: Access scope for the token
            expiry_hours: Hours until expiration
            additional_claims: Additional JWT claims

        Returns:
            Signed URL with embedded token
        """
        ...

    def verify_token(
        self,
        token: str,
        required_claims: Optional[Dict[str, Any]] = None,
        verify_ip: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Verify a signed URL token.

        Args:
            token: JWT token to verify
            required_claims: Claims that must match
            verify_ip: IP address to verify against restriction

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        ...

    def revoke_token(
        self,
        token: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Revoke a token before expiration.

        Args:
            token: Token to revoke
            reason: Optional reason for revocation

        Returns:
            True if successfully revoked
        """
        ...

    @property
    def is_token_revoked(self) -> bool:
        """Check if a token has been revoked.

        Returns:
            True if token is revoked
        """
        ...

    def check_token_revoked(self, jti: str) -> bool:
        """Check if a token has been revoked by JWT ID.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        ...

    def get_signing_key_info(self, organization_id: int) -> Dict[str, Any]:
        """Get information about the current signing key.

        Args:
            organization_id: Organization ID

        Returns:
            Key information (without sensitive data)
        """
        ...

    def rotate_signing_key(
        self,
        organization_id: int,
        reason: str,
        created_by: Optional[str] = None,
    ) -> bool:
        """Rotate the signing key for an organization.

        Args:
            organization_id: Organization ID
            reason: Reason for rotation
            created_by: Who initiated rotation

        Returns:
            True if successfully rotated
        """
        ...


class TokenPayload(Protocol):
    """Protocol for token payload structure."""

    @property
    def jti(self) -> str:
        """Unique token identifier."""
        ...

    @property
    def iss(self) -> str:
        """Token issuer."""
        ...

    @property
    def aud(self) -> str:
        """Token audience (organization)."""
        ...

    @property
    def sub(self) -> str:
        """Token subject (resource)."""
        ...

    @property
    def scope(self) -> TokenScope:
        """Access scope."""
        ...

    @property
    def exp(self) -> datetime:
        """Expiration time."""
        ...

    @property
    def iat(self) -> datetime:
        """Issued at time."""
        ...

    @property
    def organization_id(self) -> int:
        """Organization ID."""
        ...

    @property
    def ip_restriction(self) -> Optional[str]:
        """IP address restriction."""
        ...

    @property
    def usage_limit(self) -> Optional[int]:
        """Usage count limit."""
        ...

    @property
    def usage_count(self) -> int:
        """Current usage count."""
        ...
