"""URL signer interface for domain layer.

This module defines the protocol for URL signing services that infrastructure
implementations must follow. This ensures proper separation between domain
logic and infrastructure concerns.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Any, List, Protocol, runtime_checkable


class TokenScope(str, Enum):
    """Token scope enumeration for access control."""

    VIEW = "view"
    LIST = "list"
    DOWNLOAD = "download"
    ADMIN = "admin"


@runtime_checkable
class URLSignerInterface(Protocol):
    """Protocol for URL signing services.

    Infrastructure implementations must follow this protocol to provide
    URL signing functionality to the domain layer.
    """

    async def generate_highlight_url(
        self,
        stream_id: int,
        organization_id: int,
        scope: TokenScope = TokenScope.VIEW,
        expiry_hours: int = 24,
        ip_restriction: Optional[str] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for highlight access.

        Args:
            stream_id: ID of the stream
            organization_id: ID of the organization
            scope: Token scope for access control
            expiry_hours: Hours until expiration
            ip_restriction: Optional IP address restriction
            additional_claims: Additional JWT claims

        Returns:
            Signed URL string
        """
        ...

    async def generate_stream_url(
        self,
        stream_id: int,
        organization_id: int,
        scope: TokenScope = TokenScope.LIST,
        expiry_hours: int = 24,
        ip_restriction: Optional[str] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for stream access.

        Args:
            stream_id: ID of the stream
            organization_id: ID of the organization
            scope: Token scope for access control
            expiry_hours: Hours until expiration
            ip_restriction: Optional IP address restriction
            additional_claims: Additional JWT claims

        Returns:
            Signed URL string
        """
        ...

    async def generate_batch_url(
        self,
        highlight_ids: List[int],
        organization_id: int,
        scope: TokenScope = TokenScope.VIEW,
        expiry_hours: int = 24,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for batch highlight access.

        Args:
            highlight_ids: List of highlight IDs
            organization_id: ID of the organization
            scope: Token scope for access control
            expiry_hours: Hours until expiration
            additional_claims: Additional JWT claims

        Returns:
            Signed URL string
        """
        ...

    async def verify_token(
        self,
        token: str,
        required_claims: Optional[Dict[str, Any]] = None,
        verify_ip: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Verify a token and extract claims.

        Args:
            token: JWT token to verify
            required_claims: Claims that must be present
            verify_ip: IP address to verify against restrictions

        Returns:
            Tuple of (is_valid, claims, error_message)
        """
        ...

    async def revoke_token(
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

    async def check_token_revoked(self, jti: str) -> bool:
        """Check if a token has been revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        ...

    async def rotate_signing_key(
        self,
        organization_id: int,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """Rotate the signing key for an organization.

        Args:
            organization_id: Organization to rotate key for
            reason: Optional reason for rotation

        Returns:
            New key ID if successful
        """
        ...
