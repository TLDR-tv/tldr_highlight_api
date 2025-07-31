"""URL signing utilities for secure content access.

This module provides utilities for generating and verifying signed URLs
for secure content access, particularly for external streamers who are
not managed users in the system.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List
from urllib.parse import urlencode

from jose import jwt, JWTError
import redis.asyncio as redis

from src.domain.services.url_signer_interface import TokenScope
from src.domain.repositories.organization_key_repository import (
    OrganizationKeyRepository,
)
from src.domain.entities.organization_key import OrganizationKey
from src.infrastructure.security.key_service import KeyGenerationService
from src.infrastructure.security.config import SecurityConfig


logger = logging.getLogger(__name__)


class URLSigner:
    """Enhanced URL signer with organization-specific keys.

    This class provides methods for creating and validating signed URLs
    that allow secure access to content without requiring authentication.
    It supports per-organization keys, comprehensive JWT claims, and
    advanced security features.
    """

    def __init__(
        self,
        security_config: SecurityConfig,
        key_repository: OrganizationKeyRepository,
        key_service: KeyGenerationService,
        redis_client: redis.Redis,
    ):
        """Initialize URL signer.

        Args:
            security_config: Security configuration
            key_repository: Organization key repository
            key_service: Key generation service
            redis_client: Redis client for revocation
        """
        self.config = security_config
        self.key_repository = key_repository
        self.key_service = key_service
        self.redis_client = redis_client
        self._key_cache: Dict[str, Tuple[OrganizationKey, datetime]] = {}

    async def _get_signing_key(self, organization_id: int) -> OrganizationKey:
        """Get the current signing key for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            Active primary key for signing

        Raises:
            ValueError: If no signing key found
        """
        # Check cache first
        cache_key = f"org_{organization_id}_primary"
        if cache_key in self._key_cache:
            key, cached_at = self._key_cache[cache_key]
            # Cache for 5 minutes
            if datetime.utcnow() - cached_at < timedelta(minutes=5):
                return key

        # Get from repository
        key = await self.key_repository.get_primary_key(organization_id)
        if not key:
            # For backward compatibility, try master key
            if self.config.master_signing_key:
                logger.warning(
                    f"No signing key for org {organization_id}, using master key"
                )
                # Create a temporary key object
                return OrganizationKey.create_new(
                    organization_id=organization_id,
                    key_id="master",
                    key_value=self.config.master_signing_key.get_secret_value(),
                    description="Master key (backward compatibility)",
                )
            raise ValueError(f"No signing key found for organization {organization_id}")

        # Cache the key
        self._key_cache[cache_key] = (key, datetime.utcnow())

        # Increment usage
        await self.key_repository.increment_usage(key.key_id)

        return key

    async def _get_verification_keys(
        self, organization_id: int
    ) -> List[OrganizationKey]:
        """Get all keys that can verify tokens for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            List of keys that can verify tokens
        """
        keys = await self.key_repository.get_active_keys(
            organization_id, include_rotating=True
        )

        # For backward compatibility, include master key if configured
        if self.config.master_signing_key and not keys:
            master_key = OrganizationKey.create_new(
                organization_id=organization_id,
                key_id="master",
                key_value=self.config.master_signing_key.get_secret_value(),
                description="Master key (backward compatibility)",
            )
            keys.append(master_key)

        return [k for k in keys if k.can_verify]

    def _generate_jti(self) -> str:
        """Generate a unique JWT ID."""
        return str(uuid.uuid4())

    async def _check_revocation(self, jti: str) -> bool:
        """Check if a token is revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if revoked
        """
        if not self.config.enable_token_revocation:
            return False

        key = f"revoked_token:{jti}"
        return bool(await self.redis_client.exists(key))

    async def _increment_usage_count(self, jti: str) -> int:
        """Increment and return usage count for a token.

        Args:
            jti: JWT ID

        Returns:
            Updated usage count
        """
        key = f"token_usage:{jti}"
        count = await self.redis_client.incr(key)

        # Set expiry to match token lifetime
        if count == 1:
            await self.redis_client.expire(key, 86400 * 7)  # 7 days max

        return count

    async def _create_token_payload(
        self,
        organization_id: int,
        scope: TokenScope,
        expiry_hours: int,
        subject: str,
        ip_restriction: Optional[str] = None,
        usage_limit: Optional[int] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a comprehensive JWT payload.

        Args:
            organization_id: Organization ID
            scope: Access scope
            expiry_hours: Hours until expiration
            subject: Token subject (resource identifier)
            ip_restriction: Optional IP restriction
            usage_limit: Optional usage limit
            additional_claims: Additional claims

        Returns:
            JWT payload dictionary
        """
        now = datetime.utcnow()

        # Validate expiry
        if expiry_hours > self.config.max_token_lifetime_hours:
            expiry_hours = self.config.max_token_lifetime_hours

        payload = {
            # Standard JWT claims
            "jti": self._generate_jti() if self.config.require_jti else None,
            "iss": self.config.jwt_issuer,
            "aud": f"org-{organization_id}",
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(hours=expiry_hours),
            # Custom claims
            "organization_id": organization_id,
            "scope": scope.value,
            "version": "2.0",  # Token format version
        }

        # Optional claims
        if ip_restriction and self.config.require_ip_validation:
            payload["ip"] = ip_restriction

        if usage_limit:
            payload["usage_limit"] = usage_limit
            payload["usage_count"] = 0

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)

        return payload

    async def generate_highlight_url(
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
        # Get signing key
        key = await self._get_signing_key(organization_id)

        # Decrypt key value
        key_value = self.key_service.decrypt_key_value(key)

        # Create payload
        payload = await self._create_token_payload(
            organization_id=organization_id,
            scope=scope,
            expiry_hours=expiry_hours,
            subject=f"highlight-{highlight_id}",
            ip_restriction=ip_restriction,
            usage_limit=usage_limit,
            additional_claims={
                "highlight_id": highlight_id,
                "stream_id": stream_id,
                **(additional_claims or {}),
            },
        )

        # Add key identifier
        payload["kid"] = key.key_id

        # Generate token
        token = jwt.encode(payload, key_value, algorithm=key.algorithm.value)

        # Log token generation
        logger.info(
            f"Generated highlight URL for org {organization_id}, "
            f"highlight {highlight_id}, JTI: {payload.get('jti')}"
        )

        # Construct URL
        return f"{base_url.rstrip('/')}/api/v1/content/{highlight_id}?token={token}"

    async def generate_stream_url(
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
        # Get signing key
        key = await self._get_signing_key(organization_id)

        # Decrypt key value
        key_value = self.key_service.decrypt_key_value(key)

        # Create payload
        payload = await self._create_token_payload(
            organization_id=organization_id,
            scope=scope,
            expiry_hours=expiry_hours,
            subject=f"stream-{stream_id}",
            ip_restriction=ip_restriction,
            additional_claims={"stream_id": stream_id, **(additional_claims or {})},
        )

        # Add key identifier
        payload["kid"] = key.key_id

        # Generate token
        token = jwt.encode(payload, key_value, algorithm=key.algorithm.value)

        # Log token generation
        logger.info(
            f"Generated stream URL for org {organization_id}, "
            f"stream {stream_id}, JTI: {payload.get('jti')}"
        )

        # Construct URL
        return f"{base_url.rstrip('/')}/api/v1/content/stream/{stream_id}?token={token}"

    async def generate_batch_url(
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
        # Get signing key
        key = await self._get_signing_key(organization_id)

        # Decrypt key value
        key_value = self.key_service.decrypt_key_value(key)

        # Create payload
        payload = await self._create_token_payload(
            organization_id=organization_id,
            scope=scope,
            expiry_hours=expiry_hours,
            subject=f"batch-{len(highlight_ids)}",
            additional_claims={
                "highlight_ids": highlight_ids,
                **(additional_claims or {}),
            },
        )

        # Add key identifier
        payload["kid"] = key.key_id

        # Generate token
        token = jwt.encode(payload, key_value, algorithm=key.algorithm.value)

        # Log token generation
        logger.info(
            f"Generated batch URL for org {organization_id}, "
            f"{len(highlight_ids)} highlights, JTI: {payload.get('jti')}"
        )

        # Construct URL with highlight IDs as query params
        params = {"token": token, "ids": ",".join(map(str, highlight_ids))}
        return f"{base_url.rstrip('/')}/api/v1/content/batch?{urlencode(params)}"

    async def verify_token(
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
        try:
            # Decode without verification first to get headers
            unverified = jwt.get_unverified_header(token)
            kid = unverified.get("kid")

            # Decode to get organization ID
            unverified_payload = jwt.get_unverified_claims(token)
            org_id = unverified_payload.get("organization_id")

            if not org_id:
                return False, None, "Missing organization_id in token"

            # Get verification keys
            keys = await self._get_verification_keys(org_id)

            # Try each key
            payload = None
            verified = False

            for key in keys:
                # Skip if key ID doesn't match (when specified)
                if kid and key.key_id != kid:
                    continue

                try:
                    # Decrypt key value
                    key_value = self.key_service.decrypt_key_value(key)

                    # Verify token
                    payload = jwt.decode(
                        token,
                        key_value,
                        algorithms=[key.algorithm.value],
                        options={"verify_exp": True},
                    )
                    verified = True

                    # Increment key usage
                    await self.key_repository.increment_usage(key.key_id)
                    break

                except JWTError:
                    continue

            if not verified or not payload:
                return False, None, "Invalid token signature"

            # Check if revoked
            jti = payload.get("jti")
            if jti and await self._check_revocation(jti):
                return False, None, "Token has been revoked"

            # Verify IP if required
            if verify_ip and self.config.require_ip_validation:
                token_ip = payload.get("ip")
                if token_ip and token_ip != verify_ip:
                    return False, None, f"IP mismatch: {verify_ip} != {token_ip}"

            # Check usage limit
            usage_limit = payload.get("usage_limit")
            if usage_limit:
                usage_count = await self._increment_usage_count(jti)
                if usage_count > usage_limit:
                    return (
                        False,
                        None,
                        f"Usage limit exceeded: {usage_count} > {usage_limit}",
                    )

            # Verify required claims
            if required_claims:
                for key, value in required_claims.items():
                    if key not in payload:
                        return False, None, f"Missing required claim: {key}"
                    if value is not None and payload[key] != value:
                        return False, None, f"Claim mismatch: {key}"

            return True, payload, None

        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}")
            return False, None, f"Error verifying token: {str(e)}"

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
        if not self.config.enable_token_revocation:
            logger.warning("Token revocation is disabled")
            return False

        try:
            # Decode to get JTI
            payload = jwt.get_unverified_claims(token)
            jti = payload.get("jti")

            if not jti:
                logger.warning("Token has no JTI, cannot revoke")
                return False

            # Get expiration time
            exp = payload.get("exp")
            if not exp:
                logger.warning("Token has no expiration, cannot revoke")
                return False

            # Calculate TTL
            ttl = exp - datetime.utcnow().timestamp()
            if ttl <= 0:
                logger.info("Token already expired, no need to revoke")
                return True

            # Store in revocation list
            key = f"revoked_token:{jti}"
            await self.redis_client.setex(key, int(ttl), reason or "Manually revoked")

            logger.info(f"Revoked token JTI: {jti}, reason: {reason}")
            return True

        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False

    async def check_token_revoked(self, jti: str) -> bool:
        """Check if a token has been revoked by JWT ID.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        return await self._check_revocation(jti)

    @property
    def signing_key_info(self) -> Dict[str, Any]:
        """Get general information about signing key configuration.

        Returns:
            Dictionary with key configuration
        """
        # This is a property that returns general info, not org-specific
        return {
            "algorithm": self.config.jwt_default_algorithm.value,
            "issuer": self.config.jwt_issuer,
            "max_token_lifetime_hours": self.config.max_token_lifetime_hours,
            "revocation_enabled": self.config.enable_token_revocation,
        }

    async def get_organization_key_info(self, organization_id: int) -> Dict[str, Any]:
        """Get information about the current signing key for an organization.

        Args:
            organization_id: Organization ID

        Returns:
            Key information (without sensitive data)
        """
        try:
            key = await self.key_repository.get_primary_key(organization_id)
            if not key:
                return {"error": "No signing key found"}

            # Check if rotation needed
            should_rotate, rotate_reason = self.key_service.should_rotate_key(key)

            return {
                "key_id": key.key_id,
                "algorithm": key.algorithm.value,
                "version": key.key_version,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "usage_count": key.usage_count,
                "status": key.status.value,
                "should_rotate": should_rotate,
                "rotate_reason": rotate_reason,
            }
        except Exception as e:
            logger.error(f"Error getting key info: {str(e)}")
            return {"error": str(e)}

    async def rotate_signing_key(
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
        try:
            # Get current key
            current_key = await self.key_repository.get_primary_key(organization_id)
            if not current_key:
                logger.error(f"No current key found for org {organization_id}")
                return False

            # Create new key
            new_key = self.key_service.rotate_key(current_key, reason, created_by)

            # Save new key
            new_key = await self.key_repository.create(new_key)

            # Update current key status
            await self.key_repository.update(current_key)

            # Set new key as primary
            await self.key_repository.set_primary(organization_id, new_key.key_id)

            # Clear cache
            cache_key = f"org_{organization_id}_primary"
            self._key_cache.pop(cache_key, None)

            logger.info(
                f"Rotated key for org {organization_id}: "
                f"{current_key.key_id} -> {new_key.key_id}"
            )

            return True

        except Exception as e:
            logger.error(f"Error rotating key: {str(e)}")
            return False
