"""URL signing utilities for secure content access.

This module provides utilities for generating and verifying signed URLs
for secure content access, particularly for external streamers who are
not managed users in the system.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, Union

from jose import jwt, JWTError

from src.infrastructure.config import settings

logger = logging.getLogger(__name__)


class URLSigner:
    """Utility for generating and verifying signed URLs.

    This class provides methods for creating and validating signed URLs
    that allow secure access to content without requiring authentication.
    """

    def __init__(self, secret_key: str = None, algorithm: str = None):
        """Initialize the URL signer.

        Args:
            secret_key: Secret key for signing JWTs (defaults to settings.jwt_secret_key)
            algorithm: Algorithm for signing JWTs (defaults to settings.jwt_algorithm)
        """
        self.secret_key = secret_key or settings.jwt_secret_key
        self.algorithm = algorithm or settings.jwt_algorithm

    def generate_signed_url(
        self,
        base_url: str,
        highlight_id: int,
        stream_id: int,
        org_id: int,
        expiry_hours: int = 24,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for accessing a highlight.

        Args:
            base_url: Base URL for content access
            highlight_id: ID of the highlight to access
            stream_id: ID of the stream the highlight belongs to
            org_id: ID of the organization (streaming platform)
            expiry_hours: Number of hours until the URL expires
            additional_claims: Additional claims to include in the JWT

        Returns:
            Signed URL with JWT token
        """
        # Create JWT payload
        now = datetime.utcnow()
        payload = {
            "highlight_id": highlight_id,
            "stream_id": stream_id,
            "org_id": org_id,
            "iat": now,
            "exp": now + timedelta(hours=expiry_hours),
        }

        # Add additional claims if provided
        if additional_claims:
            payload.update(additional_claims)

        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Construct URL
        return f"{base_url}/api/v1/content/{highlight_id}?token={token}"

    def generate_stream_access_url(
        self,
        base_url: str,
        stream_id: int,
        org_id: int,
        expiry_hours: int = 24,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a signed URL for accessing all highlights from a stream.

        Args:
            base_url: Base URL for content access
            stream_id: ID of the stream
            org_id: ID of the organization (streaming platform)
            expiry_hours: Number of hours until the URL expires
            additional_claims: Additional claims to include in the JWT

        Returns:
            Signed URL with JWT token
        """
        # Create JWT payload
        now = datetime.utcnow()
        payload = {
            "stream_id": stream_id,
            "org_id": org_id,
            "iat": now,
            "exp": now + timedelta(hours=expiry_hours),
        }

        # Add additional claims if provided
        if additional_claims:
            payload.update(additional_claims)

        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Construct URL
        return f"{base_url}/api/v1/content/stream/{stream_id}?token={token}"

    def verify_token(
        self, token: str, required_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Verify a signed URL token.

        Args:
            token: JWT token to verify
            required_claims: Claims that must be present in the token

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        try:
            # Decode and verify token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )

            # Check if token has expired
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"])
                if exp_time < datetime.utcnow():
                    return False, None, "Token has expired"

            # Check required claims
            if required_claims:
                for key, value in required_claims.items():
                    if key not in payload:
                        return False, None, f"Missing required claim: {key}"
                    if value is not None and payload[key] != value:
                        return (
                            False,
                            None,
                            f"Invalid value for claim {key}: {payload[key]} != {value}",
                        )

            return True, payload, None

        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}")
            return False, None, f"Error verifying token: {str(e)}"

    def extract_and_verify_token(
        self, url: str, required_claims: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Extract and verify a token from a signed URL.

        Args:
            url: Signed URL containing a token
            required_claims: Claims that must be present in the token

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        try:
            # Extract token from URL
            if "?token=" not in url and "&token=" not in url:
                return False, None, "No token found in URL"

            token_part = url.split("token=")[1].split("&")[0] if "&" in url.split("token=")[1] else url.split("token=")[1]

            # Verify token
            return self.verify_token(token_part, required_claims)

        except Exception as e:
            logger.error(f"Error extracting token from URL: {str(e)}")
            return False, None, f"Error processing URL: {str(e)}"


# Create a singleton instance
url_signer = URLSigner()