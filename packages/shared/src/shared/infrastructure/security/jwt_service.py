"""JWT service for user authentication."""

from datetime import datetime, timedelta, UTC
from typing import Optional, Any
from uuid import UUID, uuid4
import jwt
from pydantic import BaseModel

from ..config.config import Settings


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str  # user_id
    org: str  # organization_id
    email: str
    role: str
    type: str  # "access" or "refresh"
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for refresh tokens


class JWTService:
    """Service for creating and validating JWT tokens for user authentication."""

    def __init__(self, settings: Settings):
        """Initialize with settings."""
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expiry = settings.jwt_expiry_seconds
        self.refresh_token_expiry = 86400 * 7  # 7 days

    def create_access_token(
        self,
        user_id: UUID,
        organization_id: UUID,
        email: str,
        role: str,
        expiry_seconds: Optional[int] = None,
    ) -> str:
        """Create access token for user authentication."""
        now = datetime.now(UTC)
        expiry = now + timedelta(seconds=expiry_seconds or self.access_token_expiry)

        payload = {
            "sub": str(user_id),
            "org": str(organization_id),
            "email": email,
            "role": role,
            "type": "access",
            "iat": now,
            "exp": expiry,
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: UUID,
        jti: Optional[str] = None,
        expiry_seconds: Optional[int] = None,
    ) -> str:
        """Create refresh token for token renewal."""
        now = datetime.now(UTC)
        expiry = now + timedelta(seconds=expiry_seconds or self.refresh_token_expiry)

        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": now,
            "exp": expiry,
            "jti": jti or str(uuid4()),  # Unique token ID for revocation
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_access_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode access token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "require": ["sub", "org", "email", "role", "type", "exp", "iat"]
                },
            )

            # Verify token type
            if payload.get("type") != "access":
                return None

            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def verify_refresh_token(self, token: str) -> Optional[dict[str, Any]]:
        """Verify and decode refresh token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["sub", "type", "exp", "iat"]},
            )

            # Verify token type
            if payload.get("type") != "refresh":
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_password_reset_token(
        self,
        user_id: UUID,
        email: str,
        expiry_seconds: int = 3600,  # 1 hour
    ) -> str:
        """Create token for password reset."""
        now = datetime.now(UTC)
        expiry = now + timedelta(seconds=expiry_seconds)

        payload = {
            "sub": str(user_id),
            "email": email,
            "type": "password_reset",
            "iat": now,
            "exp": expiry,
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_password_reset_token(self, token: str) -> Optional[dict[str, Any]]:
        """Verify password reset token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["sub", "email", "type", "exp", "iat"]},
            )

            # Verify token type
            if payload.get("type") != "password_reset":
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
