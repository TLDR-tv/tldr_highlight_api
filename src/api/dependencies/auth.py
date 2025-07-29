"""Authentication dependencies for FastAPI endpoints.

This module provides dependency functions for JWT token validation,
API key authentication, and permission checking.
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import List, Optional

from passlib.context import CryptContext
from jose import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.core.config import get_settings
from src.core.database import get_db
from src.infrastructure.persistence.models.api_key import APIKey
from src.infrastructure.persistence.models.user import User
from src.api.schemas.auth import TokenData

settings = get_settings()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationError(HTTPException):
    """Custom exception for authentication errors."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionError(HTTPException):
    """Custom exception for permission errors."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        password: Plain text password
        hashed_password: Hashed password

    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(password, hashed_password)


def generate_api_key() -> str:
    """Generate a new API key.

    Returns:
        str: Secure API key with tldr_sk_ prefix
    """
    # Generate 32 bytes of random data
    random_bytes = secrets.token_bytes(32)

    # Create a hash to ensure consistent length
    key_hash = hashlib.sha256(random_bytes).hexdigest()[:32]

    return f"tldr_sk_{key_hash}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage.

    Args:
        api_key: Plain text API key

    Returns:
        str: Hashed API key
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_access_token(user_id: int, email: str, scopes: List[str] = None) -> str:
    """Create a JWT access token.

    Args:
        user_id: User ID
        email: User email
        scopes: List of permission scopes

    Returns:
        str: JWT token
    """
    if scopes is None:
        scopes = []

    payload = {
        "user_id": user_id,
        "email": email,
        "scopes": scopes,
        "exp": datetime.now(timezone.utc).timestamp()
        + (settings.jwt_expiration_minutes * 60),
        "iat": datetime.now(timezone.utc).timestamp(),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token.

    Args:
        token: JWT token

    Returns:
        TokenData: Decoded token data

    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])

        user_id = payload.get("user_id")
        email = payload.get("email")
        scopes = payload.get("scopes", [])

        if user_id is None or email is None:
            raise AuthenticationError("Invalid token payload")

        return TokenData(user_id=user_id, email=email, scopes=scopes)

    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current user from JWT token.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        User: Current user

    Raises:
        AuthenticationError: If authentication fails
    """
    token_data = verify_token(credentials.credentials)

    # Get user from database
    result = await db.execute(select(User).where(User.id == token_data.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise AuthenticationError("User not found")

    return user


async def get_current_user_from_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db),
) -> tuple[User, APIKey]:
    """Get current user from API key.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        tuple: (User, APIKey) objects

    Raises:
        AuthenticationError: If authentication fails
    """
    api_key = credentials.credentials

    if not api_key.startswith("tldr_sk_"):
        raise AuthenticationError("Invalid API key format")

    # Hash the API key for lookup
    hashed_key = hash_api_key(api_key)

    # Get API key from database
    result = await db.execute(select(APIKey).where(APIKey.key == hashed_key))
    api_key_obj = result.scalar_one_or_none()

    if api_key_obj is None:
        raise AuthenticationError("Invalid API key")

    if not api_key_obj.active:
        raise AuthenticationError("API key is inactive")

    if api_key_obj.is_expired():
        raise AuthenticationError("API key has expired")

    # Update last used timestamp
    api_key_obj.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    # Get user
    result = await db.execute(select(User).where(User.id == api_key_obj.user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise AuthenticationError("User not found")

    return user, api_key_obj


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current user from either JWT token or API key.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

    Returns:
        User: Current user

    Raises:
        AuthenticationError: If authentication fails
    """
    token = credentials.credentials

    # Try JWT token first
    if not token.startswith("tldr_sk_"):
        try:
            return await get_current_user_from_token(credentials, db)
        except AuthenticationError:
            pass

    # Try API key
    try:
        user, _ = await get_current_user_from_api_key(credentials, db)
        return user
    except AuthenticationError:
        pass

    raise AuthenticationError("Invalid authentication credentials")


def require_scopes(required_scopes: List[str]):
    """Dependency factory for requiring specific scopes.

    Args:
        required_scopes: List of required scopes

    Returns:
        Dependency function
    """

    async def check_scopes(
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        token = credentials.credentials

        # Check if it's an API key
        if token.startswith("tldr_sk_"):
            user, api_key = await get_current_user_from_api_key(credentials, db)

            # Check if API key has required scopes
            for scope in required_scopes:
                if not api_key.has_scope(scope):
                    raise PermissionError(f"Missing required scope: {scope}")

            return user

        # Handle JWT token
        token_data = verify_token(token)

        # Check if token has required scopes
        for scope in required_scopes:
            if scope not in token_data.scopes:
                raise PermissionError(f"Missing required scope: {scope}")

        # Get user from database
        result = await db.execute(select(User).where(User.id == token_data.user_id))
        user = result.scalar_one_or_none()

        if user is None:
            raise AuthenticationError("User not found")

        return user

    return check_scopes


def require_admin():
    """Dependency for requiring admin permissions."""
    return require_scopes(["admin"])


async def get_optional_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Optional[User]: Current user or None
    """

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    try:
        token = auth_header.split(" ")[1]
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        return await get_current_user(credentials, db)
    except (AuthenticationError, IndexError):
        return None
