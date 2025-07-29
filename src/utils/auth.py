"""Authentication utilities for the TL;DR Highlight API.

This module provides utilities for:
- API key generation and verification
- Password hashing and verification
- JWT token creation and validation
- Security helper functions
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
from jose import jwt, JWTError

from src.core.config import settings


def generate_api_key(length: int = 32) -> str:
    """Generate a cryptographically secure API key.

    Args:
        length: Length of the API key in bytes (default 32)

    Returns:
        str: Hexadecimal API key string

    Example:
        >>> key = generate_api_key(32)
        >>> len(key)
        64
    """
    return secrets.token_hex(length)


def hash_api_key(api_key: str) -> str:
    """Hash an API key using bcrypt.

    Args:
        api_key: The API key to hash

    Returns:
        str: The hashed API key

    Example:
        >>> hashed = hash_api_key("my_api_key")
        >>> hashed.startswith("$2b$")
        True
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash.

    Args:
        api_key: The API key to verify
        hashed_key: The hashed API key to verify against

    Returns:
        bool: True if the API key is valid, False otherwise

    Example:
        >>> hashed = hash_api_key("my_api_key")
        >>> verify_api_key("my_api_key", hashed)
        True
        >>> verify_api_key("wrong_key", hashed)
        False
    """
    try:
        return bcrypt.checkpw(api_key.encode("utf-8"), hashed_key.encode("utf-8"))
    except Exception:
        return False


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: The password to hash

    Returns:
        str: The hashed password

    Example:
        >>> hashed = hash_password("my_password")
        >>> hashed.startswith("$2b$")
        True
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        password: The password to verify
        hashed_password: The hashed password to verify against

    Returns:
        bool: True if the password is valid, False otherwise

    Example:
        >>> hashed = hash_password("my_password")
        >>> verify_password("my_password", hashed)
        True
        >>> verify_password("wrong_password", hashed)
        False
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_jwt_token(
    payload: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT token with the given payload.

    Args:
        payload: The payload to encode in the token
        expires_delta: Optional expiration time delta from now

    Returns:
        str: The JWT token

    Example:
        >>> token = create_jwt_token({"user_id": 123})
        >>> len(token.split("."))
        3
    """
    to_encode = payload.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_expiration_minutes
        )

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def verify_jwt_token(token: str) -> bool:
    """Verify a JWT token.

    Args:
        token: The JWT token to verify

    Returns:
        bool: True if the token is valid and not expired, False otherwise

    Example:
        >>> token = create_jwt_token({"user_id": 123})
        >>> verify_jwt_token(token)
        True
        >>> verify_jwt_token("invalid.token")
        False
    """
    try:
        jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return True
    except JWTError:
        return False


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode a JWT token and return the payload.

    Args:
        token: The JWT token to decode

    Returns:
        dict: The decoded payload if valid, None otherwise

    Example:
        >>> token = create_jwt_token({"user_id": 123})
        >>> payload = decode_jwt_token(token)
        >>> payload["user_id"]
        123
    """
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None


def generate_secure_random_string(length: int = 32) -> str:
    """Generate a cryptographically secure random string.

    Args:
        length: Length of the string in characters

    Returns:
        str: Random URL-safe string

    Example:
        >>> random_str = generate_secure_random_string(16)
        >>> len(random_str)
        16
    """
    return secrets.token_urlsafe(length)[:length]


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        bool: True if strings are equal, False otherwise

    Example:
        >>> constant_time_compare("hello", "hello")
        True
        >>> constant_time_compare("hello", "world")
        False
    """
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
