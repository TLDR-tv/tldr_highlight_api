"""Authentication utilities for the API."""

from datetime import datetime, timezone, timedelta
from typing import List

import jwt
from src.infrastructure.config import settings


def create_access_token(user_id: int, email: str, scopes: List[str]) -> str:
    """Create a JWT access token.

    Args:
        user_id: User ID
        email: User email
        scopes: List of permission scopes

    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "scopes": scopes,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
