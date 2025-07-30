"""Security infrastructure for the TL;DR Highlight API.

This module provides security utilities and implementations
as infrastructure concerns, separate from business logic.
"""

from .auth_utils import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    hash_password,
    verify_password,
    create_jwt_token,
    verify_jwt_token,
    decode_jwt_token,
    generate_secure_random_string,
    constant_time_compare,
)
from .api_key_validator import APIKeyValidator
from .jwt_utils import create_access_token

__all__ = [
    # Auth utilities
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "hash_password",
    "verify_password",
    "create_jwt_token",
    "verify_jwt_token",
    "decode_jwt_token",
    "generate_secure_random_string",
    "constant_time_compare",
    # JWT utilities
    "create_access_token",
    # Validators
    "APIKeyValidator",
]
