"""Functional mapper utilities for API DTOs.

This module provides functional mapping utilities as a more Pythonic
alternative to class-based mappers, following the principle of
"functions are first-class citizens" in Python.
"""

from typing import List, Callable, TypeVar, Dict, Any
from datetime import datetime

from src.api.schemas.auth import (
    APIKeyResponse,
    APIKeyCreateResponse,
    LoginResponse
)
from src.api.schemas.users import UserRegistrationRequest
from src.application.use_cases.authentication import (
    RegisterRequest,
    LoginRequest,
    RegisterResult,
    LoginResult
)
from src.domain.entities.api_key import APIKey, APIKeyScope


# Type variable for generic mapping
T = TypeVar('T')
U = TypeVar('U')


def create_mapper(mapping_func: Callable[[T], U]) -> Callable[[T], U]:
    """Create a mapper function with optional preprocessing.
    
    This is a higher-order function that can wrap mapping logic
    with additional features like validation or logging.
    """
    def mapper(source: T) -> U:
        return mapping_func(source)
    return mapper


# Registration mappings
def registration_to_domain(dto: UserRegistrationRequest) -> RegisterRequest:
    """Convert registration DTO to domain request."""
    return RegisterRequest(
        email=dto.email,
        password=dto.password,
        company_name=dto.company_name,
        organization_name=None
    )


def registration_to_response(result: RegisterResult, api_key: str) -> APIKeyCreateResponse:
    """Convert registration result to response DTO."""
    return APIKeyCreateResponse(
        id=result.user_id,
        name="Default API Key",
        key=api_key,
        scopes=["streams:read", "streams:write", "highlights:read"],
        active=True,
        created_at=datetime.utcnow(),
        expires_at=None,
        last_used_at=None,
        is_expired=False
    )


# Login mappings
def login_to_domain(dto) -> LoginRequest:
    """Convert login DTO to domain request."""
    return LoginRequest(
        email=dto.email,
        password=dto.password,
        create_api_key=True,
        api_key_name="Login API Key"
    )


def login_to_response(result: LoginResult, token: str) -> LoginResponse:
    """Convert login result to response DTO."""
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=3600
    )


# API Key mappings
def scopes_to_domain(scopes: List[str]) -> List[APIKeyScope]:
    """Convert string scopes to domain scope enums."""
    scope_mapping = {
        "admin": APIKeyScope.ADMIN,
        # Add more mappings as needed
    }
    
    domain_scopes = []
    for scope in scopes:
        if scope in scope_mapping:
            domain_scopes.append(scope_mapping[scope])
        else:
            # Convert format like "streams:read" to "STREAMS_READ"
            try:
                enum_name = scope.upper().replace(":", "_")
                domain_scopes.append(APIKeyScope[enum_name])
            except (KeyError, ValueError):
                pass  # Skip invalid scopes
    
    return domain_scopes


def scopes_to_strings(scopes: List[APIKeyScope]) -> List[str]:
    """Convert domain scope enums to string representation."""
    return [
        "admin" if scope == APIKeyScope.ADMIN
        else scope.value.lower().replace("_", ":")
        for scope in scopes
    ]


def api_key_to_response(api_key: APIKey) -> APIKeyResponse:
    """Convert domain API key to response DTO."""
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        scopes=scopes_to_strings(api_key.scopes),
        active=api_key.is_active,
        created_at=api_key.created_at.value,
        expires_at=api_key.expires_at.value if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.value if api_key.last_used_at else None,
        is_expired=api_key.is_expired
    )


def api_key_to_create_response(api_key: APIKey, key: str) -> APIKeyCreateResponse:
    """Convert domain API key to create response DTO with actual key."""
    return APIKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        key=key,
        scopes=scopes_to_strings(api_key.scopes),
        active=api_key.is_active,
        created_at=api_key.created_at.value,
        expires_at=api_key.expires_at.value if api_key.expires_at else None,
        last_used_at=api_key.last_used_at.value if api_key.last_used_at else None,
        is_expired=api_key.is_expired
    )


# Batch mapping utility
def map_collection(items: List[T], mapper: Callable[[T], U]) -> List[U]:
    """Map a collection of items using the provided mapper function."""
    return [mapper(item) for item in items]


# Dictionary-based mapping for flexible conversions
def dict_to_dto(data: Dict[str, Any], dto_class: type[T]) -> T:
    """Convert a dictionary to a DTO instance.
    
    This is useful for flexible mappings where the source
    data might come from various sources.
    """
    return dto_class(**data)


# Partial mapping for commonly used transformations
def create_partial_mapper(mapper: Callable[..., T], **kwargs) -> Callable[..., T]:
    """Create a partial mapper with pre-filled arguments.
    
    This allows creating specialized mappers from generic ones.
    """
    def partial_mapper(**additional_kwargs):
        return mapper(**{**kwargs, **additional_kwargs})
    return partial_mapper