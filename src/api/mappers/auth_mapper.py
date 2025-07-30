"""Mapper for authentication DTOs."""

from typing import List
from datetime import datetime
from dataclasses import dataclass
from src.api.schemas.auth import (
    APIKeyResponse,
    APIKeyCreateResponse,
    LoginRequest as LoginRequestDTO,
    LoginResponse,
)
from src.api.schemas.users import UserRegistrationRequest
from src.application.use_cases.authentication import (
    RegisterRequest,
    LoginRequest,
    RegisterResult,
    LoginResult,
)
from src.domain.entities.api_key import APIKey, APIKeyScope


@dataclass
class RegisterMapper:
    """Maps registration DTOs to domain objects."""

    def to_domain(self, dto: UserRegistrationRequest) -> RegisterRequest:
        """Convert registration DTO to domain request."""
        return RegisterRequest(
            email=dto.email,
            password=dto.password,
            company_name=dto.company_name,
            organization_name=None,  # UserRegistrationRequest doesn't have organization_name
        )

    def to_dto(self, result: RegisterResult, api_key: str) -> APIKeyCreateResponse:
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
            is_expired=False,
        )


@dataclass
class LoginMapper:
    """Maps login DTOs to domain objects."""

    def to_domain(self, dto: LoginRequestDTO) -> LoginRequest:
        """Convert login DTO to domain request."""
        return LoginRequest(
            email=dto.email,
            password=dto.password,
            create_api_key=True,
            api_key_name="Login API Key",
        )

    def to_dto(self, result: LoginResult, token: str) -> LoginResponse:
        """Convert login result to response DTO."""
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=3600,  # 1 hour
        )


@dataclass
class APIKeyMapper:
    """Maps API key DTOs to domain objects."""

    def to_domain_scopes(self, scopes: List[str]) -> List[APIKeyScope]:
        """Convert string scopes to domain scope enums."""
        domain_scopes = []
        for scope in scopes:
            try:
                # Map string scope to enum
                if scope == "admin":
                    domain_scopes.append(APIKeyScope.ADMIN)
                else:
                    # Convert format like "streams:read" to "STREAMS_READ"
                    enum_name = scope.upper().replace(":", "_")
                    domain_scopes.append(APIKeyScope[enum_name])
            except (KeyError, ValueError):
                # Skip invalid scopes
                pass
        return domain_scopes

    def to_dto(self, api_key: APIKey) -> APIKeyResponse:
        """Convert domain API key to response DTO."""
        # Convert enum scopes back to strings
        scopes = []
        for scope in api_key.scopes:
            if scope == APIKeyScope.ADMIN:
                scopes.append("admin")
            else:
                # Convert format like "STREAMS_READ" to "streams:read"
                scope_str = scope.value.lower().replace("_", ":")
                scopes.append(scope_str)

        return APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            scopes=scopes,
            active=api_key.is_active,
            created_at=api_key.created_at.value,
            expires_at=api_key.expires_at.value if api_key.expires_at else None,
            last_used_at=api_key.last_used_at.value if api_key.last_used_at else None,
            is_expired=api_key.is_expired,
        )

    def to_create_response_dto(self, api_key: APIKey, key: str) -> APIKeyCreateResponse:
        """Convert domain API key to create response DTO with actual key."""
        # Convert enum scopes back to strings
        scopes = []
        for scope in api_key.scopes:
            if scope == APIKeyScope.ADMIN:
                scopes.append("admin")
            else:
                # Convert format like "STREAMS_READ" to "streams:read"
                scope_str = scope.value.lower().replace("_", ":")
                scopes.append(scope_str)

        return APIKeyCreateResponse(
            id=api_key.id,
            name=api_key.name,
            key=key,
            scopes=scopes,
            active=api_key.is_active,
            created_at=api_key.created_at.value,
            expires_at=api_key.expires_at.value if api_key.expires_at else None,
            last_used_at=api_key.last_used_at.value if api_key.last_used_at else None,
            is_expired=api_key.is_expired,
        )
