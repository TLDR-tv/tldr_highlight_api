"""Security service dependencies for FastAPI."""

from src.domain.services.security_services import (
    PasswordHashingService,
    APIKeyHashingService,
    BcryptPasswordHashingService,
    SHA256APIKeyHashingService,
)


async def get_password_hashing_service() -> PasswordHashingService:
    """Get password hashing service instance."""
    return BcryptPasswordHashingService()


async def get_api_key_hashing_service() -> APIKeyHashingService:
    """Get API key hashing service instance."""
    return SHA256APIKeyHashingService()