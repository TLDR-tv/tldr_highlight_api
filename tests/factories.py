"""Test data factories."""

from datetime import datetime, UTC
from typing import Optional
from uuid import uuid4

import factory
from factory import fuzzy
from faker import Faker

from src.domain.models.organization import Organization
from src.domain.models.user import User, UserRole
from src.domain.models.api_key import APIKey, APIScopes
from src.infrastructure.security.password_service import PasswordService

fake = Faker()
password_service = PasswordService()


class OrganizationFactory(factory.Factory):
    """Factory for creating test organizations."""
    
    class Meta:
        model = Organization
    
    id = factory.LazyFunction(uuid4)
    name = factory.LazyAttribute(lambda _: fake.company())
    slug = factory.LazyAttribute(lambda obj: Organization._generate_slug(obj.name))
    is_active = True
    webhook_url = factory.LazyAttribute(
        lambda _: fake.url(schemes=["https"]) if fake.boolean(chance_of_getting_true=50) else None
    )
    webhook_secret = factory.LazyAttribute(lambda _: fake.sha256())
    wake_words = factory.LazyFunction(set)
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
    
    # Usage tracking
    total_streams_processed = 0
    total_highlights_generated = 0
    total_processing_seconds = 0.0


class UserFactory(factory.Factory):
    """Factory for creating test users."""
    
    class Meta:
        model = User
    
    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    email = factory.LazyAttribute(lambda _: fake.email().lower())
    name = factory.LazyAttribute(lambda _: fake.name())
    hashed_password = factory.LazyAttribute(
        lambda _: password_service.hash_password(fake.password(length=12, special_chars=True))
    )
    role = fuzzy.FuzzyChoice([UserRole.MEMBER, UserRole.ADMIN])
    is_active = True
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
    last_login_at = None


class APIKeyFactory(factory.Factory):
    """Factory for creating test API keys."""
    
    class Meta:
        model = APIKey
    
    id = factory.LazyFunction(uuid4)
    organization_id = factory.LazyFunction(uuid4)
    name = factory.LazyAttribute(lambda _: f"Test Key - {fake.word()}")
    key_hash = factory.LazyAttribute(lambda _: fake.sha256())
    prefix = factory.LazyAttribute(lambda _: fake.lexify("????????"))
    scopes = factory.LazyFunction(
        lambda: {APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_READ, APIScopes.ORG_READ}
    )
    
    # Usage tracking
    last_used_at = None
    usage_count = 0
    
    # Lifecycle
    is_active = True
    expires_at = None
    revoked_at = None
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))


def create_test_organization(**kwargs) -> Organization:
    """Create a test organization with defaults."""
    return OrganizationFactory(**kwargs)


def create_test_user(
    organization_id: Optional[uuid4] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    role: UserRole = UserRole.MEMBER,
    **kwargs
) -> tuple[User, str]:
    """Create a test user with password.
    
    Returns:
        Tuple of (user, plain_password)
    """
    if not organization_id:
        organization_id = uuid4()
    
    if not email:
        email = fake.email().lower()
    
    if not password:
        password = fake.password(length=12, special_chars=True)
    
    user = UserFactory(
        organization_id=organization_id,
        email=email,
        hashed_password=password_service.hash_password(password),
        role=role,
        **kwargs
    )
    
    return user, password


def create_test_api_key(
    organization_id: Optional[uuid4] = None,
    scopes: Optional[set[str]] = None,
    **kwargs
) -> tuple[APIKey, str]:
    """Create a test API key.
    
    Returns:
        Tuple of (api_key, raw_key)
    """
    if not organization_id:
        organization_id = uuid4()
    
    if not scopes:
        scopes = {APIScopes.STREAMS_READ, APIScopes.HIGHLIGHTS_READ, APIScopes.ORG_READ}
    
    # Generate a properly formatted API key
    prefix = f"tldr_{fake.lexify('????????')}"
    key_part = fake.lexify("?" * 24)
    raw_key = f"{prefix}_{key_part}"
    
    api_key = APIKeyFactory(
        organization_id=organization_id,
        scopes=scopes,
        key_hash=password_service.hash_password(raw_key),
        prefix=prefix,
        **kwargs
    )
    
    return api_key, raw_key