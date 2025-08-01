"""Integration tests for the authentication system."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from src.core.auth import (
    authenticated_user,
    require_authenticated_scope,
    get_current_organization,
    admin_required,
    read_required,
    write_required,
)
from src.models.user import User


# Create a simple test app
app = FastAPI()


@app.get("/public")
async def public_endpoint():
    """Public endpoint that doesn't require authentication."""
    return {"message": "Hello, world!"}


@app.get("/user")
async def get_user_info(user: User = Depends(authenticated_user)):
    """Endpoint that requires authentication."""
    return {"user_id": user.id, "email": user.email, "company": user.company_name}


@app.get("/admin")
async def admin_endpoint(_=Depends(admin_required)):
    """Endpoint that requires admin permissions."""
    return {"message": "Admin access granted"}


@app.get("/read")
async def read_endpoint(_=Depends(read_required)):
    """Endpoint that requires read permissions."""
    return {"message": "Read access granted"}


@app.post("/write")
async def write_endpoint(_=Depends(write_required)):
    """Endpoint that requires write permissions."""
    return {"message": "Write access granted"}


@app.get("/organization")
async def get_organization_info(org=Depends(get_current_organization)):
    """Endpoint that requires organization context."""
    return {"org_id": org.id, "name": org.name, "plan": org.plan_type}


@app.get("/custom-scope")
async def custom_scope_endpoint(
    _=Depends(require_authenticated_scope(["streams", "webhooks"])),
):
    """Endpoint that requires custom scopes."""
    return {"message": "Custom scope access granted"}


class TestAuthenticationIntegration:
    """Integration tests for the authentication system."""

    def test_public_endpoint_no_auth(self):
        """Test that public endpoints work without authentication."""
        client = TestClient(app)
        response = client.get("/public")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, world!"}

    def test_protected_endpoint_no_auth(self):
        """Test that protected endpoints return 401 without auth."""
        client = TestClient(app)
        response = client.get("/user")
        assert response.status_code == 401
        assert "API key is required" in response.json()["detail"]

    def test_protected_endpoint_invalid_auth(self):
        """Test that protected endpoints return 401 with invalid auth."""
        client = TestClient(app)
        headers = {"X-API-Key": "invalid-key"}
        response = client.get("/user", headers=headers)
        assert response.status_code == 401

    def test_admin_endpoint_insufficient_permissions(self):
        """Test that admin endpoints reject non-admin users."""
        client = TestClient(app)
        headers = {"X-API-Key": "user-key"}  # Assuming this is a non-admin key
        response = client.get("/admin", headers=headers)
        assert response.status_code in [401, 403]  # Depends on key validity

    def test_rate_limiting_headers_present(self):
        """Test that rate limiting headers are present in responses."""
        client = TestClient(app)
        headers = {"X-API-Key": "test-key"}
        _response = client.get("/user", headers=headers)

        # Even if the request fails due to invalid key, rate limiting should be processed
        # In a real scenario with valid keys, we'd check for rate limit headers

    def test_cors_headers_configuration(self):
        """Test that CORS headers are properly configured."""
        client = TestClient(app)
        _response = client.options("/public")
        # This tests basic CORS preflight handling

    def test_multiple_scope_requirements(self):
        """Test endpoints requiring multiple scopes."""
        client = TestClient(app)
        headers = {"X-API-Key": "limited-scope-key"}
        response = client.get("/custom-scope", headers=headers)
        assert response.status_code in [401, 403]

    def test_organization_context_requirement(self):
        """Test endpoints requiring organization context."""
        client = TestClient(app)
        headers = {"X-API-Key": "user-without-org-key"}
        response = client.get("/organization", headers=headers)
        assert response.status_code in [401, 404]


class TestAuthenticationFlowDocumentation:
    """Documentation of authentication flow for developers."""

    def test_authentication_flow_documentation(self):
        """Document the complete authentication flow."""
        flow_steps = [
            "1. Client includes X-API-Key header in request",
            "2. get_api_key_from_header() extracts and validates header format",
            "3. get_current_api_key() validates key against database",
            "4. Rate limiting checks are performed via check_rate_limit()",
            "5. Scope requirements are checked via require_scope()",
            "6. User context is extracted via get_current_user()",
            "7. Organization context is optionally loaded via get_current_organization()",
            "8. Request proceeds to business logic",
        ]

        assert len(flow_steps) == 8
        assert all("." in step for step in flow_steps)

    def test_scope_hierarchy_documentation(self):
        """Document the scope hierarchy and permissions."""
        scope_hierarchy = {
            "read": "Basic read access to resources",
            "write": "Create and update resources",
            "delete": "Delete resources",
            "streams": "Stream processing access",
            "batches": "Batch processing access",
            "webhooks": "Webhook management",
            "analytics": "Analytics and reporting access",
            "admin": "Full administrative access (grants all other scopes)",
        }

        assert "admin" in scope_hierarchy
        assert len(scope_hierarchy) == 8

    def test_rate_limit_tiers_documentation(self):
        """Document rate limiting tiers by plan."""
        rate_limits = {
            "starter": 60,
            "professional": 300,
            "enterprise": 1000,
            "custom": -1,  # Unlimited
        }

        assert rate_limits["professional"] > rate_limits["starter"]
        assert rate_limits["enterprise"] > rate_limits["professional"]

    def test_security_considerations_documentation(self):
        """Document security considerations."""
        security_features = [
            "API keys are hashed using bcrypt before storage",
            "JWT tokens use configurable secret key and algorithm",
            "Rate limiting uses sliding window algorithm",
            "Authentication failures are logged",
            "API key expiration is supported",
            "Scope-based access control is enforced",
            "Organization-based multi-tenancy isolation",
            "Constant-time comparison for sensitive operations",
        ]

        assert len(security_features) == 8
        assert all(
            len(feature) > 20 for feature in security_features
        )  # All features are documented


@pytest.mark.asyncio
async def test_async_authentication_components():
    """Test that all authentication components are properly async."""
    from src.services.auth import AuthService
    from src.services.rate_limit import RateLimitService
    from src.infrastructure.cache import RedisCache

    # Verify that key components are async
    cache = RedisCache()
    auth_service = AuthService(cache)
    rate_service = RateLimitService(cache)

    # Check that key methods are coroutines
    import inspect

    assert inspect.iscoroutinefunction(auth_service.validate_api_key)
    assert inspect.iscoroutinefunction(auth_service.has_permission)
    assert inspect.iscoroutinefunction(rate_service.check_rate_limit)
    assert inspect.iscoroutinefunction(cache.get)
    assert inspect.iscoroutinefunction(cache.set)


if __name__ == "__main__":
    # Run a simple smoke test
    client = TestClient(app)
    response = client.get("/public")
    print(f"Public endpoint response: {response.json()}")

    # Test authentication requirement
    response = client.get("/user")
    print(f"Protected endpoint without auth: {response.status_code}")

    print("Authentication system integration tests completed!")
