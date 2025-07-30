"""Authentication use cases."""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import bcrypt
from datetime import datetime, timedelta, timezone

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.user import User
from src.domain.entities.api_key import APIKey, APIKeyScope
from src.domain.value_objects.email import Email
from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.timestamp import Timestamp
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.services.organization_management_service import OrganizationManagementService
from src.domain.exceptions import DuplicateEntityError, EntityNotFoundError


@dataclass
class RegisterRequest:
    """Request for user registration."""
    email: str
    password: str
    company_name: str
    organization_name: Optional[str] = None


@dataclass
class RegisterResult(UseCaseResult):
    """Result of user registration."""
    user_id: Optional[int] = None
    api_key: Optional[str] = None
    organization_id: Optional[int] = None


@dataclass
class LoginRequest:
    """Request for user login."""
    email: str
    password: str
    create_api_key: bool = False
    api_key_name: Optional[str] = None


@dataclass
class LoginResult(UseCaseResult):
    """Result of user login."""
    user_id: Optional[int] = None
    api_key: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class ValidateAPIKeyRequest:
    """Request for API key validation."""
    api_key: str
    required_scopes: Optional[List[APIKeyScope]] = None


@dataclass
class ValidateAPIKeyResult(UseCaseResult):
    """Result of API key validation."""
    user_id: Optional[int] = None
    organization_id: Optional[int] = None
    scopes: Optional[List[str]] = None
    rate_limit: Optional[int] = None


@dataclass
class CreateAPIKeyRequest:
    """Request for creating a new API key."""
    user_id: int
    name: str
    scopes: List[APIKeyScope]
    expires_at: Optional[datetime] = None


@dataclass
class CreateAPIKeyResult(UseCaseResult):
    """Result of API key creation."""
    api_key: Optional[APIKey] = None
    key: Optional[str] = None  # The actual key string (shown only once)


@dataclass
class ListAPIKeysResult(UseCaseResult):
    """Result of listing API keys."""
    api_keys: List[APIKey] = None


@dataclass
class RevokeAPIKeyResult(UseCaseResult):
    """Result of revoking an API key."""
    pass


@dataclass
class RotateAPIKeyResult(UseCaseResult):
    """Result of rotating an API key."""
    api_key: Optional[APIKey] = None
    key: Optional[str] = None  # The new key string


class AuthenticationUseCase(UseCase[RegisterRequest, RegisterResult]):
    """Use case for user authentication operations."""
    
    def __init__(
        self,
        user_repo: UserRepository,
        api_key_repo: APIKeyRepository,
        org_repo: OrganizationRepository,
        org_service: OrganizationManagementService
    ):
        """Initialize authentication use case.
        
        Args:
            user_repo: Repository for user operations
            api_key_repo: Repository for API key operations
            org_repo: Repository for organization operations
            org_service: Service for organization management
        """
        self.user_repo = user_repo
        self.api_key_repo = api_key_repo
        self.org_repo = org_repo
        self.org_service = org_service
    
    async def register(self, request: RegisterRequest) -> RegisterResult:
        """Register a new user.
        
        Args:
            request: Registration request
            
        Returns:
            Registration result
        """
        try:
            # Validate email
            try:
                email = Email(request.email)
            except ValueError as e:
                return RegisterResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[str(e)]
                )
            
            # Validate company name
            try:
                company_name = CompanyName(request.company_name)
            except ValueError as e:
                return RegisterResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[str(e)]
                )
            
            # Check if user already exists
            existing = await self.user_repo.get_by_email(email)
            if existing:
                return RegisterResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=["User with this email already exists"]
                )
            
            # Hash password
            password_hash = bcrypt.hashpw(
                request.password.encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")
            
            # Create user
            user = User(
                id=None,
                email=email,
                company_name=company_name,
                password_hash=password_hash,
                is_active=True,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now()
            )
            
            # Save user
            saved_user = await self.user_repo.save(user)
            
            # Create organization if requested
            organization_id = None
            if request.organization_name:
                org = await self.org_service.create_organization(
                    owner_id=saved_user.id,
                    name=request.organization_name
                )
                organization_id = org.id
            
            # Create default API key
            api_key = await self._create_api_key(
                user_id=saved_user.id,
                name="Default API Key",
                scopes=[APIKeyScope.STREAMS_READ, APIKeyScope.STREAMS_WRITE]
            )
            
            return RegisterResult(
                status=ResultStatus.SUCCESS,
                user_id=saved_user.id,
                api_key=api_key.key,
                organization_id=organization_id,
                message="User registered successfully"
            )
            
        except DuplicateEntityError as e:
            return RegisterResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=[str(e)]
            )
        except Exception as e:
            return RegisterResult(
                status=ResultStatus.FAILURE,
                errors=[f"Registration failed: {str(e)}"]
            )
    
    async def login(self, request: LoginRequest) -> LoginResult:
        """Authenticate user and optionally create API key.
        
        Args:
            request: Login request
            
        Returns:
            Login result
        """
        try:
            # Validate email
            try:
                email = Email(request.email)
            except ValueError as e:
                return LoginResult(
                    status=ResultStatus.VALIDATION_ERROR,
                    errors=[str(e)]
                )
            
            # Get user
            user = await self.user_repo.get_by_email(email)
            if not user:
                return LoginResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Invalid email or password"]
                )
            
            # Verify password
            if not bcrypt.checkpw(
                request.password.encode("utf-8"),
                user.password_hash.encode("utf-8")
            ):
                return LoginResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Invalid email or password"]
                )
            
            # Check if user is active
            if not user.is_active:
                return LoginResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["User account is disabled"]
                )
            
            # Create API key if requested
            api_key = None
            expires_at = None
            if request.create_api_key:
                key_name = request.api_key_name or f"Session key {datetime.utcnow().isoformat()}"
                api_key_entity = await self._create_api_key(
                    user_id=user.id,
                    name=key_name,
                    scopes=[APIKeyScope.STREAMS_READ, APIKeyScope.STREAMS_WRITE],
                    expires_in_days=30
                )
                api_key = api_key_entity.key
                expires_at = api_key_entity.expires_at.value if api_key_entity.expires_at else None
            
            return LoginResult(
                status=ResultStatus.SUCCESS,
                user_id=user.id,
                api_key=api_key,
                expires_at=expires_at,
                message="Login successful"
            )
            
        except Exception as e:
            return LoginResult(
                status=ResultStatus.FAILURE,
                errors=[f"Login failed: {str(e)}"]
            )
    
    async def validate_api_key(self, request: ValidateAPIKeyRequest) -> ValidateAPIKeyResult:
        """Validate an API key and check scopes.
        
        Args:
            request: API key validation request
            
        Returns:
            Validation result
        """
        try:
            # Get API key
            # Hash the API key for lookup
            import hashlib
            key_hash = hashlib.sha256(request.api_key.encode("utf-8")).hexdigest()
            
            # Get API key by its hash
            api_key = await self.api_key_repo.get_by_key_hash(key_hash)
            if not api_key:
                return ValidateAPIKeyResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["Invalid API key"]
                )
            
            # Check if key is active
            if not api_key.is_valid:
                return ValidateAPIKeyResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["API key is expired or revoked"]
                )
            
            # Check required scopes
            if request.required_scopes:
                missing_scopes = [
                    scope for scope in request.required_scopes
                    if scope not in api_key.scopes
                ]
                if missing_scopes:
                    return ValidateAPIKeyResult(
                        status=ResultStatus.UNAUTHORIZED,
                        errors=[f"Missing required scopes: {', '.join(s.value for s in missing_scopes)}"]
                    )
            
            # Get user's organization
            orgs = await self.org_repo.get_by_owner(api_key.user_id)
            organization_id = orgs[0].id if orgs else None
            
            # Get rate limit
            rate_limit = 60  # Default
            if organization_id:
                org = orgs[0]
                rate_limit = org.plan_limits.api_rate_limit_per_minute
            
            # Update last used timestamp
            api_key.record_usage()
            await self.api_key_repo.save(api_key)
            
            return ValidateAPIKeyResult(
                status=ResultStatus.SUCCESS,
                user_id=api_key.user_id,
                organization_id=organization_id,
                scopes=[scope.value for scope in api_key.scopes],
                rate_limit=rate_limit,
                message="API key is valid"
            )
            
        except Exception as e:
            return ValidateAPIKeyResult(
                status=ResultStatus.FAILURE,
                errors=[f"Validation failed: {str(e)}"]
            )
    
    async def execute(self, request: RegisterRequest) -> RegisterResult:
        """Execute registration (default use case method).
        
        Args:
            request: Registration request
            
        Returns:
            Registration result
        """
        return await self.register(request)
    
    # Private helper methods
    
    async def _create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: List[APIKeyScope],
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """Create a new API key for a user."""
        import secrets
        
        # Generate secure key
        key = f"sk_live_{secrets.token_urlsafe(32)}"
        
        # Calculate expiry
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = Timestamp(datetime.now(timezone.utc) + timedelta(days=expires_in_days))
        
        # Hash the key for storage
        import hashlib
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        
        # Create API key
        api_key = APIKey(
            id=None,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            expires_at=expires_at,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Save and return
        return await self.api_key_repo.save(api_key)
    
    async def create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: List[APIKeyScope],
        expires_at: Optional[datetime] = None
    ) -> CreateAPIKeyResult:
        """Create a new API key for a user."""
        try:
            # Verify user exists
            user = await self.user_repo.get(user_id)
            if not user:
                return CreateAPIKeyResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["User not found"]
                )
            
            # Generate secure key
            import secrets
            key = f"tldr_sk_{secrets.token_urlsafe(32)}"
            
            # Hash the key for storage
            key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
            
            # Create API key entity
            api_key = APIKey(
                id=None,
                user_id=user_id,
                key_hash=key_hash,
                name=name,
                scopes=scopes,
                expires_at=Timestamp(expires_at) if expires_at else None,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now()
            )
            
            # Save
            saved_key = await self.api_key_repo.save(api_key)
            
            return CreateAPIKeyResult(
                status=ResultStatus.SUCCESS,
                api_key=saved_key,
                key=key,  # Return the actual key
                message="API key created successfully"
            )
            
        except Exception as e:
            return CreateAPIKeyResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to create API key: {str(e)}"]
            )
    
    async def list_api_keys(self, user_id: int) -> ListAPIKeysResult:
        """List all API keys for a user."""
        try:
            api_keys = await self.api_key_repo.get_by_user(user_id)
            
            # Filter out expired/revoked keys unless specifically requested
            active_keys = [key for key in api_keys if key.is_valid]
            
            return ListAPIKeysResult(
                status=ResultStatus.SUCCESS,
                api_keys=active_keys,
                message=f"Found {len(active_keys)} active API keys"
            )
            
        except Exception as e:
            return ListAPIKeysResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to list API keys: {str(e)}"]
            )
    
    async def revoke_api_key(self, user_id: int, api_key_id: int) -> RevokeAPIKeyResult:
        """Revoke an API key."""
        try:
            # Get the API key
            api_key = await self.api_key_repo.get(api_key_id)
            
            if not api_key:
                return RevokeAPIKeyResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["API key not found"]
                )
            
            # Verify ownership
            if api_key.user_id != user_id:
                return RevokeAPIKeyResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You do not have permission to revoke this API key"]
                )
            
            # Revoke the key
            api_key.revoke()
            await self.api_key_repo.save(api_key)
            
            return RevokeAPIKeyResult(
                status=ResultStatus.SUCCESS,
                message="API key revoked successfully"
            )
            
        except Exception as e:
            return RevokeAPIKeyResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to revoke API key: {str(e)}"]
            )
    
    async def rotate_api_key(self, user_id: int, api_key_id: int) -> RotateAPIKeyResult:
        """Rotate an API key to generate a new key value."""
        try:
            # Get the existing API key
            old_key = await self.api_key_repo.get(api_key_id)
            
            if not old_key:
                return RotateAPIKeyResult(
                    status=ResultStatus.NOT_FOUND,
                    errors=["API key not found"]
                )
            
            # Verify ownership
            if old_key.user_id != user_id:
                return RotateAPIKeyResult(
                    status=ResultStatus.UNAUTHORIZED,
                    errors=["You do not have permission to rotate this API key"]
                )
            
            # Generate new key value
            import secrets
            new_key_value = f"tldr_sk_{secrets.token_urlsafe(32)}"
            
            # Hash the key for storage
            new_key_hash = hashlib.sha256(new_key_value.encode("utf-8")).hexdigest()
            
            # Create new API key with same properties
            new_api_key = APIKey(
                id=None,
                user_id=user_id,
                key_hash=new_key_hash,
                name=f"{old_key.name} (rotated)",
                scopes=old_key.scopes,
                expires_at=old_key.expires_at,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now()
            )
            
            # Save new key
            saved_key = await self.api_key_repo.save(new_api_key)
            
            # Schedule old key for revocation after grace period
            # In production, this would be handled by a background task
            # For now, we'll just mark it as rotated
            old_key.name = f"{old_key.name} (rotated - will be revoked)"
            await self.api_key_repo.save(old_key)
            
            return RotateAPIKeyResult(
                status=ResultStatus.SUCCESS,
                api_key=saved_key,
                key=new_key_value,
                message="API key rotated successfully"
            )
            
        except Exception as e:
            return RotateAPIKeyResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to rotate API key: {str(e)}"]
            )