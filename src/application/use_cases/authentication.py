"""Authentication use cases."""

from dataclasses import dataclass
from typing import Optional, List
import bcrypt
from datetime import datetime, timedelta

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
            api_key = await self.api_key_repo.get_by_key(request.api_key)
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
            expires_at = Timestamp.now().add_days(expires_in_days)
        
        # Create API key
        api_key = APIKey(
            id=None,
            user_id=user_id,
            key=key,
            name=name,
            scopes=scopes,
            expires_at=expires_at,
            created_at=Timestamp.now(),
            updated_at=Timestamp.now()
        )
        
        # Save and return
        return await self.api_key_repo.save(api_key)