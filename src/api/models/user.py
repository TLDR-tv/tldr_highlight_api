"""User request and response models."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field

from ...domain.models.user import UserRole


class UserResponse(BaseModel):
    """User response model."""
    
    id: UUID
    organization_id: UUID
    email: str
    name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserCreateRequest(BaseModel):
    """Create user request (admin only)."""
    
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.MEMBER


class UserUpdateRequest(BaseModel):
    """Update user profile request."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None


class UserRoleUpdateRequest(BaseModel):
    """Update user role request (admin only)."""
    
    role: UserRole


class UserListResponse(BaseModel):
    """User list response."""
    
    users: list[UserResponse]
    total: int