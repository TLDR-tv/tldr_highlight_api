"""Domain protocols - interfaces for dependency inversion."""

from typing import Protocol, TypeVar, Optional, AsyncIterator
from uuid import UUID
from pathlib import Path

from ..models.organization import Organization
from ..models.user import User
from ..models.stream import Stream
from ..models.highlight import Highlight
from ..models.api_key import APIKey
from ..models.wake_word import WakeWord


T = TypeVar("T")


class Repository(Protocol[T]):
    """Base repository protocol."""

    async def add(self, entity: T) -> T:
        """Add entity to repository."""
        ...

    async def get(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        ...

    async def update(self, entity: T) -> T:
        """Update existing entity."""
        ...

    async def delete(self, id: UUID) -> None:
        """Delete entity by ID."""
        ...

    async def list(self, **filters) -> list[T]:
        """List entities with optional filters."""
        ...


class OrganizationRepository(Repository[Organization], Protocol):
    """Organization repository protocol."""

    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        ...

    async def get_by_api_key(self, api_key: str) -> Optional[Organization]:
        """Get organization associated with API key."""
        ...


class UserRepository(Repository[User], Protocol):
    """User repository protocol."""

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        ...

    async def list_by_organization(self, org_id: UUID) -> list[User]:
        """List all users in an organization."""
        ...


class StreamRepository(Repository[Stream], Protocol):
    """Stream repository protocol."""

    async def get_by_fingerprint(
        self, fingerprint: str, org_id: UUID
    ) -> Optional[Stream]:
        """Get stream by fingerprint within an organization."""
        ...

    async def list_active(self, org_id: UUID) -> list[Stream]:
        """List active streams for an organization."""
        ...

    async def list_by_organization(
        self, org_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Stream]:
        """List streams for an organization with pagination."""
        ...


class HighlightRepository(Repository[Highlight], Protocol):
    """Highlight repository protocol."""

    async def list_by_stream(self, stream_id: UUID) -> list[Highlight]:
        """List highlights for a stream."""
        ...

    async def list_by_organization(
        self, org_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[Highlight]:
        """List highlights for an organization."""
        ...

    async def list_by_wake_word(self, org_id: UUID, wake_word: str) -> list[Highlight]:
        """List highlights triggered by a specific wake word."""
        ...


class APIKeyRepository(Repository[APIKey], Protocol):
    """API key repository protocol."""

    async def get_by_prefix(self, prefix: str) -> Optional[APIKey]:
        """Get API key by prefix."""
        ...

    async def get_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        ...

    async def list_by_organization(self, org_id: UUID) -> list[APIKey]:
        """List API keys for an organization."""
        ...


class WakeWordRepository(Repository[WakeWord], Protocol):
    """Wake word repository protocol."""

    async def list_by_organization(self, org_id: UUID) -> list[WakeWord]:
        """List wake words for an organization."""
        ...

    async def get_active_words(self, org_id: UUID) -> list[str]:
        """Get list of active wake word strings."""
        ...


class StorageService(Protocol):
    """Storage service for media files."""

    async def upload_file(self, file_path: Path, key: str) -> str:
        """Upload file and return storage path."""
        ...

    async def download_file(self, key: str, destination: Path) -> None:
        """Download file from storage."""
        ...

    async def delete_file(self, key: str) -> None:
        """Delete file from storage."""
        ...

    async def generate_signed_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate signed URL for temporary access."""
        ...


class VideoAnalyzer(Protocol):
    """Video analysis service protocol."""

    async def analyze_segment(
        self, video_path: Path, dimensions: list[str], context: Optional[dict] = None
    ) -> dict[str, tuple[float, float]]:
        """Analyze video segment and return dimension scores.
        Returns dict of dimension_name -> (score, confidence)
        """
        ...

    async def extract_transcript(self, video_path: Path) -> str:
        """Extract transcript from video audio."""
        ...

    async def identify_timestamps(
        self, video_path: Path, target_content: str
    ) -> list[tuple[float, float]]:
        """Identify timestamps for specific content.
        Returns list of (start_time, end_time) tuples.
        """
        ...


class StreamProcessor(Protocol):
    """Stream processing service protocol."""

    async def process_stream(self, stream_url: str) -> AsyncIterator[Path]:
        """Process stream and yield segment paths.
        Implements ring buffer internally.
        """
        ...

    async def extract_clip(
        self, stream_url: str, start_time: float, end_time: float, output_path: Path
    ) -> None:
        """Extract clip from stream."""
        ...

    async def generate_thumbnail(
        self, video_path: Path, timestamp: float, output_path: Path
    ) -> None:
        """Generate thumbnail from video."""
        ...


class WebhookService(Protocol):
    """Webhook delivery service protocol."""

    async def send_event(
        self, url: str, event_type: str, payload: dict, secret: Optional[str] = None
    ) -> bool:
        """Send webhook event. Returns success status."""
        ...

    async def verify_signature(
        self, payload: bytes, signature: str, secret: str
    ) -> bool:
        """Verify webhook signature."""
        ...


class AuthenticationService(Protocol):
    """Authentication service protocol."""

    async def hash_password(self, password: str) -> str:
        """Hash a password."""
        ...

    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        ...

    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate and return API key if valid."""
        ...

    async def generate_api_key(
        self, organization_id: UUID, name: str
    ) -> tuple[str, APIKey]:
        """Generate new API key. Returns (raw_key, key_entity)."""
        ...
