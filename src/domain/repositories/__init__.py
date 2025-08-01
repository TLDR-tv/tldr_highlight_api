"""Repository interfaces using Protocol classes.

Repositories provide an abstraction over data persistence,
allowing the domain layer to remain independent of the
infrastructure layer.
"""

from src.domain.repositories.base import Repository
from src.domain.repositories.user_repository import UserRepository
from src.domain.repositories.organization_repository import OrganizationRepository
from src.domain.repositories.api_key_repository import APIKeyRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.webhook_repository import WebhookRepository

# BatchRepository removed - no longer needed
from src.domain.repositories.usage_record_repository import UsageRecordRepository

__all__ = [
    "Repository",
    "UserRepository",
    "OrganizationRepository",
    "APIKeyRepository",
    "StreamRepository",
    "HighlightRepository",
    "WebhookRepository",
    "UsageRecordRepository",
]
