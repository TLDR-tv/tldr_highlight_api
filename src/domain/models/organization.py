"""Organization domain model - represents B2B customers."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class Organization:
    """B2B customer organization."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    slug: str = ""  # URL-friendly identifier
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Usage tracking (for future billing)
    total_streams_processed: int = 0
    total_highlights_generated: int = 0
    total_processing_seconds: float = 0.0
    
    # Custom configuration
    wake_words: set[str] = field(default_factory=set)
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and normalize data after initialization."""
        if not self.slug and self.name:
            self.slug = self._generate_slug(self.name)
    
    @property
    def has_webhook_configured(self) -> bool:
        """Check if organization has webhook configured."""
        return bool(self.webhook_url)
    
    @property
    def has_custom_wake_words(self) -> bool:
        """Check if org has custom wake words configured."""
        return bool(self.wake_words)
    
    def add_wake_word(self, word: str) -> None:
        """Add a custom wake word."""
        self.wake_words.add(word.lower().strip())
    
    def remove_wake_word(self, word: str) -> None:
        """Remove a custom wake word."""
        self.wake_words.discard(word.lower().strip())
    
    def record_usage(self, streams: int = 0, highlights: int = 0, seconds: float = 0.0) -> None:
        """Record usage metrics."""
        self.total_streams_processed += streams
        self.total_highlights_generated += highlights
        self.total_processing_seconds += seconds
    
    @staticmethod
    def _generate_slug(name: str) -> str:
        """Generate URL-friendly slug from name."""
        import re
        slug = name.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:50]  # Limit length