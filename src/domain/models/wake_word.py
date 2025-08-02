"""Wake word domain model - triggers for clip generation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class WakeWord:
    """Custom wake word configuration."""

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    word: str = ""
    is_active: bool = True

    # Configuration
    case_sensitive: bool = False
    exact_match: bool = True  # If false, allows partial matches
    cooldown_seconds: int = 30  # Minimum time between triggers

    # Usage tracking
    trigger_count: int = 0
    last_triggered_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Normalize wake word."""
        if not self.case_sensitive:
            self.word = self.word.lower()
        self.word = self.word.strip()

    @property
    def can_trigger(self) -> bool:
        """Check if wake word can trigger (respecting cooldown)."""
        if not self.is_active:
            return False

        if not self.last_triggered_at:
            return True

        time_since_last = (
            datetime.now(timezone.utc) - self.last_triggered_at
        ).total_seconds()
        return time_since_last >= self.cooldown_seconds

    def record_trigger(self) -> None:
        """Record that the wake word was triggered."""
        self.trigger_count += 1
        self.last_triggered_at = datetime.now(timezone.utc)

    def matches(self, text: str) -> bool:
        """Check if text contains this wake word."""
        if not self.case_sensitive:
            text = text.lower()

        if self.exact_match:
            # Look for word boundaries
            import re

            pattern = rf"\b{re.escape(self.word)}\b"
            return bool(re.search(pattern, text))
        else:
            return self.word in text
