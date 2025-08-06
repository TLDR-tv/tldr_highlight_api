"""Wake word domain model - triggers for clip generation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class WakeWord:
    """Custom wake word configuration supporting multi-word phrases."""

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    phrase: str = ""  # Can be single word or multi-word phrase
    is_active: bool = True

    # Configuration
    case_sensitive: bool = False
    exact_match: bool = True  # If false, allows partial matches
    cooldown_seconds: int = 30  # Minimum time between triggers
    
    # Fuzzy matching configuration
    max_edit_distance: int = 2  # Maximum Levenshtein-Damerau distance
    similarity_threshold: float = 0.8  # Minimum similarity score (0.0-1.0)
    
    # Clip configuration
    pre_roll_seconds: int = 10  # Seconds before wake word to include
    post_roll_seconds: int = 30  # Seconds after wake word to include

    # Usage tracking
    trigger_count: int = 0
    last_triggered_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Normalize wake phrase."""
        if not self.case_sensitive:
            self.phrase = self.phrase.lower()
        self.phrase = self.phrase.strip()

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
        """Check if text contains this wake phrase."""
        if not self.phrase:  # Handle empty phrase
            return text == ""
            
        if not self.case_sensitive:
            text = text.lower()

        if self.exact_match:
            # Look for phrase boundaries
            import re

            # Escape the phrase for regex but handle it properly
            escaped_phrase = re.escape(self.phrase)
            
            # For exact match, we want to match the phrase as a complete unit
            # Use word boundaries but be more flexible with punctuation
            pattern = rf"(?<!\w){escaped_phrase}(?!\w)"
            return bool(re.search(pattern, text))
        else:
            return self.phrase in text
