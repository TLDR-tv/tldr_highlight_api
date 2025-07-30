"""Company name value object."""

from dataclasses import dataclass
import re
from typing import ClassVar

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class CompanyName:
    """Value object representing a company/organization name.

    This is an immutable value object that ensures company names
    are valid and properly formatted.
    """

    value: str

    # Constraints
    MIN_LENGTH: ClassVar[int] = 2
    MAX_LENGTH: ClassVar[int] = 100

    # Pattern to detect potentially invalid characters
    INVALID_PATTERN: ClassVar[re.Pattern] = re.compile(r"[<>{}[\]|\\^`]")

    def __post_init__(self):
        """Validate company name after initialization."""
        # Strip and normalize whitespace
        normalized = " ".join(self.value.strip().split())

        if not normalized:
            raise InvalidValueError("Company name cannot be empty")

        if len(normalized) < self.MIN_LENGTH:
            raise InvalidValueError(
                f"Company name must be at least {self.MIN_LENGTH} characters long"
            )

        if len(normalized) > self.MAX_LENGTH:
            raise InvalidValueError(
                f"Company name cannot exceed {self.MAX_LENGTH} characters"
            )

        if self.INVALID_PATTERN.search(normalized):
            raise InvalidValueError(
                f"Company name contains invalid characters: {self.value}"
            )

        # Update value with normalized version
        if normalized != self.value:
            object.__setattr__(self, "value", normalized)

    @property
    def slug(self) -> str:
        """Get URL-safe slug version of company name."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = self.value.lower()
        # Replace spaces and common separators with hyphens
        slug = re.sub(r"[\s\-_.,/]+", "-", slug)
        # Remove non-alphanumeric characters except hyphens
        slug = re.sub(r"[^a-z0-9\-]", "", slug)
        # Remove leading/trailing hyphens and collapse multiple hyphens
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug

    @property
    def initials(self) -> str:
        """Get company initials (first letter of each word)."""
        words = self.value.split()
        # Take first letter of each significant word (skip common words)
        skip_words = {"and", "or", "the", "of", "in", "for", "to", "a", "an"}
        initials = "".join(
            word[0].upper()
            for word in words
            if word.lower() not in skip_words or len(words) == 1
        )
        return initials[:3]  # Limit to 3 characters

    @property
    def display_name(self) -> str:
        """Get properly formatted display name."""
        # Title case with special handling for common patterns
        words = self.value.split()
        formatted_words = []

        for word in words:
            # Preserve all-caps abbreviations (e.g., "IBM", "NASA")
            if word.isupper() and len(word) <= 4:
                formatted_words.append(word)
            # Handle possessives
            elif "'s" in word.lower():
                parts = word.split("'")
                formatted_words.append(parts[0].title() + "'" + parts[1])
            # Normal title case
            else:
                formatted_words.append(word.title())

        return " ".join(formatted_words)

    def contains(self, search_term: str) -> bool:
        """Case-insensitive search within company name."""
        return search_term.lower() in self.value.lower()

    def __str__(self) -> str:
        """String representation returns the display name."""
        return self.display_name

    def __len__(self) -> int:
        """Length of the company name."""
        return len(self.value)
