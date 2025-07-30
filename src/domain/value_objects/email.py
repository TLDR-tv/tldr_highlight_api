"""Email value object."""

from dataclasses import dataclass
import re
from typing import ClassVar

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class Email:
    """Value object representing an email address.

    This is an immutable value object that ensures email addresses
    are valid according to a basic pattern.
    """

    value: str

    # Email validation pattern
    _PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    def __post_init__(self):
        """Validate email format after initialization."""
        if not self._is_valid_email(self.value):
            raise InvalidValueError(f"Invalid email format: {self.value}")

    @classmethod
    def _is_valid_email(cls, email: str) -> bool:
        """Check if email matches the validation pattern."""
        if not email or not isinstance(email, str):
            return False
        return bool(cls._PATTERN.match(email.strip()))

    @property
    def domain(self) -> str:
        """Extract the domain part of the email."""
        return self.value.split("@")[1]

    @property
    def local_part(self) -> str:
        """Extract the local part (before @) of the email."""
        return self.value.split("@")[0]

    def __str__(self) -> str:
        """String representation returns the email value."""
        return self.value
