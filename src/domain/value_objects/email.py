"""Email value object."""

from dataclasses import dataclass
from email.utils import parseaddr

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class Email:
    """Value object representing an email address.

    This is an immutable value object that ensures email addresses
    are valid according to basic validation rules using stdlib.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate email format after initialization."""
        if not self._is_valid_email(self.value):
            raise InvalidValueError(f"Invalid email format: {self.value}")

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Check if email has valid format using stdlib."""
        if not email or not isinstance(email, str):
            return False
        
        # Basic validation using parseaddr
        parsed = parseaddr(email.strip())
        
        # parseaddr returns ('', '') for invalid emails
        if not parsed[1]:
            return False
            
        # Check for @ symbol
        if '@' not in parsed[1]:
            return False
            
        # Check basic structure
        local, domain = parsed[1].rsplit('@', 1)
        
        # Validate local part
        if not local or len(local) > 64:
            return False
            
        # Validate domain
        if not domain or '.' not in domain:
            return False
            
        # Check domain parts
        domain_parts = domain.split('.')
        if len(domain_parts) < 2:
            return False
            
        # Each part should have at least one character
        for part in domain_parts:
            if not part or len(part) > 63:
                return False
                
        return True

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