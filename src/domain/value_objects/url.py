"""URL value object."""

from dataclasses import dataclass
from urllib.parse import urlparse

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class Url:
    """Value object representing a URL.

    This is an immutable value object that ensures URLs are valid
    and provides utility methods for URL manipulation.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate URL format after initialization."""
        if not self._is_valid_url(self.value):
            raise InvalidValueError(f"Invalid URL format: {self.value}")

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL has valid format."""
        if not url or not isinstance(url, str):
            return False

        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @property
    def scheme(self) -> str:
        """Get the URL scheme (http, https, etc.)."""
        return urlparse(self.value).scheme

    @property
    def domain(self) -> str:
        """Get the domain/hostname from the URL."""
        return urlparse(self.value).netloc

    @property
    def path(self) -> str:
        """Get the path component of the URL."""
        return urlparse(self.value).path

    @property
    def is_secure(self) -> bool:
        """Check if URL uses HTTPS."""
        return self.scheme == "https"

    def with_path(self, new_path: str) -> "Url":
        """Create a new URL with a different path."""
        parsed = urlparse(self.value)
        new_url = parsed._replace(path=new_path).geturl()
        return Url(new_url)

    def __str__(self) -> str:
        """String representation returns the URL value."""
        return self.value
