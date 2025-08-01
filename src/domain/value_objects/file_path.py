"""Value object representing a file path."""

from dataclasses import dataclass
from pathlib import Path

from ..exceptions import InvalidValueError


@dataclass(frozen=True)
class FilePath:
    """Value object representing a file path.
    
    This ensures file paths are valid and provides
    a consistent interface for path operations.
    """
    
    value: str
    
    def __post_init__(self):
        """Validate the file path."""
        if not self.value:
            raise InvalidValueError("File path cannot be empty")
        
        # Ensure the path is absolute or can be resolved
        try:
            Path(self.value)
        except Exception as e:
            raise InvalidValueError(f"Invalid file path: {e}")
    
    @property
    def path(self) -> Path:
        """Get as pathlib Path object."""
        return Path(self.value)
    
    @property
    def exists(self) -> bool:
        """Check if the file exists."""
        return self.path.exists()
    
    @property
    def is_absolute(self) -> bool:
        """Check if this is an absolute path."""
        return self.path.is_absolute()
    
    @property
    def name(self) -> str:
        """Get the file name."""
        return self.path.name
    
    @property
    def stem(self) -> str:
        """Get the file name without extension."""
        return self.path.stem
    
    @property
    def suffix(self) -> str:
        """Get the file extension."""
        return self.path.suffix
    
    @property
    def parent_dir(self) -> str:
        """Get the parent directory path."""
        return str(self.path.parent)
    
    def with_suffix(self, suffix: str) -> "FilePath":
        """Create a new FilePath with a different suffix.
        
        Args:
            suffix: New file suffix (e.g., '.mp4')
            
        Returns:
            New FilePath instance
        """
        return FilePath(str(self.path.with_suffix(suffix)))
    
    def __str__(self) -> str:
        """String representation."""
        return self.value