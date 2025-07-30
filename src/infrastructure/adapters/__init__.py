"""Infrastructure adapters for external system integration.

This module provides adapters for integrating with external systems
like streaming platforms and chat services as infrastructure concerns.
"""

from .stream import *
from .chat import *

__all__ = [
    # Stream adapters are exported from stream submodule
    # Chat adapters are exported from chat submodule
]