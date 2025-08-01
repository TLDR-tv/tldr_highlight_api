"""Domain interfaces module.

This module contains Protocol definitions that infrastructure
implementations must follow to integrate with the domain layer.
"""

from .ai_video_analyzer import (
    AIVideoAnalyzer,
    HighlightCandidate,
    AIAnalysisError,
)

__all__ = [
    "AIVideoAnalyzer",
    "HighlightCandidate", 
    "AIAnalysisError",
]