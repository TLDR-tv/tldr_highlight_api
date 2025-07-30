"""AI and machine learning infrastructure components.

This module provides utilities for machine learning models,
natural language processing, and scoring algorithms.
"""

from .ml_utils import ModelLoader, FeatureExtractor, Predictor, ModelConfig

from .nlp_utils import (
    TextAnalyzer,
    SentimentDetector,
    KeywordExtractor,
    LanguageDetector,
)

from .scoring_utils import (
    HighlightScorer,
    ScoringAlgorithm,
    ScoreAggregator,
    ScoringConfig,
)

__all__ = [
    # ML
    "ModelLoader",
    "FeatureExtractor",
    "Predictor",
    "ModelConfig",
    # NLP
    "TextAnalyzer",
    "SentimentDetector",
    "KeywordExtractor",
    "LanguageDetector",
    # Scoring
    "HighlightScorer",
    "ScoringAlgorithm",
    "ScoreAggregator",
    "ScoringConfig",
]
