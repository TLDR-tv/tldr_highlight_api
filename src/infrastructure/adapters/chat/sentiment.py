"""Sentiment analysis infrastructure for chat messages.

This module provides sentiment analysis capabilities as an infrastructure
component using Pythonic patterns.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SentimentScore(str, Enum):
    """Sentiment score categories."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    HYPE = "hype"
    EXCITEMENT = "excitement"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    score: float  # -1.0 to 1.0
    magnitude: float  # 0.0 to 1.0 (intensity)
    category: SentimentScore
    confidence: float  # 0.0 to 1.0
    keywords: List[str]
    metadata: Dict[str, Any]


class SentimentAnalyzer:
    """Sentiment analyzer for chat messages.

    Provides basic sentiment analysis using pattern matching
    and keyword detection. Full NLP implementation would require
    additional infrastructure components.
    """

    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.positive_keywords = self._load_positive_keywords()
        self.negative_keywords = self._load_negative_keywords()
        self.hype_keywords = self._load_hype_keywords()
        self.emote_sentiments = self._load_emote_sentiments()

        logger.info("Initialized sentiment analyzer")

    def _load_positive_keywords(self) -> Dict[str, float]:
        """Load positive keywords with scores."""
        return {
            "love": 0.8,
            "awesome": 0.7,
            "amazing": 0.7,
            "great": 0.6,
            "good": 0.5,
            "nice": 0.5,
            "excellent": 0.8,
            "wonderful": 0.7,
            "fantastic": 0.7,
            "perfect": 0.8,
            "best": 0.7,
            "win": 0.6,
            "winner": 0.7,
            "yes": 0.4,
            "yay": 0.6,
            "thanks": 0.5,
            "thank you": 0.6,
            "gg": 0.5,
            "ggwp": 0.6,
        }

    def _load_negative_keywords(self) -> Dict[str, float]:
        """Load negative keywords with scores."""
        return {
            "hate": -0.8,
            "terrible": -0.7,
            "awful": -0.7,
            "bad": -0.6,
            "worst": -0.7,
            "sucks": -0.6,
            "fail": -0.5,
            "failed": -0.6,
            "lost": -0.5,
            "lose": -0.5,
            "no": -0.3,
            "nope": -0.4,
            "never": -0.5,
            "boring": -0.5,
            "sad": -0.6,
            "angry": -0.7,
            "disappointed": -0.6,
            "frustrating": -0.6,
        }

    def _load_hype_keywords(self) -> Dict[str, float]:
        """Load hype keywords with intensity scores."""
        return {
            "pog": 0.9,
            "poggers": 0.9,
            "pogchamp": 0.9,
            "hype": 0.8,
            "insane": 0.8,
            "crazy": 0.7,
            "unreal": 0.8,
            "omg": 0.7,
            "wtf": 0.6,
            "holy": 0.7,
            "lets go": 0.8,
            "letsgoo": 0.9,
            "fire": 0.7,
            "lit": 0.7,
            "goat": 0.8,
            "clutch": 0.8,
            "epic": 0.7,
        }

    def _load_emote_sentiments(self) -> Dict[str, float]:
        """Load common emote sentiments."""
        return {
            # Positive emotes
            ":)": 0.5,
            ":D": 0.7,
            "^^": 0.5,
            "xD": 0.6,
            # Negative emotes
            ":(": -0.5,
            ":/": -0.3,
            # Neutral/surprise
            ":o": 0.0,
            "o_o": 0.0,
            # Platform-specific
            "Kappa": 0.1,
            "LUL": 0.5,
            "KEKW": 0.5,
            "PepeHands": -0.5,
            "monkaS": 0.0,
        }

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult: Analysis result
        """
        text_lower = text.lower()

        # Calculate keyword scores
        positive_score = 0.0
        negative_score = 0.0
        hype_score = 0.0
        found_keywords = []

        # Check positive keywords
        for keyword, score in self.positive_keywords.items():
            if keyword in text_lower:
                positive_score += score
                found_keywords.append(keyword)

        # Check negative keywords
        for keyword, score in self.negative_keywords.items():
            if keyword in text_lower:
                negative_score += score
                found_keywords.append(keyword)

        # Check hype keywords
        for keyword, score in self.hype_keywords.items():
            if keyword in text_lower:
                hype_score += score
                found_keywords.append(keyword)

        # Check emotes
        emote_score = 0.0
        for emote, score in self.emote_sentiments.items():
            if emote in text:
                emote_score += score
                found_keywords.append(f"emote:{emote}")

        # Calculate combined score
        total_score = positive_score + negative_score + emote_score

        # Normalize to -1 to 1 range
        if abs(total_score) > 1:
            total_score = max(-1, min(1, total_score))

        # Calculate magnitude (intensity)
        magnitude = abs(total_score) + (hype_score * 0.5)
        magnitude = min(1.0, magnitude)

        # Determine category
        if hype_score > 0.5 and total_score > 0:
            category = SentimentScore.HYPE
        elif total_score >= 0.6:
            category = SentimentScore.VERY_POSITIVE
        elif total_score >= 0.2:
            category = SentimentScore.POSITIVE
        elif total_score >= -0.2:
            category = SentimentScore.NEUTRAL
        elif total_score >= -0.6:
            category = SentimentScore.NEGATIVE
        else:
            category = SentimentScore.VERY_NEGATIVE

        # Calculate confidence based on keyword matches
        confidence = min(1.0, len(found_keywords) * 0.2) if found_keywords else 0.3

        return SentimentResult(
            score=total_score,
            magnitude=magnitude,
            category=category,
            confidence=confidence,
            keywords=found_keywords,
            metadata={
                "text_length": len(text),
                "positive_score": positive_score,
                "negative_score": negative_score,
                "hype_score": hype_score,
                "emote_score": emote_score,
            },
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text) for text in texts]

    def get_aggregate_sentiment(
        self, results: List[SentimentResult]
    ) -> SentimentResult:
        """Get aggregate sentiment from multiple results.

        Args:
            results: List of sentiment results

        Returns:
            Aggregate SentimentResult
        """
        if not results:
            return SentimentResult(
                score=0.0,
                magnitude=0.0,
                category=SentimentScore.NEUTRAL,
                confidence=0.0,
                keywords=[],
                metadata={"count": 0},
            )

        # Calculate averages
        avg_score = sum(r.score for r in results) / len(results)
        avg_magnitude = sum(r.magnitude for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Collect all keywords
        all_keywords = []
        for r in results:
            all_keywords.extend(r.keywords)

        # Determine aggregate category
        if avg_score >= 0.6:
            category = SentimentScore.VERY_POSITIVE
        elif avg_score >= 0.2:
            category = SentimentScore.POSITIVE
        elif avg_score >= -0.2:
            category = SentimentScore.NEUTRAL
        elif avg_score >= -0.6:
            category = SentimentScore.NEGATIVE
        else:
            category = SentimentScore.VERY_NEGATIVE

        return SentimentResult(
            score=avg_score,
            magnitude=avg_magnitude,
            category=category,
            confidence=avg_confidence,
            keywords=list(set(all_keywords))[:10],  # Top 10 unique keywords
            metadata={
                "count": len(results),
                "score_variance": self._calculate_variance([r.score for r in results]),
            },
        )

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
