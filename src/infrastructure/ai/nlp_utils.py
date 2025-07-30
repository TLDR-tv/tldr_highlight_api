"""
NLP utilities for sentiment analysis and text processing.

This module provides natural language processing utilities for chat/comment
sentiment analysis, emotion detection, and text preprocessing.
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import nltk
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        # Try the new punkt_tab format
        nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


@dataclass
class SentimentScore:
    """Sentiment analysis result."""

    positive: float
    negative: float
    neutral: float
    compound: float
    label: str  # 'positive', 'negative', or 'neutral'
    confidence: float


@dataclass
class EmotionScore:
    """Emotion analysis result."""

    joy: float
    anger: float
    fear: float
    sadness: float
    surprise: float
    disgust: float
    primary_emotion: str
    confidence: float


@dataclass
class TextAnalysis:
    """Complete text analysis result."""

    text: str
    timestamp: float
    sentiment: SentimentScore
    emotion: Optional[EmotionScore] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    language: str = "en"
    word_count: int = 0
    toxicity_score: float = 0.0


@dataclass
class ChatMessage:
    """Chat message with metadata."""

    user_id: str
    username: str
    message: str
    timestamp: float
    platform: str  # 'twitch', 'youtube', etc.
    metadata: Optional[Dict] = None


class SentimentAnalyzer:
    """Advanced sentiment analysis with multiple models."""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english"))
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.toxicity_patterns = self._load_toxicity_patterns()

    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load emotion lexicon for emotion detection."""
        # Simplified emotion lexicon - in production, use NRC Emotion Lexicon
        return {
            "joy": {
                "happy": 0.9,
                "joy": 1.0,
                "excited": 0.8,
                "amazing": 0.7,
                "awesome": 0.8,
                "great": 0.6,
                "love": 0.7,
                "wonderful": 0.8,
                "fantastic": 0.8,
                "excellent": 0.7,
                "perfect": 0.6,
                "best": 0.6,
                "good": 0.5,
                "nice": 0.4,
                "cool": 0.4,
                "fun": 0.6,
            },
            "anger": {
                "angry": 0.9,
                "hate": 0.8,
                "mad": 0.7,
                "furious": 1.0,
                "annoyed": 0.6,
                "frustrated": 0.7,
                "irritated": 0.6,
                "terrible": 0.6,
                "awful": 0.7,
                "horrible": 0.8,
                "worst": 0.7,
                "stupid": 0.5,
                "bad": 0.4,
                "sucks": 0.6,
                "wtf": 0.6,
            },
            "fear": {
                "scared": 0.8,
                "afraid": 0.7,
                "worried": 0.6,
                "nervous": 0.6,
                "anxious": 0.7,
                "terrified": 0.9,
                "panic": 0.8,
                "concern": 0.5,
                "dangerous": 0.6,
                "risk": 0.4,
                "threat": 0.7,
            },
            "sadness": {
                "sad": 0.8,
                "cry": 0.7,
                "depressed": 0.9,
                "down": 0.5,
                "disappointed": 0.6,
                "upset": 0.6,
                "hurt": 0.6,
                "pain": 0.5,
                "sorry": 0.4,
                "tragic": 0.7,
                "unfortunate": 0.5,
            },
            "surprise": {
                "wow": 0.7,
                "amazing": 0.6,
                "incredible": 0.8,
                "unbelievable": 0.8,
                "shocking": 0.7,
                "surprised": 0.8,
                "unexpected": 0.6,
                "omg": 0.7,
                "whoa": 0.6,
            },
            "disgust": {
                "disgusting": 0.9,
                "gross": 0.7,
                "sick": 0.6,
                "nasty": 0.7,
                "revolting": 0.8,
                "repulsive": 0.8,
                "yuck": 0.6,
                "ew": 0.5,
            },
        }

    def _load_toxicity_patterns(self) -> List[re.Pattern]:
        """Load patterns for toxicity detection."""
        # Simplified toxicity patterns - in production, use more sophisticated models
        toxic_patterns = [
            r"\b(kill|die|death)\s+(yourself|urself)\b",
            r"\b(go|kys)\b",
            r"\b(f+u+c+k+|sh+i+t+|damn+|hell+)\b",
            r"\b(stupid|idiot|moron|retard)\b",
            r"\b(toxic|troll|noob|scrub)\b",
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in toxic_patterns]

    async def analyze_sentiment(self, text: str) -> SentimentScore:
        """
        Analyze sentiment using multiple approaches.

        Args:
            text: Input text

        Returns:
            SentimentScore object
        """
        try:
            # Clean text
            cleaned_text = self._preprocess_text(text)

            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)

            # TextBlob sentiment analysis
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity

            # Combine scores (weighted average)
            positive = (vader_scores["pos"] + max(0, textblob_polarity)) / 2
            negative = (vader_scores["neg"] + max(0, -textblob_polarity)) / 2
            neutral = vader_scores["neu"]
            compound = (vader_scores["compound"] + textblob_polarity) / 2

            # Determine label and confidence
            if compound >= 0.05:
                label = "positive"
                confidence = abs(compound)
            elif compound <= -0.05:
                label = "negative"
                confidence = abs(compound)
            else:
                label = "neutral"
                confidence = 1.0 - abs(compound)

            return SentimentScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=compound,
                label=label,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentScore(0.0, 0.0, 1.0, 0.0, "neutral", 0.0)

    async def analyze_emotion(self, text: str) -> EmotionScore:
        """
        Analyze emotions in text.

        Args:
            text: Input text

        Returns:
            EmotionScore object
        """
        try:
            # Clean and tokenize text
            cleaned_text = self._preprocess_text(text)
            tokens = word_tokenize(cleaned_text.lower())
            tokens = [token for token in tokens if token not in self.stop_words]

            # Calculate emotion scores
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_lexicon.keys()}
            total_words = len(tokens)

            if total_words == 0:
                primary_emotion = "neutral"
                confidence = 0.0
            else:
                for token in tokens:
                    for emotion, lexicon in self.emotion_lexicon.items():
                        if token in lexicon:
                            emotion_scores[emotion] += lexicon[token]

                # Normalize scores
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_words

                # Find primary emotion
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[primary_emotion]

                # If no emotions detected, default to neutral
                if confidence == 0.0:
                    primary_emotion = "neutral"

            return EmotionScore(
                joy=emotion_scores.get("joy", 0.0),
                anger=emotion_scores.get("anger", 0.0),
                fear=emotion_scores.get("fear", 0.0),
                sadness=emotion_scores.get("sadness", 0.0),
                surprise=emotion_scores.get("surprise", 0.0),
                disgust=emotion_scores.get("disgust", 0.0),
                primary_emotion=primary_emotion,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return EmotionScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "neutral", 0.0)

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using TF-IDF.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        try:
            # Clean text
            cleaned_text = self._preprocess_text(text)

            # Tokenize and filter words
            tokens = word_tokenize(cleaned_text.lower())
            tokens = [
                token
                for token in tokens
                if token.isalpha() and len(token) > 2 and token not in self.stop_words
            ]

            if not tokens:
                return []

            # Calculate word frequencies
            word_freq = Counter(tokens)

            # Get most common words
            keywords = [word for word, count in word_freq.most_common(max_keywords)]

            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    async def calculate_toxicity(self, text: str) -> float:
        """
        Calculate toxicity score for text.

        Args:
            text: Input text

        Returns:
            Toxicity score (0.0 to 1.0)
        """
        try:
            # Check for toxic patterns
            toxicity_score = 0.0
            text_lower = text.lower()

            for pattern in self.toxicity_patterns:
                matches = pattern.findall(text_lower)
                toxicity_score += len(matches) * 0.2  # Each match adds 0.2

            # Check for excessive caps (shouting)
            if len(text) > 10:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                if caps_ratio > 0.7:
                    toxicity_score += 0.1

            # Check for excessive punctuation
            punct_ratio = sum(1 for c in text if c in "!?") / max(len(text), 1)
            if punct_ratio > 0.2:
                toxicity_score += 0.1

            return min(toxicity_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating toxicity: {e}")
            return 0.0

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r"[@#]\w+", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove emojis (basic approach)
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text


class ChatProcessor:
    """Process chat messages for sentiment and engagement analysis."""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.message_buffer = []
        self.engagement_metrics = defaultdict(float)

    async def process_message(self, message: ChatMessage) -> TextAnalysis:
        """
        Process a single chat message.

        Args:
            message: ChatMessage object

        Returns:
            TextAnalysis result
        """
        try:
            # Analyze sentiment
            sentiment = await self.sentiment_analyzer.analyze_sentiment(message.message)

            # Analyze emotion
            emotion = await self.sentiment_analyzer.analyze_emotion(message.message)

            # Extract keywords
            keywords = await self.sentiment_analyzer.extract_keywords(message.message)

            # Calculate toxicity
            toxicity_score = await self.sentiment_analyzer.calculate_toxicity(
                message.message
            )

            # Word count
            word_count = len(message.message.split())

            return TextAnalysis(
                text=message.message,
                timestamp=message.timestamp,
                sentiment=sentiment,
                emotion=emotion,
                keywords=keywords,
                topics=[],  # Could be enhanced with topic modeling
                language="en",  # Could be enhanced with language detection
                word_count=word_count,
                toxicity_score=toxicity_score,
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Return neutral analysis on error
            return TextAnalysis(
                text=message.message,
                timestamp=message.timestamp,
                sentiment=SentimentScore(0.0, 0.0, 1.0, 0.0, "neutral", 0.0),
                word_count=len(message.message.split()),
                toxicity_score=0.0,
            )

    async def process_batch(self, messages: List[ChatMessage]) -> List[TextAnalysis]:
        """
        Process multiple messages efficiently.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of TextAnalysis results
        """
        tasks = [self.process_message(msg) for msg in messages]
        return await asyncio.gather(*tasks)

    async def analyze_engagement(
        self, messages: List[ChatMessage], window_seconds: float = 60.0
    ) -> Dict[str, float]:
        """
        Analyze chat engagement metrics.

        Args:
            messages: List of chat messages
            window_seconds: Time window for analysis

        Returns:
            Dictionary of engagement metrics
        """
        if not messages:
            return {}

        try:
            # Sort messages by timestamp
            sorted_messages = sorted(messages, key=lambda x: x.timestamp)

            # Calculate metrics
            total_messages = len(messages)
            unique_users = len(set(msg.user_id for msg in messages))

            # Messages per minute
            time_span = sorted_messages[-1].timestamp - sorted_messages[0].timestamp
            messages_per_minute = total_messages / max(time_span / 60, 1)

            # Average message length
            avg_message_length = np.mean([len(msg.message) for msg in messages])

            # Sentiment distribution
            analyses = await self.process_batch(messages)
            sentiment_counts = Counter(
                analysis.sentiment.label for analysis in analyses
            )

            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                messages_per_minute, unique_users, avg_message_length, sentiment_counts
            )

            return {
                "total_messages": total_messages,
                "unique_users": unique_users,
                "messages_per_minute": messages_per_minute,
                "avg_message_length": avg_message_length,
                "positive_ratio": sentiment_counts.get("positive", 0) / total_messages,
                "negative_ratio": sentiment_counts.get("negative", 0) / total_messages,
                "neutral_ratio": sentiment_counts.get("neutral", 0) / total_messages,
                "engagement_score": engagement_score,
                "toxicity_ratio": sum(
                    1 for analysis in analyses if analysis.toxicity_score > 0.5
                )
                / total_messages,
            }

        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return {}

    def _calculate_engagement_score(
        self,
        messages_per_minute: float,
        unique_users: int,
        avg_message_length: float,
        sentiment_counts: Counter,
    ) -> float:
        """Calculate overall engagement score."""
        try:
            # Normalize metrics
            message_rate_score = min(
                messages_per_minute / 10.0, 1.0
            )  # 10 msgs/min = 1.0
            user_diversity_score = min(unique_users / 50.0, 1.0)  # 50 users = 1.0
            message_quality_score = min(
                avg_message_length / 20.0, 1.0
            )  # 20 chars = 1.0

            # Sentiment score (positive sentiment increases engagement)
            total_messages = sum(sentiment_counts.values())
            if total_messages > 0:
                positive_ratio = sentiment_counts.get("positive", 0) / total_messages
                negative_ratio = sentiment_counts.get("negative", 0) / total_messages
                sentiment_score = positive_ratio - (
                    negative_ratio * 0.5
                )  # Negative sentiment hurts less
            else:
                sentiment_score = 0.0

            # Weighted combination
            engagement_score = (
                message_rate_score * 0.3
                + user_diversity_score * 0.3
                + message_quality_score * 0.2
                + sentiment_score * 0.2
            )

            return max(0.0, min(engagement_score, 1.0))

        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    async def detect_spam(self, messages: List[ChatMessage]) -> List[bool]:
        """
        Detect spam messages using heuristics.

        Args:
            messages: List of messages to check

        Returns:
            List of boolean flags indicating spam
        """
        spam_flags = []

        try:
            # Group messages by user
            user_messages = defaultdict(list)
            for msg in messages:
                user_messages[msg.user_id].append(msg)

            for msg in messages:
                is_spam = False
                user_msgs = user_messages[msg.user_id]

                # Check for repeated messages
                if len(user_msgs) > 1:
                    recent_msgs = [
                        m.message
                        for m in user_msgs
                        if abs(m.timestamp - msg.timestamp) < 60  # Within 1 minute
                    ]
                    if (
                        len(recent_msgs) > 3
                        and len(set(recent_msgs)) < len(recent_msgs) * 0.5
                    ):
                        is_spam = True

                # Check for excessive caps
                if len(msg.message) > 10:
                    caps_ratio = sum(1 for c in msg.message if c.isupper()) / len(
                        msg.message
                    )
                    if caps_ratio > 0.8:
                        is_spam = True

                # Check for URLs (simple spam indicator)
                if re.search(r"http[s]?://", msg.message, re.IGNORECASE):
                    is_spam = True

                spam_flags.append(is_spam)

            return spam_flags

        except Exception as e:
            logger.error(f"Error detecting spam: {e}")
            return [False] * len(messages)


# Global instances
sentiment_analyzer = SentimentAnalyzer()
chat_processor = ChatProcessor()


class TopicAnalyzer:
    """Analyze topics and trends in text collections."""

    def __init__(self, max_features: int = 100, n_clusters: int = 8):
        self.max_features = max_features
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    async def extract_topics(
        self, texts: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Extract topics from a collection of texts.

        Args:
            texts: List of text documents

        Returns:
            List of topic dictionaries with keywords and scores
        """
        if len(texts) < self.n_clusters:
            return []

        try:
            # Preprocess texts
            cleaned_texts = [
                sentiment_analyzer._preprocess_text(text) for text in texts
            ]
            cleaned_texts = [text for text in cleaned_texts if len(text) > 10]

            if len(cleaned_texts) < self.n_clusters:
                return []

            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)

            # Cluster texts
            clusters = self.clusterer.fit_predict(tfidf_matrix)

            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []

            for i in range(self.n_clusters):
                # Get cluster center
                cluster_center = self.clusterer.cluster_centers_[i]

                # Get top features for this cluster
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_features = [feature_names[idx] for idx in top_indices]
                top_scores = [cluster_center[idx] for idx in top_indices]

                # Count documents in this cluster
                cluster_size = np.sum(clusters == i)

                topics.append(
                    {
                        "topic_id": i,
                        "keywords": top_features,
                        "scores": top_scores,
                        "document_count": int(cluster_size),
                        "topic_label": " ".join(
                            top_features[:3]
                        ),  # Use top 3 keywords as label
                    }
                )

            return topics

        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []


# Global topic analyzer
topic_analyzer = TopicAnalyzer()
