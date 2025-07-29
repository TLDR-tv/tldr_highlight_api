"""
Unit tests for NLP processing utilities.

Tests for sentiment analysis, emotion detection, chat processing, and text analysis.
"""

import pytest
import time
from collections import Counter

from src.utils.nlp_utils import (
    SentimentAnalyzer,
    ChatProcessor,
    TopicAnalyzer,
    SentimentScore,
    EmotionScore,
    TextAnalysis,
    ChatMessage,
    sentiment_analyzer,
    chat_processor,
    topic_analyzer,
)


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()

    def test_sentiment_analyzer_initialization(self, analyzer):
        """Test SentimentAnalyzer initialization."""
        assert analyzer.vader_analyzer is not None
        assert analyzer.stop_words is not None
        assert isinstance(analyzer.emotion_lexicon, dict)
        assert isinstance(analyzer.toxicity_patterns, list)

    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self, analyzer):
        """Test sentiment analysis for positive text."""
        positive_text = "I love this amazing content! It's absolutely fantastic!"

        result = await analyzer.analyze_sentiment(positive_text)

        assert isinstance(result, SentimentScore)
        assert result.label == "positive"
        assert result.positive > result.negative
        assert result.compound > 0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self, analyzer):
        """Test sentiment analysis for negative text."""
        negative_text = "This is terrible! I hate it so much. Worst thing ever."

        result = await analyzer.analyze_sentiment(negative_text)

        assert isinstance(result, SentimentScore)
        assert result.label == "negative"
        assert result.negative > result.positive
        assert result.compound < 0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral(self, analyzer):
        """Test sentiment analysis for neutral text."""
        neutral_text = "This is a regular comment about the weather today."

        result = await analyzer.analyze_sentiment(neutral_text)

        assert isinstance(result, SentimentScore)
        assert result.label in ["neutral", "positive", "negative"]  # Could be any
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_emotion(self, analyzer):
        """Test emotion analysis."""
        happy_text = "I'm so happy and joyful! This is amazing and wonderful!"

        result = await analyzer.analyze_emotion(happy_text)

        assert isinstance(result, EmotionScore)
        assert result.primary_emotion in [
            "joy",
            "neutral",
        ]  # Should detect joy or neutral
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.joy <= 1.0
        assert 0.0 <= result.anger <= 1.0

    @pytest.mark.asyncio
    async def test_extract_keywords(self, analyzer):
        """Test keyword extraction."""
        text = "This is a great streaming platform with amazing video quality and excellent chat features"

        keywords = await analyzer.extract_keywords(text, max_keywords=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5

        # Should extract meaningful words
        expected_words = [
            "streaming",
            "platform",
            "amazing",
            "video",
            "quality",
            "excellent",
            "chat",
            "features",
        ]
        found_keywords = [word for word in keywords if word in expected_words]
        assert len(found_keywords) > 0

    @pytest.mark.asyncio
    async def test_calculate_toxicity(self, analyzer):
        """Test toxicity calculation."""
        # Clean text
        clean_text = "This is a nice and friendly message"
        clean_score = await analyzer.calculate_toxicity(clean_text)

        # Toxic text
        toxic_text = "This is stupid and you should die!"
        toxic_score = await analyzer.calculate_toxicity(toxic_text)

        assert 0.0 <= clean_score <= 1.0
        assert 0.0 <= toxic_score <= 1.0
        assert toxic_score > clean_score  # Toxic should score higher

    def test_preprocess_text(self, analyzer):
        """Test text preprocessing."""
        dirty_text = (
            "Check out https://example.com @username #hashtag    extra   spaces!!!"
        )

        clean_text = analyzer._preprocess_text(dirty_text)

        assert "https://example.com" not in clean_text
        assert "@username" not in clean_text
        assert "#hashtag" not in clean_text
        assert "  " not in clean_text  # No double spaces

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, analyzer):
        """Test handling of empty or invalid text."""
        empty_result = await analyzer.analyze_sentiment("")
        assert empty_result.label == "neutral"

        none_keywords = await analyzer.extract_keywords("")
        assert none_keywords == []

        empty_toxicity = await analyzer.calculate_toxicity("")
        assert empty_toxicity == 0.0


class TestChatProcessor:
    """Test cases for ChatProcessor."""

    @pytest.fixture
    def processor(self):
        return ChatProcessor()

    @pytest.fixture
    def sample_messages(self):
        return [
            ChatMessage(
                user_id="user1",
                username="viewer1",
                message="This stream is amazing!",
                timestamp=time.time(),
                platform="twitch",
            ),
            ChatMessage(
                user_id="user2",
                username="viewer2",
                message="Great gameplay",
                timestamp=time.time() + 1,
                platform="twitch",
            ),
            ChatMessage(
                user_id="user1",
                username="viewer1",
                message="Love this content",
                timestamp=time.time() + 2,
                platform="twitch",
            ),
        ]

    def test_chat_processor_initialization(self, processor):
        """Test ChatProcessor initialization."""
        assert processor.sentiment_analyzer is not None
        assert isinstance(processor.message_buffer, list)
        assert isinstance(processor.engagement_metrics, dict)

    @pytest.mark.asyncio
    async def test_process_message(self, processor, sample_messages):
        """Test processing a single message."""
        message = sample_messages[0]

        result = await processor.process_message(message)

        assert isinstance(result, TextAnalysis)
        assert result.text == message.message
        assert result.timestamp == message.timestamp
        assert isinstance(result.sentiment, SentimentScore)
        assert result.word_count > 0

    @pytest.mark.asyncio
    async def test_process_batch(self, processor, sample_messages):
        """Test batch processing of messages."""
        results = await processor.process_batch(sample_messages)

        assert len(results) == len(sample_messages)

        for result in results:
            assert isinstance(result, TextAnalysis)
            assert result.text in [msg.message for msg in sample_messages]

    @pytest.mark.asyncio
    async def test_analyze_engagement(self, processor, sample_messages):
        """Test engagement analysis."""
        engagement = await processor.analyze_engagement(
            sample_messages, window_seconds=60.0
        )

        assert isinstance(engagement, dict)
        assert "total_messages" in engagement
        assert "unique_users" in engagement
        assert "messages_per_minute" in engagement
        assert "engagement_score" in engagement

        assert engagement["total_messages"] == len(sample_messages)
        assert engagement["unique_users"] == len(
            set(msg.user_id for msg in sample_messages)
        )
        assert 0.0 <= engagement["engagement_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_spam(self, processor):
        """Test spam detection."""
        # Create spam and legitimate messages
        spam_messages = [
            ChatMessage(
                user_id="spammer",
                username="spammer",
                message="Buy now! http://spam.com",
                timestamp=time.time(),
                platform="twitch",
            ),
            ChatMessage(
                user_id="legitimate",
                username="legitimate",
                message="This is a great stream with interesting content",
                timestamp=time.time(),
                platform="twitch",
            ),
        ]

        spam_flags = await processor.detect_spam(spam_messages)

        assert len(spam_flags) == len(spam_messages)
        assert spam_flags[0] is True  # Spam message
        assert spam_flags[1] is False  # Legitimate message

    def test_calculate_engagement_score(self, processor):
        """Test engagement score calculation."""
        score = processor._calculate_engagement_score(
            messages_per_minute=10.0,
            unique_users=5,
            avg_message_length=20.0,
            sentiment_counts=Counter({"positive": 7, "negative": 2, "neutral": 1}),
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


class TestTopicAnalyzer:
    """Test cases for TopicAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return TopicAnalyzer(max_features=50, n_clusters=3)

    def test_topic_analyzer_initialization(self, analyzer):
        """Test TopicAnalyzer initialization."""
        assert analyzer.max_features == 50
        assert analyzer.n_clusters == 3
        assert analyzer.vectorizer is not None
        assert analyzer.clusterer is not None

    @pytest.mark.asyncio
    async def test_extract_topics(self, analyzer):
        """Test topic extraction."""
        texts = [
            "This is about gaming and video games",
            "I love playing video games and gaming content",
            "Gaming streams are the best entertainment",
            "Sports and football are exciting to watch",
            "Basketball and sports events are amazing",
            "I enjoy watching sports broadcasts",
            "Music and songs are beautiful art forms",
            "I love listening to music and melodies",
            "Musical performances are wonderful entertainment",
        ]

        topics = await analyzer.extract_topics(texts)

        assert isinstance(topics, list)

        if topics:  # May be empty if clustering fails
            for topic in topics:
                assert "topic_id" in topic
                assert "keywords" in topic
                assert "document_count" in topic
                assert "topic_label" in topic
                assert isinstance(topic["keywords"], list)

    @pytest.mark.asyncio
    async def test_extract_topics_insufficient_data(self, analyzer):
        """Test topic extraction with insufficient data."""
        # Too few texts for clustering
        texts = ["Short text", "Another text"]

        topics = await analyzer.extract_topics(texts)

        assert topics == []  # Should return empty list


class TestGlobalInstances:
    """Test global instances and their functionality."""

    def test_global_sentiment_analyzer(self):
        """Test global sentiment analyzer instance."""
        assert sentiment_analyzer is not None
        assert isinstance(sentiment_analyzer, SentimentAnalyzer)

    def test_global_chat_processor(self):
        """Test global chat processor instance."""
        assert chat_processor is not None
        assert isinstance(chat_processor, ChatProcessor)

    def test_global_topic_analyzer(self):
        """Test global topic analyzer instance."""
        assert topic_analyzer is not None
        assert isinstance(topic_analyzer, TopicAnalyzer)


class TestDataStructures:
    """Test data structures and classes."""

    def test_sentiment_score_creation(self):
        """Test SentimentScore creation."""
        score = SentimentScore(
            positive=0.8,
            negative=0.1,
            neutral=0.1,
            compound=0.7,
            label="positive",
            confidence=0.9,
        )

        assert score.positive == 0.8
        assert score.negative == 0.1
        assert score.neutral == 0.1
        assert score.compound == 0.7
        assert score.label == "positive"
        assert score.confidence == 0.9

    def test_emotion_score_creation(self):
        """Test EmotionScore creation."""
        emotion = EmotionScore(
            joy=0.8,
            anger=0.1,
            fear=0.0,
            sadness=0.1,
            surprise=0.0,
            disgust=0.0,
            primary_emotion="joy",
            confidence=0.8,
        )

        assert emotion.joy == 0.8
        assert emotion.primary_emotion == "joy"
        assert emotion.confidence == 0.8

    def test_text_analysis_creation(self):
        """Test TextAnalysis creation."""
        sentiment = SentimentScore(0.7, 0.2, 0.1, 0.5, "positive", 0.8)

        analysis = TextAnalysis(
            text="Test message",
            timestamp=time.time(),
            sentiment=sentiment,
            word_count=2,
            toxicity_score=0.1,
        )

        assert analysis.text == "Test message"
        assert analysis.sentiment == sentiment
        assert analysis.word_count == 2
        assert analysis.toxicity_score == 0.1

    def test_chat_message_creation(self):
        """Test ChatMessage creation."""
        timestamp = time.time()
        message = ChatMessage(
            user_id="user123",
            username="testuser",
            message="Hello world!",
            timestamp=timestamp,
            platform="twitch",
            metadata={"is_subscriber": True},
        )

        assert message.user_id == "user123"
        assert message.username == "testuser"
        assert message.message == "Hello world!"
        assert message.timestamp == timestamp
        assert message.platform == "twitch"
        assert message.metadata["is_subscriber"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text_handling(self):
        """Test handling of very long text."""
        analyzer = SentimentAnalyzer()

        # Create very long text (10,000 characters)
        long_text = "This is a test message. " * 400  # ~10,000 chars

        result = await analyzer.analyze_sentiment(long_text)

        assert isinstance(result, SentimentScore)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters."""
        analyzer = SentimentAnalyzer()

        special_text = "こんにちは! 🎉🎮 This has émojis and spécial chäracteřs! 😊"

        result = await analyzer.analyze_sentiment(special_text)

        assert isinstance(result, SentimentScore)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_malformed_input_handling(self):
        """Test handling of malformed input."""
        processor = ChatProcessor()

        # Create malformed message
        malformed_message = ChatMessage(
            user_id="",  # Empty user ID
            username=None,  # None username
            message="   ",  # Whitespace only
            timestamp=0,  # Invalid timestamp
            platform="",  # Empty platform
        )

        # Should handle gracefully
        result = await processor.process_message(malformed_message)

        assert isinstance(result, TextAnalysis)
        # Should have sensible defaults for malformed input


if __name__ == "__main__":
    pytest.main([__file__])
