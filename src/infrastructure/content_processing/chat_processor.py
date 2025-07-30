"""Chat processing infrastructure implementation.

This module provides chat analysis and sentiment detection capabilities
as an infrastructure component.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from collections import deque

from src.infrastructure.adapters.chat.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    SentimentScore
)

logger = logging.getLogger(__name__)


@dataclass
class ChatProcessorConfig:
    """Configuration for chat processor."""
    window_size_seconds: float = 10.0
    min_messages_for_analysis: int = 5
    highlight_threshold: float = 0.7
    velocity_spike_threshold: int = 50
    sentiment_weight: float = 0.3
    velocity_weight: float = 0.4
    emote_density_weight: float = 0.3


@dataclass
class ChatMessageData:
    """Raw chat message data."""
    timestamp: float
    user_id: str
    username: str
    text: str
    badges: List[str]
    emotes: List[str]
    metadata: Dict[str, Any]


@dataclass
class ChatWindow:
    """A window of chat messages for analysis."""
    start_time: float
    end_time: float
    messages: List[ChatMessageData]
    
    @property
    def duration(self) -> float:
        """Get window duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def velocity(self) -> float:
        """Get messages per second."""
        return len(self.messages) / self.duration if self.duration > 0 else 0


@dataclass
class ChatAnalysisResult:
    """Result of chat window analysis."""
    timestamp: float
    window_duration: float
    message_count: int
    velocity: float
    average_sentiment: float
    sentiment_distribution: Dict[str, int]
    emote_density: float
    highlight_score: float
    is_highlight: bool
    top_emotes: List[str]
    metadata: Dict[str, Any]


class ChatProcessor:
    """Infrastructure component for chat processing.
    
    Handles low-level chat analysis, sentiment detection, and velocity tracking.
    This is an infrastructure concern, not domain logic.
    """
    
    def __init__(self, config: ChatProcessorConfig):
        """Initialize chat processor.
        
        Args:
            config: Chat processor configuration
        """
        self.config = config
        self.sentiment_analyzer = SentimentAnalyzer()
        self._message_buffer = deque(maxlen=1000)
        self._processed_windows = 0
        
        logger.info(f"Initialized chat processor with config: {config}")
    
    async def process_chat_stream(
        self,
        chat_adapter,
        duration_seconds: Optional[float] = None
    ) -> AsyncGenerator[ChatAnalysisResult, None]:
        """Process chat messages from a stream.
        
        Args:
            chat_adapter: Chat adapter providing messages
            duration_seconds: Optional duration to process
            
        Yields:
            ChatAnalysisResult: Analysis results for chat windows
        """
        logger.info("Starting chat processing")
        
        start_time = asyncio.get_event_loop().time()
        current_window = []
        window_start = start_time
        
        async for message in chat_adapter.get_messages():
            current_time = asyncio.get_event_loop().time()
            
            # Check duration limit
            if duration_seconds and (current_time - start_time) >= duration_seconds:
                break
            
            # Convert to our data format
            msg_data = ChatMessageData(
                timestamp=current_time - start_time,
                user_id=message.user.id,
                username=message.user.username,
                text=message.text,
                badges=message.user.badges,
                emotes=self._extract_emotes(message.text),
                metadata=message.metadata
            )
            
            current_window.append(msg_data)
            self._message_buffer.append(msg_data)
            
            # Check if window is complete
            if (current_time - window_start) >= self.config.window_size_seconds:
                if len(current_window) >= self.config.min_messages_for_analysis:
                    # Analyze window
                    result = await self._analyze_window(ChatWindow(
                        start_time=window_start - start_time,
                        end_time=current_time - start_time,
                        messages=current_window
                    ))
                    
                    yield result
                    self._processed_windows += 1
                
                # Start new window
                current_window = []
                window_start = current_time
    
    async def _analyze_window(self, window: ChatWindow) -> ChatAnalysisResult:
        """Analyze a window of chat messages.
        
        Args:
            window: Chat window to analyze
            
        Returns:
            Analysis result
        """
        # Analyze sentiment for all messages
        sentiments = []
        sentiment_distribution = {
            "very_positive": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "very_negative": 0,
            "hype": 0
        }
        
        for msg in window.messages:
            sentiment = self.sentiment_analyzer.analyze(msg.text)
            sentiments.append(sentiment)
            
            # Update distribution
            category = sentiment.category.value
            if category in sentiment_distribution:
                sentiment_distribution[category] += 1
        
        # Calculate average sentiment
        avg_sentiment = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0
        
        # Calculate emote density
        total_emotes = sum(len(msg.emotes) for msg in window.messages)
        emote_density = total_emotes / len(window.messages) if window.messages else 0
        
        # Get top emotes
        emote_counts = {}
        for msg in window.messages:
            for emote in msg.emotes:
                emote_counts[emote] = emote_counts.get(emote, 0) + 1
        
        top_emotes = sorted(emote_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_emote_list = [emote for emote, _ in top_emotes]
        
        # Calculate highlight score
        highlight_score = self._calculate_highlight_score(
            velocity=window.velocity,
            avg_sentiment=avg_sentiment,
            emote_density=emote_density,
            sentiment_distribution=sentiment_distribution
        )
        
        is_highlight = highlight_score >= self.config.highlight_threshold
        
        return ChatAnalysisResult(
            timestamp=window.start_time,
            window_duration=window.duration,
            message_count=len(window.messages),
            velocity=window.velocity,
            average_sentiment=avg_sentiment,
            sentiment_distribution=sentiment_distribution,
            emote_density=emote_density,
            highlight_score=highlight_score,
            is_highlight=is_highlight,
            top_emotes=top_emote_list,
            metadata={
                "window_id": self._processed_windows,
                "unique_users": len(set(msg.user_id for msg in window.messages))
            }
        )
    
    def _calculate_highlight_score(
        self,
        velocity: float,
        avg_sentiment: float,
        emote_density: float,
        sentiment_distribution: Dict[str, int]
    ) -> float:
        """Calculate highlight score from chat metrics.
        
        Args:
            velocity: Messages per second
            avg_sentiment: Average sentiment score
            emote_density: Emotes per message
            sentiment_distribution: Distribution of sentiment categories
            
        Returns:
            Highlight score between 0.0 and 1.0
        """
        # Velocity component
        velocity_score = min(velocity / self.config.velocity_spike_threshold, 1.0)
        
        # Sentiment component (positive sentiment is good)
        sentiment_score = (avg_sentiment + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        
        # Boost for hype messages
        total_messages = sum(sentiment_distribution.values())
        if total_messages > 0:
            hype_ratio = sentiment_distribution.get("hype", 0) / total_messages
            sentiment_score = min(sentiment_score + (hype_ratio * 0.3), 1.0)
        
        # Emote density component
        emote_score = min(emote_density * 2, 1.0)  # Assume 0.5 emotes/message is high
        
        # Weighted combination
        final_score = (
            velocity_score * self.config.velocity_weight +
            sentiment_score * self.config.sentiment_weight +
            emote_score * self.config.emote_density_weight
        )
        
        return min(final_score, 1.0)
    
    def _extract_emotes(self, text: str) -> List[str]:
        """Extract emotes from message text.
        
        Args:
            text: Message text
            
        Returns:
            List of emotes found
        """
        # Simple pattern matching for common emotes
        # In a real implementation, this would use platform-specific emote data
        
        common_emotes = [
            "Kappa", "PogChamp", "LUL", "KEKW", "OMEGALUL",
            "monkaS", "Pepega", "5Head", "EZ", "GG",
            "PepeHands", "FeelsBadMan", "FeelsGoodMan"
        ]
        
        found_emotes = []
        for emote in common_emotes:
            if emote in text:
                found_emotes.append(emote)
        
        return found_emotes
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get chat processing statistics.
        
        Returns:
            Dictionary of processing statistics
        """
        return {
            "windows_processed": self._processed_windows,
            "messages_in_buffer": len(self._message_buffer),
            "config": {
                "window_size": self.config.window_size_seconds,
                "min_messages": self.config.min_messages_for_analysis,
                "highlight_threshold": self.config.highlight_threshold
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up chat processor resources."""
        self._message_buffer.clear()
        self._processed_windows = 0
        logger.info("Chat processor cleanup completed")