"""
Comprehensive chat sentiment analysis system for the TL;DR Highlight API.

This module implements a multi-modal sentiment analysis system that analyzes:
- Text sentiment using NLP (VADER, TextBlob, transformer models)
- Emote-based sentiment (platform-specific emote meanings)
- Chat velocity and acceleration (message frequency changes)
- Event impact scoring (raids, donations have high impact)
- Keyword/phrase detection for hype moments
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from .base import ChatEvent, ChatEventType, ChatMessage, ChatUser
from ...utils.nlp_utils import sentiment_analyzer as nlp_sentiment_analyzer
from ...utils.scoring_utils import normalize_score, calculate_confidence

logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentCategory(str, Enum):
    """Categories of sentiment."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    HYPE = "hype"
    EXCITEMENT = "excitement"
    DISAPPOINTMENT = "disappointment"
    SURPRISE = "surprise"


@dataclass
class EmoteSentiment:
    """Sentiment information for an emote."""
    name: str
    sentiment_value: float  # -1.0 to 1.0
    intensity: float  # 0.0 to 1.0
    category: SentimentCategory
    platform: str


@dataclass
class MessageSentiment:
    """Sentiment analysis result for a single message."""
    message_id: str
    timestamp: datetime
    text_sentiment: float  # -1.0 to 1.0
    emote_sentiment: float  # -1.0 to 1.0
    combined_sentiment: float  # -1.0 to 1.0
    intensity: float  # 0.0 to 1.0
    category: SentimentCategory
    confidence: float  # 0.0 to 1.0
    emote_count: int
    keyword_matches: List[str]
    is_spam: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowSentiment:
    """Aggregated sentiment for a time window."""
    start_time: datetime
    end_time: datetime
    message_count: int
    unique_users: int
    avg_sentiment: float
    sentiment_variance: float
    dominant_category: SentimentCategory
    intensity: float
    momentum: float  # Rate of sentiment change
    confidence: float
    emote_density: float
    keyword_density: float
    spam_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VelocityMetrics:
    """Chat velocity and acceleration metrics."""
    timestamp: datetime
    messages_per_second: float
    acceleration: float  # Change in velocity
    jerk: float  # Change in acceleration
    unique_users_per_second: float
    emotes_per_second: float
    spike_detected: bool
    spike_intensity: float  # 0.0 to 1.0


@dataclass
class EventImpact:
    """Impact scoring for special events."""
    event_type: ChatEventType
    timestamp: datetime
    impact_score: float  # 0.0 to 1.0
    duration: float  # Expected impact duration in seconds
    decay_rate: float  # How quickly impact decays


class EmoteDatabase:
    """Database of platform-specific emotes and their sentiments."""
    
    def __init__(self):
        self.emotes: Dict[str, Dict[str, EmoteSentiment]] = {
            "twitch": self._load_twitch_emotes(),
            "youtube": self._load_youtube_emotes(),
            "generic": self._load_generic_emotes()
        }
        self.custom_emotes: Dict[str, EmoteSentiment] = {}
        
    def _load_twitch_emotes(self) -> Dict[str, EmoteSentiment]:
        """Load Twitch-specific emotes with sentiment values."""
        return {
            # Positive emotes
            "PogChamp": EmoteSentiment("PogChamp", 0.9, 0.9, SentimentCategory.HYPE, "twitch"),
            "Pog": EmoteSentiment("Pog", 0.9, 0.9, SentimentCategory.HYPE, "twitch"),
            "PogU": EmoteSentiment("PogU", 0.9, 0.9, SentimentCategory.HYPE, "twitch"),
            "POGGERS": EmoteSentiment("POGGERS", 0.95, 1.0, SentimentCategory.HYPE, "twitch"),
            "Kreygasm": EmoteSentiment("Kreygasm", 0.8, 0.8, SentimentCategory.EXCITEMENT, "twitch"),
            "EZ": EmoteSentiment("EZ", 0.6, 0.5, SentimentCategory.POSITIVE, "twitch"),
            "GG": EmoteSentiment("GG", 0.7, 0.6, SentimentCategory.POSITIVE, "twitch"),
            "GGWP": EmoteSentiment("GGWP", 0.8, 0.6, SentimentCategory.POSITIVE, "twitch"),
            
            # Neutral emotes
            "Kappa": EmoteSentiment("Kappa", 0.0, 0.3, SentimentCategory.NEUTRAL, "twitch"),
            "4Head": EmoteSentiment("4Head", 0.1, 0.4, SentimentCategory.NEUTRAL, "twitch"),
            "LUL": EmoteSentiment("LUL", 0.2, 0.5, SentimentCategory.POSITIVE, "twitch"),
            "KEKW": EmoteSentiment("KEKW", 0.2, 0.6, SentimentCategory.POSITIVE, "twitch"),
            "OMEGALUL": EmoteSentiment("OMEGALUL", 0.3, 0.7, SentimentCategory.POSITIVE, "twitch"),
            
            # Negative emotes
            "ResidentSleeper": EmoteSentiment("ResidentSleeper", -0.6, 0.5, SentimentCategory.NEGATIVE, "twitch"),
            "NotLikeThis": EmoteSentiment("NotLikeThis", -0.5, 0.6, SentimentCategory.DISAPPOINTMENT, "twitch"),
            "FailFish": EmoteSentiment("FailFish", -0.4, 0.5, SentimentCategory.DISAPPOINTMENT, "twitch"),
            "BibleThump": EmoteSentiment("BibleThump", -0.3, 0.7, SentimentCategory.NEGATIVE, "twitch"),
            "PepeHands": EmoteSentiment("PepeHands", -0.4, 0.6, SentimentCategory.DISAPPOINTMENT, "twitch"),
            "Sadge": EmoteSentiment("Sadge", -0.5, 0.7, SentimentCategory.NEGATIVE, "twitch"),
            
            # Surprise emotes
            "monkaS": EmoteSentiment("monkaS", 0.0, 0.8, SentimentCategory.SURPRISE, "twitch"),
            "monkaW": EmoteSentiment("monkaW", -0.1, 0.9, SentimentCategory.SURPRISE, "twitch"),
            "WutFace": EmoteSentiment("WutFace", 0.0, 0.7, SentimentCategory.SURPRISE, "twitch"),
            "D:": EmoteSentiment("D:", -0.2, 0.6, SentimentCategory.SURPRISE, "twitch"),
        }
    
    def _load_youtube_emotes(self) -> Dict[str, EmoteSentiment]:
        """Load YouTube-specific emotes and emojis."""
        return {
            # Common YouTube emojis
            "❤️": EmoteSentiment("heart", 0.8, 0.7, SentimentCategory.POSITIVE, "youtube"),
            "👍": EmoteSentiment("thumbsup", 0.7, 0.6, SentimentCategory.POSITIVE, "youtube"),
            "🔥": EmoteSentiment("fire", 0.9, 0.8, SentimentCategory.HYPE, "youtube"),
            "😂": EmoteSentiment("joy", 0.6, 0.7, SentimentCategory.POSITIVE, "youtube"),
            "🎉": EmoteSentiment("party", 0.9, 0.9, SentimentCategory.EXCITEMENT, "youtube"),
            "💯": EmoteSentiment("100", 0.8, 0.8, SentimentCategory.HYPE, "youtube"),
            "😱": EmoteSentiment("scream", 0.1, 0.9, SentimentCategory.SURPRISE, "youtube"),
            "😭": EmoteSentiment("sob", -0.5, 0.8, SentimentCategory.NEGATIVE, "youtube"),
            "👎": EmoteSentiment("thumbsdown", -0.7, 0.6, SentimentCategory.NEGATIVE, "youtube"),
            "😴": EmoteSentiment("sleeping", -0.4, 0.4, SentimentCategory.NEGATIVE, "youtube"),
        }
    
    def _load_generic_emotes(self) -> Dict[str, EmoteSentiment]:
        """Load generic text emoticons."""
        return {
            ":)": EmoteSentiment("smile", 0.5, 0.4, SentimentCategory.POSITIVE, "generic"),
            ":D": EmoteSentiment("grin", 0.7, 0.6, SentimentCategory.POSITIVE, "generic"),
            ":(": EmoteSentiment("frown", -0.5, 0.4, SentimentCategory.NEGATIVE, "generic"),
            ":O": EmoteSentiment("surprised", 0.0, 0.6, SentimentCategory.SURPRISE, "generic"),
            "XD": EmoteSentiment("laugh", 0.6, 0.7, SentimentCategory.POSITIVE, "generic"),
            "^^": EmoteSentiment("happy", 0.4, 0.3, SentimentCategory.POSITIVE, "generic"),
            "-_-": EmoteSentiment("annoyed", -0.3, 0.4, SentimentCategory.NEGATIVE, "generic"),
            "o_o": EmoteSentiment("shocked", 0.0, 0.7, SentimentCategory.SURPRISE, "generic"),
        }
    
    def get_emote_sentiment(self, emote: str, platform: str = "generic") -> Optional[EmoteSentiment]:
        """Get sentiment for an emote."""
        # Check custom emotes first
        if emote in self.custom_emotes:
            return self.custom_emotes[emote]
        
        # Check platform-specific emotes
        if platform in self.emotes and emote in self.emotes[platform]:
            return self.emotes[platform][emote]
        
        # Check generic emotes
        if emote in self.emotes["generic"]:
            return self.emotes["generic"][emote]
        
        return None
    
    def add_custom_emote(self, emote: str, sentiment_value: float, intensity: float, 
                        category: SentimentCategory, platform: str = "custom"):
        """Add a custom emote to the database."""
        self.custom_emotes[emote] = EmoteSentiment(
            emote, sentiment_value, intensity, category, platform
        )
    
    def learn_emote_from_context(self, emote: str, contexts: List[float], platform: str = "learned"):
        """Learn emote sentiment from usage contexts."""
        if len(contexts) < 5:  # Need minimum contexts to learn
            return
        
        avg_sentiment = np.mean(contexts)
        intensity = np.std(contexts)  # Higher variance = higher intensity
        
        # Determine category based on sentiment
        if avg_sentiment >= 0.7:
            category = SentimentCategory.VERY_POSITIVE
        elif avg_sentiment >= 0.3:
            category = SentimentCategory.POSITIVE
        elif avg_sentiment >= -0.3:
            category = SentimentCategory.NEUTRAL
        elif avg_sentiment >= -0.7:
            category = SentimentCategory.NEGATIVE
        else:
            category = SentimentCategory.VERY_NEGATIVE
        
        self.add_custom_emote(emote, avg_sentiment, min(intensity * 2, 1.0), category, platform)


class HypeKeywordDetector:
    """Detector for hype keywords and phrases."""
    
    def __init__(self):
        self.hype_keywords = self._load_hype_keywords()
        self.regex_patterns = self._compile_patterns()
        
    def _load_hype_keywords(self) -> Dict[str, float]:
        """Load hype keywords with their intensity scores."""
        return {
            # High intensity hype
            "poggers": 0.9,
            "pog": 0.9,
            "holy shit": 0.95,
            "holy fuck": 0.95,
            "insane": 0.85,
            "crazy": 0.8,
            "unreal": 0.85,
            "no way": 0.9,
            "omg": 0.8,
            "wtf": 0.75,
            "lets go": 0.85,
            "let's go": 0.85,
            "letsgoo": 0.9,
            "hype": 0.8,
            "goat": 0.85,
            "godlike": 0.9,
            "legendary": 0.85,
            "epic": 0.8,
            "clutch": 0.9,
            "ace": 0.85,
            "perfect": 0.8,
            "flawless": 0.85,
            
            # Medium intensity
            "wow": 0.6,
            "nice": 0.5,
            "good": 0.4,
            "great": 0.6,
            "awesome": 0.7,
            "amazing": 0.75,
            "sick": 0.7,
            "dope": 0.65,
            "fire": 0.7,
            "lit": 0.65,
            "beast": 0.7,
            "monster": 0.7,
            "king": 0.6,
            "queen": 0.6,
            
            # Surprise/shock
            "what": 0.5,
            "how": 0.5,
            "bruh": 0.6,
            "yo": 0.5,
            "yooo": 0.7,
            "damn": 0.6,
            "shit": 0.6,
            "fuck": 0.7,
            
            # Victory/success
            "win": 0.7,
            "winner": 0.75,
            "champion": 0.8,
            "victory": 0.8,
            "gg": 0.6,
            "ggwp": 0.7,
            "ez": 0.5,
            "easy": 0.5,
            "rekt": 0.7,
            "destroyed": 0.75,
            "dominated": 0.8,
        }
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, float]]:
        """Compile regex patterns for complex hype detection."""
        patterns = [
            # Repeated characters for emphasis
            (re.compile(r'\b(\w)\1{2,}\b', re.IGNORECASE), 0.3),  # "YESSS", "NOOO"
            # Multiple exclamation marks
            (re.compile(r'!{2,}'), 0.4),
            # All caps words (longer than 3 chars)
            (re.compile(r'\b[A-Z]{4,}\b'), 0.3),
            # Number + "Head" (like 5Head)
            (re.compile(r'\b\d+head\b', re.IGNORECASE), 0.4),
            # "POG" variations
            (re.compile(r'\bpog\w*\b', re.IGNORECASE), 0.8),
            # "W" or "L" as standalone
            (re.compile(r'\b[WL]\b'), 0.5),
            # Clap emojis or text
            (re.compile(r'👏|clap', re.IGNORECASE), 0.4),
        ]
        return patterns
    
    def detect_keywords(self, text: str) -> Tuple[List[str], float]:
        """Detect hype keywords in text and return matches with intensity."""
        text_lower = text.lower()
        matches = []
        total_intensity = 0.0
        
        # Check direct keywords
        for keyword, intensity in self.hype_keywords.items():
            if keyword in text_lower:
                matches.append(keyword)
                total_intensity += intensity
        
        # Check patterns
        for pattern, intensity in self.regex_patterns:
            if pattern.search(text):
                matches.append(f"pattern:{pattern.pattern}")
                total_intensity += intensity
        
        # Normalize intensity
        if matches:
            avg_intensity = total_intensity / len(matches)
        else:
            avg_intensity = 0.0
        
        return matches, min(avg_intensity, 1.0)


class TemporalAnalyzer:
    """Analyzer for temporal patterns in chat."""
    
    def __init__(self, window_size: int = 300, sample_rate: float = 1.0):
        """
        Initialize temporal analyzer.
        
        Args:
            window_size: Size of the analysis window in seconds
            sample_rate: How often to sample metrics (seconds)
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.message_times: Deque[datetime] = deque()
        self.user_times: Dict[str, Deque[datetime]] = defaultdict(deque)
        self.emote_times: Deque[datetime] = deque()
        self.velocity_history: Deque[VelocityMetrics] = deque(maxlen=60)  # 1 minute history
        
    def add_message(self, timestamp: datetime, user_id: str, has_emotes: bool = False):
        """Add a message to the temporal analysis."""
        # Add to message times
        self.message_times.append(timestamp)
        self._cleanup_old_messages(timestamp)
        
        # Add to user times
        self.user_times[user_id].append(timestamp)
        
        # Add to emote times if applicable
        if has_emotes:
            self.emote_times.append(timestamp)
    
    def _cleanup_old_messages(self, current_time: datetime):
        """Remove messages outside the analysis window."""
        cutoff_time = current_time.timestamp() - self.window_size
        
        # Clean message times
        while self.message_times and self.message_times[0].timestamp() < cutoff_time:
            self.message_times.popleft()
        
        # Clean user times
        for user_id, times in list(self.user_times.items()):
            while times and times[0].timestamp() < cutoff_time:
                times.popleft()
            if not times:
                del self.user_times[user_id]
        
        # Clean emote times
        while self.emote_times and self.emote_times[0].timestamp() < cutoff_time:
            self.emote_times.popleft()
    
    def calculate_velocity_metrics(self, timestamp: datetime) -> VelocityMetrics:
        """Calculate current velocity metrics."""
        self._cleanup_old_messages(timestamp)
        
        # Calculate messages per second
        if len(self.message_times) > 1:
            time_span = (self.message_times[-1] - self.message_times[0]).total_seconds()
            if time_span > 0:
                messages_per_second = len(self.message_times) / time_span
            else:
                messages_per_second = 0.0
        else:
            messages_per_second = 0.0
        
        # Calculate unique users per second
        unique_users = len(self.user_times)
        unique_users_per_second = unique_users / max(self.window_size, 1)
        
        # Calculate emotes per second
        if len(self.emote_times) > 0:
            emotes_per_second = len(self.emote_times) / max(self.window_size, 1)
        else:
            emotes_per_second = 0.0
        
        # Calculate acceleration and jerk
        acceleration = 0.0
        jerk = 0.0
        
        if len(self.velocity_history) > 0:
            prev_velocity = self.velocity_history[-1]
            time_diff = (timestamp - prev_velocity.timestamp).total_seconds()
            
            if time_diff > 0:
                acceleration = (messages_per_second - prev_velocity.messages_per_second) / time_diff
                
                if len(self.velocity_history) > 1:
                    prev_acceleration = prev_velocity.acceleration
                    jerk = (acceleration - prev_acceleration) / time_diff
        
        # Detect spikes
        spike_detected = False
        spike_intensity = 0.0
        
        if len(self.velocity_history) >= 5:
            recent_velocities = [v.messages_per_second for v in list(self.velocity_history)[-5:]]
            avg_velocity = np.mean(recent_velocities)
            std_velocity = np.std(recent_velocities)
            
            if std_velocity > 0:
                z_score = (messages_per_second - avg_velocity) / std_velocity
                if z_score > 2:  # 2 standard deviations above mean
                    spike_detected = True
                    spike_intensity = min(z_score / 4, 1.0)  # Normalize to 0-1
        
        metrics = VelocityMetrics(
            timestamp=timestamp,
            messages_per_second=messages_per_second,
            acceleration=acceleration,
            jerk=jerk,
            unique_users_per_second=unique_users_per_second,
            emotes_per_second=emotes_per_second,
            spike_detected=spike_detected,
            spike_intensity=spike_intensity
        )
        
        self.velocity_history.append(metrics)
        return metrics
    
    def calculate_momentum(self, window_seconds: float = 30.0) -> float:
        """Calculate momentum (sustained velocity increase)."""
        if len(self.velocity_history) < 2:
            return 0.0
        
        # Get recent history
        recent_history = list(self.velocity_history)
        if len(recent_history) < 2:
            return 0.0
        
        # Calculate trend
        velocities = [v.messages_per_second for v in recent_history]
        times = [(v.timestamp - recent_history[0].timestamp).total_seconds() for v in recent_history]
        
        if len(set(times)) < 2:  # Need at least 2 different time points
            return 0.0
        
        # Calculate linear regression slope
        coeffs = np.polyfit(times, velocities, 1)
        slope = coeffs[0]
        
        # Normalize momentum
        avg_velocity = np.mean(velocities)
        if avg_velocity > 0:
            momentum = slope / avg_velocity
        else:
            momentum = 0.0
        
        return max(-1.0, min(1.0, momentum))


class EventImpactTracker:
    """Tracks and scores the impact of special events."""
    
    def __init__(self):
        self.active_impacts: List[EventImpact] = []
        self.impact_weights = self._load_impact_weights()
        
    def _load_impact_weights(self) -> Dict[ChatEventType, Tuple[float, float, float]]:
        """Load impact weights for different event types.
        Returns: (impact_score, duration_seconds, decay_rate)
        """
        return {
            ChatEventType.RAID: (1.0, 120.0, 0.02),  # Highest impact, lasts 2 minutes
            ChatEventType.CHEER: (0.8, 60.0, 0.03),  # High impact, lasts 1 minute
            ChatEventType.SUBSCRIBE: (0.7, 45.0, 0.04),
            ChatEventType.RESUBSCRIBE: (0.6, 30.0, 0.05),
            ChatEventType.FOLLOW: (0.4, 20.0, 0.08),
            ChatEventType.HYPE_TRAIN_BEGIN: (0.9, 180.0, 0.01),  # Long lasting
            ChatEventType.HYPE_TRAIN_PROGRESS: (0.7, 60.0, 0.02),
            ChatEventType.HYPE_TRAIN_END: (0.5, 30.0, 0.05),
        }
    
    def add_event(self, event: ChatEvent):
        """Add an event and calculate its impact."""
        if event.type not in self.impact_weights:
            return
        
        impact_score, duration, decay_rate = self.impact_weights[event.type]
        
        # Adjust impact based on event data
        if event.type == ChatEventType.RAID and event.viewers:
            # Scale raid impact by viewer count
            impact_score *= min(1.0, event.viewers / 100)  # 100+ viewers = max impact
        elif event.type == ChatEventType.CHEER and event.amount:
            # Scale cheer impact by bits amount
            impact_score *= min(1.0, event.amount / 1000)  # 1000+ bits = max impact
        elif event.type in [ChatEventType.SUBSCRIBE, ChatEventType.RESUBSCRIBE] and event.tier:
            # Scale sub impact by tier
            tier_multipliers = {"1": 1.0, "2": 1.5, "3": 2.0}
            impact_score *= tier_multipliers.get(event.tier, 1.0)
        
        impact = EventImpact(
            event_type=event.type,
            timestamp=event.timestamp,
            impact_score=impact_score,
            duration=duration,
            decay_rate=decay_rate
        )
        
        self.active_impacts.append(impact)
        self._cleanup_old_impacts()
    
    def _cleanup_old_impacts(self):
        """Remove impacts that have fully decayed."""
        current_time = datetime.now(timezone.utc)
        self.active_impacts = [
            impact for impact in self.active_impacts
            if (current_time - impact.timestamp).total_seconds() < impact.duration * 2
        ]
    
    def get_current_impact(self, timestamp: datetime) -> float:
        """Get the combined impact score at a given timestamp."""
        total_impact = 0.0
        
        for impact in self.active_impacts:
            elapsed = (timestamp - impact.timestamp).total_seconds()
            
            if elapsed < 0:  # Future event
                continue
            
            if elapsed < impact.duration:
                # Calculate decayed impact
                decay_factor = np.exp(-impact.decay_rate * elapsed)
                current_impact = impact.impact_score * decay_factor
                total_impact += current_impact
        
        return min(total_impact, 1.0)  # Cap at 1.0


class SentimentAnalyzer:
    """Comprehensive sentiment analysis system for chat."""
    
    def __init__(self, 
                 window_size_seconds: float = 10.0,
                 decay_factor: float = 0.95,
                 min_confidence: float = 0.3):
        """
        Initialize the sentiment analyzer.
        
        Args:
            window_size_seconds: Size of analysis windows in seconds
            decay_factor: Exponential decay factor for older messages
            min_confidence: Minimum confidence threshold
        """
        self.window_size = window_size_seconds
        self.decay_factor = decay_factor
        self.min_confidence = min_confidence
        
        # Initialize components
        self.vader = SentimentIntensityAnalyzer()
        self.emote_db = EmoteDatabase()
        self.keyword_detector = HypeKeywordDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        self.event_tracker = EventImpactTracker()
        
        # Message history for windowed analysis
        self.message_history: Deque[MessageSentiment] = deque()
        self.window_history: Deque[WindowSentiment] = deque(maxlen=100)
        
        # Multi-language support tracking
        self.language_sentiments: Dict[str, float] = {}
        
        logger.info("Initialized comprehensive sentiment analyzer")
    
    async def analyze_message(self, message: ChatMessage, platform: str = "generic") -> MessageSentiment:
        """Analyze sentiment of a single chat message."""
        try:
            # Extract text and emotes
            text = message.text
            emotes = message.emotes if hasattr(message, 'emotes') else []
            
            # Text sentiment analysis
            text_sentiment = await self._analyze_text_sentiment(text)
            
            # Emote sentiment analysis
            emote_sentiment, emote_count = await self._analyze_emote_sentiment(emotes, text, platform)
            
            # Keyword detection
            keywords, keyword_intensity = self.keyword_detector.detect_keywords(text)
            
            # Combine sentiments
            if emote_count > 0:
                # Weight emotes higher if present
                combined_sentiment = (text_sentiment * 0.4 + emote_sentiment * 0.6)
            else:
                combined_sentiment = text_sentiment
            
            # Add keyword boost
            if keyword_intensity > 0:
                combined_sentiment = combined_sentiment * (1 + keyword_intensity * 0.3)
                combined_sentiment = max(-1.0, min(1.0, combined_sentiment))
            
            # Calculate intensity
            intensity = abs(combined_sentiment) * (1 + keyword_intensity * 0.5)
            intensity = min(intensity, 1.0)
            
            # Determine category
            category = self._determine_category(combined_sentiment, intensity, keywords)
            
            # Calculate confidence
            confidence = self._calculate_confidence(text_sentiment, emote_sentiment, emote_count, len(keywords))
            
            # Check for spam
            is_spam = await self._is_spam(text, message.user.id if hasattr(message, 'user') else "")
            
            # Add temporal tracking
            # Handle both datetime and float timestamps
            if isinstance(message.timestamp, datetime):
                msg_timestamp = message.timestamp
            else:
                msg_timestamp = datetime.fromtimestamp(message.timestamp, tz=timezone.utc)
            
            self.temporal_analyzer.add_message(
                msg_timestamp,
                message.user.id if hasattr(message, 'user') else "",
                emote_count > 0
            )
            
            result = MessageSentiment(
                message_id=message.id,
                timestamp=msg_timestamp,
                text_sentiment=text_sentiment,
                emote_sentiment=emote_sentiment,
                combined_sentiment=combined_sentiment,
                intensity=intensity,
                category=category,
                confidence=confidence,
                emote_count=emote_count,
                keyword_matches=keywords,
                is_spam=is_spam,
                metadata={
                    "platform": platform,
                    "keyword_intensity": keyword_intensity,
                    "text_length": len(text),
                    "user_id": message.user.id if hasattr(message, 'user') else None
                }
            )
            
            # Add to history
            self.message_history.append(result)
            self._cleanup_old_messages()
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing message sentiment: {e}")
            # Return neutral sentiment on error
            # Handle both datetime and float timestamps
            if isinstance(message.timestamp, datetime):
                msg_timestamp = message.timestamp
            else:
                msg_timestamp = datetime.fromtimestamp(message.timestamp, tz=timezone.utc)
                
            return MessageSentiment(
                message_id=message.id,
                timestamp=msg_timestamp,
                text_sentiment=0.0,
                emote_sentiment=0.0,
                combined_sentiment=0.0,
                intensity=0.0,
                category=SentimentCategory.NEUTRAL,
                confidence=0.0,
                emote_count=0,
                keyword_matches=[],
                is_spam=False
            )
    
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using multiple methods."""
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
        except:
            textblob_polarity = 0.0
        
        # Use NLP utils sentiment analyzer
        try:
            nlp_sentiment = await nlp_sentiment_analyzer.analyze_sentiment(text)
            nlp_compound = nlp_sentiment.compound
        except:
            nlp_compound = 0.0
        
        # Weighted combination
        combined = (vader_compound * 0.4 + textblob_polarity * 0.3 + nlp_compound * 0.3)
        
        return max(-1.0, min(1.0, combined))
    
    async def _analyze_emote_sentiment(self, emotes: List[Dict[str, Any]], text: str, platform: str) -> Tuple[float, int]:
        """Analyze sentiment from emotes."""
        if not emotes:
            # Look for text-based emotes
            emotes = self._extract_text_emotes(text)
        
        if not emotes:
            return 0.0, 0
        
        total_sentiment = 0.0
        total_intensity = 0.0
        count = 0
        
        for emote in emotes:
            emote_name = emote.get('name', emote) if isinstance(emote, dict) else str(emote)
            emote_info = self.emote_db.get_emote_sentiment(emote_name, platform)
            
            if emote_info:
                total_sentiment += emote_info.sentiment_value * emote_info.intensity
                total_intensity += emote_info.intensity
                count += 1
        
        if count > 0:
            avg_sentiment = total_sentiment / total_intensity if total_intensity > 0 else 0
            return avg_sentiment, count
        
        return 0.0, 0
    
    def _extract_text_emotes(self, text: str) -> List[str]:
        """Extract text-based emotes from message."""
        emotes = []
        
        # Common text emoticons
        emoticon_pattern = re.compile(r'[:;=][-)DP|oO]+|[|oO]-?[-)DP]|[\^>]_*[\^<]|[xX]D+|D:')
        emotes.extend(emoticon_pattern.findall(text))
        
        # Unicode emojis (simplified)
        emoji_pattern = re.compile('[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        emotes.extend(emoji_pattern.findall(text))
        
        return emotes
    
    def _determine_category(self, sentiment: float, intensity: float, keywords: List[str]) -> SentimentCategory:
        """Determine sentiment category based on multiple factors."""
        # Check for hype keywords first
        if keywords and intensity > 0.7:
            if sentiment > 0.3:
                return SentimentCategory.HYPE
            elif sentiment < -0.3:
                return SentimentCategory.DISAPPOINTMENT
        
        # Check for excitement
        if sentiment > 0.5 and intensity > 0.6:
            return SentimentCategory.EXCITEMENT
        
        # Standard categories
        if sentiment >= 0.7:
            return SentimentCategory.VERY_POSITIVE
        elif sentiment >= 0.3:
            return SentimentCategory.POSITIVE
        elif sentiment >= -0.3:
            return SentimentCategory.NEUTRAL
        elif sentiment >= -0.7:
            return SentimentCategory.NEGATIVE
        else:
            return SentimentCategory.VERY_NEGATIVE
    
    def _calculate_confidence(self, text_sentiment: float, emote_sentiment: float, 
                            emote_count: int, keyword_count: int) -> float:
        """Calculate confidence in sentiment analysis."""
        # Base confidence from sentiment strength
        sentiment_confidence = abs(text_sentiment)
        
        # Boost confidence if emotes align with text
        if emote_count > 0:
            alignment = 1 - abs(text_sentiment - emote_sentiment)
            sentiment_confidence = (sentiment_confidence + alignment) / 2
        
        # Boost confidence for keywords
        if keyword_count > 0:
            sentiment_confidence = min(1.0, sentiment_confidence + 0.1 * keyword_count)
        
        return max(self.min_confidence, sentiment_confidence)
    
    async def _is_spam(self, text: str, user_id: str) -> bool:
        """Detect if message is likely spam."""
        # Check for repetitive characters
        if re.search(r'(.)\1{5,}', text):
            return True
        
        # Check for all caps (if long enough)
        if len(text) > 10 and text.isupper():
            return True
        
        # Check for excessive special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.7:
            return True
        
        # Check message history for repetition
        recent_messages = [m for m in self.message_history 
                          if m.metadata.get('user_id') == user_id][-5:]
        if len(recent_messages) >= 3:
            unique_texts = len(set(m.metadata.get('text', '') for m in recent_messages))
            if unique_texts == 1:  # All same message
                return True
        
        return False
    
    def _cleanup_old_messages(self):
        """Remove old messages from history."""
        if not self.message_history:
            return
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time.timestamp() - 300  # 5 minutes
        
        while self.message_history and self.message_history[0].timestamp.timestamp() < cutoff_time:
            self.message_history.popleft()
    
    async def analyze_window(self, messages: List[ChatMessage], 
                           start_time: datetime, end_time: datetime,
                           platform: str = "generic") -> WindowSentiment:
        """Analyze sentiment for a time window of messages."""
        if not messages:
            return WindowSentiment(
                start_time=start_time,
                end_time=end_time,
                message_count=0,
                unique_users=0,
                avg_sentiment=0.0,
                sentiment_variance=0.0,
                dominant_category=SentimentCategory.NEUTRAL,
                intensity=0.0,
                momentum=0.0,
                confidence=0.0,
                emote_density=0.0,
                keyword_density=0.0,
                spam_ratio=0.0
            )
        
        # Analyze all messages
        sentiments = []
        for msg in messages:
            sentiment = await self.analyze_message(msg, platform)
            if not sentiment.is_spam:
                sentiments.append(sentiment)
        
        if not sentiments:
            # All messages were spam
            return WindowSentiment(
                start_time=start_time,
                end_time=end_time,
                message_count=len(messages),
                unique_users=len(set(msg.user.id for msg in messages if hasattr(msg, 'user'))),
                avg_sentiment=0.0,
                sentiment_variance=0.0,
                dominant_category=SentimentCategory.NEUTRAL,
                intensity=0.0,
                momentum=0.0,
                confidence=0.0,
                emote_density=0.0,
                keyword_density=0.0,
                spam_ratio=1.0
            )
        
        # Calculate metrics
        sentiment_values = [s.combined_sentiment for s in sentiments]
        avg_sentiment = np.mean(sentiment_values)
        sentiment_variance = np.var(sentiment_values)
        
        # Calculate dominant category
        category_counts = Counter(s.category for s in sentiments)
        dominant_category = category_counts.most_common(1)[0][0] if category_counts else SentimentCategory.NEUTRAL
        
        # Calculate average intensity
        avg_intensity = np.mean([s.intensity for s in sentiments])
        
        # Calculate momentum
        momentum = self.temporal_analyzer.calculate_momentum()
        
        # Calculate confidence
        avg_confidence = np.mean([s.confidence for s in sentiments])
        
        # Calculate densities
        total_emotes = sum(s.emote_count for s in sentiments)
        total_keywords = sum(len(s.keyword_matches) for s in sentiments)
        emote_density = total_emotes / len(sentiments)
        keyword_density = total_keywords / len(sentiments)
        
        # Calculate spam ratio
        spam_count = len(messages) - len(sentiments)
        spam_ratio = spam_count / len(messages) if messages else 0.0
        
        # Get unique users
        unique_users = len(set(msg.user.id for msg in messages if hasattr(msg, 'user')))
        
        result = WindowSentiment(
            start_time=start_time,
            end_time=end_time,
            message_count=len(messages),
            unique_users=unique_users,
            avg_sentiment=avg_sentiment,
            sentiment_variance=sentiment_variance,
            dominant_category=dominant_category,
            intensity=avg_intensity,
            momentum=momentum,
            confidence=avg_confidence,
            emote_density=emote_density,
            keyword_density=keyword_density,
            spam_ratio=spam_ratio,
            metadata={
                "platform": platform,
                "valid_messages": len(sentiments),
                "category_distribution": dict(category_counts)
            }
        )
        
        # Add to window history
        self.window_history.append(result)
        
        return result
    
    async def process_event(self, event: ChatEvent):
        """Process a chat event for impact tracking."""
        self.event_tracker.add_event(event)
    
    def get_highlight_confidence(self, window: WindowSentiment, 
                               velocity: VelocityMetrics,
                               event_impact: float = 0.0) -> float:
        """Calculate confidence score for highlight detection."""
        scores = []
        
        # Sentiment score
        if window.dominant_category in [SentimentCategory.HYPE, SentimentCategory.EXCITEMENT]:
            sentiment_score = 0.9
        elif window.dominant_category == SentimentCategory.VERY_POSITIVE:
            sentiment_score = 0.7
        elif window.dominant_category == SentimentCategory.POSITIVE:
            sentiment_score = 0.5
        else:
            sentiment_score = 0.2
        
        sentiment_score *= window.intensity
        scores.append(sentiment_score)
        
        # Velocity score
        if velocity.spike_detected:
            velocity_score = 0.8 + (0.2 * velocity.spike_intensity)
        else:
            velocity_score = min(velocity.messages_per_second / 10, 0.5)
        scores.append(velocity_score)
        
        # Momentum score
        momentum_score = max(0, window.momentum) * 0.8
        scores.append(momentum_score)
        
        # Event impact score
        if event_impact > 0:
            scores.append(event_impact)
        
        # Engagement score (unique users)
        engagement_score = min(window.unique_users / 50, 1.0) * 0.6
        scores.append(engagement_score)
        
        # Keyword/emote density score
        density_score = min((window.keyword_density + window.emote_density) / 4, 1.0) * 0.7
        scores.append(density_score)
        
        # Calculate weighted average
        weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]
        if event_impact == 0:
            weights = weights[:3] + weights[4:]  # Skip event weight
            scores = scores[:3] + scores[4:]
        
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Apply confidence multiplier
        final_confidence = weighted_score * window.confidence
        
        # Penalty for high spam ratio
        if window.spam_ratio > 0.5:
            final_confidence *= (1 - window.spam_ratio * 0.5)
        
        return min(final_confidence, 1.0)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of analyzer metrics."""
        return {
            "total_messages_analyzed": len(self.message_history),
            "window_count": len(self.window_history),
            "active_event_impacts": len(self.event_tracker.active_impacts),
            "custom_emotes_learned": len(self.emote_db.custom_emotes),
            "velocity_metrics": {
                "current_mps": self.velocity_history[-1].messages_per_second if self.velocity_history else 0,
                "spike_detected": self.velocity_history[-1].spike_detected if self.velocity_history else False,
            } if hasattr(self, 'velocity_history') and self.velocity_history else {},
            "recent_windows": [
                {
                    "avg_sentiment": w.avg_sentiment,
                    "dominant_category": w.dominant_category,
                    "intensity": w.intensity,
                    "momentum": w.momentum
                }
                for w in list(self.window_history)[-5:]
            ]
        }


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()