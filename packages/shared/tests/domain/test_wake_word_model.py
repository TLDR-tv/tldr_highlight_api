"""Tests for the wake word domain model."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from shared.domain.models.wake_word import WakeWord


class TestWakeWordModel:
    """Test wake word model functionality."""
    
    def test_wake_word_creation(self):
        """Test basic wake word creation."""
        wake_word = WakeWord(
            phrase="hey assistant",
            organization_id=uuid4(),
        )
        
        assert wake_word.phrase == "hey assistant"
        assert wake_word.is_active is True
        assert wake_word.case_sensitive is False
        assert wake_word.exact_match is True
        assert wake_word.cooldown_seconds == 30
        assert wake_word.max_edit_distance == 2
        assert wake_word.similarity_threshold == 0.8
        assert wake_word.pre_roll_seconds == 10
        assert wake_word.post_roll_seconds == 30
        assert wake_word.trigger_count == 0
        assert wake_word.last_triggered_at is None
    
    def test_phrase_normalization_case_insensitive(self):
        """Test phrase normalization when case insensitive."""
        wake_word = WakeWord(
            phrase="  HEY Assistant  ",
            case_sensitive=False,
        )
        
        assert wake_word.phrase == "hey assistant"
    
    def test_phrase_normalization_case_sensitive(self):
        """Test phrase preservation when case sensitive."""
        wake_word = WakeWord(
            phrase="  HEY Assistant  ",
            case_sensitive=True,
        )
        
        assert wake_word.phrase == "HEY Assistant"
    
    def test_can_trigger_when_active(self):
        """Test can_trigger returns True for active wake word."""
        wake_word = WakeWord(
            phrase="activate",
            is_active=True,
        )
        
        assert wake_word.can_trigger is True
    
    def test_can_trigger_when_inactive(self):
        """Test can_trigger returns False for inactive wake word."""
        wake_word = WakeWord(
            phrase="activate",
            is_active=False,
        )
        
        assert wake_word.can_trigger is False
    
    def test_can_trigger_cooldown_not_met(self):
        """Test can_trigger respects cooldown period."""
        wake_word = WakeWord(
            phrase="activate",
            cooldown_seconds=60,
            last_triggered_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        
        assert wake_word.can_trigger is False
    
    def test_can_trigger_cooldown_met(self):
        """Test can_trigger allows trigger after cooldown."""
        wake_word = WakeWord(
            phrase="activate",
            cooldown_seconds=60,
            last_triggered_at=datetime.now(timezone.utc) - timedelta(seconds=61),
        )
        
        assert wake_word.can_trigger is True
    
    def test_record_trigger(self):
        """Test recording a trigger updates count and timestamp."""
        wake_word = WakeWord(phrase="trigger me")
        initial_count = wake_word.trigger_count
        
        wake_word.record_trigger()
        
        assert wake_word.trigger_count == initial_count + 1
        assert wake_word.last_triggered_at is not None
        assert isinstance(wake_word.last_triggered_at, datetime)
        assert wake_word.last_triggered_at.tzinfo is not None
    
    def test_exact_match_single_word(self):
        """Test exact match for single word."""
        wake_word = WakeWord(
            phrase="activate",
            exact_match=True,
            case_sensitive=False,
        )
        
        # Should match with word boundaries
        assert wake_word.matches("activate the system") is True
        assert wake_word.matches("please activate now") is True
        assert wake_word.matches("activate") is True
        
        # Should not match partial words
        assert wake_word.matches("reactivate") is False
        assert wake_word.matches("activated") is False
        assert wake_word.matches("deactivate") is False
    
    def test_exact_match_multi_word(self):
        """Test exact match for multi-word phrase."""
        wake_word = WakeWord(
            phrase="hey assistant",
            exact_match=True,
            case_sensitive=False,
        )
        
        # Should match
        assert wake_word.matches("hey assistant help me") is True
        assert wake_word.matches("say hey assistant please") is True
        
        # Should not match partial
        assert wake_word.matches("hey") is False
        assert wake_word.matches("assistant") is False
        assert wake_word.matches("hey there assistant") is False
    
    def test_partial_match(self):
        """Test partial match mode."""
        wake_word = WakeWord(
            phrase="assist",
            exact_match=False,
            case_sensitive=False,
        )
        
        # Should match partials
        assert wake_word.matches("assistant") is True
        assert wake_word.matches("assistance") is True
        assert wake_word.matches("please assist me") is True
        assert wake_word.matches("unassisted") is True
    
    def test_case_sensitive_matching(self):
        """Test case sensitive matching."""
        wake_word = WakeWord(
            phrase="JARVIS",
            exact_match=True,
            case_sensitive=True,
        )
        
        # Should match exact case
        assert wake_word.matches("Hey JARVIS") is True
        
        # Should not match different case
        assert wake_word.matches("Hey jarvis") is False
        assert wake_word.matches("Hey Jarvis") is False
    
    def test_special_characters_in_phrase(self):
        """Test wake words with special regex characters."""
        wake_word = WakeWord(
            phrase="ok, google.",
            exact_match=True,
        )
        
        # Should handle special chars
        assert wake_word.matches("say ok, google. please") is True
        
        # Test other special chars
        wake_word_special = WakeWord(
            phrase="hey (bot)",
            exact_match=True,
        )
        assert wake_word_special.matches("hey (bot) help") is True
    
    def test_empty_phrase_edge_case(self):
        """Test edge case of empty phrase."""
        wake_word = WakeWord(phrase="")
        
        assert wake_word.phrase == ""
        assert wake_word.matches("anything") is False
        assert wake_word.matches("") is True
    
    def test_unicode_phrase(self):
        """Test wake words with unicode characters."""
        wake_word = WakeWord(
            phrase="héy bøt",
            case_sensitive=False,
        )
        
        assert wake_word.phrase == "héy bøt"
        assert wake_word.matches("say héy bøt please") is True
    
    def test_very_long_phrase(self):
        """Test wake word with very long phrase."""
        long_phrase = " ".join(["word"] * 50)
        wake_word = WakeWord(phrase=long_phrase)
        
        assert wake_word.phrase == long_phrase
        assert wake_word.matches(f"start {long_phrase} end") is True
    
    def test_phrase_with_numbers(self):
        """Test wake word with numbers."""
        wake_word = WakeWord(
            phrase="agent 007",
            exact_match=True,
        )
        
        assert wake_word.matches("call agent 007 now") is True
        assert wake_word.matches("agent007") is False
    
    def test_cooldown_edge_cases(self):
        """Test cooldown edge cases."""
        # Zero cooldown
        wake_word = WakeWord(
            phrase="test",
            cooldown_seconds=0,
            last_triggered_at=datetime.now(timezone.utc),
        )
        assert wake_word.can_trigger is True
        
        # Very large cooldown
        wake_word_large = WakeWord(
            phrase="test",
            cooldown_seconds=86400,  # 24 hours
            last_triggered_at=datetime.now(timezone.utc) - timedelta(hours=23),
        )
        assert wake_word_large.can_trigger is False
    
    def test_configuration_limits(self):
        """Test configuration parameter limits."""
        # Test max edit distance limits
        wake_word = WakeWord(
            phrase="test",
            max_edit_distance=0,  # Exact match only
        )
        assert wake_word.max_edit_distance == 0
        
        # Test similarity threshold boundaries
        wake_word_sim = WakeWord(
            phrase="test",
            similarity_threshold=1.0,  # Perfect match required
        )
        assert wake_word_sim.similarity_threshold == 1.0
        
        wake_word_sim_low = WakeWord(
            phrase="test",
            similarity_threshold=0.0,  # Any match accepted
        )
        assert wake_word_sim_low.similarity_threshold == 0.0
    
    def test_clip_duration_configuration(self):
        """Test clip duration settings."""
        wake_word = WakeWord(
            phrase="record",
            pre_roll_seconds=0,  # No pre-roll
            post_roll_seconds=300,  # 5 minute post-roll
        )
        
        assert wake_word.pre_roll_seconds == 0
        assert wake_word.post_roll_seconds == 300
    
    def test_multiple_triggers_tracking(self):
        """Test multiple trigger tracking."""
        wake_word = WakeWord(phrase="test")
        
        # Record multiple triggers
        for i in range(5):
            wake_word.record_trigger()
        
        assert wake_word.trigger_count == 5
        assert wake_word.last_triggered_at is not None


class TestWakeWordEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_phrase_with_only_whitespace(self):
        """Test phrase that's only whitespace."""
        wake_word = WakeWord(phrase="   \t\n   ")
        assert wake_word.phrase == ""
    
    def test_regex_injection_attempt(self):
        """Test that regex special characters are properly escaped."""
        wake_word = WakeWord(
            phrase=".*",
            exact_match=True,
        )
        
        # Should not match everything
        assert wake_word.matches("anything") is False
        assert wake_word.matches("test .* test") is True
    
    def test_concurrent_trigger_scenario(self):
        """Test rapid concurrent triggers respect cooldown."""
        wake_word = WakeWord(
            phrase="test",
            cooldown_seconds=30,
        )
        
        # First trigger
        wake_word.record_trigger()
        first_trigger_time = wake_word.last_triggered_at
        
        # Immediate second trigger should be blocked
        assert wake_word.can_trigger is False
        
        # Force update timestamp to simulate time passing
        wake_word.last_triggered_at = first_trigger_time - timedelta(seconds=31)
        assert wake_word.can_trigger is True
    
    def test_phrase_normalization_preserves_internal_spaces(self):
        """Test that internal spaces in phrases are preserved."""
        wake_word = WakeWord(
            phrase="hey    there    assistant",
            case_sensitive=False,
        )
        
        # Should preserve internal spacing after strip
        assert wake_word.phrase == "hey    there    assistant"
    
    def test_null_and_none_handling(self):
        """Test handling of null/None values."""
        wake_word = WakeWord(
            phrase="test",
            last_triggered_at=None,
        )
        
        assert wake_word.can_trigger is True
        assert wake_word.last_triggered_at is None
    
    def test_timezone_aware_timestamps(self):
        """Test that timestamps are timezone aware."""
        wake_word = WakeWord(phrase="test")
        wake_word.record_trigger()
        
        assert wake_word.last_triggered_at.tzinfo is not None
        assert wake_word.last_triggered_at.tzinfo == timezone.utc