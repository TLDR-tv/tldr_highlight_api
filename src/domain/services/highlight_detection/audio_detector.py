"""
Audio-based highlight detection for the TL;DR Highlight API.

This module implements sophisticated audio analysis algorithms for identifying
exciting moments through volume spike detection, keyword matching, and
spectral analysis of audio content.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field

from .base_detector import (
    BaseDetector,
    ContentSegment,
    DetectionConfig,
    DetectionResult,
    ModalityType,
)
from ...utils.ml_utils import audio_feature_extractor
from ...utils.scoring_utils import (
    normalize_score,
    calculate_confidence,
)

logger = logging.getLogger(__name__)


class AudioDetectionConfig(DetectionConfig):
    """
    Configuration for audio-based highlight detection.

    Extends base configuration with audio-specific parameters
    for volume analysis, keyword detection, and spectral analysis.
    """

    # Volume spike detection parameters
    volume_spike_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Threshold for volume spike detection"
    )
    volume_spike_weight: float = Field(
        default=0.35, ge=0.0, description="Weight for volume spike features in scoring"
    )
    volume_smoothing_window: float = Field(
        default=2.0, gt=0.0, description="Window size for volume smoothing (seconds)"
    )

    # Keyword detection parameters
    keyword_detection_enabled: bool = Field(
        default=True, description="Enable keyword-based excitement detection"
    )
    keyword_weight: float = Field(
        default=0.25, ge=0.0, description="Weight for keyword features in scoring"
    )
    custom_keywords: List[str] = Field(
        default_factory=list, description="Custom excitement keywords to detect"
    )
    keyword_context_window: float = Field(
        default=5.0,
        gt=0.0,
        description="Time window around keywords to consider (seconds)",
    )

    # Spectral analysis parameters
    spectral_analysis_enabled: bool = Field(
        default=True, description="Enable spectral analysis for excitement detection"
    )
    spectral_weight: float = Field(
        default=0.25, ge=0.0, description="Weight for spectral features in scoring"
    )
    high_frequency_threshold: float = Field(
        default=4000.0,
        gt=0.0,
        description="Threshold frequency for high-frequency excitement (Hz)",
    )

    # Speech pattern analysis parameters
    speech_analysis_enabled: bool = Field(
        default=True, description="Enable speech pattern analysis"
    )
    speech_weight: float = Field(
        default=0.15,
        ge=0.0,
        description="Weight for speech pattern features in scoring",
    )
    speech_rate_threshold: float = Field(
        default=0.6,
        ge=0.0,
        description="Threshold for speech rate excitement detection",
    )

    # Audio preprocessing parameters
    sample_rate: int = Field(
        default=44100, gt=0, description="Expected audio sample rate (Hz)"
    )
    frame_size: int = Field(
        default=2048, gt=0, description="Frame size for spectral analysis"
    )
    hop_length: int = Field(
        default=512, gt=0, description="Hop length for spectral analysis"
    )

    # Noise reduction parameters
    noise_reduction_enabled: bool = Field(
        default=True, description="Enable noise reduction preprocessing"
    )
    noise_threshold: float = Field(
        default=0.05, ge=0.0, description="Threshold for noise detection"
    )


@dataclass
class AudioSegmentData:
    """
    Represents audio segment data for analysis.

    Contains audio samples, metadata, and analysis results.
    """

    start_time: float
    end_time: float
    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis results (computed lazily)
    _rms_energy: Optional[np.ndarray] = None
    _spectral_features: Optional[Dict[str, np.ndarray]] = None
    _transcription: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get audio segment duration in seconds."""
        return self.end_time - self.start_time

    @property
    def sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.samples)

    def get_rms_energy(
        self, frame_size: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Compute RMS energy over time.

        Args:
            frame_size: Size of analysis frames
            hop_length: Hop length between frames

        Returns:
            RMS energy time series
        """
        if self._rms_energy is not None:
            return self._rms_energy

        # Ensure mono audio
        if len(self.samples.shape) > 1 and self.samples.shape[1] > 1:
            audio = np.mean(self.samples, axis=1)
        else:
            audio = self.samples.flatten()

        # Compute RMS energy
        rms_frames = []
        for i in range(0, len(audio) - frame_size, hop_length):
            frame = audio[i : i + frame_size]
            rms = np.sqrt(np.mean(frame**2))
            rms_frames.append(rms)

        self._rms_energy = np.array(rms_frames)
        return self._rms_energy

    def get_spectral_features(
        self, frame_size: int = 2048, hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        Compute spectral features over time.

        Args:
            frame_size: Size of analysis frames
            hop_length: Hop length between frames

        Returns:
            Dictionary of spectral features
        """
        if self._spectral_features is not None:
            return self._spectral_features

        # Ensure mono audio
        if len(self.samples.shape) > 1 and self.samples.shape[1] > 1:
            audio = np.mean(self.samples, axis=1)
        else:
            audio = self.samples.flatten()

        # Compute spectral features
        spectral_centroids = []
        spectral_rolloffs = []
        spectral_bandwidths = []
        zero_crossing_rates = []

        for i in range(0, len(audio) - frame_size, hop_length):
            frame = audio[i : i + frame_size]

            # FFT
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft[: frame_size // 2])
            freqs = np.fft.fftfreq(frame_size, 1 / self.sample_rate)[: frame_size // 2]

            # Spectral centroid
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0.0
            spectral_centroids.append(centroid)

            # Spectral rolloff (85% of energy)
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                rolloff = freqs[rolloff_idx[0]]
            else:
                rolloff = freqs[-1]
            spectral_rolloffs.append(rolloff)

            # Spectral bandwidth
            if np.sum(magnitude) > 0:
                bandwidth = np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
                )
            else:
                bandwidth = 0.0
            spectral_bandwidths.append(bandwidth)

            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zero_crossing_rates.append(zero_crossings)

        self._spectral_features = {
            "spectral_centroid": np.array(spectral_centroids),
            "spectral_rolloff": np.array(spectral_rolloffs),
            "spectral_bandwidth": np.array(spectral_bandwidths),
            "zero_crossing_rate": np.array(zero_crossing_rates),
        }

        return self._spectral_features

    def get_transcription(self) -> Optional[str]:
        """
        Get transcription of audio segment.

        Returns:
            Transcribed text (placeholder implementation)
        """
        if self._transcription is not None:
            return self._transcription

        # Placeholder for actual speech-to-text implementation
        # In production, this would use a real STT service
        self._transcription = self.metadata.get("transcription", "")
        return self._transcription


class AudioExcitementAnalyzer:
    """
    Analyzes audio content for excitement and activity indicators.

    Implements various algorithms for detecting audio excitement,
    including volume spikes, keyword detection, and spectral analysis.
    """

    def __init__(self, config: AudioDetectionConfig):
        """
        Initialize the audio excitement analyzer.

        Args:
            config: Audio detection configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AudioExcitementAnalyzer")

        # Define excitement keywords
        self.base_keywords = {
            # Gaming keywords
            "wow",
            "epic",
            "insane",
            "incredible",
            "amazing",
            "awesome",
            "sick",
            "clutch",
            "poggers",
            "pog",
            "lets go",
            "let's go",
            "holy",
            "oh my god",
            "omg",
            "no way",
            "unbelievable",
            "crazy",
            "nuts",
            "wild",
            "fire",
            "lit",
            "hype",
            "beast",
            # Sports keywords
            "goal",
            "score",
            "touchdown",
            "home run",
            "slam dunk",
            "knockout",
            "winner",
            "champion",
            "victory",
            "comeback",
            "upset",
            "record",
            "fantastic",
            "brilliant",
            # General excitement
            "yes",
            "yeah",
            "woah",
            "whoa",
            "incredible",
            "outstanding",
            "phenomenal",
            "spectacular",
            "extraordinary",
            "perfect",
            "flawless",
            "legendary",
            "godlike",
        }

        # Add custom keywords
        if self.config.custom_keywords:
            self.base_keywords.update(self.config.custom_keywords)

        # Compile keyword patterns
        self.keyword_patterns = [
            re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            for keyword in self.base_keywords
        ]

    async def analyze_volume_spikes(
        self, audio_segment: AudioSegmentData
    ) -> Dict[str, float]:
        """
        Analyze volume spikes in audio segment.

        Args:
            audio_segment: Audio segment to analyze

        Returns:
            Dictionary with volume spike analysis results
        """
        try:
            # Get RMS energy
            rms_energy = audio_segment.get_rms_energy()

            if len(rms_energy) == 0:
                return {"volume_spike_score": 0.0, "volume_consistency": 0.0}

            # Smooth energy values
            if len(rms_energy) > 3:
                # Simple moving average
                window_size = min(5, len(rms_energy) // 2)
                smoothed_energy = np.convolve(
                    rms_energy, np.ones(window_size) / window_size, mode="same"
                )
            else:
                smoothed_energy = rms_energy

            # Find volume spikes
            mean_energy = np.mean(smoothed_energy)
            std_energy = np.std(smoothed_energy)
            spike_threshold = mean_energy + (std_energy * 2.0)  # 2 standard deviations

            # Count significant spikes
            spikes = smoothed_energy > spike_threshold
            spike_count = np.sum(spikes)

            # Calculate spike intensity
            if spike_count > 0:
                spike_intensity = np.mean(smoothed_energy[spikes]) / (
                    mean_energy + 1e-10
                )
            else:
                spike_intensity = 1.0

            # Volume consistency (inverse of variance)
            volume_consistency = 1.0 - min(1.0, std_energy / (mean_energy + 1e-10))

            # Normalize spike score
            spike_score = min(
                1.0, (spike_count / len(smoothed_energy)) * spike_intensity
            )

            return {
                "volume_spike_score": normalize_score(spike_score, method="sigmoid"),
                "volume_consistency": max(0.0, volume_consistency),
                "spike_count": spike_count,
                "spike_intensity": spike_intensity,
                "mean_energy": mean_energy,
            }

        except Exception as e:
            self.logger.error(f"Error in volume spike analysis: {e}")
            return {"volume_spike_score": 0.0, "volume_consistency": 0.0}

    async def analyze_keywords(
        self, audio_segment: AudioSegmentData
    ) -> Dict[str, float]:
        """
        Analyze keyword-based excitement in audio segment.

        Args:
            audio_segment: Audio segment to analyze

        Returns:
            Dictionary with keyword analysis results
        """
        if not self.config.keyword_detection_enabled:
            return {"keyword_score": 0.0, "keyword_count": 0}

        try:
            # Get transcription
            transcription = audio_segment.get_transcription()

            if not transcription:
                return {"keyword_score": 0.0, "keyword_count": 0}

            # Count keyword matches
            keyword_matches = 0
            matched_keywords = set()

            for pattern in self.keyword_patterns:
                matches = pattern.findall(transcription)
                keyword_matches += len(matches)
                matched_keywords.update(matches)

            # Calculate keyword density
            word_count = len(transcription.split())
            keyword_density = keyword_matches / max(1, word_count)

            # Normalize keyword score
            keyword_score = normalize_score(
                keyword_density * 10, method="sigmoid"
            )  # Scale up for normalization

            return {
                "keyword_score": keyword_score,
                "keyword_count": keyword_matches,
                "keyword_density": keyword_density,
                "unique_keywords": len(matched_keywords),
            }

        except Exception as e:
            self.logger.error(f"Error in keyword analysis: {e}")
            return {"keyword_score": 0.0, "keyword_count": 0}

    async def analyze_spectral_excitement(
        self, audio_segment: AudioSegmentData
    ) -> Dict[str, float]:
        """
        Analyze spectral features for excitement indicators.

        Args:
            audio_segment: Audio segment to analyze

        Returns:
            Dictionary with spectral analysis results
        """
        if not self.config.spectral_analysis_enabled:
            return {"spectral_score": 0.0, "high_freq_energy": 0.0}

        try:
            # Get spectral features
            spectral_features = audio_segment.get_spectral_features()

            if not spectral_features:
                return {"spectral_score": 0.0, "high_freq_energy": 0.0}

            # Analyze spectral characteristics
            spectral_centroid = spectral_features.get("spectral_centroid", np.array([]))
            _spectral_rolloff = spectral_features.get("spectral_rolloff", np.array([]))
            spectral_bandwidth = spectral_features.get(
                "spectral_bandwidth", np.array([])
            )
            zero_crossing_rate = spectral_features.get(
                "zero_crossing_rate", np.array([])
            )

            # Calculate excitement indicators
            excitement_scores = []

            # High frequency content (indicates excitement/activity)
            if len(spectral_centroid) > 0:
                high_freq_ratio = np.mean(
                    spectral_centroid > self.config.high_frequency_threshold
                )
                excitement_scores.append(high_freq_ratio)

            # Spectral variability (indicates dynamic content)
            if len(spectral_bandwidth) > 0:
                bandwidth_var = np.var(spectral_bandwidth)
                normalized_bandwidth_var = normalize_score(
                    bandwidth_var / 10000, method="sigmoid"
                )
                excitement_scores.append(normalized_bandwidth_var)

            # Zero crossing rate (indicates speech activity)
            if len(zero_crossing_rate) > 0:
                zcr_mean = np.mean(zero_crossing_rate)
                normalized_zcr = normalize_score(zcr_mean * 100, method="sigmoid")
                excitement_scores.append(normalized_zcr)

            # Combine excitement scores
            spectral_score = np.mean(excitement_scores) if excitement_scores else 0.0

            # High frequency energy
            high_freq_energy = (
                np.mean(spectral_centroid > self.config.high_frequency_threshold)
                if len(spectral_centroid) > 0
                else 0.0
            )

            return {
                "spectral_score": spectral_score,
                "high_freq_energy": high_freq_energy,
                "spectral_centroid_mean": np.mean(spectral_centroid)
                if len(spectral_centroid) > 0
                else 0.0,
                "spectral_bandwidth_var": np.var(spectral_bandwidth)
                if len(spectral_bandwidth) > 0
                else 0.0,
                "zero_crossing_rate_mean": np.mean(zero_crossing_rate)
                if len(zero_crossing_rate) > 0
                else 0.0,
            }

        except Exception as e:
            self.logger.error(f"Error in spectral analysis: {e}")
            return {"spectral_score": 0.0, "high_freq_energy": 0.0}

    async def analyze_speech_patterns(
        self, audio_segment: AudioSegmentData
    ) -> Dict[str, float]:
        """
        Analyze speech patterns for excitement indicators.

        Args:
            audio_segment: Audio segment to analyze

        Returns:
            Dictionary with speech pattern analysis results
        """
        if not self.config.speech_analysis_enabled:
            return {"speech_score": 0.0, "speech_rate": 0.0}

        try:
            # Get transcription for speech rate analysis
            transcription = audio_segment.get_transcription()

            if not transcription:
                return {"speech_score": 0.0, "speech_rate": 0.0}

            # Calculate speech rate (words per minute)
            word_count = len(transcription.split())
            duration_minutes = audio_segment.duration / 60.0
            speech_rate = word_count / max(0.1, duration_minutes)

            # Normalize speech rate (typical rate is 150-160 WPM)
            normalized_speech_rate = speech_rate / 160.0

            # Speech excitement score based on rate
            if normalized_speech_rate > self.config.speech_rate_threshold:
                speech_score = normalize_score(
                    normalized_speech_rate - self.config.speech_rate_threshold,
                    method="sigmoid",
                )
            else:
                speech_score = 0.0

            # Additional speech pattern analysis could include:
            # - Pitch variation
            # - Volume dynamics
            # - Pause patterns
            # - Speaking intensity

            return {
                "speech_score": speech_score,
                "speech_rate": speech_rate,
                "normalized_speech_rate": normalized_speech_rate,
                "word_count": word_count,
            }

        except Exception as e:
            self.logger.error(f"Error in speech pattern analysis: {e}")
            return {"speech_score": 0.0, "speech_rate": 0.0}


class AudioDetector(BaseDetector):
    """
    Audio-based highlight detector using volume analysis and keyword detection.

    Implements sophisticated algorithms for identifying exciting moments
    in audio content through volume spike detection, keyword matching,
    spectral analysis, and speech pattern recognition.
    """

    def __init__(self, config: Optional[AudioDetectionConfig] = None):
        """
        Initialize the audio detector.

        Args:
            config: Audio detection configuration
        """
        self.audio_config = config or AudioDetectionConfig()
        super().__init__(self.audio_config)

        self.excitement_analyzer = AudioExcitementAnalyzer(self.audio_config)
        self.logger = logging.getLogger(f"{__name__}.AudioDetector")

    @property
    def modality(self) -> ModalityType:
        """Get the modality this detector handles."""
        return ModalityType.AUDIO

    @property
    def algorithm_name(self) -> str:
        """Get the name of the detection algorithm."""
        return "AudioExcitementDetector"

    @property
    def algorithm_version(self) -> str:
        """Get the version of the detection algorithm."""
        return "1.0.0"

    def _validate_segment(self, segment: ContentSegment) -> bool:
        """
        Validate that a segment contains valid audio data.

        Args:
            segment: Content segment to validate

        Returns:
            True if segment contains valid audio data
        """
        if not super()._validate_segment(segment):
            return False

        # Check if data is audio samples
        if not isinstance(segment.data, (np.ndarray, dict, AudioSegmentData)):
            return False

        # Additional audio-specific validation
        if isinstance(segment.data, np.ndarray):
            if segment.data.size == 0:
                return False

        return True

    def _prepare_audio_segment(
        self, segment: ContentSegment
    ) -> Optional[AudioSegmentData]:
        """
        Prepare audio segment from segment data.

        Args:
            segment: Content segment with audio data

        Returns:
            AudioSegmentData object or None if invalid
        """
        try:
            if isinstance(segment.data, AudioSegmentData):
                return segment.data

            elif isinstance(segment.data, np.ndarray):
                # Create AudioSegmentData from numpy array
                return AudioSegmentData(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    samples=segment.data,
                    sample_rate=segment.metadata.get(
                        "sample_rate", self.audio_config.sample_rate
                    ),
                    channels=segment.metadata.get("channels", 1),
                    metadata=segment.metadata,
                )

            elif isinstance(segment.data, dict):
                # Create from dictionary
                if "samples" in segment.data:
                    return AudioSegmentData(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        samples=np.array(segment.data["samples"]),
                        sample_rate=segment.data.get(
                            "sample_rate", self.audio_config.sample_rate
                        ),
                        channels=segment.data.get("channels", 1),
                        metadata=segment.data.get("metadata", {}),
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error preparing audio segment: {e}")
            return None

    async def _detect_features(
        self, segment: ContentSegment, config: DetectionConfig
    ) -> List[DetectionResult]:
        """
        Detect audio-based highlight features in a segment.

        Args:
            segment: Audio content segment to analyze
            config: Detection configuration

        Returns:
            List of detection results for audio analysis
        """
        audio_config = (
            config if isinstance(config, AudioDetectionConfig) else self.audio_config
        )

        try:
            # Prepare audio segment
            audio_segment = self._prepare_audio_segment(segment)
            if audio_segment is None:
                self.logger.warning(f"Invalid audio segment {segment.segment_id}")
                return []

            self.logger.debug(
                f"Analyzing audio segment {segment.segment_id} ({audio_segment.duration:.2f}s)"
            )

            # Perform various analyses concurrently
            volume_task = self.excitement_analyzer.analyze_volume_spikes(audio_segment)
            keyword_task = self.excitement_analyzer.analyze_keywords(audio_segment)
            spectral_task = self.excitement_analyzer.analyze_spectral_excitement(
                audio_segment
            )
            speech_task = self.excitement_analyzer.analyze_speech_patterns(
                audio_segment
            )

            # Wait for all analyses to complete
            (
                volume_results,
                keyword_results,
                spectral_results,
                speech_results,
            ) = await asyncio.gather(
                volume_task, keyword_task, spectral_task, speech_task
            )

            # Extract ML features
            try:
                ml_features = await audio_feature_extractor.extract_features(
                    audio_segment.samples
                )
            except Exception as e:
                self.logger.warning(f"ML feature extraction failed: {e}")
                ml_features = np.array([])

            # Combine analysis results
            all_scores = {
                "volume": volume_results["volume_spike_score"],
                "keyword": keyword_results["keyword_score"],
                "spectral": spectral_results["spectral_score"],
                "speech": speech_results["speech_score"],
            }

            # Calculate weighted score
            weights = {
                "volume": audio_config.volume_spike_weight,
                "keyword": audio_config.keyword_weight,
                "spectral": audio_config.spectral_weight,
                "speech": audio_config.speech_weight,
            }

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate final score
            final_score = sum(all_scores[k] * weights[k] for k in all_scores.keys())

            # Calculate confidence based on score consistency
            score_values = list(all_scores.values())
            confidence = calculate_confidence(score_values, method="consistency")

            # Check significance thresholds
            if (
                final_score < audio_config.min_score
                or confidence < audio_config.min_confidence
            ):
                return []

            # Create detection result
            result = DetectionResult(
                segment_id=segment.segment_id,
                modality=self.modality,
                score=final_score,
                confidence=confidence,
                features={
                    **all_scores,
                    "spike_count": volume_results.get("spike_count", 0),
                    "keyword_count": keyword_results.get("keyword_count", 0),
                    "high_freq_energy": spectral_results.get("high_freq_energy", 0.0),
                    "speech_rate": speech_results.get("speech_rate", 0.0),
                    "duration": audio_segment.duration,
                },
                metadata={
                    "algorithm": self.algorithm_name,
                    "version": self.algorithm_version,
                    "config": audio_config.dict(),
                    "volume_analysis": volume_results,
                    "keyword_analysis": keyword_results,
                    "spectral_analysis": spectral_results,
                    "speech_analysis": speech_results,
                    "sample_rate": audio_segment.sample_rate,
                    "channels": audio_segment.channels,
                },
                algorithm_version=self.algorithm_version,
            )

            # Add ML features if available
            if ml_features.size > 0:
                result.metadata["ml_features"] = ml_features.tolist()

            self.logger.debug(
                f"Audio analysis complete for segment {segment.segment_id}: "
                f"score={final_score:.3f}, confidence={confidence:.3f}"
            )

            return [result]

        except Exception as e:
            self.logger.error(
                f"Error in audio detection for segment {segment.segment_id}: {e}"
            )
            return []

    async def detect_highlights_from_transcription(
        self,
        transcription_data: List[Dict[str, Any]],
        audio_segments: Optional[List[AudioSegmentData]] = None,
    ) -> List[DetectionResult]:
        """
        Detect highlights from transcription data with optional audio.

        Args:
            transcription_data: List of transcription segments with timestamps
            audio_segments: Optional corresponding audio segments

        Returns:
            List of detection results
        """
        if not transcription_data:
            return []

        segments = []

        for i, transcript in enumerate(transcription_data):
            start_time = transcript.get("start_time", 0.0)
            end_time = transcript.get("end_time", start_time + 10.0)
            text = transcript.get("text", "")

            # Create audio segment data
            audio_data = None
            if audio_segments and i < len(audio_segments):
                audio_data = audio_segments[i]
            else:
                # Create minimal audio segment for transcription analysis
                audio_data = AudioSegmentData(
                    start_time=start_time,
                    end_time=end_time,
                    samples=np.array([]),  # No audio samples
                    sample_rate=self.audio_config.sample_rate,
                    metadata={"transcription": text},
                )

            # Create content segment
            segment = ContentSegment(
                start_time=start_time,
                end_time=end_time,
                data=audio_data,
                metadata={"transcription": text, "source": "transcription"},
            )
            segments.append(segment)

        # Detect highlights in segments
        return await self.detect_highlights(segments)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for audio detection."""
        base_metrics = self.get_metrics()

        # Add audio-specific metrics
        audio_metrics = {
            **base_metrics,
            "algorithm": self.algorithm_name,
            "version": self.algorithm_version,
            "config": self.audio_config.dict(),
            "keyword_count": len(self.excitement_analyzer.base_keywords),
        }

        return audio_metrics
