"""
Machine learning utilities for the TL;DR Highlight API.

This module provides common ML functionality used across the highlight
detection system, including model loading, feature extraction, and
prediction utilities.
"""

import asyncio
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class FeatureExtractionError(Exception):
    """Exception raised when feature extraction fails."""

    pass


class MLModelWrapper:
    """
    Wrapper for machine learning models with async support.

    Provides a unified interface for different ML frameworks
    and includes caching, error handling, and metrics.
    """

    def __init__(
        self,
        model: BaseEstimator,
        model_name: str,
        version: str = "1.0",
        scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
    ):
        """
        Initialize the ML model wrapper.

        Args:
            model: Trained ML model
            model_name: Name identifier for the model
            version: Model version
            scaler: Optional data scaler
        """
        self.model = model
        self.model_name = model_name
        self.version = version
        self.scaler = scaler
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

        # Performance metrics
        self._metrics = {
            "predictions_made": 0,
            "total_prediction_time": 0.0,
            "errors": 0,
        }

    async def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions asynchronously.

        Args:
            features: Input features for prediction

        Returns:
            Model predictions

        Raises:
            FeatureExtractionError: If feature processing fails
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Validate input
            if features.size == 0:
                raise FeatureExtractionError("Empty feature array")

            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Make prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(None, self.model.predict, features)

            # Update metrics
            prediction_time = asyncio.get_event_loop().time() - start_time
            self._metrics["predictions_made"] += len(predictions)
            self._metrics["total_prediction_time"] += prediction_time

            return np.array(predictions)

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error(f"Prediction error in {self.model_name}: {e}")
            raise FeatureExtractionError(f"Prediction failed: {e}")

    async def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Make probability predictions asynchronously.

        Args:
            features: Input features for prediction

        Returns:
            Prediction probabilities
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"Model {self.model_name} does not support probability prediction"
            )

        start_time = asyncio.get_event_loop().time()

        try:
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Make prediction in thread pool
            loop = asyncio.get_event_loop()
            probabilities = await loop.run_in_executor(
                None, self.model.predict_proba, features
            )

            # Update metrics
            prediction_time = asyncio.get_event_loop().time() - start_time
            self._metrics["predictions_made"] += len(probabilities)
            self._metrics["total_prediction_time"] += prediction_time

            return np.array(probabilities)

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error(f"Probability prediction error in {self.model_name}: {e}")
            raise FeatureExtractionError(f"Probability prediction failed: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        metrics = self._metrics.copy()

        if metrics["predictions_made"] > 0:
            metrics["avg_prediction_time"] = (
                metrics["total_prediction_time"] / metrics["predictions_made"]
            )
            metrics["error_rate"] = metrics["errors"] / metrics["predictions_made"]
        else:
            metrics["avg_prediction_time"] = 0.0
            metrics["error_rate"] = 0.0

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            "predictions_made": 0,
            "total_prediction_time": 0.0,
            "errors": 0,
        }


class ModelLoader:
    """
    Utility class for loading and managing ML models.

    Supports multiple model formats and provides caching
    for efficient model reuse.
    """

    def __init__(self, model_directory: Optional[Path] = None):
        """
        Initialize the model loader.

        Args:
            model_directory: Directory containing model files
        """
        self.model_directory = model_directory or Path("models")
        self._model_cache: Dict[str, MLModelWrapper] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelLoader")

    @lru_cache(maxsize=10)
    def load_model(
        self, model_name: str, model_path: Optional[Path] = None, version: str = "1.0"
    ) -> MLModelWrapper:
        """
        Load a machine learning model.

        Args:
            model_name: Name of the model
            model_path: Path to model file (optional)
            version: Model version

        Returns:
            Loaded model wrapper

        Raises:
            ModelLoadError: If model loading fails
        """
        cache_key = f"{model_name}_{version}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            # Determine model path
            if model_path is None:
                model_path = self.model_directory / f"{model_name}_v{version}.pkl"

            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")

            # Load model from pickle
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # Extract model and scaler
            if isinstance(model_data, dict):
                model = model_data["model"]
                scaler = model_data.get("scaler")
            else:
                model = model_data
                scaler = None

            # Create wrapper
            wrapper = MLModelWrapper(
                model=model, model_name=model_name, version=version, scaler=scaler
            )

            self._model_cache[cache_key] = wrapper
            self.logger.info(f"Loaded model: {model_name} v{version}")

            return wrapper

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load model {model_name}: {e}")

    def list_available_models(self) -> List[str]:
        """List available model files."""
        if not self.model_directory.exists():
            return []

        models = []
        for model_file in self.model_directory.glob("*.pkl"):
            models.append(model_file.stem)

        return sorted(models)

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()


class FeatureExtractor:
    """
    Base class for feature extraction from different content types.

    Provides common functionality and interface for extracting
    features from video, audio, and text content.
    """

    def __init__(self, name: str):
        """
        Initialize the feature extractor.

        Args:
            name: Name of the feature extractor
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Feature extraction metrics
        self._metrics = {
            "extractions": 0,
            "total_time": 0.0,
            "errors": 0,
        }

    async def extract_features(self, data: Any) -> np.ndarray:
        """
        Extract features from input data.

        Args:
            data: Input data for feature extraction

        Returns:
            Extracted feature vector

        Raises:
            FeatureExtractionError: If extraction fails
        """
        start_time = asyncio.get_event_loop().time()

        try:
            features = await self._extract_features_impl(data)

            # Update metrics
            extraction_time = asyncio.get_event_loop().time() - start_time
            self._metrics["extractions"] += 1
            self._metrics["total_time"] += extraction_time

            return features

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error(f"Feature extraction error in {self.name}: {e}")
            raise FeatureExtractionError(f"Feature extraction failed: {e}")

    async def _extract_features_impl(self, data: Any) -> np.ndarray:
        """
        Implementation-specific feature extraction.

        Must be implemented by subclasses.

        Args:
            data: Input data

        Returns:
            Feature vector
        """
        raise NotImplementedError("Subclasses must implement _extract_features_impl")

    def get_metrics(self) -> Dict[str, float]:
        """Get feature extraction metrics."""
        metrics = self._metrics.copy()

        if metrics["extractions"] > 0:
            metrics["avg_extraction_time"] = (
                metrics["total_time"] / metrics["extractions"]
            )
            metrics["error_rate"] = metrics["errors"] / metrics["extractions"]
        else:
            metrics["avg_extraction_time"] = 0.0
            metrics["error_rate"] = 0.0

        return metrics


class VideoFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for video content.

    Extracts visual features like motion vectors, color histograms,
    and scene change indicators.
    """

    def __init__(self):
        super().__init__("VideoFeatureExtractor")

    async def _extract_features_impl(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract features from video frames.

        Args:
            frames: Array of video frames

        Returns:
            Feature vector with motion and visual characteristics
        """
        try:
            features = []

            # Basic frame statistics
            features.extend(
                [
                    np.mean(frames),  # Average pixel intensity
                    np.std(frames),  # Pixel intensity variation
                    np.max(frames) - np.min(frames),  # Dynamic range
                ]
            )

            # Motion features (simplified)
            if len(frames.shape) == 4:  # Multiple frames
                frame_diffs = np.diff(frames, axis=0)
                motion_magnitude = np.mean(np.abs(frame_diffs))
                features.append(motion_magnitude)
            else:
                features.append(0.0)  # Single frame, no motion

            # Color features (if RGB)
            if len(frames.shape) >= 3 and frames.shape[-1] == 3:
                # Color histograms
                for channel in range(3):
                    hist, _ = np.histogram(
                        frames[..., channel], bins=16, range=(0, 255)
                    )
                    features.extend(hist / np.sum(hist))  # Normalized histogram

            return np.array(features, dtype=np.float32)

        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract video features: {e}")


class AudioFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for audio content.

    Extracts audio features like spectral characteristics,
    energy distribution, and rhythm patterns.
    """

    def __init__(self):
        super().__init__("AudioFeatureExtractor")

    async def _extract_features_impl(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract features from audio data.

        Args:
            audio_data: Audio samples

        Returns:
            Feature vector with spectral and temporal characteristics
        """
        try:
            features = []

            # Time domain features
            features.extend(
                [
                    np.mean(audio_data),  # DC component
                    np.std(audio_data),  # RMS energy
                    np.max(np.abs(audio_data)),  # Peak amplitude
                    len(audio_data),  # Duration (samples)
                ]
            )

            # Frequency domain features (simplified)
            fft = np.fft.fft(audio_data)
            magnitude_spectrum = np.abs(fft[: len(fft) // 2])

            features.extend(
                [
                    np.mean(magnitude_spectrum),  # Spectral centroid (simplified)
                    np.std(magnitude_spectrum),  # Spectral spread
                    np.sum(magnitude_spectrum),  # Total energy
                ]
            )

            # Spectral rolloff (simplified)
            cumsum = np.cumsum(magnitude_spectrum)
            rolloff_85 = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_85) > 0:
                features.append(rolloff_85[0] / len(magnitude_spectrum))
            else:
                features.append(1.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract audio features: {e}")


class TextFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for text/chat content.

    Extracts features like sentiment scores, keyword frequency,
    and linguistic patterns.
    """

    def __init__(self):
        super().__init__("TextFeatureExtractor")
        self.excitement_keywords = {
            "wow",
            "amazing",
            "incredible",
            "awesome",
            "epic",
            "insane",
            "omg",
            "poggers",
            "hype",
            "clutch",
            "sick",
            "fire",
            "lit",
            "!!",
            "!!!",
            "???",
            "💯",
            "🔥",
            "😱",
            "🤯",
            "👏",
        }

    async def _extract_features_impl(self, messages: List[str]) -> np.ndarray:
        """
        Extract features from chat messages.

        Args:
            messages: List of chat messages

        Returns:
            Feature vector with text characteristics
        """
        try:
            if not messages:
                return np.zeros(10, dtype=np.float32)

            features = []

            # Basic text statistics
            total_chars = sum(len(msg) for msg in messages)
            total_words = sum(len(msg.split()) for msg in messages)

            features.extend(
                [
                    len(messages),  # Message count
                    total_chars / len(messages),  # Avg message length
                    total_words / len(messages),  # Avg words per message
                ]
            )

            # Excitement indicators
            excitement_count = 0
            caps_count = 0
            emoji_count = 0

            for msg in messages:
                msg_lower = msg.lower()

                # Count excitement keywords
                for keyword in self.excitement_keywords:
                    excitement_count += msg_lower.count(keyword)

                # Count caps and emojis
                caps_count += sum(1 for c in msg if c.isupper())
                emoji_count += sum(
                    1 for c in msg if ord(c) > 127
                )  # Simplified emoji detection

            features.extend(
                [
                    excitement_count / len(messages),  # Excitement density
                    caps_count / total_chars if total_chars > 0 else 0,  # Caps ratio
                    emoji_count / len(messages),  # Emoji density
                ]
            )

            # Message frequency features
            # TODO: Implement actual frequency analysis when timestamps are available
            # For now, using default values to maintain feature vector consistency
            features.extend(
                [
                    1.0,  # Default: normalized message frequency
                    0.5,  # Default: normalized frequency variance
                    0.3,  # Default: burst detection score
                ]
            )

            return np.array(features, dtype=np.float32)

        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract text features: {e}")


def calculate_feature_importance(
    features: np.ndarray, scores: np.ndarray, method: str = "correlation"
) -> np.ndarray:
    """
    Calculate feature importance scores.

    Args:
        features: Feature matrix (n_samples, n_features)
        scores: Target scores (n_samples,)
        method: Importance calculation method

    Returns:
        Feature importance scores
    """
    if method == "correlation":
        # Pearson correlation with target scores
        correlations = []
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], scores)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        return np.array(correlations)

    elif method == "variance":
        # Feature variance as importance proxy
        variances = np.var(features, axis=0)
        return variances / np.sum(variances) if np.sum(variances) > 0 else variances

    else:
        raise ValueError(f"Unknown importance method: {method}")


def evaluate_model_performance(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate model performance with various metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # If probabilities available
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score, log_loss

        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            metrics["auc_roc"] = 0.0

        try:
            metrics["log_loss"] = log_loss(y_true, y_prob)
        except ValueError:
            metrics["log_loss"] = float("inf")

    return metrics


# Global instances
model_loader = ModelLoader()
video_feature_extractor = VideoFeatureExtractor()
audio_feature_extractor = AudioFeatureExtractor()
text_feature_extractor = TextFeatureExtractor()
