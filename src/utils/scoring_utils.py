"""
Scoring utilities for the TL;DR Highlight API.

This module provides mathematical functions and utilities for calculating
scores, confidences, and rankings in the highlight detection system.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.signal import find_peaks, savgol_filter


def normalize_score(
    score: float, min_val: float = 0.0, max_val: float = 1.0, method: str = "linear"
) -> float:
    """
    Normalize a score to a specified range.

    Args:
        score: Raw score to normalize
        min_val: Minimum value of output range
        max_val: Maximum value of output range
        method: Normalization method ("linear", "sigmoid", "tanh")

    Returns:
        Normalized score in the specified range
    """
    if method == "linear":
        # Simple linear scaling
        return max(min_val, min(max_val, score))

    elif method == "sigmoid":
        # Sigmoid normalization
        sigmoid_val = 1 / (1 + math.exp(-score))
        return min_val + (max_val - min_val) * sigmoid_val

    elif method == "tanh":
        # Hyperbolic tangent normalization
        tanh_val = (math.tanh(score) + 1) / 2  # Scale to [0, 1]
        return min_val + (max_val - min_val) * tanh_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_weighted_score(
    scores: Dict[str, float], weights: Dict[str, float], normalize: bool = True
) -> float:
    """
    Calculate weighted combination of multiple scores.

    Args:
        scores: Dictionary of score values
        weights: Dictionary of weight values
        normalize: Whether to normalize weights to sum to 1

    Returns:
        Weighted combination score
    """
    if not scores or not weights:
        return 0.0

    # Normalize weights if requested
    if normalize:
        total_weight = sum(weights.get(key, 0.0) for key in scores.keys())
        if total_weight > 0:
            weights = {
                key: weights.get(key, 0.0) / total_weight for key in scores.keys()
            }

    # Calculate weighted sum
    weighted_sum = 0.0
    for key, score in scores.items():
        weight = weights.get(key, 0.0)
        weighted_sum += score * weight

    return weighted_sum


def calculate_confidence(
    scores: List[float], method: str = "entropy", **kwargs
) -> float:
    """
    Calculate confidence score from a list of values.

    Args:
        scores: List of score values
        method: Confidence calculation method
        **kwargs: Additional method-specific parameters

    Returns:
        Confidence score between 0 and 1
    """
    if not scores:
        return 0.0

    scores_array = np.array(scores)

    if method == "entropy":
        # Use entropy-based confidence
        # Higher entropy = lower confidence
        if len(scores) == 1:
            return 1.0

        # Normalize scores to probabilities
        if np.sum(scores_array) > 0:
            probs = scores_array / np.sum(scores_array)
        else:
            probs = np.ones_like(scores_array) / float(len(scores_array))

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(scores))

        # Convert to confidence (1 - normalized_entropy)
        return 1.0 - (float(entropy / max_entropy) if max_entropy > 0 else 0.0)

    elif method == "variance":
        # Use variance-based confidence
        # Lower variance = higher confidence
        if len(scores) == 1:
            return 1.0

        variance = np.var(scores_array)
        max_variance = kwargs.get("max_variance", 1.0)

        # Normalize variance to confidence
        normalized_var = min(float(variance / max_variance), 1.0)
        return 1.0 - normalized_var

    elif method == "consistency":
        # Use consistency-based confidence
        # More consistent scores = higher confidence
        if len(scores) <= 1:
            return 1.0

        mean_score = float(np.mean(scores_array))
        deviations = np.abs(scores_array - mean_score)
        avg_deviation = float(np.mean(deviations))

        # Normalize deviation to confidence
        max_deviation = kwargs.get("max_deviation", 1.0)
        normalized_dev = min(float(avg_deviation / max_deviation), 1.0)
        return 1.0 - normalized_dev

    elif method == "majority":
        # Use majority voting confidence
        threshold = kwargs.get("threshold", 0.5)
        above_threshold = np.sum(scores_array > threshold)
        return float(above_threshold / len(scores_array))

    else:
        raise ValueError(f"Unknown confidence method: {method}")


def calculate_temporal_correlation(
    timestamps: List[float], scores: List[float], window_size: float = 30.0
) -> float:
    """
    Calculate temporal correlation of scores.

    Args:
        timestamps: List of timestamps
        scores: List of corresponding scores
        window_size: Time window for correlation calculation

    Returns:
        Temporal correlation coefficient
    """
    if len(timestamps) != len(scores) or len(timestamps) < 2:
        return 0.0

    # Sort by timestamp
    sorted_pairs = sorted(zip(timestamps, scores))
    sorted_timestamps = [t for t, _ in sorted_pairs]
    sorted_scores = [s for _, s in sorted_pairs]

    # Calculate windowed correlations
    correlations = []

    for i in range(len(sorted_timestamps)):
        window_start = sorted_timestamps[i]
        window_end = window_start + window_size

        # Get scores in window
        window_scores = []
        window_times = []

        for j in range(i, len(sorted_timestamps)):
            if sorted_timestamps[j] <= window_end:
                window_scores.append(sorted_scores[j])
                window_times.append(sorted_timestamps[j])
            else:
                break

        # Calculate correlation if enough points
        if len(window_scores) >= 3:
            try:
                corr, _ = stats.pearsonr(window_times, window_scores)
                if not np.isnan(corr):
                    correlations.append(abs(float(corr)))
            except Exception:
                pass

    return float(np.mean(correlations)) if correlations else 0.0


def detect_score_peaks(
    timestamps: List[float],
    scores: List[float],
    prominence: float = 0.1,
    distance: float = 10.0,
    smooth: bool = True,
) -> List[Tuple[float, float]]:
    """
    Detect peaks in score timeseries.

    Args:
        timestamps: List of timestamps
        scores: List of corresponding scores
        prominence: Minimum peak prominence
        distance: Minimum distance between peaks (in seconds)
        smooth: Whether to smooth the signal before peak detection

    Returns:
        List of (timestamp, score) tuples for detected peaks
    """
    if len(timestamps) != len(scores) or len(timestamps) < 3:
        return []

    # Sort by timestamp
    sorted_pairs = sorted(zip(timestamps, scores))
    sorted_timestamps = np.array([t for t, _ in sorted_pairs])
    sorted_scores = np.array([s for _, s in sorted_pairs])

    # Smooth signal if requested
    if smooth and len(sorted_scores) >= 5:
        window_length = min(5, len(sorted_scores) // 2 * 2 + 1)  # Ensure odd
        try:
            sorted_scores = savgol_filter(sorted_scores, window_length, 2)
        except Exception:
            pass  # Use original if smoothing fails

    # Convert distance to sample distance
    if len(sorted_timestamps) > 1:
        avg_sample_rate = len(sorted_timestamps) / (
            sorted_timestamps[-1] - sorted_timestamps[0]
        )
        distance_samples = int(distance * avg_sample_rate)
    else:
        distance_samples = 1

    # Find peaks
    peaks, properties = find_peaks(
        sorted_scores, prominence=prominence, distance=max(1, distance_samples)
    )

    # Return peak timestamps and scores
    peak_results = []
    for peak_idx in peaks:
        peak_results.append((sorted_timestamps[peak_idx], sorted_scores[peak_idx]))

    return peak_results


def calculate_similarity_score(
    features1: np.ndarray, features2: np.ndarray, method: str = "cosine"
) -> float:
    """
    Calculate similarity between two feature vectors.

    Args:
        features1: First feature vector
        features2: Second feature vector
        method: Similarity calculation method

    Returns:
        Similarity score between 0 and 1
    """
    if len(features1) != len(features2):
        return 0.0

    if method == "cosine":
        # Cosine similarity
        try:
            distance = cosine(features1, features2)
            return 1.0 - distance  # Convert distance to similarity
        except Exception:
            return 0.0

    elif method == "euclidean":
        # Euclidean distance similarity
        distance: float = float(np.linalg.norm(features1 - features2))
        max_distance: float = float(np.linalg.norm(np.ones_like(features1)))
        return 1.0 - min(distance / max_distance, 1.0)

    elif method == "pearson":
        # Pearson correlation
        try:
            corr, _ = stats.pearsonr(features1, features2)
            return float((corr + 1) / 2)  # Scale from [-1, 1] to [0, 1]
        except Exception:
            return 0.0

    elif method == "jaccard":
        # Jaccard similarity (for binary features)
        binary1 = features1 > 0
        binary2 = features2 > 0
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        return float(intersection / union) if union > 0 else 0.0

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def calculate_diversity_score(
    feature_matrix: np.ndarray, method: str = "pairwise_distance"
) -> float:
    """
    Calculate diversity score for a set of feature vectors.

    Args:
        feature_matrix: Matrix of feature vectors (n_samples, n_features)
        method: Diversity calculation method

    Returns:
        Diversity score between 0 and 1
    """
    if feature_matrix.shape[0] <= 1:
        return 0.0

    if method == "pairwise_distance":
        # Average pairwise distance
        distances = []
        for i in range(feature_matrix.shape[0]):
            for j in range(i + 1, feature_matrix.shape[0]):
                dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            max_possible = np.linalg.norm(np.ones(feature_matrix.shape[1]))
            return min(float(avg_distance / max_possible), 1.0)
        else:
            return 0.0

    elif method == "determinant":
        # Determinant of covariance matrix
        try:
            cov_matrix = np.cov(feature_matrix.T)
            det = np.linalg.det(cov_matrix)
            return min(float(det), 1.0) if det >= 0 else 0.0
        except Exception:
            return 0.0

    elif method == "entropy":
        # Entropy of feature distributions
        entropies = []
        for feature_idx in range(feature_matrix.shape[1]):
            feature_values = feature_matrix[:, feature_idx]
            hist, _ = np.histogram(feature_values, bins=10)
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        avg_entropy = float(np.mean(entropies))
        max_entropy = np.log(10)  # Based on 10 bins
        return float(avg_entropy / max_entropy) if max_entropy > 0 else 0.0

    else:
        raise ValueError(f"Unknown diversity method: {method}")


def calculate_quality_score(
    detection_results: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate overall quality score from detection results.

    Args:
        detection_results: List of detection result dictionaries
        weights: Optional weights for different quality factors

    Returns:
        Quality score between 0 and 1
    """
    if not detection_results:
        return 0.0

    # Default weights
    if weights is None:
        weights = {
            "score": 0.4,
            "confidence": 0.3,
            "consistency": 0.2,
            "diversity": 0.1,
        }

    # Extract scores and calculate metrics
    scores = [result.get("score", 0.0) for result in detection_results]
    confidences = [result.get("confidence", 0.0) for result in detection_results]

    # Calculate quality components
    quality_components = {}

    # Average score
    quality_components["score"] = float(np.mean(scores)) if scores else 0.0

    # Average confidence
    quality_components["confidence"] = float(np.mean(confidences)) if confidences else 0.0

    # Consistency (inverse of score variance)
    if len(scores) > 1:
        score_var = float(np.var(scores))
        quality_components["consistency"] = 1.0 / (1.0 + score_var)
    else:
        quality_components["consistency"] = 1.0

    # Diversity (if feature vectors available)
    if all("features" in result for result in detection_results):
        feature_matrix = np.array([result["features"] for result in detection_results])
        quality_components["diversity"] = calculate_diversity_score(feature_matrix)
    else:
        quality_components["diversity"] = 0.5  # Neutral diversity

    # Calculate weighted quality score
    return float(calculate_weighted_score(quality_components, weights, normalize=True))


def apply_temporal_smoothing(
    timestamps: List[float],
    scores: List[float],
    method: str = "moving_average",
    window_size: float = 10.0,
) -> List[float]:
    """
    Apply temporal smoothing to score timeseries.

    Args:
        timestamps: List of timestamps
        scores: List of corresponding scores
        method: Smoothing method
        window_size: Smoothing window size in seconds

    Returns:
        Smoothed scores
    """
    if len(timestamps) != len(scores) or len(timestamps) < 2:
        return scores

    # Sort by timestamp
    sorted_pairs = sorted(zip(timestamps, scores))
    sorted_timestamps = [t for t, _ in sorted_pairs]
    sorted_scores = [s for _, s in sorted_pairs]

    if method == "moving_average":
        # Moving average smoothing
        smoothed_scores = []

        for i, current_time in enumerate(sorted_timestamps):
            window_start = current_time - window_size / 2
            window_end = current_time + window_size / 2

            # Find scores in window
            window_scores = []
            for j, timestamp in enumerate(sorted_timestamps):
                if window_start <= timestamp <= window_end:
                    window_scores.append(sorted_scores[j])

            # Calculate average
            if window_scores:
                smoothed_scores.append(float(np.mean(window_scores)))
            else:
                smoothed_scores.append(float(sorted_scores[i]))

        return smoothed_scores

    elif method == "exponential":
        # Exponential smoothing
        if not sorted_scores:
            return []

        alpha = 2.0 / (window_size + 1)  # Smoothing factor
        smoothed_scores = [float(sorted_scores[0])]

        for i in range(1, len(sorted_scores)):
            smoothed_value = (
                alpha * sorted_scores[i] + (1 - alpha) * smoothed_scores[-1]
            )
            smoothed_scores.append(float(smoothed_value))

        return smoothed_scores

    elif method == "savgol":
        # Savitzky-Golay smoothing
        if len(sorted_scores) < 5:
            return sorted_scores

        window_length = min(11, len(sorted_scores) // 2 * 2 + 1)  # Ensure odd
        try:
            smoothed_array = savgol_filter(sorted_scores, window_length, 2)
            return [float(x) for x in smoothed_array]
        except Exception:
            return sorted_scores

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def calculate_highlight_overlap(
    highlight1: Tuple[float, float], highlight2: Tuple[float, float]
) -> float:
    """
    Calculate overlap ratio between two highlights.

    Args:
        highlight1: (start_time, end_time) of first highlight
        highlight2: (start_time, end_time) of second highlight

    Returns:
        Overlap ratio between 0 and 1
    """
    start1, end1 = highlight1
    start2, end2 = highlight2

    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_duration = max(0, intersection_end - intersection_start)

    # Calculate union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_duration = union_end - union_start

    # Return overlap ratio
    return intersection_duration / union_duration if union_duration > 0 else 0.0


def rank_highlights(
    highlights: List[Dict[str, Union[float, List[float]]]],
    ranking_method: str = "weighted_score",
    weights: Optional[Dict[str, float]] = None,
    diversity_bonus: float = 0.1,
) -> List[int]:
    """
    Rank highlights based on various criteria.

    Args:
        highlights: List of highlight dictionaries
        ranking_method: Method for ranking
        weights: Optional weights for ranking factors
        diversity_bonus: Bonus for diverse highlights

    Returns:
        List of highlight indices in ranked order
    """
    if not highlights:
        return []

    if ranking_method == "weighted_score":
        # Rank by weighted score
        scores = []
        for highlight in highlights:
            score_val = highlight.get("score", 0.0)
            base_score = float(score_val) if not isinstance(score_val, list) else 0.0
            conf_val = highlight.get("confidence", 1.0)
            confidence = float(conf_val) if not isinstance(conf_val, list) else 1.0
            weighted_score = base_score * confidence
            scores.append(weighted_score)

        # Sort indices by score (descending)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return ranked_indices

    elif ranking_method == "multi_criteria":
        # Multi-criteria ranking
        if weights is None:
            weights = {
                "score": 0.4,
                "confidence": 0.3,
                "duration": 0.2,
                "diversity": 0.1,
            }

        # Calculate ranking scores
        ranking_scores = []

        for i, highlight in enumerate(highlights):
            criteria_scores = {}

            # Base score
            score_val = highlight.get("score", 0.0)
            criteria_scores["score"] = float(score_val) if not isinstance(score_val, list) else 0.0

            # Confidence
            conf_val = highlight.get("confidence", 0.0)
            criteria_scores["confidence"] = float(conf_val) if not isinstance(conf_val, list) else 0.0

            # Duration preference (moderate duration preferred)
            dur_val = highlight.get("duration", 30.0)
            duration = float(dur_val) if not isinstance(dur_val, list) else 30.0
            optimal_duration = 45.0  # Preferred highlight duration
            duration_score = 1.0 - abs(duration - optimal_duration) / optimal_duration
            criteria_scores["duration"] = max(0.0, duration_score)

            # Diversity (how different from other high-scoring highlights)
            diversity_score = 0.5  # Default neutral diversity
            if "features" in highlight:
                current_features = np.array(highlight["features"])
                diversity_scores = []

                for j, other_highlight in enumerate(highlights):
                    if i != j and "features" in other_highlight:
                        other_features = np.array(other_highlight["features"])
                        similarity = calculate_similarity_score(
                            current_features, other_features
                        )
                        diversity_scores.append(1.0 - similarity)

                if diversity_scores:
                    diversity_score = float(np.mean(diversity_scores))

            criteria_scores["diversity"] = diversity_score

            # Calculate weighted ranking score
            ranking_score = float(calculate_weighted_score(
                criteria_scores, weights, normalize=True
            ))
            ranking_scores.append(ranking_score)

        # Sort indices by ranking score (descending)
        ranked_indices = sorted(
            range(len(ranking_scores)), key=lambda i: ranking_scores[i], reverse=True
        )
        return ranked_indices

    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")
