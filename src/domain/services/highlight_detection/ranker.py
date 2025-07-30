"""
Highlight ranking and selection for the TL;DR Highlight API.

This module implements sophisticated algorithms for ranking highlight candidates,
performing deduplication, clustering, and final selection based on various
quality and diversity criteria.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from ...entities.highlight import HighlightCandidate
from ...utils.scoring_utils import (
    calculate_similarity_score,
    calculate_highlight_overlap,
)

logger = logging.getLogger(__name__)


class RankingMethod(str, Enum):
    """Methods for ranking highlight candidates."""

    SCORE_BASED = "score_based"
    WEIGHTED_MULTI_CRITERIA = "weighted_multi_criteria"
    DIVERSITY_AWARE = "diversity_aware"
    TEMPORAL_DISTRIBUTION = "temporal_distribution"
    USER_PREFERENCE = "user_preference"


class ClusteringMethod(str, Enum):
    """Methods for clustering similar highlights."""

    TEMPORAL = "temporal"
    FEATURE_BASED = "feature_based"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    HYBRID = "hybrid"


class SelectionStrategy(str, Enum):
    """Strategies for final highlight selection."""

    TOP_N = "top_n"
    THRESHOLD_BASED = "threshold_based"
    DIVERSE_SET = "diverse_set"
    TEMPORAL_SPREAD = "temporal_spread"
    QUALITY_BALANCED = "quality_balanced"


@dataclass
class RankingMetrics:
    """
    Metrics for evaluating ranking performance.

    Contains various quality and diversity metrics for
    ranking algorithm assessment.
    """

    total_candidates: int
    selected_count: int
    avg_score: float
    score_variance: float
    temporal_coverage: float
    diversity_score: float
    overlap_ratio: float
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def selection_ratio(self) -> float:
        """Get ratio of selected to total candidates."""
        return self.selected_count / max(1, self.total_candidates)

    @property
    def quality_score(self) -> float:
        """Get overall quality score."""
        components = [
            self.avg_score,
            1.0 - min(1.0, self.score_variance),  # Lower variance is better
            self.temporal_coverage,
            self.diversity_score,
            1.0 - self.overlap_ratio,  # Lower overlap is better
        ]
        return np.mean(components)


class RankingConfig(BaseModel):
    """
    Configuration for highlight ranking and selection.

    Defines parameters for ranking algorithms, clustering,
    and final selection strategies.
    """

    # Ranking method configuration
    ranking_method: RankingMethod = Field(
        default=RankingMethod.WEIGHTED_MULTI_CRITERIA,
        description="Method for ranking highlights",
    )
    selection_strategy: SelectionStrategy = Field(
        default=SelectionStrategy.DIVERSE_SET,
        description="Strategy for final selection",
    )

    # Selection parameters
    max_highlights: int = Field(
        default=10, ge=1, description="Maximum number of highlights to select"
    )
    min_highlights: int = Field(
        default=1, ge=1, description="Minimum number of highlights to select"
    )
    score_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum score threshold for selection"
    )

    # Ranking weights
    score_weight: float = Field(
        default=0.4, ge=0.0, description="Weight for base score in ranking"
    )
    confidence_weight: float = Field(
        default=0.3, ge=0.0, description="Weight for confidence in ranking"
    )
    diversity_weight: float = Field(
        default=0.2, ge=0.0, description="Weight for diversity in ranking"
    )
    temporal_weight: float = Field(
        default=0.1, ge=0.0, description="Weight for temporal distribution in ranking"
    )

    # Clustering configuration
    clustering_enabled: bool = Field(
        default=True, description="Enable clustering for deduplication"
    )
    clustering_method: ClusteringMethod = Field(
        default=ClusteringMethod.HYBRID, description="Method for clustering highlights"
    )
    cluster_distance_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Distance threshold for clustering"
    )

    # Deduplication parameters
    temporal_overlap_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overlap threshold for temporal deduplication",
    )
    feature_similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for feature-based deduplication",
    )

    # Diversity parameters
    min_temporal_gap: float = Field(
        default=30.0, ge=0.0, description="Minimum gap between highlights (seconds)"
    )
    max_temporal_density: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Maximum allowed temporal density"
    )
    diversity_penalty_factor: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Penalty factor for similar highlights"
    )

    # Quality parameters
    preferred_duration_min: float = Field(
        default=20.0,
        gt=0.0,
        description="Preferred minimum highlight duration (seconds)",
    )
    preferred_duration_max: float = Field(
        default=60.0,
        gt=0.0,
        description="Preferred maximum highlight duration (seconds)",
    )
    duration_penalty_factor: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Penalty for non-optimal durations"
    )


class HighlightCluster:
    """
    Represents a cluster of similar highlight candidates.

    Contains multiple candidates that are considered similar
    and provides methods for selecting the best representative.
    """

    def __init__(self, candidates: List[HighlightCandidate], cluster_id: int = 0):
        """
        Initialize highlight cluster.

        Args:
            candidates: List of highlight candidates in this cluster
            cluster_id: Unique identifier for the cluster
        """
        self.candidates = candidates
        self.cluster_id = cluster_id
        self.logger = logging.getLogger(f"{__name__}.HighlightCluster")

    @property
    def size(self) -> int:
        """Get cluster size."""
        return len(self.candidates)

    @property
    def centroid_timestamp(self) -> float:
        """Get cluster centroid timestamp."""
        if not self.candidates:
            return 0.0

        timestamps = [(c.start_time + c.end_time) / 2 for c in self.candidates]
        return np.mean(timestamps)

    @property
    def avg_score(self) -> float:
        """Get average score of candidates in cluster."""
        if not self.candidates:
            return 0.0

        return np.mean([c.score for c in self.candidates])

    @property
    def max_score(self) -> float:
        """Get maximum score of candidates in cluster."""
        if not self.candidates:
            return 0.0

        return max(c.score for c in self.candidates)

    def get_best_candidate(
        self, method: str = "weighted_score"
    ) -> Optional[HighlightCandidate]:
        """
        Get best candidate from the cluster.

        Args:
            method: Method for selecting best candidate

        Returns:
            Best candidate or None if cluster is empty
        """
        if not self.candidates:
            return None

        if method == "weighted_score":
            # Select candidate with highest weighted score
            best_candidate = max(self.candidates, key=lambda c: c.weighted_score)
        elif method == "highest_score":
            # Select candidate with highest base score
            best_candidate = max(self.candidates, key=lambda c: c.score)
        elif method == "highest_confidence":
            # Select candidate with highest confidence
            best_candidate = max(self.candidates, key=lambda c: c.confidence)
        elif method == "centroid_closest":
            # Select candidate closest to cluster centroid
            centroid_time = self.centroid_timestamp
            best_candidate = min(
                self.candidates,
                key=lambda c: abs((c.start_time + c.end_time) / 2 - centroid_time),
            )
        else:
            # Default to weighted score
            best_candidate = max(self.candidates, key=lambda c: c.weighted_score)

        return best_candidate

    def merge_cluster(self, other: "HighlightCluster") -> "HighlightCluster":
        """
        Merge this cluster with another cluster.

        Args:
            other: Other cluster to merge with

        Returns:
            New merged cluster
        """
        merged_candidates = self.candidates + other.candidates
        return HighlightCluster(
            merged_candidates, min(self.cluster_id, other.cluster_id)
        )

    def calculate_intra_cluster_similarity(self) -> float:
        """Calculate average similarity within cluster."""
        if len(self.candidates) < 2:
            return 1.0

        similarities = []
        for i, candidate1 in enumerate(self.candidates):
            for candidate2 in self.candidates[i + 1 :]:
                # Calculate temporal similarity
                overlap = calculate_highlight_overlap(
                    (candidate1.start_time, candidate1.end_time),
                    (candidate2.start_time, candidate2.end_time),
                )

                # Calculate feature similarity if available
                feature_sim = 0.5  # Default
                if (
                    hasattr(candidate1, "features")
                    and hasattr(candidate2, "features")
                    and candidate1.features
                    and candidate2.features
                ):
                    try:
                        # Convert feature dicts to arrays for similarity calculation
                        feat1 = np.array(list(candidate1.features.values()))
                        feat2 = np.array(list(candidate2.features.values()))
                        if len(feat1) == len(feat2):
                            feature_sim = calculate_similarity_score(feat1, feat2)
                    except Exception:
                        pass

                # Combined similarity
                combined_sim = (overlap + feature_sim) / 2
                similarities.append(combined_sim)

        return np.mean(similarities) if similarities else 1.0


class HighlightClustering:
    """
    Handles clustering of highlight candidates for deduplication.

    Implements various clustering algorithms to group similar
    highlights and reduce redundancy.
    """

    def __init__(self, config: RankingConfig):
        """
        Initialize highlight clustering.

        Args:
            config: Ranking configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HighlightClustering")

    async def cluster_highlights(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """
        Cluster highlight candidates.

        Args:
            candidates: List of highlight candidates to cluster

        Returns:
            List of highlight clusters
        """
        if not candidates:
            return []

        if not self.config.clustering_enabled or len(candidates) == 1:
            # No clustering - each candidate is its own cluster
            return [
                HighlightCluster([candidate], i)
                for i, candidate in enumerate(candidates)
            ]

        self.logger.info(f"Clustering {len(candidates)} highlight candidates")

        if self.config.clustering_method == ClusteringMethod.TEMPORAL:
            return await self._temporal_clustering(candidates)
        elif self.config.clustering_method == ClusteringMethod.FEATURE_BASED:
            return await self._feature_based_clustering(candidates)
        elif self.config.clustering_method == ClusteringMethod.HIERARCHICAL:
            return await self._hierarchical_clustering(candidates)
        elif self.config.clustering_method == ClusteringMethod.DBSCAN:
            return await self._dbscan_clustering(candidates)
        elif self.config.clustering_method == ClusteringMethod.HYBRID:
            return await self._hybrid_clustering(candidates)
        else:
            return await self._temporal_clustering(candidates)

    async def _temporal_clustering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """Cluster based on temporal overlap."""
        # Sort candidates by start time
        sorted_candidates = sorted(candidates, key=lambda c: c.start_time)

        clusters = []
        current_cluster = [sorted_candidates[0]]

        for candidate in sorted_candidates[1:]:
            # Check if candidate overlaps with current cluster
            cluster_end = max(c.end_time for c in current_cluster)
            overlap_ratio = calculate_highlight_overlap(
                (current_cluster[-1].start_time, cluster_end),
                (candidate.start_time, candidate.end_time),
            )

            if overlap_ratio >= self.config.temporal_overlap_threshold:
                current_cluster.append(candidate)
            else:
                # Start new cluster
                clusters.append(HighlightCluster(current_cluster, len(clusters)))
                current_cluster = [candidate]

        # Add final cluster
        if current_cluster:
            clusters.append(HighlightCluster(current_cluster, len(clusters)))

        return clusters

    async def _feature_based_clustering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """Cluster based on feature similarity."""
        if len(candidates) < 2:
            return [HighlightCluster(candidates, 0)]

        # Extract feature vectors
        feature_vectors = []
        valid_candidates = []

        for candidate in candidates:
            if hasattr(candidate, "features") and candidate.features:
                try:
                    features = np.array(list(candidate.features.values()))
                    if len(features) > 0:
                        feature_vectors.append(features)
                        valid_candidates.append(candidate)
                except Exception:
                    pass

        if len(feature_vectors) < 2:
            # Fall back to temporal clustering
            return await self._temporal_clustering(candidates)

        # Ensure all feature vectors have same length
        min_length = min(len(fv) for fv in feature_vectors)
        feature_vectors = [fv[:min_length] for fv in feature_vectors]
        feature_matrix = np.array(feature_vectors)

        # Calculate pairwise distances
        distances = pdist(feature_matrix, metric="cosine")
        distance_matrix = squareform(distances)

        # Simple clustering based on similarity threshold
        clusters = []
        assigned = set()

        for i, candidate in enumerate(valid_candidates):
            if i in assigned:
                continue

            cluster_candidates = [candidate]
            assigned.add(i)

            for j, other_candidate in enumerate(valid_candidates[i + 1 :], i + 1):
                if j in assigned:
                    continue

                similarity = 1.0 - distance_matrix[i, j]
                if similarity >= self.config.feature_similarity_threshold:
                    cluster_candidates.append(other_candidate)
                    assigned.add(j)

            clusters.append(HighlightCluster(cluster_candidates, len(clusters)))

        # Add unassigned candidates as individual clusters
        for i, candidate in enumerate(valid_candidates):
            if i not in assigned:
                clusters.append(HighlightCluster([candidate], len(clusters)))

        return clusters

    async def _hierarchical_clustering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """Hierarchical clustering of candidates."""
        if len(candidates) < 2:
            return [HighlightCluster(candidates, 0)]

        # Create combined distance matrix (temporal + feature)
        n = len(candidates)
        distance_matrix = np.zeros((n, n))

        for i, candidate1 in enumerate(candidates):
            for j, candidate2 in enumerate(candidates[i + 1 :], i + 1):
                # Temporal distance
                temporal_dist = 1.0 - calculate_highlight_overlap(
                    (candidate1.start_time, candidate1.end_time),
                    (candidate2.start_time, candidate2.end_time),
                )

                # Feature distance (if available)
                feature_dist = 0.5  # Default
                if (
                    hasattr(candidate1, "features")
                    and hasattr(candidate2, "features")
                    and candidate1.features
                    and candidate2.features
                ):
                    try:
                        feat1 = np.array(list(candidate1.features.values()))
                        feat2 = np.array(list(candidate2.features.values()))
                        if len(feat1) == len(feat2):
                            feature_sim = calculate_similarity_score(feat1, feat2)
                            feature_dist = 1.0 - feature_sim
                    except Exception:
                        pass

                # Combined distance
                combined_dist = (temporal_dist + feature_dist) / 2
                distance_matrix[i, j] = combined_dist
                distance_matrix[j, i] = combined_dist

        # Perform hierarchical clustering
        try:
            condensed_distances = distance_matrix[np.triu_indices(n, k=1)]
            linkage_matrix = linkage(condensed_distances, method="ward")
            cluster_labels = fcluster(
                linkage_matrix,
                self.config.cluster_distance_threshold,
                criterion="distance",
            )

            # Group candidates by cluster labels
            cluster_dict = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(candidates[i])

            clusters = [
                HighlightCluster(cluster_candidates, cluster_id)
                for cluster_id, cluster_candidates in cluster_dict.items()
            ]

            return clusters

        except Exception as e:
            self.logger.warning(
                f"Hierarchical clustering failed: {e}, falling back to temporal clustering"
            )
            return await self._temporal_clustering(candidates)

    async def _dbscan_clustering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """DBSCAN clustering of candidates."""
        if len(candidates) < 2:
            return [HighlightCluster(candidates, 0)]

        # Create feature matrix
        features = []
        for candidate in candidates:
            # Combine temporal and feature information
            temporal_features = [
                candidate.start_time,
                candidate.end_time,
                candidate.score,
                candidate.confidence,
            ]

            if hasattr(candidate, "features") and candidate.features:
                try:
                    feature_values = list(candidate.features.values())[
                        :10
                    ]  # Limit features
                    temporal_features.extend(feature_values)
                except Exception:
                    pass

            features.append(temporal_features)

        # Ensure all feature vectors have same length
        max_length = max(len(f) for f in features)
        normalized_features = []
        for feature_vector in features:
            if len(feature_vector) < max_length:
                feature_vector.extend([0.0] * (max_length - len(feature_vector)))
            normalized_features.append(feature_vector[:max_length])

        feature_matrix = np.array(normalized_features)

        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (
            np.std(feature_matrix, axis=0) + 1e-10
        )

        try:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=self.config.cluster_distance_threshold, min_samples=2)
            cluster_labels = dbscan.fit_predict(feature_matrix)

            # Group candidates by cluster labels
            cluster_dict = {}
            noise_candidates = []

            for i, label in enumerate(cluster_labels):
                if label == -1:  # Noise points
                    noise_candidates.append(candidates[i])
                else:
                    if label not in cluster_dict:
                        cluster_dict[label] = []
                    cluster_dict[label].append(candidates[i])

            clusters = [
                HighlightCluster(cluster_candidates, cluster_id)
                for cluster_id, cluster_candidates in cluster_dict.items()
            ]

            # Add noise points as individual clusters
            for candidate in noise_candidates:
                clusters.append(HighlightCluster([candidate], len(clusters)))

            return clusters

        except Exception as e:
            self.logger.warning(
                f"DBSCAN clustering failed: {e}, falling back to temporal clustering"
            )
            return await self._temporal_clustering(candidates)

    async def _hybrid_clustering(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCluster]:
        """Hybrid clustering combining multiple methods."""
        # Start with temporal clustering
        temporal_clusters = await self._temporal_clustering(candidates)

        # Further cluster large temporal clusters using feature similarity
        refined_clusters = []

        for temporal_cluster in temporal_clusters:
            if temporal_cluster.size <= 3:
                # Small clusters don't need further clustering
                refined_clusters.append(temporal_cluster)
            else:
                # Apply feature-based clustering to large temporal clusters
                feature_clusters = await self._feature_based_clustering(
                    temporal_cluster.candidates
                )
                refined_clusters.extend(feature_clusters)

        # Re-assign cluster IDs
        for i, cluster in enumerate(refined_clusters):
            cluster.cluster_id = i

        return refined_clusters


class HighlightRanker:
    """
    Ranks and selects final highlights from candidates.

    Implements sophisticated ranking algorithms considering
    multiple criteria including score, confidence, diversity,
    and temporal distribution.
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize highlight ranker.

        Args:
            config: Ranking configuration
        """
        self.config = config or RankingConfig()
        self.clustering = HighlightClustering(self.config)
        self.logger = logging.getLogger(f"{__name__}.HighlightRanker")

    async def rank_and_select(
        self, candidates: List[HighlightCandidate]
    ) -> Tuple[List[HighlightCandidate], RankingMetrics]:
        """
        Rank and select final highlights from candidates.

        Args:
            candidates: List of highlight candidates

        Returns:
            Tuple of (selected highlights, ranking metrics)
        """
        if not candidates:
            return [], RankingMetrics(
                total_candidates=0,
                selected_count=0,
                avg_score=0.0,
                score_variance=0.0,
                temporal_coverage=0.0,
                diversity_score=0.0,
                overlap_ratio=0.0,
            )

        self.logger.info(f"Ranking and selecting from {len(candidates)} candidates")

        # Step 1: Filter by minimum thresholds
        filtered_candidates = self._filter_candidates(candidates)

        # Step 2: Cluster for deduplication
        clusters = await self.clustering.cluster_highlights(filtered_candidates)

        # Step 3: Select best candidate from each cluster
        cluster_representatives = []
        for cluster in clusters:
            best_candidate = cluster.get_best_candidate("weighted_score")
            if best_candidate:
                cluster_representatives.append(best_candidate)

        # Step 4: Rank cluster representatives
        ranked_candidates = await self._rank_candidates(cluster_representatives)

        # Step 5: Final selection
        selected_highlights = await self._select_final_highlights(ranked_candidates)

        # Step 6: Calculate metrics
        metrics = self._calculate_metrics(candidates, selected_highlights)

        self.logger.info(
            f"Selected {len(selected_highlights)} highlights from {len(candidates)} candidates"
        )

        return selected_highlights, metrics

    def _filter_candidates(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Filter candidates by minimum thresholds."""
        filtered = []

        for candidate in candidates:
            # Check score threshold
            if candidate.score < self.config.score_threshold:
                continue

            # Check duration preferences (with penalty, not hard filter)
            duration = candidate.duration
            if (
                duration < self.config.preferred_duration_min * 0.5
                or duration > self.config.preferred_duration_max * 2.0
            ):
                continue  # Too short or too long

            filtered.append(candidate)

        return filtered

    async def _rank_candidates(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Rank candidates using the configured ranking method."""
        if not candidates:
            return []

        if self.config.ranking_method == RankingMethod.SCORE_BASED:
            return self._score_based_ranking(candidates)
        elif self.config.ranking_method == RankingMethod.WEIGHTED_MULTI_CRITERIA:
            return await self._weighted_multi_criteria_ranking(candidates)
        elif self.config.ranking_method == RankingMethod.DIVERSITY_AWARE:
            return await self._diversity_aware_ranking(candidates)
        elif self.config.ranking_method == RankingMethod.TEMPORAL_DISTRIBUTION:
            return await self._temporal_distribution_ranking(candidates)
        elif self.config.ranking_method == RankingMethod.USER_PREFERENCE:
            return await self._user_preference_ranking(candidates)
        else:
            return await self._weighted_multi_criteria_ranking(candidates)

    def _score_based_ranking(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Simple score-based ranking."""
        return sorted(candidates, key=lambda c: c.weighted_score, reverse=True)

    async def _weighted_multi_criteria_ranking(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Multi-criteria ranking with configurable weights."""
        candidate_scores = []

        for candidate in candidates:
            # Base score component
            score_component = candidate.score * self.config.score_weight

            # Confidence component
            confidence_component = candidate.confidence * self.config.confidence_weight

            # Duration preference component
            duration = candidate.duration
            optimal_duration = (
                self.config.preferred_duration_min + self.config.preferred_duration_max
            ) / 2
            duration_score = 1.0 - abs(duration - optimal_duration) / optimal_duration
            duration_score = max(0.0, duration_score)

            # Temporal distribution component (placeholder - would need full context)
            temporal_component = 0.5  # Neutral score

            # Combine components
            final_score = (
                score_component
                + confidence_component
                + duration_score * 0.1  # Small weight for duration
                + temporal_component * self.config.temporal_weight
            )

            candidate_scores.append((candidate, final_score))

        # Sort by final score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in candidate_scores]

    async def _diversity_aware_ranking(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Ranking that promotes diversity."""
        if len(candidates) <= 1:
            return candidates

        # Start with highest scoring candidate
        sorted_candidates = sorted(
            candidates, key=lambda c: c.weighted_score, reverse=True
        )
        selected = [sorted_candidates[0]]
        remaining = sorted_candidates[1:]

        # Iteratively select candidates that maximize diversity
        while remaining and len(selected) < self.config.max_highlights:
            best_candidate = None
            best_score = -1.0

            for candidate in remaining:
                # Calculate diversity score with already selected candidates
                diversity_score = self._calculate_diversity_with_selected(
                    candidate, selected
                )

                # Combined score: original score + diversity bonus
                combined_score = (
                    candidate.weighted_score * (1.0 - self.config.diversity_weight)
                    + diversity_score * self.config.diversity_weight
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _calculate_diversity_with_selected(
        self, candidate: HighlightCandidate, selected: List[HighlightCandidate]
    ) -> float:
        """Calculate diversity score of candidate with selected highlights."""
        if not selected:
            return 1.0

        diversities = []

        for selected_candidate in selected:
            # Temporal diversity
            temporal_gap = abs(
                (candidate.start_time + candidate.end_time) / 2
                - (selected_candidate.start_time + selected_candidate.end_time) / 2
            )
            temporal_diversity = min(1.0, temporal_gap / self.config.min_temporal_gap)

            # Feature diversity (if available)
            feature_diversity = 0.5  # Default
            if (
                hasattr(candidate, "features")
                and hasattr(selected_candidate, "features")
                and candidate.features
                and selected_candidate.features
            ):
                try:
                    feat1 = np.array(list(candidate.features.values()))
                    feat2 = np.array(list(selected_candidate.features.values()))
                    if len(feat1) == len(feat2):
                        similarity = calculate_similarity_score(feat1, feat2)
                        feature_diversity = 1.0 - similarity
                except Exception:
                    pass

            # Combined diversity
            combined_diversity = (temporal_diversity + feature_diversity) / 2
            diversities.append(combined_diversity)

        return np.mean(diversities)

    async def _temporal_distribution_ranking(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Ranking that optimizes temporal distribution."""
        # Sort by score first
        sorted_candidates = sorted(
            candidates, key=lambda c: c.weighted_score, reverse=True
        )

        # Apply temporal distribution optimization
        # This is a simplified implementation - could use more sophisticated algorithms
        return sorted_candidates

    async def _user_preference_ranking(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Ranking based on user preferences (placeholder)."""
        # This would incorporate user preference models
        # For now, fall back to multi-criteria ranking
        return await self._weighted_multi_criteria_ranking(candidates)

    async def _select_final_highlights(
        self, ranked_candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select final highlights using configured strategy."""
        if not ranked_candidates:
            return []

        if self.config.selection_strategy == SelectionStrategy.TOP_N:
            return self._top_n_selection(ranked_candidates)
        elif self.config.selection_strategy == SelectionStrategy.THRESHOLD_BASED:
            return self._threshold_based_selection(ranked_candidates)
        elif self.config.selection_strategy == SelectionStrategy.DIVERSE_SET:
            return self._diverse_set_selection(ranked_candidates)
        elif self.config.selection_strategy == SelectionStrategy.TEMPORAL_SPREAD:
            return self._temporal_spread_selection(ranked_candidates)
        elif self.config.selection_strategy == SelectionStrategy.QUALITY_BALANCED:
            return self._quality_balanced_selection(ranked_candidates)
        else:
            return self._top_n_selection(ranked_candidates)

    def _top_n_selection(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select top N candidates."""
        max_highlights = min(self.config.max_highlights, len(candidates))
        min_highlights = min(self.config.min_highlights, max_highlights)

        selected = candidates[:max_highlights]

        # Ensure minimum count
        if len(selected) < min_highlights:
            # Add more candidates if available
            remaining = candidates[len(selected) :]
            needed = min_highlights - len(selected)
            selected.extend(remaining[:needed])

        return selected

    def _threshold_based_selection(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select candidates above score threshold."""
        selected = []

        for candidate in candidates:
            if candidate.score >= self.config.score_threshold:
                selected.append(candidate)

                if len(selected) >= self.config.max_highlights:
                    break

        # Ensure minimum count
        if len(selected) < self.config.min_highlights:
            remaining = [c for c in candidates if c not in selected]
            needed = self.config.min_highlights - len(selected)
            selected.extend(remaining[:needed])

        return selected

    def _diverse_set_selection(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select diverse set of highlights."""
        if not candidates:
            return []

        selected = [candidates[0]]  # Start with best candidate
        remaining = candidates[1:]

        while remaining and len(selected) < self.config.max_highlights:
            # Find candidate with maximum diversity from selected set
            best_candidate = None
            best_diversity = -1.0

            for candidate in remaining:
                diversity = self._calculate_diversity_with_selected(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate and best_diversity > 0.1:  # Minimum diversity threshold
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # No more diverse candidates
                break

        # Ensure minimum count
        if len(selected) < self.config.min_highlights and remaining:
            needed = self.config.min_highlights - len(selected)
            selected.extend(remaining[:needed])

        return selected

    def _temporal_spread_selection(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select highlights with good temporal spread."""
        if not candidates:
            return []

        # Sort by timestamp
        sorted_by_time = sorted(
            candidates, key=lambda c: (c.start_time + c.end_time) / 2
        )

        # Select highlights with minimum temporal gap
        selected = [sorted_by_time[0]]

        for candidate in sorted_by_time[1:]:
            if len(selected) >= self.config.max_highlights:
                break

            # Check temporal gap with last selected
            last_selected = selected[-1]
            gap = abs(
                (candidate.start_time + candidate.end_time) / 2
                - (last_selected.start_time + last_selected.end_time) / 2
            )

            if gap >= self.config.min_temporal_gap:
                selected.append(candidate)

        # Ensure minimum count
        if len(selected) < self.config.min_highlights:
            # Add remaining candidates regardless of temporal gap
            remaining = [c for c in candidates if c not in selected]
            needed = self.config.min_highlights - len(selected)
            selected.extend(remaining[:needed])

        return selected

    def _quality_balanced_selection(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Select highlights balancing quality and diversity."""
        # Combine diverse set and threshold-based selection
        diverse_selected = self._diverse_set_selection(candidates)

        # If not enough, add high-quality candidates
        if len(diverse_selected) < self.config.max_highlights:
            remaining = [c for c in candidates if c not in diverse_selected]
            remaining_sorted = sorted(
                remaining, key=lambda c: c.weighted_score, reverse=True
            )

            needed = self.config.max_highlights - len(diverse_selected)
            diverse_selected.extend(remaining_sorted[:needed])

        return diverse_selected

    def _calculate_metrics(
        self,
        original_candidates: List[HighlightCandidate],
        selected_highlights: List[HighlightCandidate],
    ) -> RankingMetrics:
        """Calculate ranking performance metrics."""
        if not selected_highlights:
            return RankingMetrics(
                total_candidates=len(original_candidates),
                selected_count=0,
                avg_score=0.0,
                score_variance=0.0,
                temporal_coverage=0.0,
                diversity_score=0.0,
                overlap_ratio=0.0,
            )

        # Calculate metrics
        scores = [h.score for h in selected_highlights]
        avg_score = np.mean(scores)
        score_variance = np.var(scores)

        # Temporal coverage
        if original_candidates:
            original_times = [
                (c.start_time + c.end_time) / 2 for c in original_candidates
            ]
            selected_times = [
                (h.start_time + h.end_time) / 2 for h in selected_highlights
            ]

            original_span = (
                max(original_times) - min(original_times)
                if len(original_times) > 1
                else 1.0
            )
            selected_span = (
                max(selected_times) - min(selected_times)
                if len(selected_times) > 1
                else 0.0
            )

            temporal_coverage = min(1.0, selected_span / max(1.0, original_span))
        else:
            temporal_coverage = 0.0

        # Diversity score
        if len(selected_highlights) > 1:
            # Calculate pairwise diversity
            diversities = []
            for i, h1 in enumerate(selected_highlights):
                for h2 in selected_highlights[i + 1 :]:
                    gap = abs(
                        (h1.start_time + h1.end_time) / 2
                        - (h2.start_time + h2.end_time) / 2
                    )
                    temporal_diversity = min(
                        1.0, gap / max(1.0, self.config.min_temporal_gap)
                    )
                    diversities.append(temporal_diversity)

            diversity_score = np.mean(diversities) if diversities else 0.0
        else:
            diversity_score = 1.0

        # Overlap ratio
        total_overlaps = 0
        total_pairs = 0

        for i, h1 in enumerate(selected_highlights):
            for h2 in selected_highlights[i + 1 :]:
                overlap = calculate_highlight_overlap(
                    (h1.start_time, h1.end_time), (h2.start_time, h2.end_time)
                )
                total_overlaps += overlap
                total_pairs += 1

        overlap_ratio = total_overlaps / max(1, total_pairs)

        # Confidence distribution
        confidence_distribution = {
            "low": sum(1 for h in selected_highlights if h.confidence < 0.5),
            "medium": sum(1 for h in selected_highlights if 0.5 <= h.confidence < 0.8),
            "high": sum(1 for h in selected_highlights if h.confidence >= 0.8),
        }

        return RankingMetrics(
            total_candidates=len(original_candidates),
            selected_count=len(selected_highlights),
            avg_score=avg_score,
            score_variance=score_variance,
            temporal_coverage=temporal_coverage,
            diversity_score=diversity_score,
            overlap_ratio=overlap_ratio,
            confidence_distribution=confidence_distribution,
        )

    def get_ranking_config(self) -> Dict[str, Any]:
        """Get current ranking configuration."""
        return self.config.dict()

    def update_config(self, **kwargs) -> None:
        """Update ranking configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
