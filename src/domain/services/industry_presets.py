"""Industry-specific preset configurations for flexible highlight detection.

This module provides pre-configured dimension sets and type registries
for common use cases across different industries.
"""

from typing import Dict, List, Any

from src.domain.entities.dimension_set import DimensionSet
from src.domain.entities.highlight_type_registry import HighlightTypeRegistry, HighlightTypeDefinition
from src.domain.value_objects.dimension_definition import (
    DimensionDefinition, DimensionType, AggregationMethod
)
from src.domain.value_objects.processing_options import ProcessingOptions, DetectionStrategy, FusionStrategy


class IndustryPresets:
    """Factory for creating industry-specific configurations."""
    
    @staticmethod
    def gaming_preset() -> Dict[str, Any]:
        """Create gaming/esports industry preset.
        
        Returns:
            Dictionary with dimension_set, type_registry, and processing_options
        """
        # Gaming-specific dimensions
        dimensions = {
            "action_intensity": DimensionDefinition(
                id="action_intensity",
                name="Action Intensity",
                description="Level of gameplay action and excitement",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.3,
                scoring_prompt="Rate the intensity of gameplay action from 0-1",
                examples=[
                    {"score": 0.9, "description": "Multi-kill, clutch play, or boss defeat"},
                    {"score": 0.3, "description": "Normal gameplay, farming, or exploration"}
                ],
                applicable_modalities=["video", "audio"]
            ),
            "skill_display": DimensionDefinition(
                id="skill_display",
                name="Skill Display",
                description="Demonstration of exceptional gaming skill",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.25,
                scoring_prompt="Rate the level of skill demonstrated from 0-1",
                examples=[
                    {"score": 0.95, "description": "Frame-perfect execution, impossible shot"},
                    {"score": 0.4, "description": "Standard gameplay with minor skill moments"}
                ]
            ),
            "emotional_reaction": DimensionDefinition(
                id="emotional_reaction",
                name="Emotional Reaction",
                description="Streamer/player emotional response",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.2,
                scoring_prompt="Rate the emotional intensity of reactions from 0-1",
                applicable_modalities=["video", "audio", "text"]
            ),
            "humor": DimensionDefinition(
                id="humor",
                name="Humor",
                description="Funny or comedic moments",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.15,
                scoring_prompt="Rate how funny or comedic the moment is from 0-1"
            ),
            "chat_engagement": DimensionDefinition(
                id="chat_engagement",
                name="Chat Engagement",
                description="Level of viewer engagement in chat",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.1,
                scoring_prompt="Rate chat engagement spike from 0-1",
                applicable_modalities=["text", "social"]
            )
        }
        
        dimension_set = DimensionSet(
            id=None,
            name="Gaming Standard",
            description="Standard dimensions for gaming/esports content",
            organization_id=1,  # Would be set dynamically
            dimensions=dimensions,
            dimension_weights={k: d.default_weight for k, d in dimensions.items()},
            minimum_dimensions_required=2
        )
        
        # Gaming highlight types
        type_definitions = {
            "epic_play": HighlightTypeDefinition(
                id="epic_play",
                name="Epic Play",
                description="Exceptional gameplay moment",
                criteria={
                    "action_intensity": {"min": 0.8},
                    "skill_display": {"min": 0.7}
                }
            ),
            "funny_moment": HighlightTypeDefinition(
                id="funny_moment",
                name="Funny Moment",
                description="Humorous or comedic content",
                criteria={
                    "humor": {"min": 0.7}
                }
            ),
            "emotional_moment": HighlightTypeDefinition(
                id="emotional_moment",
                name="Emotional Moment",
                description="Strong emotional reaction",
                criteria={
                    "emotional_reaction": {"min": 0.8}
                }
            ),
            "chat_reaction": HighlightTypeDefinition(
                id="chat_reaction",
                name="Chat Reaction",
                description="Moment with high chat engagement",
                criteria={
                    "chat_engagement": {"min": 0.8}
                }
            )
        }
        
        type_registry = HighlightTypeRegistry(
            id=None,
            organization_id=1,  # Would be set dynamically
            types=type_definitions,
            allow_multiple_types=True,
            max_types_per_highlight=2
        )
        
        # Gaming processing options
        processing_options = ProcessingOptions(
            dimension_set_id=None,  # Would be set after saving
            type_registry_id=None,  # Would be set after saving
            detection_strategy=DetectionStrategy.AI_ONLY,
            fusion_strategy=FusionStrategy.WEIGHTED,
            enabled_modalities={"video", "audio", "text"},
            modality_weights={
                "video": 0.5,
                "audio": 0.3,
                "text": 0.2
            },
            min_confidence_threshold=0.65,
            target_confidence_threshold=0.75,
            exceptional_threshold=0.85
        )
        
        return {
            "dimension_set": dimension_set,
            "type_registry": type_registry,
            "processing_options": processing_options
        }
    
    @staticmethod
    def education_preset() -> Dict[str, Any]:
        """Create education/e-learning industry preset.
        
        Returns:
            Dictionary with dimension_set, type_registry, and processing_options
        """
        dimensions = {
            "concept_clarity": DimensionDefinition(
                id="concept_clarity",
                name="Concept Clarity",
                description="Clear explanation of educational concepts",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.3,
                scoring_prompt="Rate how clearly concepts are explained from 0-1",
                examples=[
                    {"score": 0.9, "description": "Crystal clear explanation with examples"},
                    {"score": 0.4, "description": "Basic explanation, somewhat unclear"}
                ]
            ),
            "engagement_level": DimensionDefinition(
                id="engagement_level",
                name="Engagement Level",
                description="How engaging the educational content is",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.2,
                scoring_prompt="Rate engagement level of teaching from 0-1"
            ),
            "question_answered": DimensionDefinition(
                id="question_answered",
                name="Question Answered",
                description="Important question asked and answered",
                dimension_type=DimensionType.BINARY,
                default_weight=0.25,
                scoring_prompt="Is an important question being answered? (0 or 1)"
            ),
            "key_insight": DimensionDefinition(
                id="key_insight",
                name="Key Insight",
                description="Valuable insight or 'aha' moment",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.25,
                scoring_prompt="Rate the value of insights provided from 0-1"
            )
        }
        
        dimension_set = DimensionSet(
            id=None,
            name="Education Standard",
            description="Standard dimensions for educational content",
            organization_id=1,
            dimensions=dimensions,
            dimension_weights={k: d.default_weight for k, d in dimensions.items()},
            minimum_dimensions_required=2
        )
        
        type_definitions = {
            "key_concept": HighlightTypeDefinition(
                id="key_concept",
                name="Key Concept",
                description="Important concept explanation",
                criteria={
                    "concept_clarity": {"min": 0.8}
                }
            ),
            "qa_moment": HighlightTypeDefinition(
                id="qa_moment",
                name="Q&A Moment",
                description="Question and answer exchange",
                criteria={
                    "question_answered": {"min": 1.0}
                }
            ),
            "insight_moment": HighlightTypeDefinition(
                id="insight_moment",
                name="Insight Moment",
                description="Valuable insight or revelation",
                criteria={
                    "key_insight": {"min": 0.8}
                }
            )
        }
        
        type_registry = HighlightTypeRegistry(
            id=None,
            organization_id=1,
            types=type_definitions
        )
        
        processing_options = ProcessingOptions(
            detection_strategy=DetectionStrategy.HYBRID,
            fusion_strategy=FusionStrategy.CONSENSUS,
            enabled_modalities={"video", "audio", "text"},
            modality_weights={
                "video": 0.3,
                "audio": 0.5,
                "text": 0.2
            },
            min_confidence_threshold=0.7,
            target_confidence_threshold=0.8
        )
        
        return {
            "dimension_set": dimension_set,
            "type_registry": type_registry,
            "processing_options": processing_options
        }
    
    @staticmethod
    def sports_preset() -> Dict[str, Any]:
        """Create sports broadcasting industry preset.
        
        Returns:
            Dictionary with dimension_set, type_registry, and processing_options
        """
        dimensions = {
            "play_significance": DimensionDefinition(
                id="play_significance",
                name="Play Significance",
                description="Importance of the play to the game",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.35,
                scoring_prompt="Rate the significance of this play from 0-1",
                examples=[
                    {"score": 0.95, "description": "Game-winning play, record-breaking moment"},
                    {"score": 0.3, "description": "Regular play with minor impact"}
                ]
            ),
            "athletic_excellence": DimensionDefinition(
                id="athletic_excellence",
                name="Athletic Excellence",
                description="Display of exceptional athletic ability",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.25,
                scoring_prompt="Rate the athletic excellence displayed from 0-1"
            ),
            "crowd_reaction": DimensionDefinition(
                id="crowd_reaction",
                name="Crowd Reaction",
                description="Audience excitement and reaction",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.2,
                scoring_prompt="Rate the crowd's reaction intensity from 0-1",
                applicable_modalities=["audio", "video"]
            ),
            "commentator_excitement": DimensionDefinition(
                id="commentator_excitement",
                name="Commentator Excitement",
                description="Broadcaster enthusiasm level",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.2,
                scoring_prompt="Rate commentator excitement from 0-1",
                applicable_modalities=["audio"]
            )
        }
        
        dimension_set = DimensionSet(
            id=None,
            name="Sports Standard",
            description="Standard dimensions for sports content",
            organization_id=1,
            dimensions=dimensions,
            dimension_weights={k: d.default_weight for k, d in dimensions.items()}
        )
        
        type_definitions = {
            "game_changer": HighlightTypeDefinition(
                id="game_changer",
                name="Game Changer",
                description="Play that significantly impacts the game",
                criteria={
                    "play_significance": {"min": 0.85}
                }
            ),
            "athletic_feat": HighlightTypeDefinition(
                id="athletic_feat",
                name="Athletic Feat",
                description="Exceptional athletic performance",
                criteria={
                    "athletic_excellence": {"min": 0.8}
                }
            ),
            "crowd_eruption": HighlightTypeDefinition(
                id="crowd_eruption",
                name="Crowd Eruption",
                description="Moment causing major crowd reaction",
                criteria={
                    "crowd_reaction": {"min": 0.85}
                }
            )
        }
        
        type_registry = HighlightTypeRegistry(
            id=None,
            organization_id=1,
            types=type_definitions
        )
        
        processing_options = ProcessingOptions(
            detection_strategy=DetectionStrategy.AI_ONLY,
            fusion_strategy=FusionStrategy.MAX_CONFIDENCE,
            enabled_modalities={"video", "audio"},
            modality_weights={
                "video": 0.6,
                "audio": 0.4
            },
            min_confidence_threshold=0.7,
            exceptional_threshold=0.9
        )
        
        return {
            "dimension_set": dimension_set,
            "type_registry": type_registry,
            "processing_options": processing_options
        }
    
    @staticmethod
    def corporate_preset() -> Dict[str, Any]:
        """Create corporate/business meeting industry preset.
        
        Returns:
            Dictionary with dimension_set, type_registry, and processing_options
        """
        dimensions = {
            "decision_made": DimensionDefinition(
                id="decision_made",
                name="Decision Made",
                description="Important decision or conclusion reached",
                dimension_type=DimensionType.BINARY,
                default_weight=0.3,
                scoring_prompt="Was an important decision made? (0 or 1)"
            ),
            "action_item": DimensionDefinition(
                id="action_item",
                name="Action Item",
                description="Clear action item or task assigned",
                dimension_type=DimensionType.BINARY,
                default_weight=0.25,
                scoring_prompt="Was a clear action item defined? (0 or 1)"
            ),
            "key_metric": DimensionDefinition(
                id="key_metric",
                name="Key Metric",
                description="Important metric or KPI discussed",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.25,
                scoring_prompt="Rate importance of metrics discussed from 0-1"
            ),
            "strategic_insight": DimensionDefinition(
                id="strategic_insight",
                name="Strategic Insight",
                description="Strategic insight or planning moment",
                dimension_type=DimensionType.NUMERIC,
                default_weight=0.2,
                scoring_prompt="Rate the strategic value of insights from 0-1"
            )
        }
        
        dimension_set = DimensionSet(
            id=None,
            name="Corporate Standard",
            description="Standard dimensions for business meetings",
            organization_id=1,
            dimensions=dimensions,
            dimension_weights={k: d.default_weight for k, d in dimensions.items()}
        )
        
        type_definitions = {
            "decision_point": HighlightTypeDefinition(
                id="decision_point",
                name="Decision Point",
                description="Important decision made",
                criteria={
                    "decision_made": {"min": 1.0}
                }
            ),
            "action_items": HighlightTypeDefinition(
                id="action_items",
                name="Action Items",
                description="Action items assigned",
                criteria={
                    "action_item": {"min": 1.0}
                }
            ),
            "metrics_review": HighlightTypeDefinition(
                id="metrics_review",
                name="Metrics Review",
                description="Key metrics discussion",
                criteria={
                    "key_metric": {"min": 0.8}
                }
            )
        }
        
        type_registry = HighlightTypeRegistry(
            id=None,
            organization_id=1,
            types=type_definitions
        )
        
        processing_options = ProcessingOptions(
            detection_strategy=DetectionStrategy.RULE_BASED,
            fusion_strategy=FusionStrategy.WEIGHTED,
            enabled_modalities={"audio", "text"},
            modality_weights={
                "audio": 0.6,
                "text": 0.4
            },
            min_confidence_threshold=0.75
        )
        
        return {
            "dimension_set": dimension_set,
            "type_registry": type_registry,
            "processing_options": processing_options
        }
    
    @staticmethod
    def get_preset(industry: str) -> Dict[str, Any]:
        """Get preset configuration for a specific industry.
        
        Args:
            industry: Industry identifier (gaming, education, sports, corporate)
            
        Returns:
            Preset configuration dictionary
            
        Raises:
            ValueError: If industry is not recognized
        """
        presets = {
            "gaming": IndustryPresets.gaming_preset,
            "education": IndustryPresets.education_preset,
            "sports": IndustryPresets.sports_preset,
            "corporate": IndustryPresets.corporate_preset
        }
        
        if industry not in presets:
            raise ValueError(f"Unknown industry: {industry}. Available: {list(presets.keys())}")
        
        return presets[industry]()