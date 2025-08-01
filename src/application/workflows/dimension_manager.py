"""Dimension management for B2B AI highlighting - clean Pythonic implementation.

This module handles dimension sets and highlight types, allowing businesses
to customize their AI highlighting criteria.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.domain.entities.dimension_set import DimensionSet
from src.domain.entities.highlight_type_registry import HighlightTypeRegistry
from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.domain.exceptions import EntityNotFoundError, BusinessRuleViolation


@dataclass
class DimensionManager:
    """Manages dimensions and highlight types for customizable AI highlighting.
    
    This allows B2B customers to define their own criteria for what
    constitutes a highlight in their specific domain.
    """
    
    dimension_repo: Any  # Duck typing
    registry_repo: Any
    organization_repo: Any
    
    async def create_custom_dimensions(
        self,
        organization_id: int,
        name: str,
        dimensions: List[Dict[str, Any]],
        industry: Optional[str] = None,
    ) -> DimensionSet:
        """Create a custom dimension set for an organization.
        
        Args:
            organization_id: Organization creating the dimensions
            name: Name for this dimension set
            dimensions: List of dimension definitions
            industry: Optional industry tag
            
        Returns:
            Created dimension set
        """
        # Create dimension set
        dimension_set = DimensionSet(
            id=None,
            organization_id=organization_id,
            name=name,
            description=f"Custom dimensions for {name}",
            industry=industry,
        )
        
        # Add each dimension
        for dim_data in dimensions:
            dimension = DimensionDefinition(
                id=dim_data["id"],
                name=dim_data["name"],
                type=dim_data.get("type", "numeric"),
                description=dim_data.get("description", ""),
                weight=dim_data.get("weight", 1.0),
                min_value=dim_data.get("min_value", 0.0),
                max_value=dim_data.get("max_value", 1.0),
                threshold=dim_data.get("threshold", 0.5),
            )
            dimension_set.add_dimension(dimension, weight=dimension.weight)
        
        # Save and return
        return await self.dimension_repo.save(dimension_set)
    
    async def get_or_create_default(
        self,
        organization_id: int,
        industry: str = "general"
    ) -> DimensionSet:
        """Get or create default dimensions for an organization."""
        # Check for existing sets
        existing = await self.dimension_repo.get_by_organization(
            organization_id, active_only=True
        )
        
        if existing:
            return existing[0]
        
        # Create industry-specific defaults
        return await self._create_industry_defaults(organization_id, industry)
    
    async def create_highlight_types(
        self,
        organization_id: int,
        dimension_set_id: int,
        types: List[Dict[str, Any]]
    ) -> HighlightTypeRegistry:
        """Define custom highlight types based on dimension criteria.
        
        Args:
            organization_id: Organization ID
            dimension_set_id: Associated dimension set
            types: List of type definitions with criteria
            
        Returns:
            Created type registry
        """
        # Get dimension set to validate
        dimension_set = await self.dimension_repo.get(dimension_set_id)
        if not dimension_set or dimension_set.organization_id != organization_id:
            raise BusinessRuleViolation(
                "Invalid dimension set for organization"
            )
        
        # Create registry
        registry = HighlightTypeRegistry.create(
            organization_id=organization_id,
            name=f"{dimension_set.name} Types",
        )
        
        # Add each type
        for type_data in types:
            registry.add_type(
                type_id=type_data["id"],
                name=type_data["name"],
                description=type_data.get("description", ""),
                criteria=type_data["criteria"],  # Dimension score requirements
                priority=type_data.get("priority", 50),
                color=type_data.get("color", "#007bff"),
            )
        
        return await self.registry_repo.save(registry)
    
    async def get_analysis_config(
        self,
        organization_id: int,
        dimension_set_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get complete analysis configuration for an organization.
        
        Returns dimension set and associated highlight types.
        """
        # Get dimension set
        if dimension_set_id:
            dimension_set = await self.dimension_repo.get(dimension_set_id)
            if not dimension_set:
                raise EntityNotFoundError(f"Dimension set {dimension_set_id} not found")
        else:
            dimension_set = await self.get_or_create_default(organization_id)
        
        # Get associated type registry
        registries = await self.registry_repo.get_by_organization(
            organization_id, active_only=True
        )
        
        # Find matching registry for dimension set
        type_registry = None
        for registry in registries:
            if registry.name == f"{dimension_set.name} Types":
                type_registry = registry
                break
        
        return {
            "dimension_set": {
                "id": dimension_set.id,
                "name": dimension_set.name,
                "dimensions": [
                    {
                        "id": dim.id,
                        "name": dim.name,
                        "weight": dimension_set.weights[dim.id].value,
                        "type": dim.dimension_type.value,
                        "description": dim.description,
                    }
                    for dim in dimension_set.dimensions.values()
                ],
            },
            "highlight_types": [
                {
                    "id": ht["id"],
                    "name": ht["name"],
                    "criteria": ht["criteria"],
                    "color": ht.get("color", "#007bff"),
                }
                for ht in (type_registry.types if type_registry else [])
            ] if type_registry else self._get_default_types(),
        }
    
    # Private helpers
    
    async def _create_industry_defaults(
        self,
        organization_id: int,
        industry: str
    ) -> DimensionSet:
        """Create default dimensions for an industry."""
        dimension_set = DimensionSet.create_default(organization_id, industry)
        
        # Add industry-specific dimensions
        if industry == "education":
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="educational_value",
                    name="Educational Value",
                    type="numeric",
                    description="How educational or informative",
                    weight=0.4,
                ),
                weight=0.4
            )
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="clarity",
                    name="Clarity",
                    type="numeric",
                    description="How clearly concepts are explained",
                    weight=0.3,
                ),
                weight=0.3
            )
        elif industry == "sports":
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="athletic_skill",
                    name="Athletic Skill",
                    type="numeric",
                    description="Display of athletic ability",
                    weight=0.4,
                ),
                weight=0.4
            )
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="game_impact",
                    name="Game Impact",
                    type="numeric",
                    description="Impact on game outcome",
                    weight=0.3,
                ),
                weight=0.3
            )
        # Gaming is already handled in create_default
        
        return await self.dimension_repo.save(dimension_set)
    
    def _get_default_types(self) -> List[Dict[str, Any]]:
        """Get default highlight types."""
        return [
            {
                "id": "epic_moment",
                "name": "Epic Moment",
                "criteria": {
                    "action_intensity": {"min": 0.8},
                    "emotional_peak": {"min": 0.7}
                },
                "color": "#ff6b6b",
            },
            {
                "id": "skillful_play",
                "name": "Skillful Play",
                "criteria": {
                    "skill_display": {"min": 0.8}
                },
                "color": "#4ecdc4",
            },
            {
                "id": "funny_moment",
                "name": "Funny Moment",
                "criteria": {
                    "humor": {"min": 0.7}
                },
                "color": "#ffe66d",
            },
        ]