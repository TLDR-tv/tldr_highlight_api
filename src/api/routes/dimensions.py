"""Dimension management API routes - clean Pythonic implementation.

Allows B2B customers to customize their highlight detection criteria.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.application.workflows import DimensionManager
from ..dependencies import get_current_user, get_dimension_manager


router = APIRouter(prefix="/dimensions", tags=["dimensions"])


# Request/Response models

class DimensionDefinition(BaseModel):
    """Definition of a scoring dimension."""
    id: str
    name: str
    description: str
    type: str = "numeric"
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    threshold: float = 0.5


class DimensionSetRequest(BaseModel):
    """Request to create a dimension set."""
    name: str
    dimensions: List[DimensionDefinition]
    industry: Optional[str] = None


class HighlightTypeDefinition(BaseModel):
    """Definition of a highlight type."""
    id: str
    name: str
    description: Optional[str] = None
    criteria: dict  # Dimension requirements
    priority: int = 50
    color: str = "#007bff"


class HighlightTypesRequest(BaseModel):
    """Request to create highlight types."""
    types: List[HighlightTypeDefinition]


# API Endpoints

@router.post("/sets", status_code=status.HTTP_201_CREATED)
async def create_dimension_set(
    request: DimensionSetRequest,
    user=Depends(get_current_user),
    manager: DimensionManager = Depends(get_dimension_manager)
):
    """Create a custom dimension set for your organization."""
    try:
        # Convert to dicts for the manager
        dimensions = [dim.dict() for dim in request.dimensions]
        
        dimension_set = await manager.create_custom_dimensions(
            organization_id=user.organization_id,
            name=request.name,
            dimensions=dimensions,
            industry=request.industry,
        )
        
        return {
            "id": dimension_set.id,
            "name": dimension_set.name,
            "dimensions": len(dimension_set.dimensions),
            "industry": dimension_set.industry,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sets/default")
async def get_default_dimension_set(
    industry: str = "general",
    user=Depends(get_current_user),
    manager: DimensionManager = Depends(get_dimension_manager)
):
    """Get or create default dimensions for your organization."""
    try:
        dimension_set = await manager.get_or_create_default(
            organization_id=user.organization_id,
            industry=industry
        )
        
        return {
            "id": dimension_set.id,
            "name": dimension_set.name,
            "dimensions": [
                {
                    "id": dim.id,
                    "name": dim.name,
                    "weight": dimension_set.weights[dim.id].value,
                    "description": dim.description,
                }
                for dim in dimension_set.dimensions.values()
            ],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/sets/{set_id}/types", status_code=status.HTTP_201_CREATED)
async def create_highlight_types(
    set_id: int,
    request: HighlightTypesRequest,
    user=Depends(get_current_user),
    manager: DimensionManager = Depends(get_dimension_manager)
):
    """Define custom highlight types based on dimension criteria."""
    try:
        types = [t.dict() for t in request.types]
        
        registry = await manager.create_highlight_types(
            organization_id=user.organization_id,
            dimension_set_id=set_id,
            types=types,
        )
        
        return {
            "id": registry.id,
            "name": registry.name,
            "types_count": len(registry.types),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/config")
async def get_analysis_config(
    dimension_set_id: Optional[int] = None,
    user=Depends(get_current_user),
    manager: DimensionManager = Depends(get_dimension_manager)
):
    """Get complete analysis configuration for your organization."""
    try:
        config = await manager.get_analysis_config(
            organization_id=user.organization_id,
            dimension_set_id=dimension_set_id
        )
        
        return config
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# Industry presets endpoint
@router.get("/presets")
async def get_industry_presets():
    """Get available industry presets."""
    return {
        "presets": [
            {
                "industry": "gaming",
                "name": "Gaming Highlights",
                "description": "Optimized for gaming content - action, skill, emotion",
                "dimensions": ["action_intensity", "skill_display", "emotional_peak"],
            },
            {
                "industry": "education",
                "name": "Educational Moments",
                "description": "Find key teaching moments and explanations",
                "dimensions": ["educational_value", "clarity", "engagement"],
            },
            {
                "industry": "sports",
                "name": "Sports Highlights",
                "description": "Athletic achievements and game-changing moments",
                "dimensions": ["athletic_skill", "game_impact", "crowd_reaction"],
            },
            {
                "industry": "corporate",
                "name": "Corporate Events",
                "description": "Key moments in presentations and meetings",
                "dimensions": ["key_points", "engagement", "sentiment"],
            },
        ]
    }