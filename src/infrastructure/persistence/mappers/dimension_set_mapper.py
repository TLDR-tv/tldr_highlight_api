"""Mapper for DimensionSet domain entity to persistence model conversion."""

from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.dimension_set_aggregate import (
    DimensionSetAggregate as DomainDimensionSet,
)
from src.domain.value_objects.dimension_definition import (
    DimensionDefinition,
    DimensionType,
    AggregationMethod,
)
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.dimension_set import (
    DimensionSet as PersistenceDimensionSet,
)


class DimensionSetMapper(Mapper[DomainDimensionSet, PersistenceDimensionSet]):
    """Maps between DimensionSet domain entity and persistence model."""

    def to_domain(self, model: PersistenceDimensionSet) -> DomainDimensionSet:
        """Convert DimensionSet persistence model to domain entity."""
        # Parse dimensions from JSON
        dimensions = {}
        for dim_id, dim_data in model.dimensions.items():
            dimension = DimensionDefinition(
                id=dim_data["id"],
                name=dim_data["name"],
                description=dim_data["description"],
                dimension_type=DimensionType(dim_data.get("dimension_type", "numeric")),
                default_weight=dim_data.get("default_weight", 0.1),
                min_value=dim_data.get("min_value", 0.0),
                max_value=dim_data.get("max_value", 1.0),
                threshold=dim_data.get("threshold", 0.5),
                scoring_prompt=dim_data.get("scoring_prompt", ""),
                examples=dim_data.get("examples", []),
                applicable_modalities=dim_data.get(
                    "applicable_modalities", ["video", "audio", "text"]
                ),
                aggregation_method=AggregationMethod(
                    dim_data.get("aggregation_method", "max")
                ),
            )
            dimensions[dim_id] = dimension

        # Parse dimension weights
        dimension_weights = model.dimension_weights or {}

        return DomainDimensionSet(
            id=model.id,
            name=model.name,
            description=model.description,
            organization_id=model.organization_id,
            dimensions=dimensions,
            dimension_weights=dimension_weights,
            allow_partial_scoring=model.allow_partial_scoring,
            minimum_dimensions_required=model.minimum_dimensions_required,
            weight_normalization=model.weight_normalization,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
        )

    def to_persistence(self, entity: DomainDimensionSet) -> PersistenceDimensionSet:
        """Convert DimensionSet domain entity to persistence model."""
        model = PersistenceDimensionSet()

        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id

        model.name = entity.name
        model.description = entity.description
        model.organization_id = entity.organization_id

        # Serialize dimensions to JSON
        dimensions_data = {}
        for dim_id, dimension in entity.dimensions.items():
            dim_data = {
                "id": dimension.id,
                "name": dimension.name,
                "description": dimension.description,
                "dimension_type": dimension.dimension_type.value,
                "default_weight": dimension.default_weight,
                "min_value": dimension.min_value,
                "max_value": dimension.max_value,
                "threshold": dimension.threshold,
                "scoring_prompt": dimension.scoring_prompt,
                "examples": dimension.examples,
                "applicable_modalities": list(dimension.applicable_modalities),
                "aggregation_method": dimension.aggregation_method.value,
            }
            dimensions_data[dim_id] = dim_data

        model.dimensions = dimensions_data
        model.dimension_weights = entity.dimension_weights

        # Set configuration
        model.allow_partial_scoring = entity.allow_partial_scoring
        model.minimum_dimensions_required = entity.minimum_dimensions_required
        model.weight_normalization = entity.weight_normalization

        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value

        return model
