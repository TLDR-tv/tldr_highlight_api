"""Value objects for the domain layer.

Value objects are immutable objects that represent domain concepts
without identity. They are defined by their attributes rather than
by an ID.
"""

from src.domain.value_objects.company_name import CompanyName
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.dimension_constraint import (
    DimensionConstraint,
    ConstraintType,
    ConstraintOperator,
)
from src.domain.value_objects.dimension_definition import (
    DimensionDefinition,
    DimensionType,
    AggregationMethod,
)
from src.domain.value_objects.dimension_score import DimensionScore
from src.domain.value_objects.dimension_set_version import DimensionSetVersion
from src.domain.value_objects.dimension_weight import DimensionWeight
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.email import Email
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.value_objects.prompt_template import PromptTemplate
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.url import Url

__all__ = [
    "CompanyName",
    "ConfidenceScore",
    "DimensionConstraint",
    "ConstraintType", 
    "ConstraintOperator",
    "DimensionDefinition",
    "DimensionType",
    "AggregationMethod",
    "DimensionScore",
    "DimensionSetVersion",
    "DimensionWeight",
    "Duration",
    "Email",
    "ProcessingOptions",
    "PromptTemplate",
    "Timestamp",
    "Url",
]
