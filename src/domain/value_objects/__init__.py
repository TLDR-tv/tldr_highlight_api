"""Value objects for the domain layer.

Value objects are immutable objects that represent domain concepts
without identity. They are defined by their attributes rather than
by an ID.
"""

from src.domain.value_objects.email import Email
from src.domain.value_objects.url import Url
from src.domain.value_objects.confidence_score import ConfidenceScore
from src.domain.value_objects.duration import Duration
from src.domain.value_objects.timestamp import Timestamp
from src.domain.value_objects.processing_options import ProcessingOptions
from src.domain.value_objects.company_name import CompanyName

__all__ = [
    "Email",
    "Url",
    "ConfidenceScore",
    "Duration",
    "Timestamp",
    "ProcessingOptions",
    "CompanyName",
]
