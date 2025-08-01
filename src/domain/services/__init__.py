"""Domain services have been refactored following Pythonic DDD principles.

Pure domain logic has been moved to function modules:
- src/domain/scoring.py - Dimension scoring functions
- src/domain/calibration.py - Dimension calibration functions  
- src/domain/validation.py - Dimension validation functions

Application services have been moved to:
- src/application/services/ - Workflow orchestration services

Infrastructure services have been moved to:
- src/infrastructure/webhooks/ - Webhook delivery
- src/infrastructure/agents/ - AI agent integrations
- src/infrastructure/evaluation/ - Evaluation strategies

This follows the principle that domain services should contain only
pure business logic without infrastructure or orchestration concerns.
"""

# This directory is kept for backward compatibility
# New code should import from the appropriate modules directly