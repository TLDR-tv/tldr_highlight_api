# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup and Commands

### IMPORTANT: Always Use uv for Python
**ALL Python commands must be run through uv. Never run Python or Python packages directly.**

### Dependency Management with uv
```bash
# Install dependencies
uv pip install -e .

# Add a new dependency
uv pip install <package>

# Install development dependencies
uv pip install -e ".[dev]"
```

### Code Quality Tools
```bash
# Format code with ruff (through uv)
uv run ruff format .

# Lint code with ruff (through uv)
uv run ruff check .
uv run ruff check --fix .  # Auto-fix when possible

# Type checking with mypy (through uv)
uv run mypy src/
```

### Test-Driven Development
This project follows TDD practices. Always write tests first before implementing features.

```bash
# Run all tests (through uv)
uv run pytest

# Run a specific test file
uv run pytest tests/unit/test_stream_processor.py

# Run a specific test
uv run pytest tests/unit/test_stream_processor.py::test_stream_creation

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run tests in watch mode
uv run pytest-watch
```

### Running Python Scripts
```bash
# Run any Python script (through uv)
uv run python script.py

# Run the application
uv run python -m src.api.main

# Run FastAPI development server
uv run uvicorn src.api.main:app --reload
```

## Pythonic Code Conventions

### IMPORTANT: Always Follow Pythonic Conventions
When writing or modifying code in this repository, always adhere to Pythonic idioms and best practices:

1. **Properties over getters/setters**: Use `@property` decorators instead of Java-style `get_*()` and `set_*()` methods
2. **String formatting**: Use f-strings (f"{variable}") for string interpolation
3. **Comprehensions**: Use list/dict/set comprehensions and generator expressions where appropriate
4. **PEP 8**: Follow PEP 8 style guidelines for naming, spacing, and code layout
5. **Context managers**: Use `with` statements for resource management
6. **Iteration**: Use `enumerate()`, `zip()`, and other built-in iteration tools
7. **Error handling**: Use specific exception types and EAFP (Easier to Ask for Forgiveness than Permission)
8. **Simplicity**: Prefer simple, flat structures over deep inheritance hierarchies
9. **Duck typing**: Embrace Python's dynamic typing where appropriate
10. **Standard library**: Use Python's rich standard library before reaching for external packages

### Examples:
```python
# Pythonic property usage
@property
def rate_limit(self) -> int:
    """Get rate limit for this API key."""
    return self._rate_limit

# NOT: def get_rate_limit(self) -> int:

# Pythonic string formatting
logger.info(f"Processing stream {stream_id} with {len(highlights)} highlights")

# NOT: logger.info("Processing stream %s with %d highlights" % (stream_id, len(highlights)))

# Pythonic comprehension
active_streams = [s for s in streams if s.is_active]

# NOT: 
# active_streams = []
# for s in streams:
#     if s.is_active:
#         active_streams.append(s)
```

## Temporary Files and Debug Scripts

### IMPORTANT: Use /tmp for Temporary Files
All temporary debugging scripts, test documents, or experimental files should be created in the `/tmp` directory, not in the project directory.

```bash
# Create temporary debug scripts in /tmp
/tmp/debug_stream_processing.py
/tmp/test_api_endpoints.py
/tmp/analyze_performance.py

# Create temporary documentation in /tmp
/tmp/debugging_notes.md
/tmp/api_test_results.txt
/tmp/performance_analysis.md
```

This keeps the project directory clean and prevents accidental commits of temporary files.

## Architecture Overview

This is a standalone enterprise B2B API service that provides AI-powered highlight extraction from livestreams and video content. It's designed specifically for business clients who need to integrate highlight detection into their own platforms and workflows.

### Core Components

1. **Stream Processing Pipeline**
   - Universal stream processing using FFmpeg (supports any format FFmpeg can handle)
   - Real-time processing of livestreams: RTMP, HLS, DASH, HTTP streams, local files, etc.
   - Independent implementation optimized for enterprise needs
   - Multi-modal analysis: video frames, audio transcription

2. **AI-Powered Analysis**
   - **Customizable Dimensions**: Clients define their own scoring dimensions (e.g., action_intensity, educational_value, humor)
   - **Dynamic Highlight Types**: Create custom highlight types with specific criteria
   - **AI-Based Detection**: Advanced AI-powered highlight detection using multimodal analysis
   - **Fusion Strategies**: Weighted, consensus, cascade, or max-confidence fusion of multi-modal signals
   - **Industry Presets**: Pre-configured templates for gaming, education, sports, and corporate use cases

3. **Enterprise Features**
   - API key-based authentication with scoped permissions
   - Multi-tenant architecture with isolated processing
   - SLA-backed performance guarantees
   - Detailed usage analytics and billing integration
   - Per-organization dimension sets and type registries
   - Flexible processing options with modality configuration

4. **Asynchronous Processing**
   - Celery for background task processing
   - RabbitMQ for message queuing
   - Webhook system for real-time event notifications
   - Batch processing capabilities for existing content

### Key Design Patterns

1. **Adapter Pattern**: Platform-specific adapters (Twitch, YouTube) normalize different streaming platforms
2. **Strategy Pattern**: Flexible processing strategies for different content types
3. **Repository Pattern**: Database models separated from business logic
4. **Event-Driven Architecture**: Webhooks and message queues for async communication

### Enterprise Integration Patterns

The API is designed for easy integration into existing enterprise systems:
- RESTful API with comprehensive OpenAPI documentation
- Multiple SDK options (Python, Node.js, Go, Java, C#)
- Webhook support for event-driven architectures
- Batch processing for content libraries
- Custom integration support for proprietary use cases

### Data Flow

1. Enterprise client authenticates with API key
2. Client submits stream URL or video batch
3. API validates request and creates isolated processing job
4. Job queued with tenant-specific priority
5. Processing pipeline analyzes content using AI models
6. Highlights stored in tenant-isolated S3 buckets
7. Client notified via webhook or polling
8. Usage metrics recorded for billing

### Flexible Highlight Detection System

The highlight detection system has been completely redesigned to be industry-agnostic and highly customizable:

1. **Dimension Definitions**
   - Clients define custom scoring dimensions relevant to their content
   - Each dimension includes: type (numeric/binary/categorical), weight, scoring prompt, examples
   - Dimensions can be specific to certain modalities (video, audio, text)
   - Aggregation methods control how scores are combined across time

2. **Dimension Sets**
   - Groups of related dimensions for specific use cases
   - Organizations can have multiple dimension sets
   - Built-in presets for gaming, education, sports, and corporate content
   - Weights can be normalized and minimum dimensions required can be configured

3. **Highlight Type Registry**
   - Dynamic highlight types replace hardcoded enums
   - Each type has criteria based on dimension scores
   - Supports multiple types per highlight with priority ordering
   - Auto-assignment based on dimension score patterns

4. **Analysis Strategies**
   - **AI-Only**: Uses LLM to score content against dimensions
   - **Rule-Based**: Applies deterministic rules for scoring
   - **Hybrid**: Combines AI and rule-based approaches
   - Strategies implement the same interface for easy swapping

5. **Processing Options**
   - Reference dimension sets and type registries
   - Configure detection and fusion strategies
   - Enable/disable modalities with custom weights
   - Multiple confidence thresholds for different quality levels

### Testing Strategy

Given TDD approach:
1. Write integration tests for API endpoints first
2. Mock external services (AI providers, streaming platforms)
3. Use fixtures for database and Redis state
4. Test webhook delivery with mock HTTP server
5. Load test for enterprise SLA compliance

## Important Development Guidelines

### File Naming and Versioning
- **NEVER create v2 or versioned files** (e.g., `file_v2.py`, `processor_v3.py`)
- **ALWAYS replace the original file** with the new implementation
- If you need to refactor or rewrite a component, update the existing file
- Use Git for version control, not file naming