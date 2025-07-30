# Development Setup Guide

This guide will help you set up a local development environment for the TL;DR Highlight API.

## 📋 Prerequisites

### System Requirements
- **Python 3.11+** (Python 3.12 recommended)
- **Docker & Docker Compose** (for local services)
- **Git** (latest version)
- **FFmpeg 4.4+** (for video processing)

### Operating System Support
- **macOS** 12+ (recommended for development)
- **Linux** (Ubuntu 20.04+, Fedora 35+)
- **Windows** 11 with WSL2

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/tldr-tv/tldr-highlight-api.git
cd tldr-highlight-api
```

### 2. Install uv (Python Package Manager)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

### 3. Install Dependencies
```bash
# Install all dependencies including dev dependencies
uv pip install -e ".[dev]"
```

### 4. Start Infrastructure Services
```bash
# Start PostgreSQL, Redis, and RabbitMQ
docker-compose up -d postgres redis rabbitmq
```

### 5. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration (see Configuration section below)
vim .env
```

### 6. Initialize Database
```bash
# Run database migrations
uv run alembic upgrade head

# Optional: Seed test data
uv run python scripts/seed_test_data.py
```

### 7. Start the Application
```bash
# Start FastAPI development server
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Start Background Workers
```bash
# In a separate terminal, start Celery workers
uv run celery -A src.infrastructure.queue.celery_app worker --loglevel=info
```

## ⚙️ Detailed Setup

### Python Environment Setup

#### Using uv (Recommended)
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

#### Using Traditional pip/venv
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### Infrastructure Services Setup

#### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres redis rabbitmq

# View logs
docker-compose logs -f postgres

# Stop all services
docker-compose down
```

#### Local Installation (Alternative)

**PostgreSQL 13+**
```bash
# macOS (Homebrew)
brew install postgresql@13
brew services start postgresql@13

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-13 postgresql-contrib

# Create database
createdb tldr_highlight_api
```

**Redis 6+**
```bash
# macOS (Homebrew)
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server
```

**RabbitMQ (for Celery)**
```bash
# macOS (Homebrew)
brew install rabbitmq
brew services start rabbitmq

# Ubuntu/Debian
sudo apt install rabbitmq-server
sudo systemctl start rabbitmq-server
```

### FFmpeg Installation

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu/Debian**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows**
```bash
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

### External Service Setup

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to your `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

#### AWS S3 (Optional for local development)
1. Create AWS account and S3 bucket
2. Create IAM user with S3 permissions
3. Add credentials to `.env`:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
```

#### Logfire (Optional)
1. Sign up at [Logfire](https://logfire.pydantic.dev/)
2. Create a project and get token
3. Add to `.env`:
```bash
LOGFIRE_TOKEN=your_logfire_token
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application Settings
APP_NAME="TL;DR Highlight API"
APP_VERSION="1.0.0"
APP_ENVIRONMENT="development"
DEBUG=true
LOG_LEVEL="DEBUG"

# API Configuration
API_V1_PREFIX="/api/v1"
API_KEY_HEADER="X-API-Key"

# Database Configuration
DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/tldr_highlight_api"

# Redis Configuration
REDIS_URL="redis://localhost:6379/0"
CACHE_TTL=3600

# Celery Configuration
CELERY_BROKER_URL="redis://localhost:6379/1"
CELERY_RESULT_BACKEND="redis://localhost:6379/2"

# AI Services
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="gemini-2.0-flash-exp"

# Storage
AWS_ACCESS_KEY_ID="your_access_key"
AWS_SECRET_ACCESS_KEY="your_secret_key"
AWS_S3_BUCKET="tldr-dev-highlights"
AWS_REGION="us-east-1"

# Security
SECRET_KEY="your-super-secret-key-change-this-in-production"
JWT_SECRET_KEY="your-jwt-secret-key"
JWT_ALGORITHM="HS256"
JWT_EXPIRE_MINUTES=1440

# CORS Settings
CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# Observability
LOGFIRE_TOKEN="your_logfire_token"
LOGFIRE_PROJECT_NAME="tldr-highlight-api-dev"
LOGFIRE_CAPTURE_HEADERS=true
LOGFIRE_CAPTURE_BODY=false

# Webhook Settings
WEBHOOK_SECRET_KEY="your-webhook-secret-key"
WEBHOOK_TIMEOUT_SECONDS=30
WEBHOOK_MAX_RETRIES=5
```

### Docker Compose Configuration

The included `docker-compose.yml` provides local development services:

```yaml
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: tldr_highlight_api
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
```

## 🧪 Development Workflow

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_stream_processor.py

# Run in watch mode
uv run pytest-watch
```

### Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .
uv run ruff check --fix .

# Type checking
uv run mypy src/
```

### Database Operations
```bash
# Create migration
uv run alembic revision --autogenerate -m "Description of changes"

# Apply migrations
uv run alembic upgrade head

# Downgrade migration
uv run alembic downgrade -1

# View migration history
uv run alembic history
```

### Running the Application

#### Development Server
```bash
# Start with auto-reload
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# With custom log level
uv run uvicorn src.api.main:app --reload --log-level debug
```

#### Background Workers
```bash
# Start Celery worker
uv run celery -A src.infrastructure.queue.celery_app worker --loglevel=info

# Start with auto-reload (development)
uv run watchfiles --ignore-paths alembic/ 'celery -A src.infrastructure.queue.celery_app worker --loglevel=info'

# Monitor tasks
uv run celery -A src.infrastructure.queue.celery_app events
```

#### Flower (Celery Monitoring)
```bash
# Start Flower dashboard
uv run celery -A src.infrastructure.queue.celery_app flower

# Access at http://localhost:5555
```

## 🔍 Development Tools

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Database Tools
- **pgAdmin**: Use with connection string `postgresql://postgres:password@localhost:5432/tldr_highlight_api`
- **DBeaver**: Community edition works well
- **Database IDE**: Built into PyCharm Professional

### Monitoring Tools
- **Flower**: http://localhost:5555 (Celery monitoring)
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)
- **Redis Commander**: `npm install -g redis-commander && redis-commander`

## 🐛 Debugging

### Debug Configuration

**VS Code (.vscode/launch.json)**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Debug",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/.venv/bin/uvicorn",
            "args": [
                "src.api.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

**PyCharm Configuration**
1. Create new Python configuration
2. Script path: `uvicorn`
3. Parameters: `src.api.main:app --reload --host 0.0.0.0 --port 8000`
4. Environment variables: Load from `.env` file

### Common Debug Scenarios

#### Database Connection Issues
```bash
# Test database connection
uv run python -c "
import asyncio
from src.infrastructure.database import init_db
asyncio.run(init_db())
print('Database connection successful!')
"
```

#### Redis Connection Issues
```bash
# Test Redis connection
uv run python -c "
import asyncio
from src.infrastructure.cache import get_redis_cache
async def test():
    cache = await get_redis_cache()
    await cache.set('test', 'value')
    result = await cache.get('test')
    print(f'Redis test: {result}')
asyncio.run(test())
"
```

#### Celery Worker Issues
```bash
# Check Celery configuration
uv run celery -A src.infrastructure.queue.celery_app inspect stats

# Test task execution
uv run python -c "
from src.infrastructure.async_processing.tasks import health_check_task
result = health_check_task.delay()
print(f'Task result: {result.get()}')
"
```

## 🔒 Security Considerations

### Development Security
- Use strong, unique values for `SECRET_KEY` and `JWT_SECRET_KEY`
- Never commit real API keys or secrets to version control
- Use `.env.local` for sensitive local overrides (add to `.gitignore`)
- Enable HTTPS in production environments

### API Key Management
```bash
# Generate secure API key
uv run python -c "
import secrets
print(f'tldr_sk_{secrets.token_urlsafe(32)}')
"

# Test API authentication
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/v1/streams
```

## 📝 Common Issues

### Port Conflicts
```bash
# Check what's using port 8000
lsof -i :8000

# Kill process on port
kill -9 $(lsof -t -i:8000)

# Use different port
uv run uvicorn src.api.main:app --reload --port 8001
```

### Permission Issues
```bash
# Fix uv permissions (macOS/Linux)
chmod +x ~/.cargo/bin/uv

# Reset virtual environment
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Database Migration Issues
```bash
# Reset database (development only!)
docker-compose down -v
docker-compose up -d postgres
uv run alembic upgrade head
```

## 🚀 Next Steps

Once you have the development environment running:

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Run tests**: Execute `uv run pytest` to ensure everything works
3. **Check the examples**: Review files in the `examples/` directory
4. **Read the architecture docs**: Understand the system design
5. **Start developing**: Follow the [contributing guide](./development/contributing.md)

## 📞 Getting Help

If you encounter issues:

1. Check the [troubleshooting guide](./troubleshooting/common-issues.md)
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Join our developer Discord community

---

*Happy coding! 🎉*