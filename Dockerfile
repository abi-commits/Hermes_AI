# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.13-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies (without dev dependencies)
RUN uv sync --no-dev --no-install-project --extra all

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.13-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    PORT=8000

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set work directory
WORKDIR $APP_HOME

# Copy virtualenv from builder
COPY --from=builder /app/.venv ./.venv

# Copy application code
COPY --chown=appuser:appgroup hermes/ ./hermes/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup pyproject.toml ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD [".venv/bin/uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM builder AS development

# Install dev dependencies
RUN uv sync --extra all

# Set work directory
WORKDIR /app

# Copy application code
COPY . .

# Run in development mode with reload
CMD ["uv", "run", "uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
