FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_HOME=/app \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libffi-dev \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/


# Builder stage: install dependencies
FROM base AS builder

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra all


# Production stage: minimal image with only runtime dependencies
FROM base AS production

RUN groupadd --system appgroup && useradd --system --gid appgroup --create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appgroup hermes/ ./hermes/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup modal_deploy/ ./modal_deploy/

ENV PATH="/app/.venv/bin:${PATH}"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage: full environment with dev tools and hot-reload
FROM base AS development

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra all

COPY . .

# Keep development as root for convenience (shared volumes may need root permissions)
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["uv", "run", "uvicorn", "hermes.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
