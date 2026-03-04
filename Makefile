.PHONY: help install update clean test test-cov lint format type-check check run run-dev build up down logs shell

PYTHON := python3
POETRY := poetry
DOCKER := docker
COMPOSE := docker-compose

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Setup & Dependencies
# =============================================================================

install: ## Install dependencies
	$(POETRY) install --extras all

update: ## Update dependencies
	$(POETRY) update

lock: ## Update poetry.lock
	$(POETRY) lock

clean: ## Clean cache and temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

# =============================================================================
# Testing
# =============================================================================

test: ## Run unit tests
	$(POETRY) run pytest -m "not integration and not slow"

test-all: ## Run all tests including integration
	$(POETRY) run pytest

test-cov: ## Run tests with coverage report
	$(POETRY) run pytest --cov=hermes --cov=config --cov-report=term-missing --cov-report=html

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linters (flake8)
	$(POETRY) run flake8 hermes config tests

format: ## Format code with black and isort
	$(POETRY) run black hermes config tests
	$(POETRY) run isort hermes config tests

format-check: ## Check code formatting without making changes
	$(POETRY) run black --check hermes config tests
	$(POETRY) run isort --check-only hermes config tests

type-check: ## Run type checking with mypy
	$(POETRY) run mypy hermes config

check: format-check lint type-check test ## Run all checks (format, lint, type-check, test)

# =============================================================================
# Development
# =============================================================================

run: ## Run the application
	$(POETRY) run uvicorn hermes.main:app --host 0.0.0.0 --port 8000

run-dev: ## Run the application in development mode with reload
	$(POETRY) run uvicorn hermes.main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Run the TTS worker
	$(POETRY) run python -m hermes.workers.tts_worker

seed-kb: ## Seed the knowledge base
	$(POETRY) run python scripts/seed_knowledge_base.py

# =============================================================================
# Docker
# =============================================================================

build: ## Build Docker image
	$(DOCKER) build -t hermes:latest .

up: ## Start services with docker-compose
	$(COMPOSE) up -d

down: ## Stop services with docker-compose
	$(COMPOSE) down

logs: ## View docker-compose logs
	$(COMPOSE) logs -f

shell: ## Open a shell in the app container
	$(COMPOSE) exec app /bin/bash

# =============================================================================
# Database
# =============================================================================

db-migrate: ## Run database migrations
	$(POETRY) run alembic upgrade head

db-rollback: ## Rollback database migrations
	$(POETRY) run alembic downgrade -1

db-revision: ## Create a new migration revision (usage: make db-revision MSG="description")
	$(POETRY) run alembic revision --autogenerate -m "$(MSG)"

# =============================================================================
# Utilities
# =============================================================================

benchmark-tts: ## Benchmark TTS latency
	$(POETRY) run python scripts/benchmark_tts.py

security-check: ## Run security checks
	$(POETRY) run bandit -r hermes

pre-commit: ## Run pre-commit hooks
	$(POETRY) run pre-commit run --all-files
