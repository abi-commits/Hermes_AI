.PHONY: help install update clean test test-cov lint format type-check check run run-dev build up down logs shell

PYTHON := python3
DOCKER := docker
COMPOSE := docker-compose

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Setup & Dependencies
# =============================================================================

install: ## Install dependencies
	uv sync --extra all

update: ## Update dependencies
	uv lock --upgrade

lock: ## Update uv.lock
	uv lock

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
	uv run pytest -m "not integration and not slow"

test-all: ## Run all tests including integration
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=hermes --cov=config --cov-report=term-missing --cov-report=html

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linters (flake8)
	uv run flake8 hermes config tests

format: ## Format code with black and isort
	uv run black hermes config tests
	uv run isort hermes config tests

format-check: ## Check code formatting without making changes
	uv run black --check hermes config tests
	uv run isort --check-only hermes config tests

type-check: ## Run type checking with mypy
	uv run mypy hermes config

check: format-check lint type-check test ## Run all checks (format, lint, type-check, test)

# =============================================================================
# Development
# =============================================================================

run: ## Run the application
	uv run uvicorn hermes.main:app --host 0.0.0.0 --port 8000

run-dev: ## Run the application in development mode with reload
	uv run uvicorn hermes.main:app --host 0.0.0.0 --port 8000 --reload

seed-kb: ## Seed the knowledge base
	uv run python scripts/seed_knowledge_base.py

# =============================================================================
# Docker
# =============================================================================

build: ## Build Docker image
	$(DOCKER) build -t hermes:latest .

build-cosyvoice2: ## Build the CosyVoice2 TTS server image (GPU required)
	$(DOCKER) build -t hermes-cosyvoice2:latest -f docker/cosyvoice2/Dockerfile .

up: ## Start services with docker-compose
	$(COMPOSE) up -d

up-gpu: ## Start all services including CosyVoice2 (GPU required)
	$(COMPOSE) --profile gpu up -d

down: ## Stop services with docker-compose
	$(COMPOSE) down

down-gpu: ## Stop all services including GPU profile
	$(COMPOSE) --profile gpu down

logs: ## View docker-compose logs
	$(COMPOSE) logs -f

shell: ## Open a shell in the app container
	$(COMPOSE) exec app /bin/bash

# =============================================================================
# Utilities
# =============================================================================

benchmark-tts: ## Benchmark TTS latency
	uv run python scripts/benchmark_tts.py

security-check: ## Run security checks
	uv run bandit -r hermes

pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files
