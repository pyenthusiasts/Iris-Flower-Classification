.PHONY: help install install-dev test lint format clean docker-build docker-run api docs

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
FLAKE8 := $(PYTHON) -m flake8
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy
DOCKER_IMAGE := iris-classifier
DOCKER_TAG := latest

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)Iris Flower Classification - Makefile Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	@echo '$(BLUE)Installing production dependencies...$(NC)'
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-api.txt
	$(PIP) install -e .
	@echo '$(GREEN)Installation complete!$(NC)'

install-dev: ## Install development dependencies
	@echo '$(BLUE)Installing development dependencies...$(NC)'
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-api.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install
	@echo '$(GREEN)Development installation complete!$(NC)'

# Testing targets
test: ## Run tests with coverage
	@echo '$(BLUE)Running tests...$(NC)'
	$(PYTEST) tests/ -v --cov=iris_classifier --cov-report=html --cov-report=term-missing
	@echo '$(GREEN)Tests complete! Coverage report: htmlcov/index.html$(NC)'

test-fast: ## Run tests without coverage
	@echo '$(BLUE)Running fast tests...$(NC)'
	$(PYTEST) tests/ -v
	@echo '$(GREEN)Tests complete!$(NC)'

test-watch: ## Run tests in watch mode
	@echo '$(BLUE)Running tests in watch mode...$(NC)'
	$(PYTEST) tests/ -v --looponfail

# Code quality targets
lint: ## Run all linters
	@echo '$(BLUE)Running linters...$(NC)'
	$(FLAKE8) src/iris_classifier --max-line-length=100 --statistics
	$(MYPY) src/iris_classifier --ignore-missing-imports
	$(PYTHON) -m bandit -r src/iris_classifier -c pyproject.toml
	@echo '$(GREEN)Linting complete!$(NC)'

format: ## Format code with black and isort
	@echo '$(BLUE)Formatting code...$(NC)'
	$(BLACK) src/iris_classifier tests/
	$(ISORT) src/iris_classifier tests/
	@echo '$(GREEN)Formatting complete!$(NC)'

format-check: ## Check code formatting without changes
	@echo '$(BLUE)Checking code formatting...$(NC)'
	$(BLACK) --check src/iris_classifier tests/
	$(ISORT) --check src/iris_classifier tests/

security: ## Run security checks
	@echo '$(BLUE)Running security checks...$(NC)'
	$(PYTHON) -m bandit -r src/iris_classifier -c pyproject.toml
	$(PYTHON) -m safety check
	@echo '$(GREEN)Security checks complete!$(NC)'

# Cleaning targets
clean: ## Clean build artifacts and caches
	@echo '$(BLUE)Cleaning...$(NC)'
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	@echo '$(GREEN)Cleaned!$(NC)'

clean-models: ## Remove saved models
	@echo '$(BLUE)Cleaning models...$(NC)'
	rm -f models/*.pkl models/*.joblib
	@echo '$(GREEN)Models cleaned!$(NC)'

# Docker targets
docker-build: ## Build Docker image
	@echo '$(BLUE)Building Docker image...$(NC)'
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo '$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)'

docker-run: ## Run Docker container
	@echo '$(BLUE)Running Docker container...$(NC)'
	docker run -d -p 8000:8000 --name iris-classifier-api $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo '$(GREEN)Container running on http://localhost:8000$(NC)'

docker-stop: ## Stop Docker container
	@echo '$(BLUE)Stopping Docker container...$(NC)'
	docker stop iris-classifier-api
	docker rm iris-classifier-api
	@echo '$(GREEN)Container stopped$(NC)'

docker-compose-up: ## Start all services with docker-compose
	@echo '$(BLUE)Starting services...$(NC)'
	docker-compose up -d
	@echo '$(GREEN)Services started!$(NC)'
	@echo 'API: http://localhost:8000'
	@echo 'Prometheus: http://localhost:9090'
	@echo 'Grafana: http://localhost:3000 (admin/admin)'

docker-compose-down: ## Stop all services
	@echo '$(BLUE)Stopping services...$(NC)'
	docker-compose down
	@echo '$(GREEN)Services stopped!$(NC)'

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# API targets
api: ## Run the API locally
	@echo '$(BLUE)Starting API server...$(NC)'
	$(PYTHON) -m uvicorn iris_classifier.api:app --host 0.0.0.0 --port 8000 --reload

api-prod: ## Run the API in production mode
	@echo '$(BLUE)Starting API server (production)...$(NC)'
	$(PYTHON) -m uvicorn iris_classifier.api:app --host 0.0.0.0 --port 8000 --workers 4

# Analysis targets
run: ## Run the main analysis script
	@echo '$(BLUE)Running main analysis...$(NC)'
	$(PYTHON) main.py

train: ## Train a specific model (usage: make train MODEL=random_forest)
	@echo '$(BLUE)Training model: $(MODEL)$(NC)'
	$(PYTHON) -m iris_classifier.cli train --model $(MODEL) --save

compare: ## Compare all models
	@echo '$(BLUE)Comparing models...$(NC)'
	$(PYTHON) -m iris_classifier.cli compare --plot

# Documentation targets
docs: ## Build documentation
	@echo '$(BLUE)Building documentation...$(NC)'
	cd docs && make html
	@echo '$(GREEN)Documentation built: docs/_build/html/index.html$(NC)'

docs-serve: ## Serve documentation locally
	@echo '$(BLUE)Serving documentation...$(NC)'
	$(PYTHON) -m http.server 8080 --directory docs/_build/html

# Development targets
notebook: ## Start Jupyter notebook server
	@echo '$(BLUE)Starting Jupyter notebook...$(NC)'
	jupyter notebook notebooks/

pre-commit: ## Run pre-commit hooks on all files
	@echo '$(BLUE)Running pre-commit hooks...$(NC)'
	pre-commit run --all-files
	@echo '$(GREEN)Pre-commit checks complete!$(NC)'

setup-hooks: ## Setup git hooks
	@echo '$(BLUE)Setting up git hooks...$(NC)'
	pre-commit install
	@echo '$(GREEN)Git hooks installed!$(NC)'

# Performance targets
benchmark: ## Run performance benchmarks
	@echo '$(BLUE)Running benchmarks...$(NC)'
	$(PYTHON) scripts/benchmark.py

load-test: ## Run load tests
	@echo '$(BLUE)Running load tests...$(NC)'
	locust -f tests/load_test.py --headless -u 100 -r 10 -t 1m

# Package targets
build: ## Build distribution packages
	@echo '$(BLUE)Building packages...$(NC)'
	$(PYTHON) -m build
	@echo '$(GREEN)Packages built in dist/$(NC)'

publish-test: build ## Publish to TestPyPI
	@echo '$(BLUE)Publishing to TestPyPI...$(NC)'
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo '$(BLUE)Publishing to PyPI...$(NC)'
	$(PYTHON) -m twine upload dist/*

# Info targets
info: ## Show project information
	@echo '$(BLUE)Project Information$(NC)'
	@echo 'Python version: $(shell $(PYTHON) --version)'
	@echo 'Pip version: $(shell $(PIP) --version)'
	@echo 'Installed packages:'
	@$(PIP) list | grep -E 'scikit-learn|pandas|numpy|fastapi|uvicorn'

version: ## Show version
	@$(PYTHON) -c "from iris_classifier import __version__; print(__version__)"

# All-in-one targets
all: clean install test lint ## Clean, install, test, and lint

ci: clean install-dev lint test ## Run CI pipeline locally

# Default target
.DEFAULT_GOAL := help
