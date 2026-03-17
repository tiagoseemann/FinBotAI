.PHONY: help install clean test train evaluate demo api format lint typecheck all

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python
PIP := pip

# Directories
SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs
REPORTS_DIR := reports

help: ## Show this help message
	@echo "FinBot - AI Sales Agent with Lead Scoring"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

install-dev: install ## Install dev dependencies
	$(PIP) install black ruff mypy pytest-cov ipython jupyter
	@echo "✓ Dev dependencies installed"

setup: install ## Setup project (install + create directories)
	@mkdir -p $(DATA_DIR) $(MODELS_DIR) $(LOGS_DIR) $(REPORTS_DIR)
	@echo "✓ Project setup complete"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Add your ANTHROPIC_API_KEY or OPENAI_API_KEY"
	@echo "  3. Run: make train"

clean: ## Clean generated files
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	@echo "✓ Cleaned"

clean-data: ## Clean generated data (conversations, models, logs)
	@echo "⚠️  This will delete:"
	@echo "  - Generated conversations"
	@echo "  - Trained models"
	@echo "  - Logs and reports"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f $(DATA_DIR)/synthetic_conversations.json
	rm -f $(DATA_DIR)/features.duckdb
	rm -f $(MODELS_DIR)/*.pkl
	rm -f $(MODELS_DIR)/*.json
	rm -f $(LOGS_DIR)/*.log
	rm -rf $(REPORTS_DIR)/*
	@echo "✓ Data cleaned"

# Data Pipeline
generate-data: ## Generate synthetic conversations
	$(PYTHON) -c "from src.data.generator import generate_dataset; generate_dataset(n_conversations=500)"
	@echo "✓ Generated 500 conversations"

extract-features: ## Extract features from conversations
	$(PYTHON) -c "from src.data.extractor import extract_features_from_file; df = extract_features_from_file(); print(f'Extracted {len(df)} feature sets')"
	@echo "✓ Features extracted"

# Training
train: ## Train ML model with default settings
	$(PYTHON) $(SCRIPTS_DIR)/train.py --n-trials 50
	@echo "✓ Training complete"

train-fast: ## Quick training (10 trials, for testing)
	$(PYTHON) $(SCRIPTS_DIR)/train.py --n-trials 10
	@echo "✓ Quick training complete"

train-full: ## Full training (100 trials, best performance)
	$(PYTHON) $(SCRIPTS_DIR)/train.py --n-trials 100
	@echo "✓ Full training complete"

train-no-opt: ## Train without hyperparameter optimization
	$(PYTHON) $(SCRIPTS_DIR)/train.py --no-optimize
	@echo "✓ Training complete (no optimization)"

retrain: clean-data train ## Clean data and retrain from scratch

# Evaluation
evaluate: ## Run full evaluation (ML + LLM + integration)
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py
	@echo "✓ Evaluation complete"

evaluate-ml: ## Evaluate only ML model
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py --ml-only
	@echo "✓ ML evaluation complete"

evaluate-llm: ## Evaluate only LLM agent
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py --llm-only
	@echo "✓ LLM evaluation complete"

# Demo & API
demo: ## Run interactive demo
	$(PYTHON) $(SCRIPTS_DIR)/demo.py

api: ## Start API server
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Start API in production mode
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Testing
test: ## Run tests
	pytest $(TEST_DIR) -v

test-cov: ## Run tests with coverage
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "✓ Coverage report: htmlcov/index.html"

test-fast: ## Run tests without coverage (faster)
	pytest $(TEST_DIR) -v -x

test-data: ## Run data pipeline tests only
	pytest $(TEST_DIR)/test_data.py -v

test-ml: ## Run ML tests only
	pytest $(TEST_DIR)/test_ml.py -v

test-api: ## Run API tests only
	pytest $(TEST_DIR)/test_api.py -v

# Code Quality
format: ## Format code with black
	black $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "✓ Code formatted"

lint: ## Lint code with ruff
	ruff check $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "✓ Linting complete"

lint-fix: ## Fix linting issues automatically
	ruff check --fix $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@echo "✓ Linting issues fixed"

typecheck: ## Type check with mypy
	mypy $(SRC_DIR) --ignore-missing-imports
	@echo "✓ Type checking complete"

check: format lint typecheck test ## Run all checks (format, lint, typecheck, test)
	@echo "✓ All checks passed"

# Utilities
logs: ## Show recent logs
	@echo "Recent logs:"
	@tail -n 50 $(LOGS_DIR)/*.log 2>/dev/null || echo "No logs found"

stats: ## Show project statistics
	@echo "Project Statistics:"
	@echo "  Python files:  $$(find $(SRC_DIR) -name '*.py' | wc -l)"
	@echo "  Lines of code: $$(find $(SRC_DIR) -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "  Test files:    $$(find $(TEST_DIR) -name '*.py' | wc -l)"
	@echo ""
	@echo "Data:"
	@[ -f $(DATA_DIR)/synthetic_conversations.json ] && echo "  ✓ Conversations generated" || echo "  ✗ No conversations"
	@[ -f $(MODELS_DIR)/lead_scorer.pkl ] && echo "  ✓ Model trained" || echo "  ✗ No trained model"

deps: ## Show installed dependencies
	$(PIP) list

deps-tree: ## Show dependency tree
	pipdeptree

version: ## Show Python and package versions
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@$(PYTHON) -c "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')"
	@$(PYTHON) -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
	@$(PYTHON) -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Docker (bonus)
docker-build: ## Build Docker image
	docker build -t finbot:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env finbot:latest

# Documentation
docs: ## Generate documentation
	@echo "Documentation available in:"
	@echo "  - README.md"
	@echo "  - CLAUDE.md"
	@echo "  - API Docs: http://localhost:8000/docs (when API running)"

jupyter: ## Start Jupyter notebook
	jupyter notebook notebooks/

# All-in-one commands
all: setup train evaluate ## Setup, train, and evaluate
	@echo "✓ Complete pipeline finished"

quick-start: install generate-data train-fast demo ## Quick start (minimal training)
	@echo "✓ Quick start complete"

production: install train-full test api-prod ## Full production setup
	@echo "✓ Production setup complete"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	@$(PYTHON) -m timeit -n 100 -s "from src.ml.model import LeadScorer; scorer = LeadScorer()" "scorer.predict_proba([{'f1': 0.5}])"

# Health checks
health-check: ## Check system health
	@echo "System Health Check:"
	@echo "  Python: $$($(PYTHON) --version)"
	@[ -f .env ] && echo "  ✓ .env exists" || echo "  ✗ .env missing"
	@[ -f $(DATA_DIR)/products.json ] && echo "  ✓ Products exist" || echo "  ✗ Products missing"
	@[ -f $(MODELS_DIR)/lead_scorer.pkl ] && echo "  ✓ Model trained" || echo "  ✗ Model not trained"
	@$(PYTHON) -c "from src.config import validate_settings; errors = validate_settings(); exit(1 if errors else 0)" && echo "  ✓ Config valid" || echo "  ✗ Config issues"
