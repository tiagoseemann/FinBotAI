"""
Configuration management for FinBot project.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # LLM Configuration
    llm_provider: str = "anthropic"  # "anthropic", "openai", or "ollama"
    llm_model: str = "claude-3-5-sonnet-20241022"  # or "gpt-4", "llama2", etc
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500
    ollama_base_url: str = "http://localhost:11434"

    # ML Configuration
    model_path: Path = MODELS_DIR / "lead_scorer.pkl"
    scaler_path: Path = MODELS_DIR / "scaler.pkl"
    feature_names_path: Path = MODELS_DIR / "feature_names.json"

    # Data Configuration
    data_path: Path = DATA_DIR
    synthetic_conversations_path: Path = DATA_DIR / "synthetic_conversations.json"
    products_path: Path = DATA_DIR / "products.json"
    features_db_path: Path = DATA_DIR / "features.duckdb"

    # Training Configuration
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    optuna_n_trials: int = 50
    optuna_timeout: Optional[int] = None  # seconds, None = no timeout

    # Agent Configuration
    lead_score_threshold: float = 0.7  # Threshold for aggressive recommendations
    conversation_max_history: int = 10  # Max messages to keep in context

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"

    # Evaluation
    eval_test_conversations: int = 20
    eval_rouge_types: list[str] = ["rouge1", "rouge2", "rougeL"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_settings() -> list[str]:
    """
    Validate critical settings and return list of errors.

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    # Check LLM configuration
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY not set (required for Claude)")

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        errors.append("OPENAI_API_KEY not set (required for OpenAI)")

    # Check required files
    if not settings.products_path.exists():
        errors.append(f"Products file not found: {settings.products_path}")

    # Check paths are writable
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
        if not os.access(directory, os.W_OK):
            errors.append(f"Directory not writable: {directory}")

    return errors


if __name__ == "__main__":
    # Test configuration
    print("FinBot Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Model Path: {settings.model_path}")
    print(f"Data Path: {settings.data_path}")
    print(f"Log Level: {settings.log_level}")

    errors = validate_settings()
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ All settings valid")
