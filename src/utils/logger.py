"""
Centralized logging configuration for FinBot.
Provides structured logging with JSON and text formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from pythonjsonlogger import jsonlogger

from src.config import settings, LOGS_DIR


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_type: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file handler
        format_type: "json" or "text" (defaults to settings.log_format)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Use settings defaults if not provided
    level = level or settings.log_level
    format_type = format_type or settings.log_format

    logger.setLevel(getattr(logging, level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if format_type == "json":
        console_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = LOGS_DIR / log_file if not Path(log_file).is_absolute() else log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        # Always use JSON for file logs for easier parsing
        file_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)


# Default loggers for each module
data_logger = setup_logger("finbot.data", log_file="data.log")
ml_logger = setup_logger("finbot.ml", log_file="ml.log")
llm_logger = setup_logger("finbot.llm", log_file="llm.log")
api_logger = setup_logger("finbot.api", log_file="api.log")


if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("test", format_type="text")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning")
    test_logger.error("This is an error")

    test_logger_json = setup_logger("test_json", format_type="json")
    test_logger_json.info("JSON formatted message", extra={"user": "test", "action": "login"})
