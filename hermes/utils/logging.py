"""Structured logging configuration for Hermes."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def configure_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """Configure unified structured logging for the entire application.
    
    Integrates standard Python logging with structlog to ensure all logs
    (including those from libraries) are consistent.
    """
    is_production = environment.lower() == "production"
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 1. Standard library logging configuration
    # We pipe all standard library logs through structlog
    shared_processors: list[Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if is_production:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer())

    # Configure the standard logging to use structlog's formatting
    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 2. Final plumbing for the standard library
    # Redirect standard logging to our shared processors
    handler = logging.StreamHandler(sys.stdout)
    
    # Simple formatter for the handler as structlog does the heavy lifting
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # Optional: Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("twilio").setLevel(logging.INFO)
    
    # Special case: ChromaDB can be very noisy
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # 3. Add global context if needed
    structlog.contextvars.clear_contextvars()
