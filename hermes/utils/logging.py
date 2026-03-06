"""Custom logging utilities."""

import json
import logging
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the log record serialised as a JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "asctime",
            ):
                log_data[key] = value

        return json.dumps(log_data, default=str)


def configure_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with the given level and optionally JSON output."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
