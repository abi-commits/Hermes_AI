"""Timing utilities for profiling."""

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

F = TypeVar("F", bound=Callable[..., Any])

logger = structlog.get_logger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, log_level: str = "debug"):
        """Initialize timer.

        Args:
            name: Timer name for identification.
            log_level: Logging level (debug, info, warning, error).
        """
        self.name = name
        self.log_level = log_level
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer and log elapsed time."""
        self.elapsed = time.perf_counter() - self.start_time  # type: ignore

        log_func = getattr(logger, self.log_level)
        log_func(
            "timer_elapsed",
            name=self.name,
            elapsed_seconds=self.elapsed,
        )


def timed(func: F | None = None, *, name: str | None = None, log_level: str = "debug") -> F:
    """Decorator for timing function execution.

    Args:
        func: Function to decorate.
        name: Optional timer name (defaults to function name).
        log_level: Logging level.

    Returns:
        Decorated function.
    """

    def decorator(fn: F) -> F:
        timer_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log_func = getattr(logger, log_level)
                log_func(
                    "function_elapsed",
                    name=timer_name,
                    elapsed_seconds=elapsed,
                )

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                log_func = getattr(logger, log_level)
                log_func(
                    "function_elapsed",
                    name=timer_name,
                    elapsed_seconds=elapsed,
                )

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    if func is None:
        return decorator  # type: ignore
    return decorator(func)


# Need to import asyncio here for the check above
import asyncio
