"""Prometheus metrics endpoint.

Exposes application metrics for monitoring and alerting.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

if TYPE_CHECKING:
    pass

router = APIRouter(tags=["metrics"])

# Application info
APP_INFO = Info("hermes_info", "Hermes application information")

# Active calls gauge
ACTIVE_CALLS = Gauge(
    "hermes_active_calls",
    "Number of currently active calls",
)

# Call duration histogram
CALL_DURATION = Histogram(
    "hermes_call_duration_seconds",
    "Call duration in seconds",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# Call counter
CALLS_TOTAL = Counter(
    "hermes_calls_total",
    "Total number of calls",
    ["status"],
)

# STT latency
STT_LATENCY = Histogram(
    "hermes_stt_latency_seconds",
    "Speech-to-text latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# LLM latency
LLM_LATENCY = Histogram(
    "hermes_llm_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# TTS latency
TTS_LATENCY = Histogram(
    "hermes_tts_latency_seconds",
    "Text-to-speech latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# LLM tokens
LLM_TOKENS = Counter(
    "hermes_llm_tokens_total",
    "Total LLM tokens generated",
    ["type"],  # prompt, completion
)

# STT errors
STT_ERRORS = Counter(
    "hermes_stt_errors_total",
    "Total STT errors",
    ["error_type"],
)

# TTS errors
TTS_ERRORS = Counter(
    "hermes_tts_errors_total",
    "Total TTS errors",
    ["error_type"],
)

# LLM errors
LLM_ERRORS = Counter(
    "hermes_llm_errors_total",
    "Total LLM errors",
    ["error_type"],
)

# WebSocket connections
WS_CONNECTIONS = Gauge(
    "hermes_websocket_connections",
    "Number of active WebSocket connections",
)

# Audio bytes processed
AUDIO_BYTES = Counter(
    "hermes_audio_bytes_total",
    "Total audio bytes processed",
    ["direction"],  # inbound, outbound
)


# Set application info on startup
APP_INFO.info({"version": "0.1.0", "name": "hermes"})


class MetricsCollector:
    """Helper class for collecting and exposing metrics."""

    @staticmethod
    def record_call_started() -> None:
        """Record a call started."""
        ACTIVE_CALLS.inc()

    @staticmethod
    def record_call_ended(status: str, duration: float) -> None:
        """Record a call ended.

        Args:
            status: Call status (completed, failed, etc.).
            duration: Call duration in seconds.
        """
        ACTIVE_CALLS.dec()
        CALL_DURATION.observe(duration)
        CALLS_TOTAL.labels(status=status).inc()

    @staticmethod
    def record_stt_latency(seconds: float) -> None:
        """Record STT latency.

        Args:
            seconds: Latency in seconds.
        """
        STT_LATENCY.observe(seconds)

    @staticmethod
    def record_llm_latency(seconds: float) -> None:
        """Record LLM latency.

        Args:
            seconds: Latency in seconds.
        """
        LLM_LATENCY.observe(seconds)

    @staticmethod
    def record_tts_latency(seconds: float) -> None:
        """Record TTS latency.

        Args:
            seconds: Latency in seconds.
        """
        TTS_LATENCY.observe(seconds)

    @staticmethod
    def record_llm_tokens(prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token usage.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
        """
        LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
        LLM_TOKENS.labels(type="completion").inc(completion_tokens)

    @staticmethod
    def record_stt_error(error_type: str) -> None:
        """Record an STT error.

        Args:
            error_type: Type of error.
        """
        STT_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_tts_error(error_type: str) -> None:
        """Record a TTS error.

        Args:
            error_type: Type of error.
        """
        TTS_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_llm_error(error_type: str) -> None:
        """Record an LLM error.

        Args:
            error_type: Type of error.
        """
        LLM_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_websocket_connected() -> None:
        """Record a WebSocket connection."""
        WS_CONNECTIONS.inc()

    @staticmethod
    def record_websocket_disconnected() -> None:
        """Record a WebSocket disconnection."""
        WS_CONNECTIONS.dec()

    @staticmethod
    def record_audio_bytes(direction: str, bytes_count: int) -> None:
        """Record audio bytes processed.

        Args:
            direction: "inbound" or "outbound".
            bytes_count: Number of bytes.
        """
        AUDIO_BYTES.labels(direction=direction).inc(bytes_count)


@router.get("")
async def metrics() -> Response:
    """Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/json")
async def metrics_json() -> dict:
    """Metrics in JSON format for debugging.

    Returns:
        Metrics as JSON.
    """
    return {
        "active_calls": ACTIVE_CALLS._value.get(),  # type: ignore
        "websocket_connections": WS_CONNECTIONS._value.get(),  # type: ignore
    }
