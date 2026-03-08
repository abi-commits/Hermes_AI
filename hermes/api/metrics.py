"""Prometheus metrics endpoint."""

from typing import Any

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

router = APIRouter(tags=["System"])

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

# Barge-in interrupts
CALL_INTERRUPTS = Counter(
    "hermes_call_interrupts_total",
    "Total barge-in interrupts applied to active calls",
)


# Set application info on startup (updated in lifespan with real version)
APP_INFO.info({"version": "0.0.0", "name": "hermes"})


class MetricsCollector:
    """Static helpers for recording Prometheus metrics."""

    @staticmethod
    def record_call_started() -> None:
        """Increment the active-calls gauge."""
        ACTIVE_CALLS.inc()

    @staticmethod
    def record_call_ended(status: str, duration: float) -> None:
        """Decrement active-calls gauge and record duration/status."""
        ACTIVE_CALLS.dec()
        CALL_DURATION.observe(duration)
        CALLS_TOTAL.labels(status=status).inc()

    @staticmethod
    def record_stt_latency(seconds: float) -> None:
        """Record STT latency."""
        STT_LATENCY.observe(seconds)

    @staticmethod
    def record_llm_latency(seconds: float) -> None:
        """Record LLM latency."""
        LLM_LATENCY.observe(seconds)

    @staticmethod
    def record_tts_latency(seconds: float) -> None:
        """Record TTS latency."""
        TTS_LATENCY.observe(seconds)

    @staticmethod
    def record_llm_tokens(prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM prompt and completion token counts."""
        LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
        LLM_TOKENS.labels(type="completion").inc(completion_tokens)

    @staticmethod
    def record_stt_error(error_type: str) -> None:
        """Increment the STT error counter."""
        STT_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_tts_error(error_type: str) -> None:
        """Increment the TTS error counter."""
        TTS_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_llm_error(error_type: str) -> None:
        """Increment the LLM error counter."""
        LLM_ERRORS.labels(error_type=error_type).inc()

    @staticmethod
    def record_websocket_connected() -> None:
        """Increment the active WebSocket connections gauge."""
        WS_CONNECTIONS.inc()

    @staticmethod
    def record_websocket_disconnected() -> None:
        """Decrement the active WebSocket connections gauge."""
        WS_CONNECTIONS.dec()

    @staticmethod
    def record_audio_bytes(direction: str, bytes_count: int) -> None:
        """Record audio bytes processed (direction: ``inbound`` or ``outbound``)."""
        AUDIO_BYTES.labels(direction=direction).inc(bytes_count)

    @staticmethod
    def record_call_interrupted(call_sid: str) -> None:  # noqa: ARG004
        """Increment the barge-in interrupt counter."""
        CALL_INTERRUPTS.inc()


@router.get("")
async def metrics() -> Response:
    """Return Prometheus-formatted metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/json")
async def metrics_json() -> dict:
    """Return active call and WebSocket counts as JSON (for debugging)."""
    return {
        "active_calls": ACTIVE_CALLS._value.get(),  # type: ignore
        "websocket_connections": WS_CONNECTIONS._value.get(),  # type: ignore
    }
