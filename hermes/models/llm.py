"""LLM-related data models, enums, and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ======================================================================
# Exceptions
# ======================================================================


class LLMGenerationError(Exception):
    """Raised when LLM generation fails."""

    pass


# ======================================================================
# Markers
# ======================================================================


class InterruptMarker:
    """Special marker to signal an interruption event to the LLM stream."""

    pass


# ======================================================================
# Enums
# ======================================================================


class ResponseModality(Enum):
    """Output modalities supported by Gemini."""

    TEXT = "TEXT"
    AUDIO = "AUDIO"  # Requires Live API with native audio


class StreamingMode(Enum):
    """Different streaming approaches for different latency requirements."""

    STANDARD = "standard"  # Regular streaming via generate_content
    LIVE = "live"  # Live API for bidirectional audio (lowest latency)


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation history."""

    role: str  # "user" or "assistant"
    content: str
    interrupted: bool = False  # Whether this assistant turn was interrupted


@dataclass
class LLMConfig:
    """Configuration for the LLM service."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    response_modality: ResponseModality = ResponseModality.TEXT
    streaming_mode: StreamingMode = StreamingMode.STANDARD

    # Live API specific settings
    voice_name: str | None = "charon"  # Chirp 3 HD voices: charon, puck, etc.
    language_code: str = "en-US"

    # VAD settings for interruption handling
    vad_enabled: bool = True
    vad_start_sensitivity: str = "START_SENSITIVITY_LOW"  # or HIGH
    vad_end_sensitivity: str = "END_SENSITIVITY_LOW"  # or HIGH
    vad_prefix_padding_ms: int = 20
    vad_silence_duration_ms: int = 100
