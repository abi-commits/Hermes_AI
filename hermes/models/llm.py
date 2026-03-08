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
    """Sentinel yielded by the LLM stream to signal a barge-in interrupt."""

    pass


class FillerMarker:
    """Sentinel yielded by the LLM stream to indicate a filler phrase."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    role: str  # "user" or "assistant"
    content: str
    interrupted: bool = False  # Whether this assistant turn was interrupted


@dataclass
class LLMConfig:
    """Generation parameters for the LLM service."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40

    # Reliability
    timeout_s: float = 60.0  # Per-request timeout in seconds
