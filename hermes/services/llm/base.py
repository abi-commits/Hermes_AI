"""Abstract base class for LLM services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from hermes.models.llm import ConversationTurn, InterruptMarker


class AbstractLLMService(ABC):
    """Interface contract for all LLM service implementations."""

    # ------------------------------------------------------------------
    # Unary generation
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: "list[ConversationTurn] | None" = None,
        call_sid: str | None = None,
    ) -> str:
        """Return the complete response for *prompt* (non-streaming)."""

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    @abstractmethod
    def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: "list[ConversationTurn] | None" = None,
        call_sid: str | None = None,
        interruption_check: "Callable[[], bool] | None" = None,
    ) -> "AsyncIterator[str | InterruptMarker]":
        """Stream the response sentence-by-sentence with optional barge-in support."""
