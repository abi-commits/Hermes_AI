"""Mock LLM service for testing and local development."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable

import structlog

from hermes.models.llm import ConversationTurn, InterruptMarker, LLMConfig
from hermes.prompts.prompt_manager import PromptManager
from hermes.services.llm.base import AbstractLLMService

logger = structlog.get_logger(__name__)


class MockGeminiLLMService(AbstractLLMService):
    """Deterministic LLM stub for testing; cycles through pre-set response strings."""

    def __init__(
        self,
        responses: list[str] | None = None,
        prompt_manager: PromptManager | None = None,
        prompt_name: str = "default",
        **_kwargs,  # absorb any extra kwargs callers may pass
    ) -> None:
        """Initialise the mock LLM service."""
        # Expose the same attributes as GeminiLLMService so duck-typed code works.
        self.config = LLMConfig()
        self.prompt_manager = prompt_manager
        self.tools: list[Callable] | None = None
        self.client = None  # type: ignore[assignment]

        # Load system instruction from PromptManager if available
        if prompt_manager:
            system_prompt = prompt_manager.get_system_prompt(prompt_name)
            self.system_instruction = system_prompt.system_prompt
        else:
            self.system_instruction = None

        self._responses = responses or ["Mock response from Hermes."]
        self._index = 0
        self._logger = structlog.get_logger(__name__)

    def _build_prompt(
        self,
        query: str,
        context: str | None,
        history: list[ConversationTurn] | None,
    ) -> str:
        """Match the prompt-building shape of ``GeminiLLMService`` for tests."""
        parts: list[str] = []

        if context:
            parts.append(f"Context information:\n{context}\n")

        if history:
            history_lines = []
            for turn in history:
                prefix = "[INTERRUPTED] " if turn.interrupted else ""
                history_lines.append(f"{turn.role.capitalize()}: {prefix}{turn.content}")
            parts.append("Conversation history:\n" + "\n".join(history_lines) + "\n")

        parts.append(f"User: {query}\nAssistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Unary generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
    ) -> str:
        """Return the next pre-set response, ignoring all arguments."""
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        self._logger.debug("mock_llm_generate", response=response[:50])
        return response

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def stream_sentences(  # type: ignore[override]
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
        interruption_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str | InterruptMarker]:
        """Yield the next pre-set response word-by-word to simulate streaming."""
        response = self._responses[self._index % len(self._responses)]
        self._index += 1

        for word in response.split():
            if interruption_check and interruption_check():
                self._logger.debug("mock_llm_interrupted")
                yield InterruptMarker()
                return
            yield word + " "
