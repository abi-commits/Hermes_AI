"""Gemini LLM service implementation."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Callable

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hermes.models.llm import (
    ConversationTurn,
    InterruptMarker,
    LLMConfig,
    LLMGenerationError,
)
from hermes.prompts.prompt_manager import PromptManager
from hermes.services.llm.base import AbstractLLMService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation while avoiding false
# positives on common abbreviations (Mr., Dr., U.S., e.g., 3.14 …).
# Python requires fixed-width lookbehinds, so each abbreviation is listed
# separately instead of using alternation.
_SENTENCE_END_RE = re.compile(
    r"""
    (?<!\bMr)    # Mr.
    (?<!\bMs)    # Ms.
    (?<!\bDr)    # Dr.
    (?<!\bSr)    # Sr.
    (?<!\bJr)    # Jr.
    (?<!\bSt)    # St.
    (?<!\bvs)    # vs.
    (?<!\b\d)    # not after a lone digit (3.14)
    [.?!]        # sentence-ending punctuation
    (?=\s|$)     # followed by whitespace or end-of-string
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GeminiLLMService(AbstractLLMService):
    """LLM service backed by Google Gemini with retry and sentence-streaming support."""

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        system_instruction: str | None = None,
        tools: list[Callable] | None = None,
        prompt_manager: PromptManager | None = None,
        prompt_name: str = "default",
    ) -> None:
        """Initialise the Gemini LLM service.

        Args:
            api_key: Gemini API key (uses GOOGLE_API_KEY env var if not provided).
            config: Generation parameters for the LLM.
            system_instruction: Explicit system instruction (overrides prompt_manager).
            tools: List of Gemini function-calling tools.
            prompt_manager: PromptManager instance to load system prompts from.
            prompt_name: Name of the system prompt to load (default: "default").
        """
        self.config = config or LLMConfig()
        self.prompt_manager = prompt_manager
        self.tools = tools

        # Load system instruction from PromptManager if available and not explicitly set
        if system_instruction:
            self.system_instruction = system_instruction
        elif prompt_manager:
            system_prompt = prompt_manager.get_system_prompt(prompt_name)
            self.system_instruction = system_prompt.system_prompt
            logger.info(
                "GeminiLLMService loaded system prompt: %s (temperature: %.1f)",
                prompt_name,
                system_prompt.temperature,
            )
        else:
            self.system_instruction = None

        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

        logger.info("GeminiLLMService initialised with model %s", self.config.model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_generate_config(self) -> types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from current settings."""
        cfg = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=self.system_instruction,
        )
        if self.tools:
            cfg.tools = self._convert_tools(self.tools)
        return cfg

    def _build_prompt(
        self,
        query: str,
        context: str | None,
        history: list[ConversationTurn] | None,
    ) -> str:
        """Assemble the full prompt string from query, context, and history."""
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

    def _convert_tools(self, tools: list[Callable]) -> list[types.Tool]:
        """Convert ``create_function_tool``-decorated callables to Gemini tool objects."""
        declarations = []
        for tool in tools:
            if hasattr(tool, "function_declaration"):
                declarations.append(tool.function_declaration)
            else:
                logger.warning("Tool %s missing function_declaration attribute", tool.__name__)

        return [types.Tool(function_declarations=declarations)] if declarations else []

    # ------------------------------------------------------------------
    # Unary generation
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(LLMGenerationError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
    ) -> str:
        """Generate a complete response (non-streaming); retried up to 3× on error."""
        full_prompt = self._build_prompt(prompt, context, conversation_history)
        cfg = self._make_generate_config()

        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.config.model_name,
                    contents=full_prompt,
                    config=cfg,
                ),
                timeout=self.config.timeout_s,
            )
            if not response.text:
                raise LLMGenerationError("Empty response from Gemini")
            return response.text

        except asyncio.TimeoutError as exc:
            logger.error("Gemini generation timed out (call=%s)", call_sid)
            raise LLMGenerationError("Gemini API timed out") from exc
        except ClientError as exc:
            logger.error("Gemini generation error (call=%s): %s", call_sid, exc)
            raise LLMGenerationError(f"Gemini API error: {exc}") from exc

    # ------------------------------------------------------------------
    # Streaming generation (sentence-level)
    # ------------------------------------------------------------------

    async def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
        interruption_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream the response sentence-by-sentence with barge-in support."""
        full_prompt = self._build_prompt(prompt, context, conversation_history)
        cfg = self._make_generate_config()

        try:
            buffer = ""

            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.config.model_name,
                contents=full_prompt,
                config=cfg,
            ):
                if interruption_check and interruption_check():
                    logger.info("Barge-in detected (call=%s)", call_sid)
                    yield InterruptMarker()
                    return

                if chunk.text:
                    buffer += chunk.text

                    while True:
                        match = _SENTENCE_END_RE.search(buffer)
                        if not match:
                            break
                        sentence = buffer[: match.end()].strip()
                        buffer = buffer[match.end() :]
                        if sentence:
                            yield sentence

            # Yield any leftover text that didn't end with punctuation
            if buffer.strip():
                yield buffer.strip()

        except ClientError as exc:
            logger.error("Gemini streaming error (call=%s): %s", call_sid, exc)
            raise LLMGenerationError(f"Gemini API error: {exc}") from exc
