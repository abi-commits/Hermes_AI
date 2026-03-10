"""Gemini LLM service implementation."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator
from typing import Callable

import structlog
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
    FillerMarker,
    InterruptMarker,
    LLMConfig,
    LLMGenerationError,
)
from hermes.prompts.prompt_manager import PromptManager
from hermes.services.llm.base import AbstractLLMService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation while avoiding false
# positives on common abbreviations (Mr., Dr., U.S., e.g., 3.14 …).
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

_SOFT_FRAGMENT_MIN_CHARS = 80
_HARD_FRAGMENT_MIN_CHARS = 140
_EARLY_BREAK_MARKERS = (",", ";", ":", "\n")


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
        prompt_manager: PromptManager | None = None,
        prompt_name: str = "default",
        filler_phrases: list[str] | None = None,
    ) -> None:
        """Initialise the Gemini LLM service.

        Args:
            api_key: Gemini API key (uses GOOGLE_API_KEY env var if not provided).
            config: Generation parameters for the LLM.
            system_instruction: Explicit system instruction (overrides prompt_manager).
            prompt_manager: PromptManager instance to load system prompts from.
            prompt_name: Name of the system prompt to load (default: "default").
            filler_phrases: Phrases used to bridge silence during long RAG retrievals.
        """
        self.config = config or LLMConfig()
        self.prompt_manager = prompt_manager
        self.filler_phrases = filler_phrases or []

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

    def _make_generate_config(self, tools: list[Callable] | None = None) -> types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from current settings."""
        cfg = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=self.system_instruction,
        )
        if tools:
            cfg.tools = self._convert_tools_to_declarations(tools)
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

    def _convert_tools_to_declarations(self, tools: list[Callable]) -> list[types.Tool]:
        """Convert ``create_function_tool``-decorated callables to Gemini tool declarations."""
        declarations = []
        for tool in tools:
            if hasattr(tool, "function_declaration"):
                declarations.append(tool.function_declaration)
            else:
                logger.warning("Tool %s missing function_declaration attribute", tool.__name__)

        return [types.Tool(function_declarations=declarations)] if declarations else []

    @staticmethod
    def _pop_ready_fragment(buffer: str) -> tuple[str | None, str]:
        """Return the next TTS-ready fragment from *buffer*, if any."""
        sentence_match = _SENTENCE_END_RE.search(buffer)
        if sentence_match:
            end = sentence_match.end()
            return buffer[:end].strip(), buffer[end:].lstrip()

        if len(buffer) >= _SOFT_FRAGMENT_MIN_CHARS:
            split_at = max(buffer.rfind(marker) for marker in _EARLY_BREAK_MARKERS)
            if split_at >= 40:
                end = split_at + 1
                return buffer[:end].strip(), buffer[end:].lstrip()

        if len(buffer) >= _HARD_FRAGMENT_MIN_CHARS:
            split_at = buffer.rfind(" ")
            if split_at > 0:
                return buffer[:split_at].strip(), buffer[split_at + 1 :].lstrip()

        return None, buffer

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
        """Generate a complete response (non-streaming)."""
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

    def _convert_history(self, history: list[ConversationTurn] | None) -> list[types.Content]:
        """Convert Hermes history to Gemini content objects."""
        if not history:
            return []
        return [
            types.Content(
                role="user" if turn.role == "user" else "model",
                parts=[types.Part(text=turn.content)],
            )
            for turn in history
        ]

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
        tools: list[Callable] | None = None,
    ) -> AsyncIterator[str | InterruptMarker | FillerMarker]:
        """Stream the response sentence-by-sentence with barge-in support."""
        cfg = self._make_generate_config(tools=tools)
        
        # Local tool map to avoid cross-call race conditions
        local_tool_map = {}
        if tools:
            for tool in tools:
                if hasattr(tool, "function_declaration"):
                    local_tool_map[tool.function_declaration.name] = tool

        try:
            buffer = ""
            history = self._convert_history(conversation_history)
            
            if not tools:
                # No tools: standard prompt flow
                full_prompt = self._build_prompt(prompt, context, conversation_history)
                chat = self.client.aio.chats.create(model=self.config.model_name, config=cfg)
                message = full_prompt
            else:
                # Tool mode: separate history and context
                if context:
                    history.append(types.Content(role="user", parts=[types.Part(text=f"Context information:\n{context}")]))
                    history.append(types.Content(role="model", parts=[types.Part(text="Understood.")]))
                
                chat = self.client.aio.chats.create(model=self.config.model_name, config=cfg, history=history)
                message = prompt

            response_iter = await chat.send_message_stream(message)
            
            while True:
                found_tool_calls = False
                filler_sent = False # Reset per hop
                
                async for chunk in response_iter:
                    if interruption_check and interruption_check():
                        yield InterruptMarker()
                        return

                    # Detect tool calls
                    if chunk.candidates and chunk.candidates[0].content.parts:
                        f_calls = [p.function_call for p in chunk.candidates[0].content.parts if p.function_call]
                        if f_calls:
                            found_tool_calls = True
                            if not filler_sent and self.filler_phrases:
                                filler = random.choice(self.filler_phrases)
                                yield FillerMarker(filler)
                                filler_sent = True

                            # Execute tools in parallel
                            tasks = []
                            for fc in f_calls:
                                tool_func = local_tool_map.get(fc.name)
                                if tool_func:
                                    tasks.append(tool_func(**fc.args))
                            
                            if tasks:
                                results = await asyncio.gather(*tasks)
                                tool_responses = []
                                for i, fc in enumerate(f_calls):
                                    tool_responses.append(
                                        types.Part(
                                            function_response=types.FunctionResponse(
                                                name=fc.name,
                                                response={"result": results[i]},
                                            )
                                        )
                                    )
                                # FIX: Wrap tool responses in Content object
                                response_iter = await chat.send_message_stream(
                                    types.Content(role="user", parts=tool_responses)
                                )
                            break

                    if chunk.text:
                        buffer += chunk.text
                        while True:
                            fragment, buffer = self._pop_ready_fragment(buffer)
                            if not fragment:
                                break
                            yield fragment

                if not found_tool_calls:
                    break

            if buffer.strip():
                yield buffer.strip()

        except ClientError as exc:
            logger.error("Gemini streaming error (call=%s): %s", call_sid, exc)
            raise LLMGenerationError(f"Gemini API error: {exc}") from exc
