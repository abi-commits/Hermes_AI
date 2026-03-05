"""
LLM Service Module for Hermes.

This module provides a wrapper around Google's Gemini 2.5 Flash model using the
google-genai SDK. It supports both standard generation and low-latency streaming
with integrated barge-in handling via the Live API.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from hermes.models.llm import (
    ConversationTurn,
    InterruptMarker,
    LLMConfig,
    LLMGenerationError,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Main service
# ======================================================================


class GeminiLLMService:
    """Service class for Google's Gemini 2.5 Flash LLM.

    Supports both standard generation and Live API streaming with native
    interruption handling via Voice Activity Detection (VAD).
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        system_instruction: str | None = None,
        tools: list[Callable] | None = None,
    ) -> None:
        """Initialise the LLM service.

        Args:
            api_key: Google API key. If ``None``, reads from ``GOOGLE_API_KEY`` env var.
            config: LLM configuration parameters.
            system_instruction: System prompt for the model.
            tools: List of callable tools for function calling.
        """
        self.config = config or LLMConfig()
        self.system_instruction = system_instruction
        self.tools = tools

        # Initialise the Google Gen AI client
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

        logger.info("LLM Service initialised with model %s", self.config.model_name)

        # Store active Live API sessions for cancellation
        self._active_sessions: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Standard (unary) generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
    ) -> str:
        """Generate a complete response (non-streaming).

        Useful for offline processing or when latency isn't critical.

        Args:
            prompt: The user's query.
            context: Retrieved context from RAG system.
            conversation_history: Previous turns in the conversation.
            call_sid: Optional call identifier for logging.

        Returns:
            Complete response text.

        Raises:
            LLMGenerationError: On upstream Gemini API errors.
        """
        full_prompt = self._build_prompt(prompt, context, conversation_history)

        generate_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=self.system_instruction,
        )

        if self.tools:
            generate_config.tools = self._convert_tools(self.tools)

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=full_prompt,
                config=generate_config,
            )
            if not response.text:
                raise LLMGenerationError("Empty response from model")
            return response.text

        except ClientError as e:
            logger.error("LLM generation failed for call %s: %s", call_sid, e)
            raise LLMGenerationError(f"Gemini API error: {e}") from e

    # ------------------------------------------------------------------
    # Streaming generation (sentence-level with barge-in)
    # ------------------------------------------------------------------

    async def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
        turn_id: int | None = None,
        interruption_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream the response sentence by sentence with interruption support.

        Yields complete sentences as soon as they're available, allowing the
        TTS system to start generating audio immediately.

        Args:
            prompt: The user's query.
            context: Retrieved context from RAG system.
            conversation_history: Previous turns in the conversation.
            call_sid: Optional call identifier for logging.
            turn_id: Current turn identifier for cancellation.
            interruption_check: Callable that returns ``True`` if interrupted.

        Yields:
            Sentences as strings, or ``InterruptMarker`` if interrupted.

        Raises:
            LLMGenerationError: On upstream Gemini API errors.
        """
        full_prompt = self._build_prompt(prompt, context, conversation_history)

        generate_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            system_instruction=self.system_instruction,
        )

        if self.tools:
            generate_config.tools = self._convert_tools(self.tools)

        try:
            current_sentence = ""
            sentence_endings = {".", "?", "!", "\n"}

            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.config.model_name,
                contents=full_prompt,
                config=generate_config,
            ):
                # Check for interruption
                if interruption_check and interruption_check():
                    logger.info("Call %s: Interruption detected in LLM stream", call_sid)
                    yield InterruptMarker()
                    return

                if chunk.text:
                    current_sentence += chunk.text

                    # Yield when we have a complete sentence
                    if any(
                        current_sentence.rstrip().endswith(e) for e in sentence_endings
                    ):
                        yield current_sentence.strip()
                        current_sentence = ""

            # Yield any remaining text
            if current_sentence.strip():
                yield current_sentence.strip()

        except ClientError as e:
            logger.error("LLM streaming failed for call %s: %s", call_sid, e)
            raise LLMGenerationError(f"Gemini API error: {e}") from e

    # ------------------------------------------------------------------
    # Live API session (bidirectional WebSocket with barge-in)
    # ------------------------------------------------------------------

    async def live_conversation(
        self,
        call_sid: str,
        audio_input_queue: asyncio.Queue,
        audio_output_queue: asyncio.Queue,
        text_output_queue: asyncio.Queue,
        context_provider: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        """Run a full bidirectional Live API session with native audio.

        This is the lowest-latency approach, ideal for voice agents.
        The model receives audio directly and outputs audio, with built-in
        VAD and interruption handling.

        Args:
            call_sid: Unique call identifier.
            audio_input_queue: Queue with incoming audio chunks (16 kHz PCM).
            audio_output_queue: Queue to put outgoing audio chunks (24 kHz PCM).
            text_output_queue: Queue to put transcribed text for logging/history.
            context_provider: Async function to retrieve context for queries.
        """
        live_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        # Configure voice
        if self.config.voice_name:
            live_config.speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice_name,
                    )
                ),
            )

        # Configure VAD for interruption handling
        if self.config.vad_enabled:
            live_config.realtime_input_config = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=getattr(
                        types.StartSensitivity, self.config.vad_start_sensitivity, None
                    ),
                    end_of_speech_sensitivity=getattr(
                        types.EndSensitivity, self.config.vad_end_sensitivity, None
                    ),
                    prefix_padding_ms=self.config.vad_prefix_padding_ms,
                    silence_duration_ms=self.config.vad_silence_duration_ms,
                )
            )

        # System instruction
        if self.system_instruction:
            live_config.system_instruction = types.Content(
                parts=[types.Part(text=self.system_instruction)]
            )

        # Tools
        if self.tools:
            live_config.tools = self._convert_tools(self.tools)

        model_name = "gemini-live-2.5-flash"

        try:
            async with self.client.aio.live.connect(
                model=model_name,
                config=live_config,
            ) as session:
                self._active_sessions[call_sid] = session

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        self._send_audio_task(session, audio_input_queue)
                    )
                    tg.create_task(
                        self._receive_live_task(
                            session,
                            audio_output_queue,
                            text_output_queue,
                            context_provider,
                            call_sid,
                        )
                    )

        except asyncio.CancelledError:
            logger.info("Live session for call %s cancelled", call_sid)
        except Exception as e:
            logger.error("Live session failed for call %s: %s", call_sid, e)
            raise
        finally:
            self._active_sessions.pop(call_sid, None)

    # ------------------------------------------------------------------
    # Live API internal tasks
    # ------------------------------------------------------------------

    async def _send_audio_task(
        self,
        session: Any,
        audio_input_queue: asyncio.Queue,
    ) -> None:
        """Send audio chunks from queue to the Live API session."""
        while True:
            chunk = await audio_input_queue.get()
            if chunk is None:  # Sentinel to end
                await session.send_realtime_input(audio_stream_end=True)
                break

            await session.send_realtime_input(
                audio={"data": chunk, "mime_type": "audio/pcm;rate=16000"}
            )

    async def _receive_live_task(
        self,
        session: Any,
        audio_output_queue: asyncio.Queue,
        text_output_queue: asyncio.Queue,
        context_provider: Callable | None,
        call_sid: str,
    ) -> None:
        """Receive and process responses from the Live API."""
        turn = session.receive()

        async for message in turn:
            # Handle interruption
            if message.server_content and message.server_content.interrupted:
                logger.info("Call %s: Model generation interrupted", call_sid)
                while not audio_output_queue.empty():
                    audio_output_queue.get_nowait()
                continue

            # Handle input transcription (what the user said)
            if message.server_content and message.server_content.input_transcription:
                transcript = message.server_content.input_transcription.transcript
                logger.info("Call %s User: %s", call_sid, transcript)
                await text_output_queue.put(("user", transcript))

            # Handle output transcription (what the model said)
            if message.server_content and message.server_content.output_transcription:
                transcript = message.server_content.output_transcription.transcript
                logger.info("Call %s Assistant: %s", call_sid, transcript)
                await text_output_queue.put(("assistant", transcript))

            # Handle model turn (audio data)
            if message.server_content and message.server_content.model_turn:
                for part in message.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        await audio_output_queue.put(part.inline_data.data)

    def cancel_session(self, call_sid: str) -> None:
        """Cancel an active Live API session (e.g. on barge-in)."""
        if call_sid in self._active_sessions:
            logger.info("Cancelling Live API session for call %s", call_sid)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        context: str | None,
        history: list[ConversationTurn] | None,
    ) -> str:
        """Build the full prompt with context and conversation history."""
        parts: list[str] = []

        if context:
            parts.append(f"Context information:\n{context}\n")

        if history:
            history_text = []
            for turn in history:
                prefix = "[INTERRUPTED] " if turn.interrupted else ""
                history_text.append(
                    f"{turn.role.capitalize()}: {prefix}{turn.content}"
                )
            parts.append("Conversation history:\n" + "\n".join(history_text) + "\n")

        parts.append(f"User: {query}\nAssistant:")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list[Callable]) -> list[types.Tool]:
        """Convert Python callables to Gemini Tool objects.

        In production you would use function introspection to generate proper
        schemas.  Here we assume each tool has a ``function_declaration`` attr.
        """
        function_declarations = []

        for tool in tools:
            if hasattr(tool, "function_declaration"):
                function_declarations.append(tool.function_declaration)
            else:
                logger.warning("Tool %s missing function declaration", tool.__name__)

        if function_declarations:
            return [types.Tool(function_declarations=function_declarations)]
        return []


# ======================================================================
# Tool decorator helper
# ======================================================================


def create_function_tool(
    name: str, description: str, parameters: types.Schema | None = None
) -> Callable:
    """Decorator to create a function tool with proper Gemini schema.

    Example::

        @create_function_tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        )
        def get_weather(location: str) -> str:
            return f"The weather in {location} is sunny."
    """

    def decorator(func: Callable) -> Callable:
        func.function_declaration = types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )
        return func

    return decorator


# ======================================================================
# Mock implementation (for tests)
# ======================================================================


class MockGeminiLLMService(GeminiLLMService):
    """Deterministic mock used in tests.

    Cycles through a fixed list of responses and avoids all network calls.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialise with predetermined responses.

        Args:
            responses: Responses to return, cycled in order.
        """
        # Intentionally skip super().__init__() to avoid requiring an API key.
        self.config = LLMConfig()
        self.system_instruction = None
        self.tools = None
        self.client = None  # type: ignore[assignment]
        self._active_sessions: dict[str, Any] = {}
        self.responses = responses or ["This is a mock response from the LLM."]
        self._index = 0

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
    ) -> str:
        """Return the next mock response."""
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        logger.debug("mock_llm_response: %s", response)
        return response

    async def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        call_sid: str | None = None,
        turn_id: int | None = None,
        interruption_check: Callable[[], bool] | None = None,
    ) -> AsyncIterator[str | InterruptMarker]:
        """Yield mock response word-by-word to simulate streaming."""
        response = await self.generate(prompt, context, conversation_history, call_sid)
        for word in response.split():
            yield word + " "
