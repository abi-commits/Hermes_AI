"""Call state machine and session management."""

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from hermes.api.metrics import MetricsCollector
from hermes.core.audio import decode_mulaw, encode_mulaw
from hermes.utils.timing import Timer

if TYPE_CHECKING:
    from fastapi import WebSocket

    from hermes.services.llm import LLMService
    from hermes.services.rag import RAGService
    from hermes.services.stt import STTService
    from hermes.services.tts import TTSService

logger = structlog.get_logger(__name__)


class CallState(Enum):
    """Call state machine states."""

    IDLE = auto()
    CONNECTING = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    DISCONNECTING = auto()
    ENDED = auto()


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class Call:
    """Manages the state and lifecycle of a single voice call.

    This class coordinates:
    - Audio processing queues
    - STT, LLM, and TTS services
    - Conversation history
    - State transitions
    """

    def __init__(
        self,
        call_sid: str,
        stream_sid: str,
        websocket: "WebSocket",
        account_sid: str,
        *,
        stt_service: "STTService | None" = None,
        llm_service: "LLMService | None" = None,
        tts_service: "TTSService | None" = None,
        rag_service: "RAGService | None" = None,
    ) -> None:
        """Initialize a new call.

        Args:
            call_sid: Twilio call SID.
            stream_sid: Twilio media stream SID.
            websocket: WebSocket connection.
            account_sid: Twilio account SID.
            stt_service: Speech-to-text service (injected).
            llm_service: LLM service (injected).
            tts_service: TTS service (injected).
            rag_service: RAG service (injected).
        """
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.websocket = websocket
        self.account_sid = account_sid

        # State
        self._state = CallState.IDLE
        self._state_lock = asyncio.Lock()

        # Timing
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

        # Queues for async processing
        self.audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self.text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)
        self.tts_text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)

        # Conversation history
        self.conversation: list[ConversationTurn] = []
        self.max_history = 20

        # Services (injected)
        self.stt_service: "STTService | None" = stt_service
        self.llm_service: "LLMService | None" = llm_service
        self.tts_service: "TTSService | None" = tts_service
        self.rag_service: "RAGService | None" = rag_service

        # Background tasks
        self._tasks: set[asyncio.Task] = set()
        self._running = False

        self._logger = structlog.get_logger(__name__).bind(call_sid=call_sid)

    @property
    def state(self) -> CallState:
        """Current call state."""
        return self._state

    @property
    def duration_seconds(self) -> float:
        """Call duration in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    async def start(self) -> None:
        """Start the call and initialize services."""
        async with self._state_lock:
            if self._state != CallState.IDLE:
                raise RuntimeError(f"Cannot start call from state {self._state}")

            self._state = CallState.CONNECTING
            self.started_at = datetime.now(UTC)
            self._running = True

            self._logger.info("call_started")

        # Record metrics
        MetricsCollector.record_call_started()
        MetricsCollector.record_websocket_connected()

        # Start background processing tasks
        self._start_tasks()

        # Transition to listening state
        await self._transition_to(CallState.LISTENING)

    def _start_tasks(self) -> None:
        """Start background processing tasks."""
        tasks = [
            asyncio.create_task(self._stt_task(), name=f"stt-{self.call_sid}"),
            asyncio.create_task(self._llm_task(), name=f"llm-{self.call_sid}"),
            asyncio.create_task(self._tts_task(), name=f"tts-{self.call_sid}"),
        ]
        for task in tasks:
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _transition_to(self, new_state: CallState) -> None:
        """Transition to a new state.

        Args:
            new_state: The new state to transition to.
        """
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._logger.info(
                "state_transition",
                from_state=old_state.name,
                to_state=new_state.name,
            )

    async def process_audio_chunk(self, payload: str) -> None:
        """Process an incoming audio chunk from Twilio.

        Args:
            payload: Base64 encoded mu-law audio.
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(payload)

            # Put in queue for processing (drop if full)
            try:
                self.audio_in_queue.put_nowait(audio_bytes)
            except asyncio.QueueFull:
                self._logger.warning("audio_queue_full", dropped_bytes=len(audio_bytes))

        except Exception as e:
            self._logger.error("audio_processing_error", error=str(e))

    async def _stt_task(self) -> None:
        """Background task: Process audio with STT."""
        if not self.stt_service:
            return

        self._logger.info("stt_task_started")

        try:
            # Accumulate audio buffer for speech detection
            audio_buffer: list[bytes] = []
            silence_frames = 0
            max_silence = 50  # ~1 second of silence

            while self._running:
                try:
                    # Wait for audio with timeout
                    audio = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1,
                    )
                    audio_buffer.append(audio)
                    silence_frames = 0

                except asyncio.TimeoutError:
                    # Check if we have accumulated speech
                    if audio_buffer and silence_frames < max_silence:
                        silence_frames += 1
                        continue

                    if audio_buffer:
                        # Process accumulated audio
                        combined = b"".join(audio_buffer)
                        audio_buffer = []
                        silence_frames = 0

                        # Decode mu-law to PCM
                        pcm_audio = decode_mulaw(combined)

                        # Send to STT service with timing
                        with Timer("stt_transcribe") as t:
                            transcript = await self.stt_service.transcribe(pcm_audio)
                        MetricsCollector.record_stt_latency(t.elapsed or 0.0)

                        if transcript.strip():
                            await self.text_queue.put(transcript)
                            self._logger.debug("stt_transcript", text=transcript)

        except asyncio.CancelledError:
            self._logger.info("stt_task_cancelled")
        except Exception as e:
            MetricsCollector.record_stt_error(type(e).__name__)
            self._logger.error("stt_task_error", error=str(e))

    async def _llm_task(self) -> None:
        """Background task: Generate responses with LLM."""
        if not self.llm_service:
            return

        self._logger.info("llm_task_started")

        try:
            while self._running:
                # Wait for user input
                user_text = await self.text_queue.get()

                # Transition to processing state
                await self._transition_to(CallState.PROCESSING)

                # Add to conversation history
                self.conversation.append(
                    ConversationTurn(role="user", content=user_text)
                )

                # Build context from history and RAG (async)
                context = await self._build_context()

                # Generate response with timing
                response = ""
                with Timer("llm_generate") as t:
                    async for chunk in self.llm_service.generate_stream(context, user_text):
                        response += chunk
                MetricsCollector.record_llm_latency(t.elapsed or 0.0)

                # Add assistant response to history
                self.conversation.append(
                    ConversationTurn(role="assistant", content=response)
                )

                # Trim history if needed
                if len(self.conversation) > self.max_history:
                    self.conversation = self.conversation[-self.max_history :]

                # Queue text for TTS (typed str queue)
                await self.tts_text_queue.put(response)

                self._logger.debug("llm_response_generated", response=response[:100])

                # Transition back to listening
                await self._transition_to(CallState.LISTENING)

        except asyncio.CancelledError:
            self._logger.info("llm_task_cancelled")
        except Exception as e:
            MetricsCollector.record_llm_error(type(e).__name__)
            self._logger.error("llm_task_error", error=str(e))

    async def _build_context(self) -> str:
        """Build conversation context from history and RAG.

        Returns:
            Context string for LLM.
        """
        context_parts = []

        # Add RAG context if available (properly awaited)
        if self.rag_service and self.conversation:
            last_query = self.conversation[-1].content
            try:
                rag_results = await self.rag_service.retrieve(last_query)
                if rag_results:
                    context_parts.append(
                        self.rag_service.format_context(rag_results)
                    )
            except Exception as e:
                self._logger.warning("rag_retrieval_skipped", error=str(e))

        # Add conversation history
        if len(self.conversation) > 1:
            context_parts.append("\nConversation history:")
            for turn in self.conversation[:-1]:  # Exclude last user message
                context_parts.append(f"{turn.role}: {turn.content}")

        return "\n".join(context_parts)

    async def _tts_task(self) -> None:
        """Background task: Generate audio with TTS."""
        if not self.tts_service:
            return

        self._logger.info("tts_task_started")

        try:
            while self._running:
                # Wait for text to synthesize (str queue)
                text = await self.tts_text_queue.get()

                # Transition to speaking state
                await self._transition_to(CallState.SPEAKING)

                # Generate audio with timing
                with Timer("tts_synthesize") as t:
                    audio = await self.tts_service.synthesize(text)
                MetricsCollector.record_tts_latency(t.elapsed or 0.0)

                # Encode to mu-law and send to Twilio
                mulaw_audio = encode_mulaw(audio)
                await self._send_audio(mulaw_audio)
                MetricsCollector.record_audio_bytes("outbound", len(mulaw_audio))

                self._logger.debug("tts_audio_sent", text=text[:100])

                # Transition back to listening
                await self._transition_to(CallState.LISTENING)

        except asyncio.CancelledError:
            self._logger.info("tts_task_cancelled")
        except Exception as e:
            MetricsCollector.record_tts_error(type(e).__name__)
            self._logger.error("tts_task_error", error=str(e))

    async def _send_audio(self, audio: bytes) -> None:
        """Send audio back to Twilio.

        Args:
            audio: Mu-law encoded audio bytes.
        """
        import json

        payload = base64.b64encode(audio).decode()
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload},
        }

        try:
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            self._logger.error("audio_send_error", error=str(e))

    async def handle_dtmf(self, digit: str) -> None:
        """Handle DTMF (touch tone) input.

        Args:
            digit: The DTMF digit pressed.
        """
        self._logger.info("dtmf_received", digit=digit)

        # Handle special digits
        if digit == "0":
            # Transfer to human agent
            await self._handle_transfer()
        elif digit == "*":
            # Repeat last message
            await self._repeat_last_message()
        elif digit == "#":
            # End call
            await self.stop()

    async def _handle_transfer(self) -> None:
        """Transfer the call to a human agent."""
        self._logger.info("transfer_requested")
        # Implementation depends on your call center integration

    async def _repeat_last_message(self) -> None:
        """Repeat the last assistant message."""
        for turn in reversed(self.conversation):
            if turn.role == "assistant":
                await self.tts_text_queue.put(turn.content)
                break

    async def stop(self) -> None:
        """Stop the call and cleanup resources."""
        await self._transition_to(CallState.DISCONNECTING)
        self._running = False

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self.ended_at = datetime.now(UTC)
        await self._transition_to(CallState.ENDED)

        MetricsCollector.record_call_ended(
            status="completed", duration=self.duration_seconds
        )
        MetricsCollector.record_websocket_disconnected()

        self._logger.info(
            "call_ended",
            duration_seconds=self.duration_seconds,
            total_turns=len(self.conversation),
        )
