"""Call state machine and session management."""

import asyncio
import base64
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from hermes.api.metrics import MetricsCollector
from hermes.core.audio import decode_mulaw
from hermes.models.call import CallState, ConversationTurn
from hermes.services.tts.audio import convert_to_ulaw, resample_to_8khz

if TYPE_CHECKING:
    from fastapi import WebSocket

    from hermes.services.llm.base import AbstractLLMService
    from hermes.services.rag.base import AbstractRAGService
    from hermes.services.stt.base import AbstractSTTService
    from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper (avoids lambda in run_in_executor)
# ---------------------------------------------------------------------------

def _do_twilio_redirect(
    account_sid: str,
    auth_token: str,
    call_sid: str,
    twiml: str,
) -> None:
    """Update the Twilio call with new TwiML (blocking — run in executor)."""
    from twilio.rest import Client  # type: ignore[import-untyped]

    client = Client(account_sid, auth_token)
    client.calls(call_sid).update(twiml=twiml)


class Call:
    """State machine and session manager for a single voice call."""

    def __init__(
        self,
        call_sid: str,
        stream_sid: str,
        websocket: "WebSocket",
        account_sid: str,
        *,
        # --- Injected services (optional — created lazily if None) ---
        stt_service: "AbstractSTTService | None" = None,
        llm_service: "AbstractLLMService | None" = None,
        tts_service: "AbstractTTSService | None" = None,
        rag_service: "AbstractRAGService | None" = None,
        # --- Per-call configuration ---
        persona: str = "default",
        rag_metadata_filter: dict[str, Any] | None = None,
        max_history: int = 20,
        fallback_phrase: str = "I'm sorry, I had a problem. Could you repeat that?",
    ) -> None:
        """Initialise a call."""
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
        self.text_out_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)
        self.audio_out_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=50)

        # Conversation history
        self.conversation: list[ConversationTurn] = []
        self.max_history = max_history

        # Services (injected or created lazily in start())
        self.stt_service: "AbstractSTTService | None" = stt_service
        self.llm_service: "AbstractLLMService | None" = llm_service
        self.tts_service: "AbstractTTSService | None" = tts_service
        self.rag_service: "AbstractRAGService | None" = rag_service

        # Per-call configuration
        self._persona = persona
        self.rag_metadata_filter: dict[str, Any] | None = rag_metadata_filter
        self._fallback_phrase = fallback_phrase

        # Barge-in / interrupt signal
        #   Set by interrupt() → checked by _tts_task between chunks
        self._interrupt_event = asyncio.Event()

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
        end = self.ended_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    async def start(self) -> None:
        """Start the call, initialise services lazily if not injected, and begin background tasks."""
        async with self._state_lock:
            if self._state != CallState.IDLE:
                raise RuntimeError(f"Cannot start call from state {self._state}")

            self._state = CallState.CONNECTING
            self.started_at = datetime.now(UTC)
            self._running = True

            # ── Lazy service creation (skipped when orchestrator injected them) ──
            if self.stt_service is None:
                from hermes.services.stt import STTService
                self.stt_service = STTService()

            if self.llm_service is None:
                from config import get_settings
                from hermes.models.llm import LLMConfig
                from hermes.services.llm import GeminiLLMService
                settings = get_settings()
                self.llm_service = GeminiLLMService(
                    api_key=settings.gemini_api_key,
                    config=LLMConfig(
                        model_name=settings.gemini_model,
                        temperature=settings.gemini_temperature,
                        max_output_tokens=settings.gemini_max_tokens,
                    ),
                )

            if self.tts_service is None:
                from hermes.services.tts import ChatterboxTTSService
                self.tts_service = ChatterboxTTSService()

            if self.rag_service is None:
                from hermes.services.rag import ChromaRAGService
                self.rag_service = ChromaRAGService()
                try:
                    await self.rag_service.warm_up()
                except Exception as e:
                    self._logger.warning("rag_warm_up_failed", error=str(e))

            MetricsCollector.record_call_started()
            self._logger.info("call_started", persona=self._persona)

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
        """Transition to *new_state* and log the change."""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            self._logger.info(
                "state_transition",
                from_state=old_state.name,
                to_state=new_state.name,
            )

    async def process_audio_chunk(self, payload: str) -> None:
        """Decode a base64 mu-law audio chunk from Twilio and enqueue it."""
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
        """Background task: stream audio to STT and forward transcripts to the LLM queue."""
        if not self.stt_service:
            return

        self._logger.info("stt_task_started")

        # Queue of decoded PCM tensors fed into the Deepgram live connection.
        tensor_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        async def _feed_audio() -> None:
            """Decode µ-law frames and push PCM tensors to the tensor queue."""
            while self._running:
                try:
                    audio_bytes = await asyncio.wait_for(
                        self.audio_in_queue.get(), timeout=0.5
                    )
                    pcm_tensor = decode_mulaw(audio_bytes)
                    MetricsCollector.record_audio_bytes("inbound", len(audio_bytes))
                    await tensor_queue.put(pcm_tensor)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise

        feeder = asyncio.create_task(
            _feed_audio(), name=f"stt-feeder-{self.call_sid}"
        )
        try:
            stt_start = time.perf_counter()
            async for transcript in self.stt_service.stream_transcribe(tensor_queue):
                if transcript.strip():
                    MetricsCollector.record_stt_latency(
                        time.perf_counter() - stt_start
                    )
                    await self.text_out_queue.put(transcript)
                    self._logger.debug("stt_transcript", text=transcript)
                    stt_start = time.perf_counter()

        except asyncio.CancelledError:
            self._logger.info("stt_task_cancelled")
        except Exception as e:
            self._logger.error("stt_task_error", error=str(e))
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def _llm_task(self) -> None:
        """Background task: consume transcripts, retrieve RAG context, and stream LLM sentences to TTS."""
        if not self.llm_service:
            return

        self._logger.info("llm_task_started")

        try:
            while self._running:
                # Wait for user input
                user_text = await self.text_out_queue.get()

                # Transition to processing state
                await self._transition_to(CallState.PROCESSING)

                # Add to conversation history
                self.conversation.append(
                    ConversationTurn(role="user", content=user_text)
                )

                # Build context (RAG + history) with a time budget
                context = await self._build_context()

                # Stream sentences and forward each one to TTS immediately
                llm_start = time.perf_counter()
                response_parts: list[str] = []

                async for chunk in self.llm_service.stream_sentences(
                    prompt=user_text, context=context
                ):
                    sentence = str(chunk)
                    response_parts.append(sentence)
                    # Feed each sentence to TTS as soon as it's ready
                    await self.audio_out_queue.put(sentence)

                MetricsCollector.record_llm_latency(time.perf_counter() - llm_start)

                # Store the full response in conversation history
                full_response = " ".join(response_parts)
                self.conversation.append(
                    ConversationTurn(role="assistant", content=full_response)
                )

                # Trim history if needed
                if len(self.conversation) > self.max_history:
                    self.conversation = self.conversation[-self.max_history :]

                self._logger.debug(
                    "llm_response_generated",
                    response=full_response[:100],
                    sentences=len(response_parts),
                )

                # Transition back to listening
                await self._transition_to(CallState.LISTENING)

        except asyncio.CancelledError:
            self._logger.info("llm_task_cancelled")
        except Exception as e:
            self._logger.error("llm_task_error", error=str(e))

    async def _build_context(self) -> str:
        """Assemble LLM context from RAG results and conversation history."""
        context_parts: list[str] = []

        # Add RAG context if available (time-budgeted)
        if self.rag_service and self.conversation:
            last_query = self.conversation[-1].content
            try:
                rag_results = await self.rag_service.retrieve_with_timeout(
                    last_query,
                    where=self.rag_metadata_filter,
                )
                if rag_results:
                    context_parts.append("Relevant information:")
                    for result in rag_results:
                        context_parts.append(f"- {result}")
            except Exception as e:
                self._logger.warning("rag_context_failed", error=str(e))

        # Add conversation history
        if len(self.conversation) > 1:
            context_parts.append("\nConversation history:")
            for turn in self.conversation[:-1]:  # Exclude last user message
                context_parts.append(f"{turn.role}: {turn.content}")

        return "\n".join(context_parts)

    async def _tts_task(self) -> None:
        """Background task: synthesise LLM sentences and stream µ-law audio to Twilio."""
        if not self.tts_service:
            return

        self._logger.info("tts_task_started")

        try:
            while self._running:
                # Wait for a sentence to synthesize
                text = await self.audio_out_queue.get()

                # Transition to speaking state (idempotent if already SPEAKING)
                if self._state != CallState.SPEAKING:
                    await self._transition_to(CallState.SPEAKING)

                tts_start = time.perf_counter()
                chunk_count = 0

                # Clear any pending interrupt from a previous sentence before
                # starting synthesis for this one.
                self._interrupt_event.clear()

                # Stream synthesis: each chunk is 16-bit PCM at model.sr (24 kHz)
                async for chunk_bytes in self.tts_service.generate_stream(text):
                    # Barge-in check: exit synthesis loop immediately
                    if self._interrupt_event.is_set():
                        self._logger.info("tts_stream_interrupted", text=text[:60])
                        break
                    # 1. Resample 24 kHz → 8 kHz (Twilio expects 8 kHz PCMU)
                    resampled = resample_to_8khz(
                        chunk_bytes, self.tts_service.sample_rate
                    )
                    # 2. Convert 16-bit PCM → 8-bit µ-law
                    mulaw_audio = convert_to_ulaw(resampled)
                    # 3. Send frame immediately
                    await self._send_audio(mulaw_audio)
                    MetricsCollector.record_audio_bytes("outbound", len(mulaw_audio))
                    chunk_count += 1

                MetricsCollector.record_tts_latency(time.perf_counter() - tts_start)
                self._logger.debug(
                    "tts_audio_sent", text=text[:100], chunks=chunk_count
                )

                # Only transition back to listening when queue is empty
                # (no more sentences pending from the LLM)
                if self.audio_out_queue.empty():
                    await self._transition_to(CallState.LISTENING)

        except asyncio.CancelledError:
            self._logger.info("tts_task_cancelled")
        except Exception as e:
            MetricsCollector.record_tts_error(type(e).__name__)
            self._logger.error("tts_task_error", error=str(e))

    async def _send_audio(self, audio: bytes) -> None:
        """Base64-encode *audio* and send it to Twilio as a media event."""
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

    async def _send_twilio_clear(self) -> None:
        """Send a Twilio ``clear`` event to flush buffered audio mid-utterance."""
        message = {"event": "clear", "streamSid": self.stream_sid}
        try:
            await self.websocket.send_text(json.dumps(message))
            self._logger.debug("twilio_clear_sent")
        except Exception as e:
            self._logger.warning("twilio_clear_failed", error=str(e))

    async def interrupt(self) -> None:
        """Stop current TTS, drain pending sentences, clear Twilio buffer, return to LISTENING."""
        self._interrupt_event.set()

        # Drain pending sentences (future TTS jobs)
        drained = 0
        while not self.audio_out_queue.empty():
            try:
                self.audio_out_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        await self._send_twilio_clear()
        await self._transition_to(CallState.LISTENING)
        self._logger.info("barge_in_interrupt", drained_sentences=drained)

    async def handle_dtmf(self, digit: str) -> None:
        """Route a DTMF digit to the appropriate handler."""
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
        """Transfer the call to a human agent via the Twilio REST API.

        Announces the transfer via TTS, then updates the in-progress Twilio
        call with ``<Dial>`` TwiML so the caller is connected directly to the
        configured ``TWILIO_TRANSFER_NUMBER``.  The local AI pipeline is
        stopped after the redirect is issued — Twilio takes over from there.
        """
        from config import get_settings

        settings = get_settings()
        transfer_number = settings.twilio_transfer_number

        if not transfer_number:
            self._logger.warning(
                "transfer_skipped",
                reason="TWILIO_TRANSFER_NUMBER not configured",
            )
            await self.audio_out_queue.put(
                "I'm sorry, I'm unable to transfer your call right now. "
                "Please call back and ask to speak with an agent."
            )
            return

        if not settings.twilio_account_sid or not settings.twilio_auth_token:
            self._logger.error(
                "transfer_failed",
                reason="Twilio credentials not configured",
            )
            return

        self._logger.info(
            "transfer_requested",
            call_sid=self.call_sid,
            transfer_to=transfer_number,
        )

        # 1. Announce the transfer before handing off
        announcement = (
            "Please hold while I transfer you to a human agent."
        )
        await self.audio_out_queue.put(announcement)

        # Give TTS time to finish the announcement before redirecting
        await asyncio.sleep(3.0)

        # 2. Build TwiML that dials the transfer number
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Dial callerId=\"{settings.twilio_phone_number or ''}\">"
            f"{transfer_number}"
            "</Dial>"
            "</Response>"
        )

        # 3. Redirect the Twilio call via REST API (runs in thread pool to avoid blocking)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                _do_twilio_redirect,
                settings.twilio_account_sid,
                settings.twilio_auth_token,
                self.call_sid,
                twiml,
            )
            self._logger.info(
                "transfer_completed",
                call_sid=self.call_sid,
                transfer_to=transfer_number,
            )
        except Exception as exc:
            self._logger.error(
                "transfer_failed",
                call_sid=self.call_sid,
                error=str(exc),
            )
            await self.audio_out_queue.put(
                "I'm sorry, the transfer failed. Please try again or call back."
            )
            return

        # 4. Stop the local AI pipeline — Twilio now owns the call
        await self.stop()

    async def _repeat_last_message(self) -> None:
        """Repeat the last assistant message."""
        for turn in reversed(self.conversation):
            if turn.role == "assistant":
                await self.audio_out_queue.put(turn.content)
                break

    async def stop(self) -> None:
        """Cancel background tasks and transition to ENDED state."""
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
            status="completed",
            duration=self.duration_seconds,
        )

        self._logger.info(
            "call_ended",
            duration_seconds=self.duration_seconds,
            total_turns=len(self.conversation),
        )
