"""Call state machine and session management."""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from hermes.api.metrics import MetricsCollector
from hermes.core.adapters import ServiceAdapters
from hermes.core.audio import decode_mulaw
from hermes.models.call import CallState, ConversationTurn
from hermes.models.llm import FillerMarker, InterruptMarker
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
        task_error_handler: "Callable[[str, Exception], Awaitable[None]] | None" = None,
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
        self._stop_lock = asyncio.Lock()

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
        self._task_error_handler = task_error_handler

        # Barge-in / interrupt signal
        #   Set by interrupt() → checked by _tts_task between chunks
        self._interrupt_event = asyncio.Event()

        # Background tasks
        self._tasks: set[asyncio.Task] = set()
        self._running = False
        self._background_failure_reported = False

        # Service adapters — built in start() once services are confirmed present
        self._adapters: ServiceAdapters | None = None

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

    async def start(self, greeting: str | None = None) -> None:
        """Start the call, initialise services lazily if not injected, and begin background tasks."""
        from config import get_settings

        async with self._state_lock:
            if self._state != CallState.IDLE:
                raise RuntimeError(f"Cannot start call from state {self._state}")

            self._state = CallState.CONNECTING
            self.started_at = datetime.now(UTC)
            self._running = True

            # ── Services must be injected externally (via CallOrchestrator/ServiceBundle) ──
            # Warn if any service is missing — tasks will be skipped gracefully.
            missing = [
                name for name, svc in (
                    ("stt", self.stt_service),
                    ("llm", self.llm_service),
                    ("tts", self.tts_service),
                    ("rag", self.rag_service),
                )
                if svc is None
            ]
            if missing:
                self._logger.warning(
                    "call_services_missing",
                    missing=missing,
                    hint="Inject services via CallOrchestrator and ServiceBundle",
                )

            # Build service adapters now that services + interrupt_event are ready
            settings = get_settings()
            self._adapters = ServiceAdapters.build(
                call_sid=self.call_sid,
                interrupt_event=self._interrupt_event,
                stt_service=self.stt_service,
                llm_service=self.llm_service,
                tts_service=self.tts_service,
                rag_service=self.rag_service,
                rag_timeout_s=settings.rag_query_timeout_s,
            )

            self._logger.info("adapters_built", persona=self._persona)

        # Record metrics
        MetricsCollector.record_call_started()
        MetricsCollector.record_websocket_connected()

        # Start background processing tasks BEFORE sending greeting
        # This ensures we are listening for barge-in even during the initial greeting.
        self._start_tasks()

        # Optional initial greeting
        if greeting:
            await self.audio_out_queue.put(greeting)
            self.conversation.append(ConversationTurn(role="assistant", content=greeting))
            self._logger.info("initial_greeting_sent", text=greeting)

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
            self._register_task(task)

    def _register_task(self, task: asyncio.Task) -> None:
        """Track a background task and supervise unexpected failures."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._handle_task_completion)

    def _handle_task_completion(self, task: asyncio.Task) -> None:
        """Report unexpected task failures back into the call supervision path."""
        if task.cancelled() or not self._running or self._background_failure_reported:
            return

        try:
            error = task.exception()
        except asyncio.CancelledError:
            return

        if error is None:
            return

        self._background_failure_reported = True
        asyncio.create_task(
            self._handle_background_task_failure(task.get_name(), error),
            name=f"call-failure-{self.call_sid}",
        )

    async def _handle_background_task_failure(
        self,
        task_name: str,
        error: Exception,
    ) -> None:
        """Terminate the call cleanly when a supervised background task crashes."""
        self._logger.error(
            "background_task_failed",
            task_name=task_name,
            error_type=type(error).__name__,
            error=str(error),
        )

        if self._task_error_handler is not None:
            try:
                await self._task_error_handler(self.call_sid, error)
                return
            except Exception as handler_exc:
                self._logger.error(
                    "background_task_failure_handler_failed",
                    task_name=task_name,
                    error=str(handler_exc),
                )

        await self.stop(status="failed")

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
        if not self.stt_service or not self._adapters:
            return

        self._logger.info("stt_task_started")

        # Queue of decoded PCM arrays fed into the Deepgram live connection.
        pcm_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=200)

        async def _feed_audio() -> None:
            """Decode µ-law frames and push PCM arrays to the queue."""
            while self._running:
                try:
                    audio_bytes = await asyncio.wait_for(
                        self.audio_in_queue.get(), timeout=0.5
                    )
                    pcm_array = decode_mulaw(audio_bytes)
                    MetricsCollector.record_audio_bytes("inbound", len(audio_bytes))
                    await pcm_queue.put(pcm_array)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    await pcm_queue.put(None)
                    raise

        feeder = asyncio.create_task(
            _feed_audio(), name=f"stt-feeder-{self.call_sid}"
        )
        try:
            async for item in self._adapters.stt.stream_transcribe(pcm_queue):
                if isinstance(item, InterruptMarker):
                    # Immediate barge-in on speech detection
                    await self.interrupt()
                    continue

                await self.text_out_queue.put(item)
                self._logger.debug("stt_transcript", text=item)

        except asyncio.CancelledError:
            self._logger.info("stt_task_cancelled")
        except Exception as e:
            self._logger.error("stt_task_error", error=str(e))
            raise
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def _llm_task(self) -> None:
        """Background task: consume transcripts, retrieve RAG context, and stream LLM sentences to TTS."""
        if not self.llm_service or not self._adapters:
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

                # Stream sentences and forward each one to TTS immediately
                # (Agentic RAG handles retrieval via tools inside stream_sentences)
                response_parts: list[str] = []
                turn_interrupted = False

                # Create per-call RAG tool bound to this call's adapter and metadata filter
                from hermes.services.llm.rag_tool import get_rag_tool
                rag_tool = get_rag_tool(
                    rag_adapter=self._adapters.rag,
                    metadata_filter=self.rag_metadata_filter,
                )

                async for item in self._adapters.llm.stream_sentences(
                    prompt=user_text,
                    context=None,
                    conversation_history=self.conversation[:-1],
                    tools=[rag_tool],
                ):
                    if isinstance(item, InterruptMarker):
                        self._logger.info("llm_turn_interrupted")
                        turn_interrupted = True
                        break

                    if isinstance(item, FillerMarker):
                        # Play filler speech but don't add to history
                        await self.audio_out_queue.put(str(item))
                        continue

                    sentence = str(item)
                    response_parts.append(sentence)
                    # Feed each sentence to TTS as soon as it's ready
                    await self.audio_out_queue.put(sentence)

                # Store the response in conversation history
                full_response = " ".join(response_parts)
                self.conversation.append(
                    ConversationTurn(
                        role="assistant",
                        content=full_response,
                        interrupted=turn_interrupted,
                    )
                )

                # Trim history if needed
                if len(self.conversation) > self.max_history:
                    self.conversation = self.conversation[-self.max_history :]

                self._logger.debug(
                    "llm_response_generated",
                    response=full_response[:100],
                    sentences=len(response_parts),
                    interrupted=turn_interrupted,
                )

                # Transition back to listening
                await self._transition_to(CallState.LISTENING)

        except asyncio.CancelledError:
            self._logger.info("llm_task_cancelled")
        except Exception as e:
            self._logger.error("llm_task_error", error=str(e))
            raise

    async def _build_context(self) -> str:
        """Assemble LLM context from RAG results and conversation history."""
        context_parts: list[str] = []

        # Add RAG context via adapter (handles timeout + errors internally)
        if self._adapters and self.conversation:
            last_query = self.conversation[-1].content
            rag_results = await self._adapters.rag.retrieve(
                last_query,
                where=self.rag_metadata_filter,
            )
            if rag_results:
                context_parts.append("Relevant information:")
                for result in rag_results:
                    context_parts.append(f"- {result}")

        # Add conversation history
        if len(self.conversation) > 1:
            context_parts.append("\nConversation history:")
            for turn in self.conversation[:-1]:  # Exclude last user message
                context_parts.append(f"{turn.role}: {turn.content}")

        return "\n".join(context_parts)

    async def _tts_task(self) -> None:
        """Background task: synthesise LLM sentences and stream µ-law audio to Twilio."""
        if not self.tts_service or not self._adapters:
            return

        self._logger.info("tts_task_started")

        try:
            while self._running:
                # Wait for a sentence to synthesize
                text = await self.audio_out_queue.get()

                # Transition to speaking state (idempotent if already SPEAKING)
                if self._state != CallState.SPEAKING:
                    await self._transition_to(CallState.SPEAKING)

                # Clear any pending interrupt before starting this sentence
                self._interrupt_event.clear()

                chunk_count = 0

                # Stream synthesis via adapter (handles timing, interrupt, errors)
                async for chunk_bytes in self._adapters.tts.generate_stream(text):
                    # 1. Resample 24 kHz → 8 kHz (Twilio expects 8 kHz PCMU)
                    resampled = resample_to_8khz(
                        chunk_bytes, self._adapters.tts.sample_rate
                    )
                    # 2. Convert 16-bit PCM → 8-bit µ-law
                    mulaw_audio = convert_to_ulaw(resampled)
                    # 3. Send frame immediately
                    await self._send_audio(mulaw_audio)
                    MetricsCollector.record_audio_bytes("outbound", len(mulaw_audio))
                    chunk_count += 1

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
            self._logger.error("tts_task_error", error=str(e))
            raise

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
            raise

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
        await self.stop(status="transferred")

    async def _repeat_last_message(self) -> None:
        """Repeat the last assistant message."""
        for turn in reversed(self.conversation):
            if turn.role == "assistant":
                await self.audio_out_queue.put(turn.content)
                break

    async def stop(self, status: str = "completed") -> None:
        """Cancel background tasks and transition to ENDED state."""
        async with self._stop_lock:
            if self._state == CallState.ENDED:
                return
            if self._state == CallState.DISCONNECTING:
                return

            await self._transition_to(CallState.DISCONNECTING)
            self._running = False

            current_task = asyncio.current_task()
            tasks_to_cancel = [task for task in self._tasks if task is not current_task]

            for task in tasks_to_cancel:
                task.cancel()

            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            self.ended_at = datetime.now(UTC)
            await self._transition_to(CallState.ENDED)

            MetricsCollector.record_call_ended(
                status=status,
                duration=self.duration_seconds,
            )
            MetricsCollector.record_websocket_disconnected()

            self._logger.info(
                "call_ended",
                status=status,
                duration_seconds=self.duration_seconds,
                total_turns=len(self.conversation),
            )
