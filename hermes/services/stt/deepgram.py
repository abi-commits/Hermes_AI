"""Deepgram Speech-to-Text service implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    # Modern Deepgram SDK (v3+)
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
    )
    # The response types are often returned as objects with attributes
    HAS_DEEPGRAM = True
except ImportError:
    DeepgramClient = None
    DeepgramClientOptions = None
    LiveTranscriptionEvents = None
    LiveOptions = None
    HAS_DEEPGRAM = False

from config import get_settings
from hermes.core.exceptions import STTError, ServiceUnavailableError
from hermes.models.llm import InterruptMarker
from hermes.services.stt.base import AbstractSTTService

if TYPE_CHECKING:
    from asyncio import Queue

logger = structlog.get_logger(__name__)


class DeepgramSTTService(AbstractSTTService):
    """STT service backed by the Deepgram API."""

    def __init__(self) -> None:
        """Initialise the Deepgram STT service."""
        self.settings = get_settings()
        self._client: Any | None = None
        self._logger = structlog.get_logger(__name__)

        if not HAS_DEEPGRAM:
            self._logger.warning("deepgram_sdk_not_installed")
        if not self.settings.deepgram_api_key:
            self._logger.warning("deepgram_api_key_not_set")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the Deepgram client."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed or import failed")
        if not self.settings.deepgram_api_key:
            raise ServiceUnavailableError("Deepgram", "DEEPGRAM_API_KEY not configured")

        try:
            config = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            self._client = DeepgramClient(self.settings.deepgram_api_key, config)
            self._logger.info("deepgram_client_created", version="v3+")
        except Exception as e:
            self._logger.error("deepgram_init_failed", error=str(e))
            raise ServiceUnavailableError("Deepgram", f"Failed to initialize client: {e}")

    async def disconnect(self) -> None:
        """Discard the client reference."""
        self._client = None
        self._logger.info("deepgram_disconnected")

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_transcript(message: Any) -> str:
        """Return the top transcript alternative, or an empty string."""
        try:
            # Handle both object-style and dict-style responses
            channel = getattr(message, "channel", None)
            if channel is None and isinstance(message, dict):
                channel = message.get("channel")
            
            alternatives = getattr(channel, "alternatives", None)
            if alternatives is None and isinstance(channel, dict):
                alternatives = channel.get("alternatives")

            if not alternatives:
                return ""
            
            alt = alternatives[0]
            transcript = getattr(alt, "transcript", "")
            if transcript == "" and isinstance(alt, dict):
                transcript = alt.get("transcript", "")
                
            return (transcript or "").strip()
        except Exception:
            return ""

    def _consume_result(
        self,
        message: Any,
        finalized_segments: list[str],
    ) -> str | None:
        """Buffer final transcript segments and emit a full utterance when ready."""
        is_final = getattr(message, "is_final", False)
        if not is_final and isinstance(message, dict):
            is_final = message.get("is_final", False)
            
        if not is_final:
            return None

        transcript = self._extract_transcript(message)
        if not transcript:
            return None

        finalized_segments.append(transcript)

        speech_final = getattr(message, "speech_final", False)
        if not speech_final and isinstance(message, dict):
            speech_final = message.get("speech_final", False)

        if speech_final:
            return self._flush_segments(finalized_segments)

        return None

    @staticmethod
    def _flush_segments(finalized_segments: list[str]) -> str | None:
        """Join buffered final segments into a single utterance."""
        if not finalized_segments:
            return None

        utterance = " ".join(segment for segment in finalized_segments if segment).strip()
        finalized_segments.clear()
        return utterance or None

    def _live_transcription_options(self) -> dict[str, Any]:
        """Return the live transcription options."""
        return {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "smart_format": True,
            "interim_results": True,
            "utterance_end_ms": self.settings.deepgram_utterance_end_ms,
            "encoding": "linear16",
            "sample_rate": 8000,
        }

    async def _stream_live_events(
        self,
        audio_queue: "Queue[np.ndarray | None]",
        transcript_queue: "Queue[str | InterruptMarker | None]",
        connection: Any,
    ) -> AsyncIterator[str | InterruptMarker]:
        """Forward audio while yielding transcripts."""
        audio_task = asyncio.create_task(audio_queue.get())
        transcript_task = asyncio.create_task(transcript_queue.get())

        try:
            while True:
                done, _pending = await asyncio.wait(
                    {audio_task, transcript_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if transcript_task in done:
                    item = transcript_task.result()
                    if item:
                        yield item
                    transcript_task = asyncio.create_task(transcript_queue.get())

                if audio_task in done:
                    audio = audio_task.result()
                    if audio is None:
                        break

                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await connection.send(audio_bytes)
                    audio_task = asyncio.create_task(audio_queue.get())
        finally:
            for task in (audio_task, transcript_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(audio_task, transcript_task, return_exceptions=True)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe via pre-recorded API."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")
        if self._client is None:
            await self.connect()

        try:
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            options = {
                "model": self.settings.deepgram_model,
                "language": self.settings.deepgram_language,
                "smart_format": True,
            }
            
            response = await self._client.listen.asyncwebsocket.v("1").transcribe_file(
                {"buffer": audio_bytes},
                options,
            )
            
            return self._extract_transcript(response)
        except Exception as exc:
            self._logger.error("transcription_failed", error=str(exc))
            raise STTError(f"Transcription failed: {exc}") from exc

    async def stream_transcribe(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream audio to Deepgram."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")
        if self._client is None:
            await self.connect()

        transcript_queue: asyncio.Queue[str | InterruptMarker | None] = asyncio.Queue()
        finalized_segments: list[str] = []
        
        options = LiveOptions(**self._live_transcription_options())
        connection = self._client.listen.asyncwebsocket.v("1")

        async def _on_message(self, result, **kwargs) -> None:
            utterance = self._consume_result(result, finalized_segments)
            if utterance:
                await transcript_queue.put(utterance)

        async def _on_speech_started(self, speech_started, **kwargs) -> None:
            self._logger.debug("deepgram_speech_started")
            await transcript_queue.put(InterruptMarker())

        async def _on_error(self, error, **kwargs) -> None:
            self._logger.error("deepgram_live_error", error=str(error))

        connection.on(LiveTranscriptionEvents.Transcript, _on_message)
        connection.on(LiveTranscriptionEvents.SpeechStarted, _on_speech_started)
        connection.on(LiveTranscriptionEvents.Error, _on_error)

        if not await connection.start(options):
            raise STTError("Failed to start Deepgram connection")

        try:
            async for item in self._stream_live_events(audio_queue, transcript_queue, connection):
                yield item
        finally:
            await connection.finish()
            self._logger.info("deepgram_live_connection_closed")
