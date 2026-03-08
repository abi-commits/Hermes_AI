"""Deepgram Speech-to-Text service implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from deepgram import AsyncDeepgramClient, LiveTranscriptionEvents  # type: ignore[import-untyped]
    from deepgram.listen.v1.types import (  # type: ignore[import-untyped]
        ListenV1Metadata,
        ListenV1Results,
        ListenV1SpeechStarted,
        ListenV1UtteranceEnd,
    )

    _DEEPGRAM_SDK_VARIANT = "v5"
    HAS_DEEPGRAM = True
except ImportError:
    try:
        from deepgram import DeepgramClient, LiveTranscriptionEvents  # type: ignore[import-untyped]
        from deepgram.clients.listen.v1.websocket.response import (  # type: ignore[import-untyped]
            LiveResultResponse as ListenV1Results,
            MetadataResponse as ListenV1Metadata,
            SpeechStartedResponse as ListenV1SpeechStarted,
            UtteranceEndResponse as ListenV1UtteranceEnd,
        )

        _DEEPGRAM_SDK_VARIANT = "v3"
        HAS_DEEPGRAM = True
    except ImportError:
        AsyncDeepgramClient = None  # type: ignore[assignment]
        DeepgramClient = None  # type: ignore[assignment]
        LiveTranscriptionEvents = None  # type: ignore[assignment]
        ListenV1Metadata = ListenV1Results = ListenV1SpeechStarted = ListenV1UtteranceEnd = Any
        _DEEPGRAM_SDK_VARIANT = None
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
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")
        if not self.settings.deepgram_api_key:
            raise ServiceUnavailableError("Deepgram", "DEEPGRAM_API_KEY not configured")

        if _DEEPGRAM_SDK_VARIANT == "v5":
            self._client = AsyncDeepgramClient(api_key=self.settings.deepgram_api_key)
        elif _DEEPGRAM_SDK_VARIANT == "v3":
            self._client = DeepgramClient(api_key=self.settings.deepgram_api_key)
        else:
            raise ServiceUnavailableError("Deepgram", "Unsupported Deepgram SDK variant")

        self._logger.info("deepgram_client_created", sdk_variant=_DEEPGRAM_SDK_VARIANT)

    async def disconnect(self) -> None:
        """Discard the client reference (connections are managed per request)."""
        self._client = None
        self._logger.info("deepgram_disconnected")

    # ------------------------------------------------------------------
    # Final-transcript helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_transcript(message: Any) -> str:
        """Return the top transcript alternative, or an empty string."""
        channel = getattr(message, "channel", None)
        alternatives = getattr(channel, "alternatives", None) if channel else None
        if not alternatives:
            return ""
        transcript = getattr(alternatives[0], "transcript", "") or ""
        return transcript.strip()

    def _consume_result(
        self,
        message: Any,
        finalized_segments: list[str],
    ) -> str | None:
        """Buffer final transcript segments and emit a full utterance when ready."""
        if not getattr(message, "is_final", False):
            return None

        transcript = self._extract_transcript(message)
        if not transcript:
            return None

        finalized_segments.append(transcript)

        if getattr(message, "speech_final", False):
            return self._flush_segments(finalized_segments)

        return None

    @staticmethod
    def _flush_segments(finalized_segments: list[str]) -> str | None:
        """Join buffered final segments into a single utterance and clear the buffer."""
        if not finalized_segments:
            return None

        utterance = " ".join(segment for segment in finalized_segments if segment).strip()
        finalized_segments.clear()
        return utterance or None

    def _live_transcription_options(self) -> dict[str, str | int]:
        """Return the live transcription options for the current settings."""
        return {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "smart_format": "true",
            "interim_results": "true",
            "utterance_end_ms": str(self.settings.deepgram_utterance_end_ms),
            "encoding": "linear16",
            "sample_rate": 8000,
        }

    async def _stream_live_events(
        self,
        audio_queue: "Queue[np.ndarray | None]",
        transcript_queue: "Queue[str | InterruptMarker | None]",
        send_audio: "Callable[[bytes], Awaitable[None]]",
    ) -> AsyncIterator[str | InterruptMarker]:
        """Forward audio upstream while yielding transcripts or markers as soon as they arrive."""
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

                    await send_audio((audio * 32767).astype(np.int16).tobytes())
                    audio_task = asyncio.create_task(audio_queue.get())
        finally:
            for task in (audio_task, transcript_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(audio_task, transcript_task, return_exceptions=True)

    async def _wait_for_finalize_flush(self) -> None:
        """Allow a short grace period for trailing final events after finalize."""
        grace_ms = self.settings.deepgram_finalize_grace_ms
        if grace_ms > 0:
            await asyncio.sleep(grace_ms / 1000)

    # ------------------------------------------------------------------
    # Single-shot transcription
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((STTError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a PCM array via the Deepgram pre-recorded API (retried up to 3x)."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")
        if not self.settings.deepgram_api_key:
            raise STTError("Deepgram API key not configured")

        if self._client is None:
            await self.connect()

        try:
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            options = {
                "model": self.settings.deepgram_model,
                "language": self.settings.deepgram_language,
                "smart_format": True,
                "encoding": "linear16",
            }

            if _DEEPGRAM_SDK_VARIANT == "v5":
                response = await self._client.listen.v1.media.transcribe_file(
                    {"buffer": audio_bytes},
                    options,
                )
            else:
                response = await self._client.listen.asyncrest.v("1").transcribe_file(
                    {"buffer": audio_bytes},
                    options,
                )

            transcript = ""
            results = getattr(response, "results", None)
            channels = getattr(results, "channels", None) if results else None
            alternatives = channels[0].alternatives if channels else None
            if alternatives:
                transcript = getattr(alternatives[0], "transcript", None) or ""

            self._logger.debug("transcription_complete", transcript=transcript[:50])
            return transcript

        except STTError:
            raise
        except Exception as exc:
            self._logger.error("transcription_failed", error=str(exc))
            raise STTError(f"Transcription failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    async def _drain_transcript_queue(
        self,
        transcript_queue: "Queue[str | InterruptMarker | None]",
    ) -> AsyncIterator[str | InterruptMarker]:
        """Yield all queued transcripts without blocking."""
        while not transcript_queue.empty():
            item = transcript_queue.get_nowait()
            if item:
                yield item

    async def _stream_transcribe_v5(
        self,
        audio_queue: "Queue[np.ndarray | None]",
        transcript_queue: "Queue[str | InterruptMarker | None]",
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream audio using the Deepgram v5 client."""
        options = self._live_transcription_options()
        finalized_segments: list[str] = []

        async with self._client.listen.v1.connect(options) as connection:

            async def _on_message(self, message: ListenV1Results, **kwargs) -> None:  # type: ignore[name-defined]
                utterance = self._consume_result(message, finalized_segments)
                if utterance:
                    await transcript_queue.put(utterance)

            async def _on_speech_started(self, speech_started: ListenV1SpeechStarted, **kwargs) -> None:  # type: ignore[name-defined]
                self._logger.debug("deepgram_speech_started")
                await transcript_queue.put(InterruptMarker())

            async def _on_utterance_end(self, message: ListenV1UtteranceEnd, **kwargs) -> None:  # type: ignore[name-defined]
                utterance = self._flush_segments(finalized_segments)
                if utterance:
                    await transcript_queue.put(utterance)

            async def _on_error(self, error: object, **kwargs) -> None:
                self._logger.error("deepgram_live_error", error=str(error))

            connection.on(LiveTranscriptionEvents.Transcript, _on_message)
            connection.on(LiveTranscriptionEvents.SpeechStarted, _on_speech_started)
            connection.on(LiveTranscriptionEvents.UtteranceEnd, _on_utterance_end)
            connection.on(LiveTranscriptionEvents.Error, _on_error)

            listen_task = asyncio.create_task(connection.start_listening())
            self._logger.info("deepgram_live_connection_opened", sdk_variant="v5")

            try:
                async for item in self._stream_live_events(
                    audio_queue,
                    transcript_queue,
                    connection.send_media,
                ):
                    yield item

                await connection.send_finalize()
                self._logger.debug("deepgram_finalize_sent")
                await self._wait_for_finalize_flush()

                utterance = self._flush_segments(finalized_segments)
                if utterance:
                    await transcript_queue.put(utterance)

            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                self._logger.info("deepgram_live_connection_closed", sdk_variant="v5")

        async for item in self._drain_transcript_queue(transcript_queue):
            yield item

    async def _stream_transcribe_v3(
        self,
        audio_queue: "Queue[np.ndarray | None]",
        transcript_queue: "Queue[str | InterruptMarker | None]",
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream audio using the Deepgram v3 async websocket client."""
        options = self._live_transcription_options()
        finalized_segments: list[str] = []
        connection = self._client.listen.asyncwebsocket.v("1")

        async def _on_message(self, message: ListenV1Results, **kwargs) -> None:  # type: ignore[name-defined]
            utterance = self._consume_result(message, finalized_segments)
            if utterance:
                await transcript_queue.put(utterance)

        async def _on_speech_started(self, speech_started: ListenV1SpeechStarted, **kwargs) -> None:  # type: ignore[name-defined]
            self._logger.debug("deepgram_speech_started")
            await transcript_queue.put(InterruptMarker())

        async def _on_utterance_end(self, message: ListenV1UtteranceEnd, **kwargs) -> None:  # type: ignore[name-defined]
            utterance = self._flush_segments(finalized_segments)
            if utterance:
                await transcript_queue.put(utterance)

        async def _on_error(self, error: object, **kwargs) -> None:
            self._logger.error("deepgram_live_error", error=str(error))

        connection.on(LiveTranscriptionEvents.Transcript, _on_message)
        connection.on(LiveTranscriptionEvents.SpeechStarted, _on_speech_started)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, _on_utterance_end)
        connection.on(LiveTranscriptionEvents.Error, _on_error)

        started = await connection.start(options)
        if not started:
            raise STTError("Failed to start Deepgram live transcription")

        self._logger.info("deepgram_live_connection_opened", sdk_variant="v3")

        try:
            async for item in self._stream_live_events(
                audio_queue,
                transcript_queue,
                connection.send,
            ):
                yield item

            await connection.finalize()
            self._logger.debug("deepgram_finalize_sent")
            await self._wait_for_finalize_flush()

            utterance = self._flush_segments(finalized_segments)
            if utterance:
                await transcript_queue.put(utterance)

        finally:
            await connection.finish()
            self._logger.info("deepgram_live_connection_closed", sdk_variant="v3")

        async for item in self._drain_transcript_queue(transcript_queue):
            yield item

    async def stream_transcribe(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
    ) -> AsyncIterator[str | InterruptMarker]:
        """Stream audio to Deepgram and yield transcripts or InterruptMarkers."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")

        if self._client is None:
            await self.connect()

        transcript_queue: asyncio.Queue[str | InterruptMarker | None] = asyncio.Queue()

        if _DEEPGRAM_SDK_VARIANT == "v5":
            async for item in self._stream_transcribe_v5(audio_queue, transcript_queue):
                yield item
            return

        if _DEEPGRAM_SDK_VARIANT == "v3":
            async for item in self._stream_transcribe_v3(audio_queue, transcript_queue):
                yield item
            return

        raise ServiceUnavailableError("Deepgram", "Unsupported Deepgram SDK variant")
