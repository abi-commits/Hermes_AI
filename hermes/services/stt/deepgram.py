"""Deepgram Speech-to-Text service implementation (Deepgram Python SDK v5+)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Union

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

    ListenV1SocketClientResponse = Union[
        ListenV1Results,
        ListenV1Metadata,
        ListenV1UtteranceEnd,
        ListenV1SpeechStarted,
    ]

    HAS_DEEPGRAM = True
except ImportError:
    HAS_DEEPGRAM = False

from config import get_settings
from hermes.core.exceptions import ServiceUnavailableError, STTError
from hermes.services.stt.base import AbstractSTTService

logger = structlog.get_logger(__name__)


class DeepgramSTTService(AbstractSTTService):
    """STT service backed by the Deepgram API v5+ (pre-recorded REST + live WebSocket streaming)."""

    def __init__(self) -> None:
        """Initialise the Deepgram STT service."""
        self.settings = get_settings()
        self._client: "AsyncDeepgramClient | None" = None
        self._logger = structlog.get_logger(__name__)

        if not HAS_DEEPGRAM:
            self._logger.warning("deepgram_sdk_not_installed")
        if not self.settings.deepgram_api_key:
            self._logger.warning("deepgram_api_key_not_set")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the async Deepgram client."""
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")
        if not self.settings.deepgram_api_key:
            raise ServiceUnavailableError("Deepgram", "DEEPGRAM_API_KEY not configured")

        self._client = AsyncDeepgramClient(api_key=self.settings.deepgram_api_key)
        self._logger.info("deepgram_client_created")

    async def disconnect(self) -> None:
        """Discard the client reference (connections are context-managed per request)."""
        self._client = None
        self._logger.info("deepgram_disconnected")

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
            audio_bytes: bytes = (audio * 32767).astype(np.int16).tobytes()

            # v5 SDK: client.listen.v1.media.transcribe_file
            response = await self._client.listen.v1.media.transcribe_file(  # type: ignore[union-attr]
                {"buffer": audio_bytes},
                {
                    "model": self.settings.deepgram_model,
                    "language": self.settings.deepgram_language,
                    "smart_format": True,
                    "encoding": "linear16",
                }
            )

            transcript: str = ""
            _results = getattr(response, "results", None)
            _channels = getattr(_results, "channels", None) if _results else None
            _alts = _channels[0].alternatives if _channels else None
            if _alts:
                transcript = getattr(_alts[0], "transcript", None) or ""

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

    async def stream_transcribe(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
    ) -> AsyncIterator[str]:
        """Stream audio to Deepgram Live API (v5) and yield transcripts in real-time.

        A sentinel ``None`` placed in *audio_queue* signals end of stream.

        The method opens a WebSocket connection via ``client.listen.v1.connect``,
        starts the listener in a background task, drains *audio_queue* sending
        PCM chunks as they arrive, then signals end-of-stream with
        ``send_finalize`` before tearing down.
        """
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram", "Deepgram SDK not installed")

        if self._client is None:
            await self.connect()

        # Queue used to ferry transcripts from the synchronous event callback
        # back into this async generator.
        transcript_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        options = {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "smart_format": "true",
            "interim_results": "true",
            "utterance_end_ms": "1000",
            "encoding": "linear16",
            "sample_rate": 8000, # Twilio rate
        }

        async with self._client.listen.v1.connect(options) as connection:  # type: ignore[union-attr]

            # ----------------------------------------------------------------
            # Event handlers
            # ----------------------------------------------------------------

            async def _on_message(self, message: ListenV1Results, **kwargs) -> None:  # type: ignore[name-defined]
                transcript = message.channel.alternatives[0].transcript
                if transcript:
                    await transcript_queue.put(transcript)

            async def _on_error(self, error: object, **kwargs) -> None:
                self._logger.error("deepgram_live_error", error=str(error))

            connection.on(LiveTranscriptionEvents.Transcript, _on_message)  # type: ignore[possibly-undefined]
            connection.on(LiveTranscriptionEvents.Error, _on_error)  # type: ignore[possibly-undefined]

            # Start listener as a background task so audio sending is concurrent.
            listen_task = asyncio.create_task(connection.start_listening())
            self._logger.info("deepgram_live_connection_opened")

            # ----------------------------------------------------------------
            # Send audio chunks; None sentinel stops the loop.
            # ----------------------------------------------------------------
            try:
                while True:
                    audio = await audio_queue.get()
                    if audio is None:
                        break
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                    await connection.send_media(audio_bytes)

                    # Drain any already-arrived transcripts
                    while not transcript_queue.empty():
                        item = transcript_queue.get_nowait()
                        if item is not None:
                            yield item

                # Signal end of audio to Deepgram and wait for final responses.
                await connection.send_finalize()
                self._logger.debug("deepgram_finalize_sent")
                await asyncio.sleep(1.5)

            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                self._logger.info("deepgram_live_connection_closed")

            # Yield any remaining transcripts collected after finalize.
            while not transcript_queue.empty():
                item = transcript_queue.get_nowait()
                if item is not None:
                    yield item
