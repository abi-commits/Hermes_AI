"""Speech-to-Text service integration."""

import base64
import struct
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

    HAS_DEEPGRAM = True
except ImportError:
    HAS_DEEPGRAM = False

from config import get_settings
from hermes.core.exceptions import ServiceUnavailableError, STTError

if TYPE_CHECKING:
    import torch

logger = structlog.get_logger(__name__)


class STTService:
    """Speech-to-Text service using Deepgram.

    This class provides streaming transcription capabilities
    for converting audio to text in real-time.
    """

    def __init__(self) -> None:
        """Initialize the STT service."""
        self.settings = get_settings()
        self._client: DeepgramClient | None = None
        self._connection = None
        self._logger = structlog.get_logger(__name__)

        if not HAS_DEEPGRAM:
            self._logger.warning("deepgram_sdk_not_installed")
        if not self.settings.deepgram_api_key:
            self._logger.warning("deepgram_api_key_not_set")

    async def connect(self) -> None:
        """Connect to the Deepgram streaming API."""
        if not HAS_DEEPGRAM or not self.settings.deepgram_api_key:
            raise ServiceUnavailableError("Deepgram")

        self._client = DeepgramClient(self.settings.deepgram_api_key)

    async def disconnect(self) -> None:
        """Disconnect from Deepgram."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @retry(
        retry=retry_if_exception_type((STTError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: "torch.Tensor") -> str:
        """Transcribe audio to text.

        This is a simplified non-streaming transcription.
        For real-time streaming, use the stream_transcribe method.

        Args:
            audio: Audio tensor (PCM float32).

        Returns:
            Transcribed text.
        """
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError(
                "Deepgram", "Deepgram SDK not installed"
            )

        if not self.settings.deepgram_api_key:
            raise STTError("Deepgram API key not configured")

        try:
            # Convert tensor to bytes
            import numpy as np

            audio_bytes = (audio.numpy() * 32767).astype(np.int16).tobytes()

            # Use Deepgram REST API for non-streaming transcription
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    headers={
                        "Authorization": f"Token {self.settings.deepgram_api_key}",
                        "Content-Type": "audio/wav",
                    },
                    params={
                        "model": self.settings.deepgram_model,
                        "language": self.settings.deepgram_language,
                        "smart_format": True,
                    },
                    content=audio_bytes,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    raise STTError(
                        f"Deepgram API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                transcript = (
                    data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )

                self._logger.debug("transcription_complete", transcript=transcript[:50])
                return transcript

        except Exception as e:
            self._logger.error("transcription_failed", error=str(e))
            raise STTError(f"Transcription failed: {e}")

    async def stream_transcribe(
        self,
        audio_queue: "asyncio.Queue[torch.Tensor]",
    ) -> AsyncIterator[str]:
        """Stream audio and yield transcriptions in real-time.

        Args:
            audio_queue: Queue of audio tensors to transcribe.

        Yields:
            Transcription text chunks.
        """
        if not HAS_DEEPGRAM:
            raise ServiceUnavailableError("Deepgram")

        # Set up live transcription options
        options = LiveOptions(
            model=self.settings.deepgram_model,
            language=self.settings.deepgram_language,
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
        )

        # Create connection
        connection = self._client.listen.live.v("1")

        transcripts: list[str] = []

        # Set up event handlers
        def on_transcript(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                transcripts.append(transcript)

        def on_error(self, error, **kwargs):
            self._logger.error("deepgram_error", error=str(error))

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        # Start connection
        await connection.start(options)

        try:
            while True:
                # Get audio from queue
                audio = await audio_queue.get()

                # Convert to bytes
                import numpy as np

                audio_bytes = (audio.numpy() * 32767).astype(np.int16).tobytes()

                # Send to Deepgram
                connection.send(audio_bytes)

                # Yield any new transcripts
                while transcripts:
                    yield transcripts.pop(0)

        finally:
            connection.finish()


class MockSTTService(STTService):
    """Mock STT service for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize mock service.

        Args:
            responses: List of mock responses to cycle through.
        """
        self.responses = responses or ["Hello, this is a test transcript."]
        self._index = 0
        self._logger = structlog.get_logger(__name__)

    async def connect(self) -> None:
        """No-op connect."""
        pass

    async def disconnect(self) -> None:
        """No-op disconnect."""
        pass

    async def transcribe(self, audio: "torch.Tensor") -> str:
        """Return mock transcription."""
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        self._logger.debug("mock_transcription", response=response)
        return response

    async def stream_transcribe(
        self,
        audio_queue: "asyncio.Queue[torch.Tensor]",
    ) -> AsyncIterator[str]:
        """Yield mock transcriptions."""
        while True:
            await audio_queue.get()
            yield self.responses[0]
