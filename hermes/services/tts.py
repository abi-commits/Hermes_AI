"""Text-to-Speech service integration."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
import structlog
import torch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings
from hermes.core.exceptions import ServiceUnavailableError, TTSGenerationError

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class TTSService:
    """Text-to-Speech service using Chatterbox or alternative providers.

    This class provides text-to-speech synthesis for generating
    natural-sounding voice responses.
    """

    def __init__(self) -> None:
        """Initialize the TTS service."""
        self.settings = get_settings()
        self._logger = structlog.get_logger(__name__)

    @retry(
        retry=retry_if_exception_type((TTSGenerationError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def synthesize(self, text: str) -> "torch.Tensor":
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.

        Returns:
            Audio tensor (PCM float32).
        """
        # Try Chatterbox first
        try:
            return await self._synthesize_chatterbox(text)
        except Exception as e:
            self._logger.warning("chatterbox_failed", error=str(e))

        # Fallback to OpenAI TTS
        if self.settings.openai_api_key:
            try:
                return await self._synthesize_openai(text)
            except Exception as e:
                self._logger.warning("openai_tts_failed", error=str(e))

        # If all fail, raise error
        raise ServiceUnavailableError("TTS", "No TTS provider available")

    async def _synthesize_chatterbox(self, text: str) -> "torch.Tensor":
        """Synthesize using Chatterbox API.

        Args:
            text: Text to synthesize.

        Returns:
            Audio tensor.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.chatterbox_api_url}/synthesize",
                json={
                    "text": text,
                    "voice": self.settings.chatterbox_voice,
                    "speed": self.settings.chatterbox_speed,
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise TTSGenerationError(
                    f"Chatterbox API error: {response.status_code}"
                )

            # Parse response
            data = response.json()
            audio_data = data.get("audio")

            if not audio_data:
                raise TTSGenerationError("No audio data in response")

            # Convert base64 to tensor
            import base64
            import numpy as np

            audio_bytes = base64.b64decode(audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            return torch.from_numpy(audio_array)

    async def _synthesize_openai(self, text: str) -> "torch.Tensor":
        """Synthesize using OpenAI TTS API.

        Args:
            text: Text to synthesize.

        Returns:
            Audio tensor.
        """
        if not self.settings.openai_api_key:
            raise ServiceUnavailableError("OpenAI TTS")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": "alloy",
                    "response_format": "pcm",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise TTSGenerationError(
                    f"OpenAI TTS API error: {response.status_code}"
                )

            # Parse PCM audio
            import numpy as np

            audio_bytes = response.content
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            return torch.from_numpy(audio_array)

    async def synthesize_stream(
        self,
        text_iterator: AsyncIterator[str],
    ) -> AsyncIterator["torch.Tensor"]:
        """Stream text and yield audio chunks.

        Args:
            text_iterator: Iterator yielding text chunks.

        Yields:
            Audio tensor chunks.
        """
        # Buffer text until we have a complete sentence
        buffer = ""

        async for text_chunk in text_iterator:
            buffer += text_chunk

            # Check for sentence end
            if any(c in buffer for c in ".!?"):
                # Split on sentence boundaries
                while "." in buffer or "!" in buffer or "?" in buffer:
                    for delim in [".", "!", "?"]:
                        if delim in buffer:
                            idx = buffer.find(delim) + 1
                            sentence = buffer[:idx].strip()
                            buffer = buffer[idx:].strip()

                            if sentence:
                                audio = await self.synthesize(sentence)
                                yield audio
                            break

        # Process remaining buffer
        if buffer.strip():
            audio = await self.synthesize(buffer.strip())
            yield audio


class MockTTSService(TTSService):
    """Mock TTS service for testing."""

    def __init__(self, duration_seconds: float = 1.0) -> None:
        """Initialize mock service.

        Args:
            duration_seconds: Duration of mock audio in seconds.
        """
        self.duration_seconds = duration_seconds
        self.sample_rate = 16000
        self._logger = structlog.get_logger(__name__)

    async def synthesize(self, text: str) -> "torch.Tensor":
        """Return mock audio (sine wave)."""
        import numpy as np

        duration_samples = int(self.duration_seconds * self.sample_rate)
        t = np.linspace(0, self.duration_seconds, duration_samples)
        # Generate a simple sine wave at 440 Hz
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3

        self._logger.debug("mock_tts_generated", text=text[:50])
        return torch.from_numpy(audio)
