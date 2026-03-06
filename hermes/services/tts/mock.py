"""Mock TTS service for testing and local development."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import structlog

from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)

# Default chunk size (number of samples per yielded buffer).
_DEFAULT_CHUNK_SIZE: int = 50


class MockTTSService(AbstractTTSService):
    """Deterministic TTS stub for testing; returns a 440 Hz sine-wave without model loading."""

    def __init__(
        self,
        duration_seconds: float = 1.0,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Initialise the mock TTS service."""
        self.duration_seconds = duration_seconds
        self.chunk_size = chunk_size
        self._logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = False,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield 16-bit PCM chunks of a 440 Hz sine wave."""
        sr = self.sample_rate
        n_samples = int(self.duration_seconds * sr)
        t = np.linspace(0, self.duration_seconds, n_samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32_767).astype(np.int16)

        _chunk = chunk_size if chunk_size is not None else self.chunk_size
        for start in range(0, len(audio), _chunk):
            yield audio[start : start + _chunk].tobytes()
            await asyncio.sleep(0)  # yield control back to the event loop

        self._logger.debug("mock_tts_stream_complete", text=text[:50])

    # ------------------------------------------------------------------
    # Full-audio synthesis
    # ------------------------------------------------------------------

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = False,
    ) -> bytes:
        """Return the complete sine-wave audio in a single buffer."""
        chunks: list[bytes] = []
        async for chunk in self.generate_stream(text, audio_prompt_path, embed_watermark):
            chunks.append(chunk)
        self._logger.debug("mock_tts_generated", text=text[:50])
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Executor management (no-op)
    # ------------------------------------------------------------------

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """No-op — the mock does not use a thread pool."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Fixed at 16 kHz for the mock service."""
        return 16_000
