"""Abstract base class for Text-to-Speech services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class AbstractTTSService(ABC):
    """Interface contract for all TTS service implementations."""

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesise *text* and yield 16-bit PCM chunks as produced."""

    # ------------------------------------------------------------------
    # Full-audio synthesis (convenience wrapper)
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
    ) -> bytes:
        """Synthesise *text* and return the complete audio as a single buffer."""

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    @abstractmethod
    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """Attach a thread-pool executor for CPU-bound synthesis work."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of the loaded model in Hz."""
