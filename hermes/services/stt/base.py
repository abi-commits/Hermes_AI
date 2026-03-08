"""Abstract base class for Speech-to-Text services."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AbstractSTTService(ABC):
    """Interface contract for all STT service implementations."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open a connection / warm up the underlying service."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close any open connections and release resources."""

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @abstractmethod
    async def transcribe(self, audio: "np.ndarray") -> str:
        """Transcribe a complete PCM float32 audio array (non-streaming)."""

    @abstractmethod
    def stream_transcribe(
        self,
        audio_queue: "asyncio.Queue[np.ndarray]",
    ) -> AsyncIterator[str | InterruptMarker]:
        """Yield transcript fragments or InterruptMarkers in real-time."""
...
from hermes.models.llm import InterruptMarker
