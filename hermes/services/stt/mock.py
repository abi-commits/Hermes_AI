"""Mock STT service for testing and local development."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np
import structlog

from hermes.services.stt.base import AbstractSTTService

logger = structlog.get_logger(__name__)


class MockSTTService(AbstractSTTService):
    """Deterministic STT stub for testing; cycles through pre-set transcript strings."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialise the mock STT service."""
        self.responses = responses or ["Hello, this is a test transcript."]
        self._index = 0
        self._logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Lifecycle (no-ops)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """No-op — nothing to connect to."""

    async def disconnect(self) -> None:
        """No-op — nothing to disconnect from."""

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """Return the next pre-set response, ignoring the audio array."""
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        self._logger.debug("mock_transcription_returned", response=response)
        return response

    async def stream_transcribe(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
    ) -> AsyncIterator[str]:
        """Yield one pre-set response per audio array received from the queue."""
        while True:
            audio = await audio_queue.get()
            if audio is None:
                break
            yield self.responses[self._index % len(self.responses)]
            self._index += 1
