"""Speech-to-Text service package.

Public API
----------
``DeepgramSTTService``
    Production STT backed by Deepgram's live WebSocket API.

``STTService``
    Alias for ``DeepgramSTTService`` — kept for backward compatibility.

``MockSTTService``
    Deterministic stub for tests and local development.

``AbstractSTTService``
    ABC defining the interface all implementations must fulfil.
"""

from hermes.services.stt.base import AbstractSTTService
from hermes.services.stt.deepgram import DeepgramSTTService
from hermes.services.stt.mock import MockSTTService

# Backward-compatible alias — existing code that imports STTService still works.
STTService = DeepgramSTTService

__all__ = [
    "AbstractSTTService",
    "DeepgramSTTService",
    "MockSTTService",
    "STTService",
]
