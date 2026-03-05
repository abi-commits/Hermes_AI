"""Core domain logic for Hermes."""

from hermes.core.audio import decode_mulaw, encode_mulaw, resample_audio
from hermes.core.call import Call
from hermes.core.exceptions import (
    AudioProcessingError,
    CallError,
    HermesError,
    LLMError,
    STTError,
    TTSGenerationError,
)
from hermes.models.call import CallState

__all__ = [
    "Call",
    "CallState",
    "decode_mulaw",
    "encode_mulaw",
    "resample_audio",
    "HermesError",
    "CallError",
    "STTError",
    "LLMError",
    "TTSGenerationError",
    "AudioProcessingError",
]
