"""Core domain logic for Hermes."""

from hermes.core.audio import decode_mulaw
from hermes.core.call import Call
from hermes.core.exceptions import (
    AudioProcessingError,
    CallError,
    HermesError,
    LLMError,
    STTError,
    TTSGenerationError,
)
from hermes.core.orchestrator import CallConfig, CallOrchestrator, OrchestratorHooks, ServiceBundle
from hermes.models.call import CallState

__all__ = [
    # Call state machine
    "Call",
    "CallState",
    # Orchestrator
    "CallConfig",
    "CallOrchestrator",
    "OrchestratorHooks",
    "ServiceBundle",
    # Audio helpers
    "decode_mulaw",
    # Exceptions
    "AudioProcessingError",
    "CallError",
    "HermesError",
    "LLMError",
    "STTError",
    "TTSGenerationError",
]
