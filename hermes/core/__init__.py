"""Lazy core exports for Hermes."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Call",
    "CallState",
    "CallConfig",
    "CallOrchestrator",
    "OrchestratorHooks",
    "ServiceBundle",
    "decode_mulaw",
    "AudioProcessingError",
    "CallError",
    "HermesError",
    "LLMError",
    "STTError",
    "TTSGenerationError",
]


_EXPORTS = {
    "Call": ("hermes.core.call", "Call"),
    "CallState": ("hermes.models.call", "CallState"),
    "CallConfig": ("hermes.core.orchestrator", "CallConfig"),
    "CallOrchestrator": ("hermes.core.orchestrator", "CallOrchestrator"),
    "OrchestratorHooks": ("hermes.core.orchestrator", "OrchestratorHooks"),
    "ServiceBundle": ("hermes.core.orchestrator", "ServiceBundle"),
    "decode_mulaw": ("hermes.core.audio", "decode_mulaw"),
    "AudioProcessingError": ("hermes.core.exceptions", "AudioProcessingError"),
    "CallError": ("hermes.core.exceptions", "CallError"),
    "HermesError": ("hermes.core.exceptions", "HermesError"),
    "LLMError": ("hermes.core.exceptions", "LLMError"),
    "STTError": ("hermes.core.exceptions", "STTError"),
    "TTSGenerationError": ("hermes.core.exceptions", "TTSGenerationError"),
}


def __getattr__(name: str):
    """Load core exports on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
