"""TTS service package.

Public API
----------
``ChatterboxTTSService``
    Production TTS backed by the Chatterbox Streaming model.

``TTSWorkerPool``
    Round-robin pool of ``ChatterboxTTSService`` instances for concurrent synthesis.

``MockTTSService``
    Deterministic sine-wave stub for tests and local development.

``AbstractTTSService``
    ABC defining the interface all implementations must fulfil.

Audio helpers
~~~~~~~~~~~~~
``resample_to_8khz(audio_bytes, orig_sr)``
    Resample 16-bit PCM from *orig_sr* to 8 kHz (Twilio requirement).

``convert_to_ulaw(pcm16_bytes)``
    Encode 16-bit PCM to 8-bit µ-law (G.711 PCMU, Twilio format).
"""

from __future__ import annotations

from typing import Any
from importlib import import_module

class ChatterboxTurboTTS:
    """Lazy proxy around the Chatterbox model class.

    Keeping this import lazy avoids making ``import hermes.services.tts`` fail in
    environments where heavyweight optional dependencies are partially present.
    """

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        """Load and return the concrete Chatterbox model."""
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(*args, **kwargs)


# Lazy loading mapping
_EXPORTS = {
    "AbstractTTSService": ("hermes.services.tts.base", "AbstractTTSService"),
    "ChatterboxTTSService": ("hermes.services.tts.chatterbox", "ChatterboxTTSService"),
    "ModalRemoteTTSService": ("hermes.services.tts.modal_remote", "ModalRemoteTTSService"),
    "MockTTSService": ("hermes.services.tts.mock", "MockTTSService"),
    "TTSWorkerPool": ("hermes.services.tts.worker_pool", "TTSWorkerPool"),
    "convert_to_ulaw": ("hermes.services.tts.audio", "convert_to_ulaw"),
    "resample_to_8khz": ("hermes.services.tts.audio", "resample_to_8khz"),
}

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr_name = _EXPORTS[name]
        module = import_module(module_path)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "AbstractTTSService",
    "ChatterboxTurboTTS",
    "ChatterboxTTSService",
    "ModalRemoteTTSService",
    "MockTTSService",
    "TTSWorkerPool",
    "convert_to_ulaw",
    "resample_to_8khz",
]
