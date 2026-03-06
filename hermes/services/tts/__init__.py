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

from hermes.services.tts.audio import convert_to_ulaw, resample_to_8khz
from hermes.services.tts.base import AbstractTTSService
from hermes.services.tts.chatterbox import ChatterboxTTSService
from hermes.services.tts.mock import MockTTSService
from hermes.services.tts.worker_pool import TTSWorkerPool

__all__ = [
    "AbstractTTSService",
    "ChatterboxTTSService",
    "MockTTSService",
    "TTSWorkerPool",
    "convert_to_ulaw",
    "resample_to_8khz",
]
