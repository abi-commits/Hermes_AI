"""Audio format conversion helpers (16-bit PCM → 8 kHz → µ-law) for the TTS pipeline."""

from __future__ import annotations

import numpy as np
import soxr

from hermes.core.audio import pcm16_bytes_to_mulaw


def resample_to_8khz(audio_bytes: bytes, orig_sr: int) -> bytes:
    """Resample 16-bit PCM from *orig_sr* to 8 kHz (required by Twilio)."""
    if orig_sr == 8_000:
        return audio_bytes

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    resampled = soxr.resample(audio_np, orig_sr, 8000)
    resampled_int16 = np.clip(np.round(resampled * 32767.0), -32768, 32767).astype(np.int16)
    return resampled_int16.tobytes()


def convert_to_ulaw(pcm16_bytes: bytes) -> bytes:
    """Encode 16-bit PCM bytes to 8-bit µ-law (G.711 PCMU) for Twilio."""
    return pcm16_bytes_to_mulaw(pcm16_bytes)
