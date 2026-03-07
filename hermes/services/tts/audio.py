"""Audio format conversion helpers (16-bit PCM → 8 kHz → µ-law) for the TTS pipeline."""

from __future__ import annotations

import numpy as np
import soxr


def resample_to_8khz(audio_bytes: bytes, orig_sr: int) -> bytes:
    """Resample 16-bit PCM from *orig_sr* to 8 kHz (required by Twilio)."""
    if orig_sr == 8_000:
        return audio_bytes

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    resampled = soxr.resample(audio_np, orig_sr, 8000)
    resampled_int16 = (resampled * 32767.0).astype(np.int16)
    return resampled_int16.tobytes()


def convert_to_ulaw(pcm16_bytes: bytes) -> bytes:
    """Encode 16-bit PCM bytes to 8-bit µ-law (G.711 PCMU) for Twilio."""
    audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    mu = 255
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = np.sign(audio) * magnitude
    
    quantized = ((1.0 - encoded) * 127.5).astype(np.uint8)
    return quantized.tobytes()
