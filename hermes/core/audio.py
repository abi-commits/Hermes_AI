"""Audio processing utilities (NumPy-based with Soxr and G.711 PCMU support)."""

from __future__ import annotations

import math

import numpy as np
import soxr

try:
    import audioop
except ImportError:  # pragma: no cover - covered indirectly in environments without audioop
    audioop = None


def _require_audioop():
    """Return the available audioop module or raise a clear runtime error."""
    if audioop is None:
        raise RuntimeError(
            "G.711 mu-law support requires 'audioop'. "
            "Install 'audioop-lts' on Python 3.13+ environments."
        )
    return audioop


def pcm16_bytes_to_mulaw(data: bytes) -> bytes:
    """Encode 16-bit PCM bytes to G.711 PCMU bytes."""
    return _require_audioop().lin2ulaw(data, 2)


def mulaw_bytes_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 PCMU bytes into 16-bit PCM bytes."""
    return _require_audioop().ulaw2lin(data, 2)


def encode_mulaw(audio: np.ndarray) -> bytes:
    """Encode float PCM [-1.0, 1.0] to G.711 PCMU bytes."""
    return pcm16_bytes_to_mulaw(float_to_int16(audio))


def decode_mulaw(data: bytes) -> np.ndarray:
    """Decode G.711 PCMU bytes to float PCM [-1.0, 1.0]."""
    return int16_to_float(mulaw_bytes_to_pcm16(data))


def resample_audio(audio: np.ndarray, orig_freq: int, new_freq: int) -> np.ndarray:
    """Resample PCM audio using Soxr."""
    if orig_freq == new_freq:
        return audio
    return soxr.resample(audio, orig_freq, new_freq).astype(np.float32)


def float_to_int16(audio: np.ndarray) -> bytes:
    """Convert float PCM [-1,1] to int16 bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()


def int16_to_float(data: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to float PCM."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in decibels."""
    gain = math.pow(10.0, gain_db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize to target dBFS."""
    if len(audio) == 0:
        return audio

    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db

    return apply_gain(audio, gain_db)
