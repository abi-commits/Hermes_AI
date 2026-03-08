"""Audio processing utilities (NumPy-based with Soxr)."""

from __future__ import annotations

import math
import numpy as np
import soxr


def encode_mulaw(audio: np.ndarray) -> bytes:
    """Encode float PCM [-1.0, 1.0] to μ-law bytes using NumPy."""
    # Clip to avoid log of zero or negative
    mu = 255
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = np.sign(audio) * magnitude
    
    # Map from [-1, 1] to [0, 255]
    # This is a simplified linear mapping; for production telephony, 
    # audioop.lin2ulaw or a lookup table is more standard, 
    # but this satisfies the requirement without audioop.
    quantized = ((1.0 - encoded) * 127.5).astype(np.uint8)
    return quantized.tobytes()


def decode_mulaw(data: bytes) -> np.ndarray:
    """Decode μ-law bytes to float PCM [-1.0, 1.0] using NumPy."""
    mu = 255
    quantized = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    
    # Inverse mapping
    encoded = 1.0 - (quantized / 127.5)
    audio = np.sign(encoded) * (1.0 / mu) * ((1.0 + mu) ** np.abs(encoded) - 1.0)
    return audio.astype(np.float32)


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
