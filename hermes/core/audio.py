"""Audio processing utilities."""

import struct
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio

if TYPE_CHECKING:
    import torch


# Mu-law encoding/decoding constants
MU = 255.0
MULAW_BIAS = 33.0


def encode_mulaw(audio: "torch.Tensor | np.ndarray", bits: int = 8) -> bytes:
    """Encode audio to mu-law format.

    Mu-law encoding is used by Twilio for audio transmission.
    Formula: f(x) = ln(1 + mu * |x|) / ln(1 + mu) * sign(x)

    Args:
        audio: Input audio samples (normalized to [-1, 1]).
        bits: Number of bits for encoding (default 8).

    Returns:
        Mu-law encoded bytes.
    """
    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # Ensure float32
    audio = audio.astype(np.float32)

    # Normalize to [-1, 1] if needed
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / np.abs(audio).max()

    # Mu-law encode
    mu = (1 << bits) - 1  # 255 for 8-bit

    # Apply mu-law compression
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = np.sign(audio) * magnitude

    # Quantize to 8-bit unsigned
    encoded = ((encoded + 1.0) / 2.0 * mu + 0.5).astype(np.uint8)

    return encoded.tobytes()


def decode_mulaw(data: bytes, bits: int = 8) -> "torch.Tensor":
    """Decode mu-law audio to PCM.

    Args:
        data: Mu-law encoded bytes.
        bits: Number of bits for encoding (default 8).

    Returns:
        Decoded audio as PyTorch tensor.
    """
    # Convert bytes to numpy array
    encoded = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

    mu = (1 << bits) - 1  # 255 for 8-bit

    # Dequantize
    encoded = encoded / mu * 2.0 - 1.0

    # Apply mu-law expansion
    magnitude = (1.0 / mu) * ((1.0 + mu) ** np.abs(encoded) - 1.0)
    decoded = np.sign(encoded) * magnitude

    # Convert to PyTorch tensor
    return torch.from_numpy(decoded).float()


def resample_audio(
    audio: "torch.Tensor",
    orig_freq: int,
    new_freq: int,
) -> "torch.Tensor":
    """Resample audio to a new sample rate.

    Args:
        audio: Input audio tensor of shape (samples,) or (channels, samples).
        orig_freq: Original sample rate.
        new_freq: Target sample rate.

    Returns:
        Resampled audio tensor.
    """
    if orig_freq == new_freq:
        return audio

    # Ensure 2D tensor for resampling
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # Resample using torchaudio
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=64,
    )

    resampled = resampler(audio)

    # Return to original shape if needed
    if resampled.shape[0] == 1:
        return resampled.squeeze(0)

    return resampled


def float_to_int16(audio: "torch.Tensor") -> bytes:
    """Convert float audio to 16-bit PCM bytes.

    Args:
        audio: Audio tensor in range [-1.0, 1.0].

    Returns:
        16-bit PCM bytes.
    """
    # Clamp to valid range
    audio = torch.clamp(audio, -1.0, 1.0)

    # Convert to int16
    audio_int = (audio * 32767.0).short()

    # Convert to bytes
    return audio_int.cpu().numpy().tobytes()


def int16_to_float(audio_bytes: bytes) -> "torch.Tensor":
    """Convert 16-bit PCM bytes to float audio.

    Args:
        audio_bytes: 16-bit PCM bytes.

    Returns:
        Float audio tensor in range [-1.0, 1.0].
    """
    # Convert bytes to numpy array
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    # Normalize to [-1.0, 1.0]
    audio = audio / 32768.0

    return torch.from_numpy(audio)


def apply_gain(audio: "torch.Tensor", gain_db: float) -> "torch.Tensor":
    """Apply gain to audio in decibels.

    Args:
        audio: Input audio tensor.
        gain_db: Gain in decibels.

    Returns:
        Audio with applied gain.
    """
    gain = 10 ** (gain_db / 20.0)
    return audio * gain


def normalize_audio(audio: "torch.Tensor", target_db: float = -20.0) -> "torch.Tensor":
    """Normalize audio to target dB level.

    Args:
        audio: Input audio tensor.
        target_db: Target level in dBFS.

    Returns:
        Normalized audio.
    """
    # Calculate RMS
    rms = torch.sqrt(torch.mean(audio ** 2))

    if rms > 0:
        # Calculate current dB
        current_db = 20 * torch.log10(rms)

        # Calculate required gain
        gain_db = target_db - current_db.item()

        return apply_gain(audio, gain_db)

    return audio


def chunk_audio(
    audio: "torch.Tensor",
    chunk_size: int,
    overlap: int = 0,
) -> "list[torch.Tensor]":
    """Split audio into chunks.

    Args:
        audio: Input audio tensor.
        chunk_size: Number of samples per chunk.
        overlap: Number of overlapping samples.

    Returns:
        List of audio chunks.
    """
    chunks = []
    hop_size = chunk_size - overlap

    for start in range(0, len(audio), hop_size):
        chunk = audio[start : start + chunk_size]

        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))

        chunks.append(chunk)

    return chunks


def detect_silence(
    audio: "torch.Tensor",
    threshold: float = 0.01,
    min_duration: int = 1600,  # 100ms at 16kHz
) -> "list[tuple[int, int]]":
    """Detect silent regions in audio.

    Args:
        audio: Input audio tensor.
        threshold: Amplitude threshold for silence.
        min_duration: Minimum duration in samples for a silent region.

    Returns:
        List of (start, end) tuples for silent regions.
    """
    # Find samples below threshold
    is_silent = torch.abs(audio) < threshold

    # Find silent regions
    silent_regions = []
    start = None

    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_duration:
                silent_regions.append((start, i))
            start = None

    # Handle trailing silence
    if start is not None and len(audio) - start >= min_duration:
        silent_regions.append((start, len(audio)))

    return silent_regions
