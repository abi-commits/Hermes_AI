"""Audio processing utilities."""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio

if TYPE_CHECKING:
    import torch


def encode_mulaw(audio: "torch.Tensor | np.ndarray", bits: int = 8) -> bytes:
    """Encode audio to mu-law format.

    Uses torchaudio's built-in mu-law encoding, which Twilio uses
    for audio transmission.

    Args:
        audio: Input audio samples (normalized to [-1, 1]).
        bits: Number of bits for encoding (default 8).

    Returns:
        Mu-law encoded bytes.
    """
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    audio = audio.float()

    # Normalize to [-1, 1] if needed
    peak = audio.abs().max()
    if peak > 1.0:
        audio = audio / peak

    quantization_channels = 1 << bits  # 256 for 8-bit
    encoded = torchaudio.functional.mu_law_encoding(audio, quantization_channels)

    return encoded.to(torch.uint8).cpu().numpy().tobytes()


def decode_mulaw(data: bytes, bits: int = 8) -> "torch.Tensor":
    """Decode mu-law audio to PCM.

    Uses torchaudio's built-in mu-law decoding.

    Args:
        data: Mu-law encoded bytes.
        bits: Number of bits for encoding (default 8).

    Returns:
        Decoded audio as PyTorch tensor in range [-1.0, 1.0].
    """
    encoded = torch.from_numpy(
        np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    )
    quantization_channels = 1 << bits  # 256 for 8-bit

    return torchaudio.functional.mu_law_decoding(encoded, quantization_channels)


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
    return torchaudio.functional.gain(audio, gain_db)


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
    # Vectorised edge detection instead of per-sample Python loop
    is_silent = (torch.abs(audio) < threshold).int()

    # Pad with zeros so edges at the very start / end are detected
    padded = torch.nn.functional.pad(is_silent, (1, 1), value=0)
    diff = padded[1:] - padded[:-1]

    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]

    # Filter by minimum duration
    durations = ends - starts
    mask = durations >= min_duration

    return list(zip(starts[mask].tolist(), ends[mask].tolist()))
