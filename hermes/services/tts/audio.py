"""Audio format conversion helpers (16-bit PCM → 8 kHz → µ-law) for the TTS pipeline."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


def resample_to_8khz(audio_bytes: bytes, orig_sr: int) -> bytes:
    """Resample 16-bit PCM from *orig_sr* to 8 kHz (required by Twilio)."""
    if orig_sr == 8_000:
        return audio_bytes

    audio_tensor = torch.from_numpy(
        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32_767.0
    ).unsqueeze(0)  # shape: (1, T)

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=8_000,
        lowpass_filter_width=64,
    )
    resampled = resampler(audio_tensor)  # shape: (1, T')
    resampled_int16 = (resampled.squeeze(0).numpy() * 32_767).astype(np.int16)
    return resampled_int16.tobytes()


def convert_to_ulaw(pcm16_bytes: bytes) -> bytes:
    """Encode 16-bit PCM bytes to 8-bit µ-law (G.711 PCMU) for Twilio."""
    audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32_768.0
    tensor = torch.from_numpy(audio_np)
    encoded = torchaudio.functional.mu_law_encoding(tensor, quantization_channels=256)
    return encoded.to(torch.uint8).numpy().tobytes()
