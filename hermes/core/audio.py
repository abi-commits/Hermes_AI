"""Audio processing utilities."""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio

if TYPE_CHECKING:
    import torch


def decode_mulaw(data: bytes, bits: int = 8) -> "torch.Tensor":
    """Decode mu-law encoded bytes to a PCM tensor in ``[-1.0, 1.0]``."""
    encoded = torch.from_numpy(
        np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    )
    quantization_channels = 1 << bits  # 256 for 8-bit

    return torchaudio.functional.mu_law_decoding(encoded, quantization_channels)
