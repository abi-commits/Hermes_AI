"""Tests for audio processing utilities."""

import pytest
import torch
import numpy as np

from hermes.core.audio import (
    decode_mulaw,
    encode_mulaw,
    resample_audio,
    float_to_int16,
    int16_to_float,
    apply_gain,
    normalize_audio,
)


class TestMuLawCodec:
    """Tests for mu-law encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Test that encode/decode is lossy but reversible."""
        # Create test audio
        original = torch.sin(torch.linspace(0, 2 * np.pi, 8000)) * 0.5

        # Encode then decode
        encoded = encode_mulaw(original)
        decoded = decode_mulaw(encoded)

        # Should be approximately the same (mu-law is lossy)
        assert len(decoded) == len(original)
        assert decoded.abs().max() <= 1.0

    def test_encode_returns_bytes(self):
        """Test that encoding returns bytes."""
        audio = torch.zeros(100)
        encoded = encode_mulaw(audio)

        assert isinstance(encoded, bytes)
        assert len(encoded) == 100

    def test_decode_returns_tensor(self):
        """Test that decoding returns tensor."""
        encoded = b"\xff" * 100
        decoded = decode_mulaw(encoded)

        assert isinstance(decoded, torch.Tensor)
        assert len(decoded) == 100


class TestResampling:
    """Tests for audio resampling."""

    def test_resample_changes_sample_rate(self):
        """Test that resampling changes the sample count."""
        # 1 second at 16kHz
        audio = torch.randn(16000)

        # Resample to 8kHz
        resampled = resample_audio(audio, orig_freq=16000, new_freq=8000)

        # Should be approximately 8000 samples
        assert len(resampled) == 8000

    def test_resample_no_change_same_rate(self):
        """Test that same rate returns unchanged."""
        audio = torch.randn(16000)
        resampled = resample_audio(audio, orig_freq=16000, new_freq=16000)

        assert torch.allclose(audio, resampled)


class TestFormatConversion:
    """Tests for audio format conversion."""

    def test_float_to_int16(self):
        """Test float to int16 conversion."""
        audio = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        encoded = float_to_int16(audio)

        assert isinstance(encoded, bytes)
        assert len(encoded) == 10  # 5 samples * 2 bytes

    def test_int16_to_float(self):
        """Test int16 to float conversion."""
        # Create int16 bytes
        int16_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        encoded = int16_data.tobytes()

        decoded = int16_to_float(encoded)

        assert isinstance(decoded, torch.Tensor)
        assert len(decoded) == 5
        assert decoded.abs().max() <= 1.0

    def test_format_roundtrip(self):
        """Test float -> int16 -> float roundtrip."""
        original = torch.tensor([0.0, 0.5, -0.5, 0.25, -0.25])

        int16 = float_to_int16(original)
        recovered = int16_to_float(int16)

        # Should be approximately equal
        assert torch.allclose(original, recovered, atol=0.001)


class TestGain:
    """Tests for gain operations."""

    def test_apply_gain_increases_amplitude(self):
        """Test that positive gain increases amplitude."""
        audio = torch.tensor([0.5, 0.5, 0.5])
        gained = apply_gain(audio, 6.0)  # +6dB = ~2x

        assert gained.abs().mean() > audio.abs().mean()

    def test_apply_gain_decreases_amplitude(self):
        """Test that negative gain decreases amplitude."""
        audio = torch.tensor([0.5, 0.5, 0.5])
        gained = apply_gain(audio, -6.0)  # -6dB = ~0.5x

        assert gained.abs().mean() < audio.abs().mean()

    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = torch.tensor([0.1, 0.1, 0.1])
        normalized = normalize_audio(audio, target_db=-20.0)

        # Should be scaled up
        assert normalized.abs().mean() > audio.abs().mean()
