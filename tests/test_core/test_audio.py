"""Tests for audio processing utilities."""

import pytest
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
        original = np.sin(np.linspace(0, 2 * np.pi, 8000)).astype(np.float32) * 0.5

        # Encode then decode
        encoded = encode_mulaw(original)
        decoded = decode_mulaw(encoded)

        # Should be approximately the same (mu-law is lossy)
        assert len(decoded) == len(original)
        assert np.abs(decoded).max() <= 1.0

    def test_encode_silence_matches_g711_pcmu(self):
        """Digital silence should encode to 0xFF in G.711 PCMU."""
        audio = np.zeros(4, dtype=np.float32)

        assert encode_mulaw(audio) == b"\xff\xff\xff\xff"

    def test_decode_known_pcmu_levels(self):
        """Known PCMU bytes should decode to the expected signed levels."""
        decoded = decode_mulaw(bytes([0xFF, 0xFE, 0x7E]))

        assert decoded[0] == pytest.approx(0.0, abs=1e-6)
        assert decoded[1] == pytest.approx(8 / 32768.0, abs=1e-6)
        assert decoded[2] == pytest.approx(-8 / 32768.0, abs=1e-6)

    def test_encode_returns_bytes(self):
        """Test that encoding returns bytes."""
        audio = np.zeros(100, dtype=np.float32)
        encoded = encode_mulaw(audio)

        assert isinstance(encoded, bytes)
        assert len(encoded) == 100

    def test_decode_returns_array(self):
        """Test that decoding returns numpy array."""
        encoded = b"\xff" * 100
        decoded = decode_mulaw(encoded)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 100


class TestResampling:
    """Tests for audio resampling."""

    def test_resample_changes_sample_rate(self):
        """Test that resampling changes the sample count."""
        # 1 second at 16kHz
        audio = np.random.randn(16000).astype(np.float32)

        # Resample to 8kHz
        resampled = resample_audio(audio, orig_freq=16000, new_freq=8000)

        # Should be approximately 8000 samples
        assert len(resampled) == 8000

    def test_resample_no_change_same_rate(self):
        """Test that same rate returns unchanged."""
        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample_audio(audio, orig_freq=16000, new_freq=16000)

        assert np.allclose(audio, resampled)


class TestFormatConversion:
    """Tests for audio format conversion."""

    def test_float_to_int16(self):
        """Test float to int16 conversion."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        encoded = float_to_int16(audio)

        assert isinstance(encoded, bytes)
        assert len(encoded) == 10  # 5 samples * 2 bytes

    def test_int16_to_float(self):
        """Test int16 to float conversion."""
        # Create int16 bytes
        int16_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        encoded = int16_data.tobytes()

        decoded = int16_to_float(encoded)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 5
        assert np.abs(decoded).max() <= 1.0

    def test_format_roundtrip(self):
        """Test float -> int16 -> float roundtrip."""
        original = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)

        int16 = float_to_int16(original)
        recovered = int16_to_float(int16)

        # Should be approximately equal
        assert np.allclose(original, recovered, atol=0.001)


class TestGain:
    """Tests for gain operations."""

    def test_apply_gain_increases_amplitude(self):
        """Test that positive gain increases amplitude."""
        audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        gained = apply_gain(audio, 6.0)  # +6dB = ~2x

        assert np.mean(np.abs(gained)) > np.mean(np.abs(audio))

    def test_apply_gain_decreases_amplitude(self):
        """Test that negative gain decreases amplitude."""
        audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        gained = apply_gain(audio, -6.0)  # -6dB = ~0.5x

        assert np.mean(np.abs(gained)) < np.mean(np.abs(audio))

    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        normalized = normalize_audio(audio, target_db=-20.0)

        # Should be scaled up (original is much quieter than -20 dBFS)
        assert np.mean(np.abs(normalized)) > np.mean(np.abs(audio))
