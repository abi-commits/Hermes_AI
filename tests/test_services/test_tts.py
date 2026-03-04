"""Tests for TTS service."""

import pytest
import torch

from hermes.services.tts import MockTTSService


class TestMockTTSService:
    """Tests for Mock TTS service."""

    @pytest.mark.asyncio
    async def test_mock_synthesize_returns_audio(self):
        """Test that mock service returns audio tensor."""
        service = MockTTSService(duration_seconds=0.5)

        result = await service.synthesize("Hello world")

        assert isinstance(result, torch.Tensor)
        assert len(result) == 8000  # 0.5s * 16kHz

    @pytest.mark.asyncio
    async def test_mock_synthesize_different_durations(self):
        """Test different audio durations."""
        for duration in [0.1, 0.5, 1.0]:
            service = MockTTSService(duration_seconds=duration)
            result = await service.synthesize("Test")

            expected_samples = int(duration * 16000)
            assert len(result) == expected_samples
