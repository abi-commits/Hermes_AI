"""Tests for Chatterbox Turbo TTS service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hermes.services.tts import ChatterboxTTSService, MockTTSService, TTSWorkerPool


class TestMockTTSService:
    """Tests for Mock TTS service."""

    @pytest.mark.asyncio
    async def test_mock_generate_returns_bytes(self):
        """Test that mock service returns PCM bytes."""
        service = MockTTSService(duration_seconds=0.5)

        result = await service.generate("Hello world")

        assert isinstance(result, bytes)
        # 0.5s * 16kHz = 8000 samples * 2 bytes/sample = 16000 bytes
        assert len(result) == 16000

    @pytest.mark.asyncio
    async def test_mock_generate_different_durations(self):
        """Test different audio durations."""
        for duration in [0.1, 0.5, 1.0]:
            service = MockTTSService(duration_seconds=duration)
            result = await service.generate("Test")

            expected_samples = int(duration * 16000)
            expected_bytes = expected_samples * 2  # int16 = 2 bytes
            assert len(result) == expected_bytes

    @pytest.mark.asyncio
    async def test_mock_sample_rate(self):
        """Mock service runs at 16 kHz."""
        service = MockTTSService()
        assert service.sample_rate == 16_000

    @pytest.mark.asyncio
    async def test_mock_audio_is_valid_pcm(self):
        """Verify the bytes decode to valid int16 PCM."""
        service = MockTTSService(duration_seconds=0.1)
        result = await service.generate("Test")

        audio = np.frombuffer(result, dtype=np.int16).astype(np.float32) / 32767.0
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0


class TestChatterboxTTSService:
    """Tests for ChatterboxTTSService (model mocked)."""

    @pytest.mark.asyncio
    async def test_generate_calls_model(self):
        """generate() dispatches to the model via thread pool."""
        fake_wav = torch.randn(1, 24000)

        with patch("hermes.services.tts.ChatterboxTurboTTS") as MockModel:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = fake_wav
            mock_instance.sr = 24000
            MockModel.from_pretrained.return_value = mock_instance

            service = ChatterboxTTSService(device="cpu")
            audio_bytes = await service.generate("Hello there")

        mock_instance.generate.assert_called_once()
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

    @pytest.mark.asyncio
    async def test_generate_with_voice_prompt(self):
        """generate() passes audio_prompt_path to model."""
        fake_wav = torch.randn(1, 24000)

        with patch("hermes.services.tts.ChatterboxTurboTTS") as MockModel:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = fake_wav
            mock_instance.sr = 24000
            MockModel.from_pretrained.return_value = mock_instance

            service = ChatterboxTTSService(device="cpu")
            await service.generate("Hello there", audio_prompt_path="/tmp/ref.wav")

        call_args = mock_instance.generate.call_args
        assert call_args[1].get("audio_prompt_path") == "/tmp/ref.wav" or \
               call_args[0][1] == "/tmp/ref.wav"

    @pytest.mark.asyncio
    async def test_generate_raises_on_no_model(self):
        """generate() raises TTSGenerationError when model is None."""
        with patch("hermes.services.tts.ChatterboxTurboTTS") as MockModel:
            mock_instance = MagicMock()
            mock_instance.sr = 24000
            MockModel.from_pretrained.return_value = mock_instance

            service = ChatterboxTTSService(device="cpu")
            service._model = None

            from hermes.core.exceptions import TTSGenerationError

            with pytest.raises(TTSGenerationError, match="Model not loaded"):
                await service.generate("Test")


class TestResampleAndUlaw:
    """Tests for audio format conversion helpers."""

    def test_resample_to_8khz(self):
        """Resampling 16 kHz PCM to 8 kHz halves the sample count."""
        with patch("hermes.services.tts.ChatterboxTurboTTS") as MockModel:
            mock_instance = MagicMock()
            mock_instance.sr = 24000
            MockModel.from_pretrained.return_value = mock_instance

            service = ChatterboxTTSService(device="cpu")

        # Create 1s of 16 kHz silence (16000 int16 samples)
        pcm = np.zeros(16000, dtype=np.int16).tobytes()
        resampled = service.resample_to_8khz(pcm, orig_sr=16000)

        # 8000 samples * 2 bytes = 16000 bytes
        assert len(resampled) == 16000

    def test_convert_to_ulaw(self):
        """µ-law conversion produces 1 byte per sample."""
        pcm = np.zeros(100, dtype=np.int16).tobytes()  # 200 bytes
        ulaw = ChatterboxTTSService.convert_to_ulaw(pcm)
        assert len(ulaw) == 100  # 1 byte per sample


class TestTTSWorkerPool:
    """Tests for TTSWorkerPool."""

    def test_pool_round_robin(self):
        """Workers are selected round-robin."""
        with patch("hermes.services.tts.ChatterboxTurboTTS") as MockModel:
            mock_instance = MagicMock()
            mock_instance.sr = 24000
            MockModel.from_pretrained.return_value = mock_instance

            pool = TTSWorkerPool(num_workers=2, device_ids=["cpu", "cpu"])

        assert len(pool.workers) == 2
        assert pool._next_worker == 0
