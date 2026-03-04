"""Tests for STT service."""

import pytest
import torch
import numpy as np

from hermes.services.stt import MockSTTService


class TestMockSTTService:
    """Tests for Mock STT service."""

    @pytest.mark.asyncio
    async def test_mock_transcribe_returns_text(self):
        """Test that mock service returns transcription."""
        service = MockSTTService(responses=["Test transcription"])

        # Create dummy audio
        audio = torch.zeros(16000)  # 1 second of silence

        result = await service.transcribe(audio)

        assert result == "Test transcription"

    @pytest.mark.asyncio
    async def test_mock_transcribe_cycles_responses(self):
        """Test that mock service cycles through responses."""
        service = MockSTTService(responses=["First", "Second", "Third"])

        audio = torch.zeros(16000)

        results = []
        for _ in range(4):
            result = await service.transcribe(audio)
            results.append(result)

        assert results == ["First", "Second", "Third", "First"]

    @pytest.mark.asyncio
    async def test_mock_connect_disconnect(self):
        """Test mock connect and disconnect."""
        service = MockSTTService()

        await service.connect()
        await service.disconnect()

        # Should complete without error
