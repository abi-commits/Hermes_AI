"""Tests for Speech-to-Text services."""

import asyncio
import numpy as np
import pytest

from hermes.services.stt import MockSTTService


class TestMockSTTService:
    """Tests for the deterministic STT mock."""

    @pytest.mark.asyncio
    async def test_transcribe_cycles_responses(self):
        """Test that transcribe returns responses in order."""
        service = MockSTTService(responses=["Test transcription"])
        
        # Audio array doesn't matter for mock
        audio = np.zeros(16000, dtype=np.float32)
        
        result = await service.transcribe(audio)
        assert result == "Test transcription"

    @pytest.mark.asyncio
    async def test_stream_transcribe(self):
        """Test that streaming yields responses."""
        responses = ["First", "Second", "Third"]
        service = MockSTTService(responses=responses)
        
        queue = asyncio.Queue()
        # Feed 3 audio arrays
        for _ in range(3):
            await queue.put(np.zeros(1000, dtype=np.float32))
        await queue.put(None)  # Stop sentinel
        
        results = []
        async for transcript in service.stream_transcribe(queue):
            results.append(transcript)
            
        assert results == responses

    def test_default_responses(self):
        """Test that mock has a default response."""
        service = MockSTTService()
        assert len(service.responses) > 0
        assert isinstance(service.responses[0], str)
