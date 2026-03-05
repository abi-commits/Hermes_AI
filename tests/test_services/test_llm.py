"""Tests for LLM service."""

import pytest

from hermes.models.llm import InterruptMarker
from hermes.services.llm import MockGeminiLLMService


class TestMockGeminiLLMService:
    """Tests for Mock Gemini LLM service."""

    @pytest.mark.asyncio
    async def test_mock_generate_returns_response(self):
        """Test that mock service returns response."""
        service = MockGeminiLLMService(responses=["Test response"])

        result = await service.generate(
            prompt="Test query",
            context="Some context",
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_mock_generate_cycles_responses(self):
        """Test that mock service cycles through responses."""
        service = MockGeminiLLMService(responses=["A", "B", "C"])

        results = []
        for _ in range(4):
            result = await service.generate(prompt="query")
            results.append(result)

        assert results == ["A", "B", "C", "A"]

    @pytest.mark.asyncio
    async def test_mock_stream_sentences_yields_chunks(self):
        """Test that mock stream yields chunks."""
        service = MockGeminiLLMService(responses=["Hello world"])

        chunks = []
        async for chunk in service.stream_sentences(prompt="query"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(str(c) for c in chunks).strip() == "Hello world"

    @pytest.mark.asyncio
    async def test_mock_stream_no_interrupt_marker_by_default(self):
        """Test that no InterruptMarker appears without an interruption check."""
        service = MockGeminiLLMService(responses=["Quick reply"])

        async for chunk in service.stream_sentences(prompt="q"):
            assert not isinstance(chunk, InterruptMarker)

