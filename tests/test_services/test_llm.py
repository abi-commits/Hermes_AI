"""Tests for LLM service."""

import pytest

from hermes.services.llm import MockLLMService


class TestMockLLMService:
    """Tests for Mock LLM service."""

    @pytest.mark.asyncio
    async def test_mock_generate_returns_response(self):
        """Test that mock service returns response."""
        service = MockLLMService(responses=["Test response"])

        result = await service.generate(
            context="Some context",
            query="Test query"
        )

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_mock_generate_cycles_responses(self):
        """Test that mock service cycles through responses."""
        service = MockLLMService(responses=["A", "B", "C"])

        results = []
        for _ in range(4):
            result = await service.generate("ctx", "query")
            results.append(result)

        assert results == ["A", "B", "C", "A"]

    @pytest.mark.asyncio
    async def test_mock_generate_stream_yields_chunks(self):
        """Test that mock stream yields chunks."""
        service = MockLLMService(responses=["Hello world"])

        chunks = []
        async for chunk in service.generate_stream("ctx", "query"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks).strip() == "Hello world"
