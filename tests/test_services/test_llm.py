"""Tests for LLM service."""

import pytest

from hermes.models.llm import ConversationTurn, InterruptMarker, LLMConfig
from hermes.services.llm import GeminiLLMService, MockGeminiLLMService


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
        async for chunk in service.stream_sentences(prompt="query", conversation_history=[]):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(str(c) for c in chunks).strip() == "Hello world"

    @pytest.mark.asyncio
    async def test_mock_stream_no_interrupt_marker_by_default(self):
        """Test that no InterruptMarker appears without an interruption check."""
        service = MockGeminiLLMService(responses=["Quick reply"])

        async for chunk in service.stream_sentences(prompt="q", conversation_history=[]):
            assert not isinstance(chunk, InterruptMarker)

    @pytest.mark.asyncio
    async def test_mock_stream_with_interruption(self):
        """Test that mock stream yields InterruptMarker when interrupted."""
        service = MockGeminiLLMService(responses=["This is a long response"])

        interrupted = False
        call_count = 0

        def interruption_check():
            nonlocal call_count
            call_count += 1
            return call_count > 2  # Interrupt after 2 chunks

        chunks = []
        async for chunk in service.stream_sentences(
            prompt="query", 
            conversation_history=[],
            interruption_check=interruption_check
        ):
            chunks.append(chunk)
            if isinstance(chunk, InterruptMarker):
                interrupted = True
                break

        assert interrupted


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = LLMConfig()
        assert config.model_name == "gemini-2.5-flash"
        assert config.temperature == 0.7
        assert config.max_output_tokens == 2048
        assert config.timeout_s == 60.0

    def test_custom_config(self):
        """Test custom config values."""
        config = LLMConfig(
            model_name="gemini-2.5-pro",
            temperature=0.3,
            max_output_tokens=1024,
            timeout_s=30.0,
        )
        assert config.model_name == "gemini-2.5-pro"
        assert config.temperature == 0.3
        assert config.timeout_s == 30.0


class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_prompt_with_query_only(self):
        """Test prompt with just a query."""
        service = MockGeminiLLMService()
        result = service._build_prompt("What is Hermes?", None, None)
        assert "User: What is Hermes?" in result
        assert "Assistant:" in result

    def test_prompt_with_context(self):
        """Test prompt with RAG context."""
        service = MockGeminiLLMService()
        result = service._build_prompt("query", "relevant docs here", None)
        assert "Context information:" in result
        assert "relevant docs here" in result

    def test_prompt_with_history(self):
        """Test prompt with conversation history."""
        service = MockGeminiLLMService()
        history = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ]
        result = service._build_prompt("follow up", None, history)
        assert "Conversation history:" in result
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result

    def test_prompt_with_interrupted_turn(self):
        """Test prompt marks interrupted turns."""
        service = MockGeminiLLMService()
        history = [
            ConversationTurn(
                role="assistant", content="I was saying...", interrupted=True
            ),
        ]
        result = service._build_prompt("go on", None, history)
        assert "[INTERRUPTED]" in result


class TestStreamingChunking:
    """Tests for low-latency streaming chunk extraction."""

    def test_pop_ready_fragment_prefers_complete_sentences(self):
        """Sentence-ending punctuation should flush immediately."""
        fragment, remainder = GeminiLLMService._pop_ready_fragment(
            "Thanks for calling. Let me check that for you"
        )

        assert fragment == "Thanks for calling."
        assert remainder == "Let me check that for you"

    def test_pop_ready_fragment_flushes_long_clauses(self):
        """Long comma-delimited clauses should flush before sentence end."""
        fragment, remainder = GeminiLLMService._pop_ready_fragment(
            "Let me pull up your account details and confirm the latest payment, "
            "then I can walk you through the next step"
        )

        assert fragment == "Let me pull up your account details and confirm the latest payment,"
        assert remainder == "then I can walk you through the next step"

    def test_pop_ready_fragment_hard_flushes_long_unpunctuated_buffers(self):
        """Very long streams without punctuation should still flush on whitespace."""
        fragment, remainder = GeminiLLMService._pop_ready_fragment(
            "this response keeps streaming without punctuation so we should still hand off "
            "a meaningful fragment to tts before the model finally decides to stop"
        )

        assert fragment is not None
        assert len(fragment) >= 80
        assert remainder
