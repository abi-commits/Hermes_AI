"""Tests for call orchestration and failure supervision."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from hermes.core.orchestrator import CallConfig, CallOrchestrator, ServiceBundle
from hermes.models.call import CallState
from hermes.services.llm.base import AbstractLLMService
from hermes.services.rag.base import AbstractRAGService
from hermes.services.stt import MockSTTService
from hermes.services.tts import MockTTSService


class FailingLLMService(AbstractLLMService):
    """LLM stub that fails as soon as generation starts."""

    async def generate(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history=None,
        call_sid: str | None = None,
    ) -> str:
        raise RuntimeError("llm crashed")

    async def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
        conversation_history=None,
        call_sid: str | None = None,
        interruption_check=None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("llm crashed")
        yield ""


class DummyRAGService(AbstractRAGService):
    """No-op RAG stub for orchestrator tests."""

    async def warm_up(self) -> None:
        return None

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> list[str]:
        return []

    async def retrieve_with_timeout(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
        timeout_s: float | None = None,
    ) -> list[str]:
        return []

    async def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        return ids or []

    async def delete_documents(self, ids: list[str]) -> None:
        return None

    async def get_collection_stats(self) -> dict[str, int | str]:
        return {"name": "kb", "count": 0}


@pytest.mark.asyncio
async def test_background_task_failures_terminate_the_call(mock_websocket):
    """A crashing background task should remove the call from the orchestrator."""
    orchestrator = CallOrchestrator(
        ServiceBundle(
            stt_factory=MockSTTService,
            llm_service=FailingLLMService(),
            tts_service=MockTTSService(duration_seconds=0.01),
            rag_service=DummyRAGService(),
        )
    )

    call = await orchestrator.create_call(
        websocket=mock_websocket,
        call_sid="CA-failing",
        stream_sid="MZ-failing",
        account_sid="AC-failing",
    )

    await call.text_out_queue.put("hello")

    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        if orchestrator.get_call(call.call_sid) is None and call.state == CallState.ENDED:
            break
        await asyncio.sleep(0.05)

    assert orchestrator.get_call(call.call_sid) is None
    assert call.state == CallState.ENDED


@pytest.mark.asyncio
async def test_immediate_greeting_is_sent(mock_websocket):
    """Test that if a greeting is provided in config, it is sent to TTS immediately."""
    orchestrator = CallOrchestrator(
        ServiceBundle(
            stt_factory=MockSTTService,
            llm_service=FailingLLMService(), # Won't be reached
            tts_service=MockTTSService(duration_seconds=0.01),
            rag_service=DummyRAGService(),
        )
    )

    greeting_text = "Welcome to Hermes support!"
    config = CallConfig(greeting=greeting_text)

    call = await orchestrator.create_call(
        websocket=mock_websocket,
        call_sid="CA-greeting",
        stream_sid="MZ-greeting",
        account_sid="AC-greeting",
        config=config,
    )

    # Check that greeting was put in the audio_out_queue
    sent_greeting = await asyncio.wait_for(call.audio_out_queue.get(), timeout=1.0)
    assert sent_greeting == greeting_text
    assert call.conversation[0].content == greeting_text
    assert call.conversation[0].role == "assistant"

    await orchestrator.terminate_call(call.call_sid)
