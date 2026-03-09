"""Regression tests for the Modal-backed remote TTS adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes.services.tts.modal_remote import ModalRemoteTTSService


async def _async_stream(*chunks: bytes):
    """Yield the provided chunks asynchronously."""
    for chunk in chunks:
        yield chunk


class FakeRemoteCall:
    """Callable test double that intentionally does not expose `.aio`."""

    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


@pytest.mark.asyncio
async def test_generate_stream_uses_remote_gen_aio_when_available():
    """Streaming should prefer Modal's async generator surface."""
    service = ModalRemoteTTSService("hermes-tts", "RemoteWorker")
    remote_gen = SimpleNamespace(aio=MagicMock(return_value=_async_stream(b"a", b"b")))
    instance = SimpleNamespace(
        generate_stream=SimpleNamespace(remote_gen=remote_gen),
    )
    service._get_remote_instance = AsyncMock(return_value=instance)  # type: ignore[method-assign]

    chunks = [chunk async for chunk in service.generate_stream("hello", chunk_size=32)]

    assert chunks == [b"a", b"b"]
    remote_gen.aio.assert_called_once_with(
        text="hello",
        audio_prompt_path=None,
        embed_watermark=True,
        chunk_size=32,
    )


@pytest.mark.asyncio
async def test_generate_stream_falls_back_to_remote_gen_without_aio():
    """Streaming should still work when the Modal client exposes only .remote_gen()."""
    service = ModalRemoteTTSService("hermes-tts", "RemoteWorker")
    remote_gen = FakeRemoteCall([b"a", bytearray(b"b")])
    instance = SimpleNamespace(
        generate_stream=SimpleNamespace(remote_gen=remote_gen),
    )
    service._get_remote_instance = AsyncMock(return_value=instance)  # type: ignore[method-assign]

    chunks = [chunk async for chunk in service.generate_stream("fallback")]

    assert chunks == [b"a", b"b"]
    assert remote_gen.calls == [{
        "text": "fallback",
        "audio_prompt_path": None,
        "embed_watermark": True,
        "chunk_size": 50,
    }]


@pytest.mark.asyncio
async def test_generate_falls_back_to_remote_without_aio():
    """Unary generation should tolerate Modal clients that only expose .remote()."""
    service = ModalRemoteTTSService("hermes-tts", "RemoteWorker")
    remote = FakeRemoteCall(b"pcm-data")
    instance = SimpleNamespace(
        generate=SimpleNamespace(remote=remote),
    )
    service._get_remote_instance = AsyncMock(return_value=instance)  # type: ignore[method-assign]

    result = await service.generate("hello")

    assert result == b"pcm-data"
    assert remote.calls == [{
        "text": "hello",
        "audio_prompt_path": None,
        "embed_watermark": True,
    }]


@pytest.mark.asyncio
async def test_ping_falls_back_to_remote_without_aio():
    """Health checks should not depend on one exact Modal client dialect."""
    service = ModalRemoteTTSService("hermes-tts", "RemoteWorker")
    remote = FakeRemoteCall(24_000)
    instance = SimpleNamespace(
        get_sample_rate=SimpleNamespace(remote=remote),
    )
    service._get_remote_instance = AsyncMock(return_value=instance)  # type: ignore[method-assign]

    assert await service.ping() is True
    assert remote.calls == [{}]
