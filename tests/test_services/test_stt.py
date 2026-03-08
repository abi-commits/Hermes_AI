"""Tests for Speech-to-Text services."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from hermes.services.stt import DeepgramSTTService, MockSTTService


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


class TestDeepgramTranscriptAggregation:
    """Tests for finalized-utterance gating in the Deepgram adapter."""

    def test_ignores_interim_results(self):
        """Interim transcripts should not reach the LLM pipeline."""
        service = DeepgramSTTService()
        buffered: list[str] = []
        message = SimpleNamespace(
            is_final=False,
            speech_final=False,
            channel=SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="hello there")]
            ),
        )

        utterance = service._consume_result(message, buffered)

        assert utterance is None
        assert buffered == []

    def test_emits_once_when_speech_turn_finishes(self):
        """Final transcript chunks should be combined into one utterance."""
        service = DeepgramSTTService()
        buffered: list[str] = []
        first = SimpleNamespace(
            is_final=True,
            speech_final=False,
            channel=SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="I need help")]
            ),
        )
        second = SimpleNamespace(
            is_final=True,
            speech_final=True,
            channel=SimpleNamespace(
                alternatives=[SimpleNamespace(transcript="with billing")]
            ),
        )

        assert service._consume_result(first, buffered) is None
        assert buffered == ["I need help"]
        assert service._consume_result(second, buffered) == "I need help with billing"
        assert buffered == []

    def test_flushes_buffered_segments_on_utterance_end(self):
        """Utterance-end events should release any buffered final segments."""
        buffered = ["Can you", "repeat that"]

        assert DeepgramSTTService._flush_segments(buffered) == "Can you repeat that"
        assert buffered == []

    def test_live_options_use_configured_turn_detection(self):
        """Live transcription options should respect the latency tuning settings."""
        service = DeepgramSTTService()
        service.settings.deepgram_utterance_end_ms = 650

        assert service._live_transcription_options()["utterance_end_ms"] == "650"

    @pytest.mark.asyncio
    async def test_stream_live_events_yields_transcript_without_waiting_for_more_audio(self):
        """A finalized transcript should be yielded as soon as it is queued."""
        service = DeepgramSTTService()
        audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        transcript_queue: asyncio.Queue[str | None] = asyncio.Queue()
        sent_frames: list[bytes] = []

        async def send_audio(frame: bytes) -> None:
            sent_frames.append(frame)

        async def produce() -> None:
            await audio_queue.put(np.zeros(16, dtype=np.float32))
            await asyncio.sleep(0.01)
            await transcript_queue.put("hello there")
            await asyncio.sleep(0.01)
            await audio_queue.put(None)

        producer = asyncio.create_task(produce())
        results: list[str] = []

        async for transcript in service._stream_live_events(
            audio_queue,
            transcript_queue,
            send_audio,
        ):
            results.append(transcript)

        await producer

        assert results == ["hello there"]
        assert sent_frames
