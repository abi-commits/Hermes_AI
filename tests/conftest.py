"""Pytest configuration and fixtures."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hermes.main import create_app
from hermes.services.llm import MockLLMService
from hermes.services.stt import MockSTTService
from hermes.services.tts import MockTTSService


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def app() -> AsyncGenerator[FastAPI, None]:
    """Create a test application instance."""
    app = create_app()
    yield app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_stt_service() -> MockSTTService:
    """Create a mock STT service."""
    return MockSTTService(
        responses=["Hello, this is a test transcription.", "How can I help you?"]
    )


@pytest.fixture
def mock_llm_service() -> MockLLMService:
    """Create a mock LLM service."""
    return MockLLMService(
        responses=[
            "I can help you with that. Let me assist you.",
            "Is there anything else you need?",
        ]
    )


@pytest.fixture
def mock_tts_service() -> MockTTSService:
    """Create a mock TTS service."""
    return MockTTSService(duration_seconds=0.5)


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Create a mock WebSocket."""
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture(autouse=True)
def disable_external_calls() -> Generator[None, None, None]:
    """Disable external API calls during tests."""
    with patch("httpx.AsyncClient") as mock:
        mock.return_value.__aenter__ = AsyncMock()
        mock.return_value.__aexit__ = AsyncMock()
        yield


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Sample mu-law audio bytes for testing."""
    # Sample 8kHz mu-law encoded audio
    return b"\xff\xff\xff\xff" * 160  # 20ms of silence at 8kHz


@pytest.fixture
def sample_start_message() -> dict:
    """Sample Twilio start message."""
    return {
        "event": "start",
        "sequenceNumber": 1,
        "start": {
            "callSid": "CA1234567890abcdef",
            "accountSid": "AC1234567890abcdef",
            "streamSid": "MZ1234567890abcdef",
        },
    }


@pytest.fixture
def sample_media_message() -> dict:
    """Sample Twilio media message."""
    import base64

    # Sample mu-law audio (all 0xff is silence in mu-law)
    audio_data = base64.b64encode(b"\xff" * 160).decode()

    return {
        "event": "media",
        "sequenceNumber": 2,
        "streamSid": "MZ1234567890abcdef",
        "media": {
            "track": "inbound",
            "chunk": "1",
            "timestamp": "1234567890",
            "payload": audio_data,
        },
    }


@pytest.fixture
def sample_stop_message() -> dict:
    """Sample Twilio stop message."""
    return {
        "event": "stop",
        "sequenceNumber": 3,
        "streamSid": "MZ1234567890abcdef",
        "stop": {
            "callSid": "CA1234567890abcdef",
            "accountSid": "AC1234567890abcdef",
        },
    }


class TestMetricsCollector:
    """Helper for collecting metrics during tests."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics: dict[str, Any] = {}

    def record(self, name: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get a metric value."""
        return self.metrics.get(name, default)


@pytest.fixture
def metrics_collector() -> TestMetricsCollector:
    """Create a metrics collector for testing."""
    return TestMetricsCollector()
