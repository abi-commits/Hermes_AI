"""Global test configuration and fixtures."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# 1. Mock chatterbox and its transitive dependencies that cause issues
mock_chatterbox = MagicMock()
mock_chatterbox_tts = MagicMock()
mock_chatterbox.tts = mock_chatterbox_tts

sys.modules["chatterbox"] = mock_chatterbox
sys.modules["chatterbox.tts"] = mock_chatterbox_tts
sys.modules["perth"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["onnx"] = MagicMock()
sys.modules["ml_dtypes"] = MagicMock()
sys.modules["s3tokenizer"] = MagicMock()


@pytest.fixture
def mock_websocket():
    """Minimal FastAPI WebSocket double used by websocket handler tests."""
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.close = AsyncMock()
    websocket.app = SimpleNamespace(state=SimpleNamespace())
    return websocket


@pytest.fixture
def sample_start_message() -> dict:
    """Representative Twilio start event payload."""
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
    """Representative Twilio media event payload."""
    return {
        "event": "media",
        "sequenceNumber": 2,
        "streamSid": "MZ1234567890abcdef",
        "media": {
            "track": "inbound",
            "chunk": "1",
            "timestamp": "20",
            "payload": "dGVzdA==",
        },
    }


@pytest.fixture
def sample_stop_message() -> dict:
    """Representative Twilio stop event payload."""
    return {
        "event": "stop",
        "sequenceNumber": 3,
        "streamSid": "MZ1234567890abcdef",
        "stop": {},
    }
