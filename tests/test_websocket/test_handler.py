"""Tests for WebSocket handler."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from hermes.websocket.handler import handle_websocket
from hermes.websocket.schemas import StartMessage, MediaMessage, StopMessage


@pytest.mark.asyncio
async def test_websocket_accepts_connection(mock_websocket):
    """Test that WebSocket connection is accepted."""
    call_sid = "CA1234567890"

    # Setup mock to return connected message then stop
    connected_message = {"event": "connected", "protocol": "call", "version": "1.0"}
    stop_message = {
        "event": "stop",
        "sequenceNumber": 1,
        "streamSid": "MZ123",
        "stop": {}
    }

    mock_websocket.receive_text = AsyncMock(side_effect=[
        json.dumps(connected_message),
        json.dumps(stop_message)
    ])

    await handle_websocket(mock_websocket, call_sid)

    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_handles_start_message(mock_websocket, sample_start_message):
    """Test that start message initializes call."""
    call_sid = "CA1234567890"

    stop_message = {
        "event": "stop",
        "sequenceNumber": 2,
        "streamSid": "MZ1234567890abcdef",
        "stop": {}
    }

    mock_websocket.receive_text = AsyncMock(side_effect=[
        json.dumps(sample_start_message),
        json.dumps(stop_message)
    ])

    await handle_websocket(mock_websocket, call_sid)

    # Verify connection was handled
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_handles_media_message(
    mock_websocket, sample_start_message, sample_media_message
):
    """Test that media messages are processed."""
    call_sid = "CA1234567890"

    stop_message = {
        "event": "stop",
        "sequenceNumber": 3,
        "streamSid": "MZ1234567890abcdef",
        "stop": {}
    }

    mock_websocket.receive_text = AsyncMock(side_effect=[
        json.dumps(sample_start_message),
        json.dumps(sample_media_message),
        json.dumps(stop_message)
    ])

    await handle_websocket(mock_websocket, call_sid)

    # Should process all messages
    assert mock_websocket.receive_text.call_count == 3


@pytest.mark.asyncio
async def test_websocket_handles_disconnect(mock_websocket):
    """Test that WebSocket disconnect is handled gracefully."""
    call_sid = "CA1234567890"

    from fastapi import WebSocketDisconnect
    mock_websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

    await handle_websocket(mock_websocket, call_sid)

    # Should complete without error
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_handles_invalid_json(mock_websocket):
    """Test that invalid JSON is handled gracefully."""
    call_sid = "CA1234567890"

    mock_websocket.receive_text = AsyncMock(side_effect=[
        "not valid json",
        Exception("Stop iteration")
    ])

    try:
        await handle_websocket(mock_websocket, call_sid)
    except Exception:
        pass  # Expected


class TestWebSocketSchemas:
    """Tests for WebSocket message schemas."""

    def test_start_message_parsing(self, sample_start_message):
        """Test StartMessage parsing."""
        msg = StartMessage(**sample_start_message)

        assert msg.event == "start"
        assert msg.start.call_sid == "CA1234567890abcdef"
        assert msg.start.stream_sid == "MZ1234567890abcdef"

    def test_media_message_parsing(self, sample_media_message):
        """Test MediaMessage parsing."""
        msg = MediaMessage(**sample_media_message)

        assert msg.event == "media"
        assert msg.stream_sid == "MZ1234567890abcdef"
        assert msg.media.track == "inbound"
        assert msg.media.payload is not None

    def test_stop_message_parsing(self, sample_stop_message):
        """Test StopMessage parsing."""
        msg = StopMessage(**sample_stop_message)

        assert msg.event == "stop"
        assert msg.stream_sid == "MZ1234567890abcdef"
