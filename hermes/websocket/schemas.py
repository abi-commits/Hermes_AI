"""Pydantic models for Twilio WebSocket messages."""

from typing import Literal

from pydantic import BaseModel, Field


class StreamParameters(BaseModel):
    """Parameters for the media stream."""

    call_sid: str = Field(..., alias="callSid", description="Twilio call SID")
    account_sid: str = Field(..., alias="accountSid", description="Twilio account SID")
    stream_sid: str = Field(..., alias="streamSid", description="Media stream SID")
    custom_parameters: dict[str, str] = Field(
        default_factory=dict,
        alias="customParameters",
        description="Custom parameters passed from TwiML <Parameter> tags",
    )

    model_config = {"populate_by_name": True}


class StartMessage(BaseModel):
    """Twilio start event message."""

    event: Literal["start"]
    sequence_number: int = Field(..., alias="sequenceNumber")
    start: StreamParameters

    model_config = {"populate_by_name": True}


class MediaPayload(BaseModel):
    """Media payload from Twilio."""

    track: Literal["inbound", "outbound"]
    chunk: str
    timestamp: str
    payload: str  # Base64 encoded audio


class MediaMessage(BaseModel):
    """Twilio media event message."""

    event: Literal["media"]
    sequence_number: int = Field(..., alias="sequenceNumber")
    media: MediaPayload
    stream_sid: str = Field(..., alias="streamSid")

    model_config = {"populate_by_name": True}


class StopMessage(BaseModel):
    """Twilio stop event message."""

    event: Literal["stop"]
    sequence_number: int = Field(..., alias="sequenceNumber")
    stop: dict
    stream_sid: str = Field(..., alias="streamSid")

    model_config = {"populate_by_name": True}


class ConnectedMessage(BaseModel):
    """Twilio connected event message."""

    event: Literal["connected"]
    protocol: str
    version: str


class DtmfMessage(BaseModel):
    """Twilio DTMF (touch tone) event message."""

    event: Literal["dtmf"]
    stream_sid: str = Field(..., alias="streamSid")
    sequence_number: int = Field(..., alias="sequenceNumber")
    dtmf: dict

    model_config = {"populate_by_name": True}


class MarkMessage(BaseModel):
    """Twilio mark event message (for TTS timing)."""

    event: Literal["mark"]
    stream_sid: str = Field(..., alias="streamSid")
    sequence_number: int = Field(..., alias="sequenceNumber")
    mark: dict

    model_config = {"populate_by_name": True}


class ClearMessage(BaseModel):
    """Twilio clear event message."""

    event: Literal["clear"]
    stream_sid: str = Field(..., alias="streamSid")

    model_config = {"populate_by_name": True}


# Union type for incoming Twilio messages
TwilioMessage = StartMessage | MediaMessage | StopMessage | ConnectedMessage | DtmfMessage


# Outbound messages to Twilio
class TwilioMediaResponse(BaseModel):
    """Send audio back to Twilio."""

    event: Literal["media"]
    stream_sid: str = Field(..., alias="streamSid")
    media: dict  # Contains "payload" with base64 audio

    model_config = {"populate_by_name": True}


class TwilioMarkResponse(BaseModel):
    """Send mark event to Twilio for timing."""

    event: Literal["mark"]
    stream_sid: str = Field(..., alias="streamSid")
    mark: dict  # Contains "name" for the mark

    model_config = {"populate_by_name": True}


class TwilioClearResponse(BaseModel):
    """Send clear event to Twilio to stop audio playback."""

    event: Literal["clear"]
    stream_sid: str = Field(..., alias="streamSid")

    model_config = {"populate_by_name": True}
