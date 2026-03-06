"""Call management REST endpoints."""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from hermes.websocket.manager import connection_manager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/calls", tags=["calls"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ConversationTurnResponse(BaseModel):
    """A single conversation turn."""

    role: str
    content: str
    timestamp: datetime


class CallDetailResponse(BaseModel):
    """Detailed call info including full conversation history."""

    call_sid: str
    stream_sid: str
    state: str
    duration_seconds: float
    started_at: datetime | None
    account_sid: str
    total_turns: int
    conversation: list[ConversationTurnResponse]


class CallSummaryResponse(BaseModel):
    """Brief call info for list responses."""

    call_sid: str
    stream_sid: str
    state: str
    duration_seconds: float
    started_at: datetime | None
    total_turns: int


class ActiveCallsResponse(BaseModel):
    """Response for the list-calls endpoint."""

    total: int
    calls: list[CallSummaryResponse]


class EndCallResponse(BaseModel):
    """Response after requesting a call to end."""

    call_sid: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ActiveCallsResponse,
    summary="List active calls",
    description="Returns all currently active voice calls.",
)
async def list_calls() -> ActiveCallsResponse:
    """Return all active calls with summary information."""
    calls = []
    for call_sid, call in connection_manager._active_calls.items():
        calls.append(
            CallSummaryResponse(
                call_sid=call_sid,
                stream_sid=call.stream_sid,
                state=call.state.name,
                duration_seconds=round(call.duration_seconds, 2),
                started_at=call.started_at,
                total_turns=len(call.conversation),
            )
        )

    return ActiveCallsResponse(total=len(calls), calls=calls)


@router.get(
    "/{call_sid}",
    response_model=CallDetailResponse,
    summary="Get call details",
    description="Returns full details for a specific active call including conversation history.",
)
async def get_call(call_sid: str) -> CallDetailResponse:
    """Return full detail for a single active call or 404."""
    call = connection_manager.get_call(call_sid)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active call with SID '{call_sid}'",
        )

    conversation = [
        ConversationTurnResponse(
            role=turn.role,
            content=turn.content,
            timestamp=turn.timestamp,
        )
        for turn in call.conversation
    ]

    return CallDetailResponse(
        call_sid=call.call_sid,
        stream_sid=call.stream_sid,
        state=call.state.name,
        duration_seconds=round(call.duration_seconds, 2),
        started_at=call.started_at,
        account_sid=call.account_sid,
        total_turns=len(call.conversation),
        conversation=conversation,
    )


@router.delete(
    "/{call_sid}",
    response_model=EndCallResponse,
    summary="End a call",
    description="Gracefully terminates an active call.",
)
async def end_call(call_sid: str) -> EndCallResponse:
    """Terminate an active call or 404."""
    call = connection_manager.get_call(call_sid)
    if call is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active call with SID '{call_sid}'",
        )

    stream_sid = call.stream_sid
    await connection_manager.disconnect(stream_sid)

    logger.info("call_terminated_via_api", call_sid=call_sid)

    return EndCallResponse(
        call_sid=call_sid,
        message="Call terminated successfully",
    )
