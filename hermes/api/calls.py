"""API endpoints for monitoring and managing active voice calls."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hermes.core.call import Call

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/calls")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConversationTurnSchema(BaseModel):
    """A single turn in a conversation history."""

    role: str = Field(..., example="user")
    content: str = Field(..., example="Hello, I need help with my account.")
    timestamp: datetime


class CallSummarySchema(BaseModel):
    """Brief summary of an active call."""

    call_sid: str = Field(..., example="CA12345")
    stream_sid: str = Field(..., example="MZ67890")
    state: str = Field(..., example="SPEAKING")
    duration_seconds: float = Field(..., example=45.2)
    started_at: datetime | None
    total_turns: int = Field(..., example=4)


class CallDetailSchema(CallSummarySchema):
    """Full details of an active call including transcript."""

    account_sid: str = Field(..., example="ACabcde")
    conversation: list[ConversationTurnSchema]


class ActiveCallsResponse(BaseModel):
    """Response containing all active calls."""

    total: int = Field(..., example=1)
    calls: list[CallSummarySchema]


class EndCallResponse(BaseModel):
    """Confirmation response after terminating a call."""

    call_sid: str
    message: str = "Call terminated successfully"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ActiveCallsResponse,
    summary="List Active Calls",
    description="Returns a list of all currently active voice calls and their basic status.",
)
async def list_calls(request: Request) -> ActiveCallsResponse:
    """Return all active calls from the orchestrator registry."""
    manager = request.app.state.connection_manager
    calls = []
    
    # We iterate over the orchestrator's active calls
    active_map = getattr(request.app.state.orchestrator, "_active_calls", {})
    
    for call_sid, call in active_map.items():
        calls.append(
            CallSummarySchema(
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
    response_model=CallDetailSchema,
    summary="Get Call Details",
    description="Returns full details for a specific active call, including the real-time transcript.",
)
async def get_call(call_sid: str, request: Request) -> CallDetailSchema:
    """Find an active call by SID and return its full state and history."""
    orchestrator = request.app.state.orchestrator
    call: Call | None = orchestrator.get_call(call_sid)
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active call found with SID: {call_sid}",
        )

    conversation = [
        ConversationTurnSchema(
            role=turn.role,
            content=turn.content,
            timestamp=turn.timestamp,
        )
        for turn in call.conversation
    ]

    return CallDetailSchema(
        call_sid=call.call_sid,
        stream_sid=call.stream_sid,
        state=call.state.name,
        duration_seconds=round(call.duration_seconds, 2),
        started_at=call.started_at,
        total_turns=len(call.conversation),
        account_sid=call.account_sid,
        conversation=conversation,
    )


@router.delete(
    "/{call_sid}",
    response_model=EndCallResponse,
    summary="Terminate Call",
    description="Forcefully terminates an active call and closes its WebSocket stream.",
)
async def terminate_call(call_sid: str, request: Request) -> EndCallResponse:
    """Signal the orchestrator to shut down a specific call."""
    orchestrator = request.app.state.orchestrator
    call = orchestrator.get_call(call_sid)
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active call found with SID: {call_sid}",
        )

    await orchestrator.terminate_call(call_sid, reason="api_request")
    logger.info("call_terminated_via_api", call_sid=call_sid)

    return EndCallResponse(call_sid=call_sid)
