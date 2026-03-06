"""Call-related data models and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any


class CallState(Enum):
    """States in the call state machine."""

    IDLE = auto()
    CONNECTING = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    DISCONNECTING = auto()
    ENDED = auto()


@dataclass
class ConversationTurn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
