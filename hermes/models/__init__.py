"""Hermes data models.

Centralised package for all dataclasses, enums, and value objects used
across the application.  Keeping models separate from service logic
improves testability, prevents circular imports, and makes the domain
language explicit.
"""

from hermes.models.call import CallState, ConversationTurn
from hermes.models.llm import (
    InterruptMarker,
    LLMConfig,
    LLMGenerationError,
    ResponseModality,
    StreamingMode,
)
from hermes.models.llm import ConversationTurn as LLMConversationTurn
from hermes.models.prompts import FewShotExample, SystemPrompt

__all__ = [
    # call
    "CallState",
    "ConversationTurn",
    # llm
    "InterruptMarker",
    "LLMConfig",
    "LLMConversationTurn",
    "LLMGenerationError",
    "ResponseModality",
    "StreamingMode",
    # prompts
    "FewShotExample",
    "SystemPrompt",
]
