"""Prompt-related data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemPrompt:
    """Parsed representation of a system prompt YAML file."""

    name: str
    description: str
    system_prompt: str
    temperature: float = 0.7
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class FewShotExample:
    """A single input → output example used for few-shot prompting."""

    input: str
    output: str
    label: str | None = None
