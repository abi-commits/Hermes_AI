"""Prompt Manager — load, cache, and render prompt templates.

Prompts are stored as YAML (system personas) or plain-text Jinja2 templates
(user / few-shot) under the ``hermes/prompts/`` directory tree.

Directory layout::

    hermes/prompts/
    ├── system/          # YAML persona files (system_prompt + gen params)
    │   ├── default.yaml
    │   ├── polite.yaml
    │   ├── technical.yaml
    │   └── concise.yaml
    ├── user/            # Jinja2 .txt templates for user-turn formatting
    │   └── query_with_context.txt
    ├── few_shot/        # YAML few-shot example banks
    │   └── examples.yaml
    └── prompt_manager.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import Any

import yaml

from hermes.models.prompts import FewShotExample, SystemPrompt

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


# ======================================================================
# Manager
# ======================================================================


class PromptManager:
    """Central registry for system personas, user templates, and few-shot banks.

    All templates are lazily loaded from disk on first access and cached in
    memory.  Re-instantiating the manager or calling ``reload()`` clears
    the cache.

    Usage::

        pm = PromptManager()
        persona = pm.get_system_prompt("polite")
        user_msg = pm.render_user_prompt(
            "query_with_context",
            query="What is my balance?",
            context="Account #1234 has a balance of $42.00.",
        )
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or _PROMPTS_DIR
        self._system_cache: dict[str, SystemPrompt] = {}
        self._user_cache: dict[str, str] = {}
        self._few_shot_cache: dict[str, list[FewShotExample]] = {}

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    def get_system_prompt(self, name: str = "default") -> SystemPrompt:
        """Return a ``SystemPrompt`` by persona *name*.

        Raises ``FileNotFoundError`` if the YAML file doesn't exist.
        """
        if name not in self._system_cache:
            self._system_cache[name] = self._load_system_prompt(name)
        return self._system_cache[name]

    def list_system_prompts(self) -> list[str]:
        """Return names of all available system-prompt personas."""
        system_dir = self._dir / "system"
        if not system_dir.is_dir():
            return []
        return sorted(p.stem for p in system_dir.glob("*.yaml"))

    # ------------------------------------------------------------------
    # User templates
    # ------------------------------------------------------------------

    def render_user_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Render a user-turn template with the given variables.

        Templates use Python ``string.Template`` syntax (``$var`` or
        ``${var}``).  Unknown placeholders are left as-is.

        Raises ``FileNotFoundError`` if the template doesn't exist.
        """
        if template_name not in self._user_cache:
            self._user_cache[template_name] = self._load_user_template(template_name)

        return Template(self._user_cache[template_name]).safe_substitute(**kwargs)

    def list_user_templates(self) -> list[str]:
        """Return names of all available user-turn templates."""
        user_dir = self._dir / "user"
        if not user_dir.is_dir():
            return []
        return sorted(p.stem for p in user_dir.glob("*.txt"))

    # ------------------------------------------------------------------
    # Few-shot examples
    # ------------------------------------------------------------------

    def get_few_shot_examples(
        self, bank_name: str = "examples", label: str | None = None
    ) -> list[FewShotExample]:
        """Return few-shot examples, optionally filtered by *label*."""
        if bank_name not in self._few_shot_cache:
            self._few_shot_cache[bank_name] = self._load_few_shot(bank_name)

        examples = self._few_shot_cache[bank_name]
        if label is not None:
            examples = [e for e in examples if e.label == label]
        return examples

    def format_few_shot_block(
        self, bank_name: str = "examples", label: str | None = None
    ) -> str:
        """Return a pre-formatted few-shot examples block ready for injection."""
        examples = self.get_few_shot_examples(bank_name, label)
        if not examples:
            return ""

        lines: list[str] = ["Here are some examples:", ""]
        for ex in examples:
            lines.append(f"User: {ex.input}")
            lines.append(f"Assistant: {ex.output}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Clear all caches so prompts are re-read from disk on next access."""
        self._system_cache.clear()
        self._user_cache.clear()
        self._few_shot_cache.clear()
        logger.info("Prompt caches cleared")

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_system_prompt(self, name: str) -> SystemPrompt:
        path = self._dir / "system" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return SystemPrompt(
            name=data.get("name", name),
            description=data.get("description", ""),
            system_prompt=data["system_prompt"].strip(),
            temperature=float(data.get("temperature", 0.7)),
            max_output_tokens=int(data.get("max_output_tokens", 2048)),
        )

    def _load_user_template(self, name: str) -> str:
        path = self._dir / "user" / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"User template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _load_few_shot(self, bank_name: str) -> list[FewShotExample]:
        path = self._dir / "few_shot" / f"{bank_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Few-shot bank not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        examples_raw = data.get("examples", [])
        return [
            FewShotExample(
                input=ex["input"],
                output=ex["output"],
                label=ex.get("label"),
            )
            for ex in examples_raw
        ]
