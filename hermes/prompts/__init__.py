"""Prompt management for Hermes.

This package provides centralised prompt management with YAML-based templates,
variable substitution, and persona switching.

Typical usage::

    from hermes.prompts import PromptManager

    pm = PromptManager()
    system = pm.get_system_prompt("polite")
    user   = pm.render_user_prompt("query_with_context", query=q, context=ctx)
"""

from hermes.models.prompts import FewShotExample, SystemPrompt
from hermes.prompts.prompt_manager import PromptManager

__all__ = ["FewShotExample", "PromptManager", "SystemPrompt"]
