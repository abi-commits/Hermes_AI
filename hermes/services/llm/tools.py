"""Function-tool decorator helper for Gemini function calling."""

from __future__ import annotations

from typing import Callable

from google.genai import types


def create_function_tool(
    name: str,
    description: str,
    parameters: types.Schema | None = None,
) -> Callable:
    """Decorator factory that attaches a Gemini ``FunctionDeclaration`` to a callable."""

    def decorator(func: Callable) -> Callable:
        func.function_declaration = types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )
        return func

    return decorator
