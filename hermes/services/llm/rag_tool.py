"""Knowledge base search tool for Gemini function calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai import types

from hermes.services.llm.tools import create_function_tool

if TYPE_CHECKING:
    from hermes.core.adapters import RAGAdapter


def get_rag_tool(rag_adapter: RAGAdapter, metadata_filter: dict[str, Any] | None = None) -> Any:
    """Return a Gemini-compatible tool function bound to a specific call's RAG adapter."""

    @create_function_tool(
        name="search_knowledge_base",
        description=(
            "Search the internal knowledge base for relevant information about "
            "company policies, procedures, products, or general support info. "
            "Use this tool when you need facts to answer a user's question accurately."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query to look up in the knowledge base.",
                ),
            },
            required=["query"],
        ),
    )
    async def search_knowledge_base(query: str) -> str:
        """Execute a RAG search via the call's adapter and return formatted results."""
        try:
            # Uses the adapter's configured timeout and per-call metadata filter
            results = await rag_adapter.retrieve(query, where=metadata_filter)
            if not results:
                return "No relevant information found in the knowledge base."

            formatted_results = "Relevant information found:\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. {result}\n"
            return formatted_results
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    return search_knowledge_base
