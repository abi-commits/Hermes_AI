"""Retrieval-Augmented Generation service."""

from typing import TYPE_CHECKING

import structlog

from config import get_settings
from hermes.core.exceptions import RAGError
from hermes.services.vector_db import VectorDB

if TYPE_CHECKING:
    import httpx

logger = structlog.get_logger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service.

    This class handles retrieval of relevant documents from a vector
    database to augment LLM responses with domain knowledge.
    """

    def __init__(
        self,
        vector_db: VectorDB | None = None,
        http_client: "httpx.AsyncClient | None" = None,
    ) -> None:
        """Initialize the RAG service.

        Args:
            vector_db: Vector database instance. If None, creates a new one.
            http_client: Shared HTTP client for connection pooling.
        """
        self.settings = get_settings()
        self.vector_db = vector_db or VectorDB(http_client=http_client)
        self._logger = structlog.get_logger(__name__)
        self._http_client = http_client

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve. Uses settings default if None.
            threshold: Minimum similarity score. Uses settings default if None.

        Returns:
            List of relevant documents with metadata.
        """
        top_k = top_k or self.settings.rag_top_k
        threshold = threshold or self.settings.rag_similarity_threshold

        try:
            results = await self.vector_db.search(
                query=query,
                n_results=top_k,
            )

            # Filter by threshold
            filtered_results = [
                result
                for result in results
                if result.get("score", 0) >= threshold
            ]

            self._logger.debug(
                "rag_retrieved",
                query=query[:50],
                results_count=len(filtered_results),
            )

            return filtered_results

        except Exception as e:
            self._logger.error("rag_retrieval_failed", error=str(e), query=query[:50])
            raise RAGError(f"Retrieval failed: {e}")

    def format_context(self, documents: list[dict]) -> str:
        """Format retrieved documents as context string.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        if not documents:
            return ""

        context_parts = ["Relevant information:"]

        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")

            context_parts.append(f"\n[{i}] Source: {source}\n{content}")

        return "\n".join(context_parts)

    async def query_with_context(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Query with RAG and return context documents.

        This is a helper method that retrieves documents and formats
        them for use with an LLM.

        Args:
            query: The user query.
            system_prompt: Optional system prompt template.

        Returns:
            Tuple of (formatted context, raw documents).
        """
        documents = await self.retrieve(query)
        context = self.format_context(documents)

        return context, documents
