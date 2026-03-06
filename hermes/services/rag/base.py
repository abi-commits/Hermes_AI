"""Abstract base class for RAG (Retrieval-Augmented Generation) services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractRAGService(ABC):
    """Interface contract for all RAG / vector-database service implementations."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def warm_up(self) -> None:
        """Connect to the backend; call at startup to eliminate cold-start latency."""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return the top-*k* most relevant document chunks for *query*."""

    @abstractmethod
    async def retrieve_with_timeout(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> list[str]:
        """Retrieve with a time budget; return ``[]`` on timeout."""

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    @abstractmethod
    async def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add *texts* to the vector store and return their IDs."""

    @abstractmethod
    async def delete_documents(self, ids: list[str]) -> None:
        """Remove documents by ID."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics (name and count) about the backing collection."""
