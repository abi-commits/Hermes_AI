"""API endpoints for managing the RAG knowledge base."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/knowledge")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DocumentIngestRequest(BaseModel):
    """Request to add new documents to the knowledge base."""

    texts: list[str] = Field(..., min_items=1, example=["Hermes is a voice AI."])
    ids: list[str] | None = Field(None, example=["doc_1"])
    metadatas: list[dict[str, Any]] | None = Field(None, example=[{"source": "wiki"}])


class DocumentIngestResponse(BaseModel):
    """Response after ingesting documents."""

    count: int
    ids: list[str]


class DocumentDeleteRequest(BaseModel):
    """Request to remove documents from the knowledge base."""

    ids: list[str] = Field(..., min_items=1, example=["doc_1"])


class QueryRequest(BaseModel):
    """Request to test retrieval from the knowledge base."""

    query: str = Field(..., example="What is Hermes?")
    k: int = Field(5, ge=1, le=20)


class QueryResultSchema(BaseModel):
    """A single retrieved document chunk."""

    content: str
    metadata: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    """Response containing retrieved chunks."""

    query: str
    results: list[QueryResultSchema]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    summary="Get Collection Stats",
    description="Returns metadata and document counts for the vector database collection.",
)
async def get_stats(request: Request) -> dict[str, Any]:
    """Fetch statistics from the active RAG service."""
    rag = request.app.state.rag_service
    try:
        return await rag.get_collection_stats()
    except Exception as exc:
        logger.error("rag_stats_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch vector DB stats: {exc}",
        )


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Documents",
    description="Adds text documents to the vector store for RAG retrieval.",
)
async def ingest_documents(
    body: DocumentIngestRequest, request: Request
) -> DocumentIngestResponse:
    """Ingest documents into ChromaDB."""
    rag = request.app.state.rag_service
    try:
        doc_ids = await rag.add_documents(
            texts=body.texts,
            ids=body.ids,
            metadatas=body.metadatas,
        )
        return DocumentIngestResponse(count=len(doc_ids), ids=doc_ids)
    except Exception as exc:
        logger.error("rag_ingest_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


@router.delete(
    "/documents",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Documents",
    description="Removes documents from the vector store by their IDs.",
)
async def delete_documents(
    body: DocumentDeleteRequest, request: Request
) -> None:
    """Delete specific documents from ChromaDB."""
    rag = request.app.state.rag_service
    try:
        await rag.delete_documents(ids=body.ids)
    except Exception as exc:
        logger.error("rag_delete_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Test Retrieval",
    description="Utility endpoint to test what chunks the RAG system returns for a given query.",
)
async def query_knowledge_base(
    body: QueryRequest, request: Request
) -> QueryResponse:
    """Perform a direct retrieval search."""
    rag = request.app.state.rag_service
    try:
        results = await rag.retrieve(body.query, k=body.k)
        
        return QueryResponse(
            query=body.query,
            results=[QueryResultSchema(content=r) for r in results],
            count=len(results)
        )
    except Exception as exc:
        logger.error("rag_query_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Retrieval failed: {exc}",
        )
