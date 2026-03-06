"""Knowledge base (RAG) management endpoints."""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AddDocumentsRequest(BaseModel):
    """Request body for adding documents."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of raw text documents to add.",
    )
    ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of unique document IDs. "
            "Auto-generated from content hash when omitted."
        ),
    )
    metadatas: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of metadata dicts, one per document.",
    )
    split: bool = Field(
        default=True,
        description=(
            "When True (default), documents are split into chunks before "
            "being stored.  Set to False to store each text as a single chunk."
        ),
    )
    chunk_size: int | None = Field(
        default=None,
        ge=50,
        description="Override the default chunk size (tokens or chars depending on settings).",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        description="Override the default chunk overlap.",
    )


class AddDocumentsResponse(BaseModel):
    """Response after adding documents."""

    added: int
    ids: list[str]


class DeleteDocumentsRequest(BaseModel):
    """Request body for deleting documents."""

    ids: list[str] = Field(..., min_length=1, description="IDs of documents to delete.")


class DeleteDocumentsResponse(BaseModel):
    """Response after deleting documents."""

    deleted: int
    ids: list[str]


class KnowledgeStatsResponse(BaseModel):
    """Collection statistics."""

    collection: str
    document_count: int


class QueryRequest(BaseModel):
    """Request body for testing retrieval."""

    query: str = Field(..., min_length=1, description="Query text.")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
    where: dict[str, Any] | None = Field(
        default=None,
        description="Optional Chroma metadata filter.",
    )


class QueryResponse(BaseModel):
    """Response for a knowledge base query."""

    query: str
    results: list[str]
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_rag(request: Request):
    """Return the RAG service from app state, or raise 503 if not initialised."""
    rag = getattr(request.app.state, "rag_service", None)
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not initialised",
        )
    return rag


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    response_model=KnowledgeStatsResponse,
    summary="Knowledge base statistics",
    description="Returns the document count and collection name for the active Chroma collection.",
)
async def get_stats(request: Request) -> KnowledgeStatsResponse:
    """Return basic collection statistics."""
    rag = _get_rag(request)
    try:
        stats = await rag.get_collection_stats()
    except Exception as exc:
        logger.error("knowledge_stats_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach vector database: {exc}",
        )
    return KnowledgeStatsResponse(
        collection=stats["name"],
        document_count=stats["count"],
    )


@router.post(
    "/documents",
    response_model=AddDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add documents",
    description=(
        "Add one or more text documents to the knowledge base. "
        "Documents are chunked (with optional metadata) and stored in Chroma."
    ),
)
async def add_documents(
    request: Request,
    body: AddDocumentsRequest,
) -> AddDocumentsResponse:
    """Ingest documents; splits into chunks when ``split=True`` (default)."""
    rag = _get_rag(request)
    try:
        if body.split:
            added_ids = await rag.split_and_add_documents(
                texts=body.texts,
                metadatas=body.metadatas,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
            )
        else:
            added_ids = await rag.add_documents(
                texts=body.texts,
                ids=body.ids,
                metadatas=body.metadatas,
            )
    except Exception as exc:
        logger.error("knowledge_add_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to add documents: {exc}",
        )

    logger.info("knowledge_documents_added", count=len(added_ids))
    return AddDocumentsResponse(added=len(added_ids), ids=added_ids)


@router.delete(
    "/documents",
    response_model=DeleteDocumentsResponse,
    summary="Delete documents",
    description="Remove documents from the knowledge base by their IDs.",
)
async def delete_documents(
    request: Request,
    body: DeleteDocumentsRequest,
) -> DeleteDocumentsResponse:
    """Delete documents from the vector database by ID."""
    rag = _get_rag(request)
    try:
        await rag.delete_documents(ids=body.ids)
    except Exception as exc:
        logger.error("knowledge_delete_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to delete documents: {exc}",
        )

    logger.info("knowledge_documents_deleted", count=len(body.ids))
    return DeleteDocumentsResponse(deleted=len(body.ids), ids=body.ids)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query knowledge base",
    description=(
        "Test retrieval from the knowledge base. "
        "Returns the top-k most relevant document chunks for the given query."
    ),
)
async def query_knowledge(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    """Run a retrieval query for debugging RAG quality."""
    rag = _get_rag(request)
    try:
        results = await rag.retrieve(query=body.query, k=body.k, where=body.where)
    except Exception as exc:
        logger.error("knowledge_query_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Retrieval failed: {exc}",
        )

    return QueryResponse(query=body.query, results=results, count=len(results))
