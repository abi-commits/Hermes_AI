"""Text splitting utilities for the RAG ingestion pipeline."""

from __future__ import annotations

from typing import Any


class TextSplitter:
    """Thin wrapper around LangChain's ``RecursiveCharacterTextSplitter``."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_token_splitting: bool = False,
        token_encoding: str = "cl100k_base",
    ) -> None:
        """Initialise the text splitter with the given parameters."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._use_token_splitting = use_token_splitting
        self._token_encoding = token_encoding

    # ------------------------------------------------------------------
    # Internal splitter construction
    # ------------------------------------------------------------------

    def _build(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):  # -> RecursiveCharacterTextSplitter
        """Instantiate a LangChain splitter with the configured parameters."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        size = chunk_size or self._chunk_size
        overlap = chunk_overlap or self._chunk_overlap
        separators = ["\n\n", "\n", ". ", " ", ""]

        if self._use_token_splitting:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self._token_encoding,
                chunk_size=size,
                chunk_overlap=overlap,
                separators=separators,
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=separators,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split a single document into chunks."""
        splitter = self._build(chunk_size, chunk_overlap)
        return splitter.split_text(text)

    def split_many(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Split multiple documents and pair each chunk with enriched metadata."""
        splitter = self._build(chunk_size, chunk_overlap)
        all_chunks: list[str] = []
        all_metas: list[dict[str, Any]] = []

        for i, text in enumerate(texts):
            chunks = splitter.split_text(text)
            base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metas.append({
                    **base_meta,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                })

        return all_chunks, all_metas
