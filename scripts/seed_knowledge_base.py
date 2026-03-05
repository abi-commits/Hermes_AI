"""Script to populate the Chroma Cloud vector database from documents.

Uses LangChain document loaders and text splitters for robust ingestion.

Usage:
    python scripts/seed_knowledge_base.py [options] <documents_dir>

Examples:
    python scripts/seed_knowledge_base.py ./docs
    python scripts/seed_knowledge_base.py --chunk-size 500 --chunk-overlap 50 ./docs
    python scripts/seed_knowledge_base.py --strategy markdown ./docs
    python scripts/seed_knowledge_base.py --collection my_collection --clear ./docs
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes.services.rag import ChromaRAGService

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the knowledge base from documents using LangChain",
    )
    parser.add_argument(
        "documents_dir",
        type=str,
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of each text chunk in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks (default: 200)",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "markdown", "code", "token"],
        default="recursive",
        help="Splitting strategy: 'recursive' (general text), "
        "'markdown' (header-aware), 'code' (language-aware), "
        "'token' (tiktoken token-based) (default: recursive)",
    )
    parser.add_argument(
        "--code-language",
        type=str,
        default="python",
        help="Programming language for 'code' strategy (default: python)",
    )
    parser.add_argument(
        "--token-encoding",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding name for 'token' strategy (default: cl100k_base)",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".txt", ".md", ".pdf", ".html", ".htm", ".rst", ".json", ".csv"],
        help="File extensions to index (default: .txt .md .pdf .html .htm .rst .json .csv)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Chroma collection name (default: from settings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to upsert per batch (default: 100)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing documents in the collection before seeding",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Document loading  (LangChain loaders)
# ──────────────────────────────────────────────────────────────────────

# Maps file suffixes → LangChain loader class + kwargs.
# Lazy imports are used so that optional heavy packages (e.g. pypdf,
# unstructured) are only required if their file types are encountered.

_LOADER_REGISTRY: dict[str, tuple[str, str, dict[str, Any]]] = {
    # (module_path, class_name, extra_kwargs)
    ".txt": ("langchain_community.document_loaders", "TextLoader", {"encoding": "utf-8"}),
    ".md": ("langchain_community.document_loaders", "TextLoader", {"encoding": "utf-8"}),
    ".rst": ("langchain_community.document_loaders", "TextLoader", {"encoding": "utf-8"}),
    ".json": ("langchain_community.document_loaders", "TextLoader", {"encoding": "utf-8"}),
    ".html": (
        "langchain_community.document_loaders",
        "UnstructuredHTMLLoader",
        {},
    ),
    ".htm": (
        "langchain_community.document_loaders",
        "UnstructuredHTMLLoader",
        {},
    ),
    ".pdf": ("langchain_community.document_loaders", "PyPDFLoader", {}),
    ".csv": ("langchain_community.document_loaders", "CSVLoader", {}),
}


def load_document(path: Path) -> list[Document]:
    """Load a single file into LangChain Documents via the appropriate loader.

    Args:
        path: Path to the document file.

    Returns:
        List of LangChain Document objects (may be >1 for multi-page PDFs).
    """
    suffix = path.suffix.lower()
    entry = _LOADER_REGISTRY.get(suffix)

    if entry is None:
        logger.warning("unsupported_file_type", path=str(path), suffix=suffix)
        return []

    module_path, class_name, kwargs = entry
    try:
        import importlib

        module = importlib.import_module(module_path)
        loader_cls = getattr(module, class_name)
        loader = loader_cls(str(path), **kwargs)
        docs = loader.load()

        # Ensure every document carries its source in metadata
        for doc in docs:
            doc.metadata.setdefault("source", str(path))

        return docs

    except ImportError as exc:
        logger.error(
            "loader_import_failed",
            loader=f"{module_path}.{class_name}",
            error=str(exc),
            hint="Install the required extra, e.g. `pip install pypdf` for PDFs",
        )
        return []
    except Exception as exc:
        logger.error("document_load_failed", path=str(path), error=str(exc))
        return []


def load_directory(
    documents_dir: Path,
    file_types: list[str],
) -> list[Document]:
    """Recursively load all matching files from a directory.

    Args:
        documents_dir: Root directory to scan.
        file_types: List of allowed file extensions (e.g. [".txt", ".md"]).

    Returns:
        Flat list of LangChain Documents from all files.
    """
    all_docs: list[Document] = []

    for file_type in file_types:
        for path in sorted(documents_dir.rglob(f"*{file_type}")):
            docs = load_document(path)
            if docs:
                # Store the relative path for cleaner metadata
                rel = str(path.relative_to(documents_dir))
                for doc in docs:
                    doc.metadata["source"] = rel

                all_docs.extend(docs)
                logger.debug("loaded_document", path=rel, pages=len(docs))

    logger.info("documents_loaded", total=len(all_docs), directory=str(documents_dir))
    return all_docs


# ──────────────────────────────────────────────────────────────────────
# Text splitting  (LangChain splitters)
# ──────────────────────────────────────────────────────────────────────

_MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def build_text_splitter(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    code_language: str = "python",
    token_encoding: str = "cl100k_base",
) -> RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter:
    """Build a LangChain text splitter from CLI args.

    Args:
        strategy: One of "recursive", "markdown", "code", "token".
        chunk_size: Maximum chunk length in characters (or tokens for "token").
        chunk_overlap: Character/token overlap between consecutive chunks.
        code_language: Language hint for the "code" strategy.
        token_encoding: Tiktoken encoding name for the "token" strategy.

    Returns:
        A configured LangChain text splitter instance.
    """
    if strategy == "markdown":
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS,
            strip_headers=False,
        )

    if strategy == "code":
        lang_enum = Language(code_language)
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if strategy == "token":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=token_encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # Default: recursive character splitter (good for prose)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def split_documents(
    docs: list[Document],
    splitter: RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter,
) -> list[Document]:
    """Split documents into chunks, preserving metadata.

    The MarkdownHeaderTextSplitter operates on raw text, so we handle it
    slightly differently from the other RecursiveCharacterTextSplitter.

    Args:
        docs: List of loaded LangChain Documents.
        splitter: A configured text splitter.

    Returns:
        List of chunked Documents with inherited + chunk-level metadata.
    """
    if isinstance(splitter, MarkdownHeaderTextSplitter):
        chunks: list[Document] = []
        for doc in docs:
            md_chunks = splitter.split_text(doc.page_content)
            for i, chunk in enumerate(md_chunks):
                # Merge header metadata from the splitter with the original
                merged_meta = {**doc.metadata, **chunk.metadata, "chunk_index": i}
                chunks.append(
                    Document(page_content=chunk.page_content, metadata=merged_meta)
                )
        return chunks

    return splitter.split_documents(docs)


# ──────────────────────────────────────────────────────────────────────
# ID generation
# ──────────────────────────────────────────────────────────────────────


def deterministic_id(source: str, chunk_index: int, content: str) -> str:
    """Generate a reproducible document ID from source + index + content hash.

    This ensures that re-running the script with the same content produces
    the same IDs, making upserts idempotent.

    Args:
        source: Source file relative path.
        chunk_index: Index of the chunk within the source.
        content: The chunk text.

    Returns:
        A 32-character hex string ID.
    """
    payload = f"{source}::{chunk_index}::{content[:200]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ──────────────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ──────────────────────────────────────────────────────────────────────


async def seed_knowledge_base(args: argparse.Namespace) -> dict[str, Any]:
    """Seed the knowledge base.

    Pipeline:
        1. Load documents via LangChain loaders
        2. Split into chunks via LangChain text splitters
        3. Upsert chunks into Chroma Cloud via ChromaRAGService

    Args:
        args: Parsed command line arguments.

    Returns:
        Statistics about the operation.
    """
    documents_dir = Path(args.documents_dir)
    if not documents_dir.exists():
        logger.error("documents_directory_not_found", path=str(documents_dir))
        return {"error": f"Directory not found: {documents_dir}"}

    # 1. Load ─────────────────────────────────────────────────────────
    raw_docs = load_directory(documents_dir, args.file_types)
    if not raw_docs:
        logger.warning("no_documents_found")
        return {"error": "No documents found"}

    # 2. Split ────────────────────────────────────────────────────────
    splitter = build_text_splitter(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        code_language=args.code_language,
        token_encoding=args.token_encoding,
    )
    chunks = split_documents(raw_docs, splitter)

    # Inject chunk indices per source
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["total_chunks"] = 0  # filled below
        source_counters[src] = idx + 1

    # Back-fill total_chunks per source
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["total_chunks"] = source_counters[src]

    logger.info(
        "documents_split",
        raw_documents=len(raw_docs),
        chunks=len(chunks),
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # ── Dry-run ──────────────────────────────────────────────────────
    if args.dry_run:
        unique_sources = sorted({c.metadata.get("source", "?") for c in chunks})
        for src in unique_sources:
            src_chunks = [c for c in chunks if c.metadata.get("source") == src]
            logger.info(
                "would_index",
                source=src,
                chunks=len(src_chunks),
                characters=sum(len(c.page_content) for c in src_chunks),
            )
        return {
            "documents_found": len(raw_docs),
            "chunks_prepared": len(chunks),
            "mode": "dry_run",
        }

    # 3. Upsert to Chroma Cloud ──────────────────────────────────────
    rag = ChromaRAGService(collection_name=args.collection)

    # Optionally clear the collection first
    if args.clear:
        stats = await rag.get_collection_stats()
        existing_count = stats.get("count", 0)
        if existing_count:
            logger.info("clearing_collection", count=existing_count)
            # The Chroma API does not expose a "delete all" — we need to
            # recreate the collection. Handled by deleting then re-ensuring.
            await rag._ensure_client()
            assert rag._collection is not None
            assert rag._client is not None
            loop = asyncio.get_event_loop()
            client = rag._client
            name = rag.collection_name
            await loop.run_in_executor(None, lambda: client.delete_collection(name))
            rag._collection = None  # force re-creation on next call
            logger.info("collection_cleared")

    # Batch upserts
    batch_size = args.batch_size
    total_upserted = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        texts = [c.page_content for c in batch]
        ids = [
            deterministic_id(
                c.metadata.get("source", ""),
                c.metadata.get("chunk_index", 0),
                c.page_content,
            )
            for c in batch
        ]
        metadatas = [
            {
                "source": str(c.metadata.get("source", "")),
                "chunk_index": c.metadata.get("chunk_index", 0),
                "total_chunks": c.metadata.get("total_chunks", 0),
                **(
                    {"h1": c.metadata["h1"]}
                    if "h1" in c.metadata
                    else {}
                ),
                **(
                    {"h2": c.metadata["h2"]}
                    if "h2" in c.metadata
                    else {}
                ),
            }
            for c in batch
        ]

        await rag.add_documents(texts=texts, ids=ids, metadatas=metadatas)  # type: ignore[arg-type]
        total_upserted += len(batch)

        logger.info(
            "batch_upserted",
            batch=batch_start // batch_size + 1,
            chunks=len(batch),
            total_so_far=total_upserted,
        )

    # Final stats
    collection_stats = await rag.get_collection_stats()
    stats: dict[str, Any] = {
        "documents_loaded": len(raw_docs),
        "chunks_indexed": total_upserted,
        "strategy": args.strategy,
        "collection": rag.collection_name,
        "collection_total": collection_stats.get("count", "?"),
    }
    logger.info("knowledge_base_seeded", **stats)
    return stats


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()

    try:
        stats = await seed_knowledge_base(args)

        if "error" in stats:
            print(f"Error: {stats['error']}", file=sys.stderr)
            return 1

        print("\nKnowledge base seeded successfully!")
        print(f"  Documents loaded: {stats['documents_loaded']}")
        print(f"  Chunks indexed:   {stats['chunks_indexed']}")
        print(f"  Strategy:         {stats['strategy']}")
        print(f"  Collection:       {stats['collection']}")
        print(f"  Total in collection: {stats['collection_total']}")

        return 0

    except Exception as e:
        logger.exception("seed_failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
