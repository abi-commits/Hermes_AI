"""Script to populate the vector database from documents.

Usage:
    python scripts/seed_knowledge_base.py [options] <documents_dir>

Examples:
    python scripts/seed_knowledge_base.py ./docs
    python scripts/seed_knowledge_base.py --chunk-size 500 --chunk-overlap 50 ./docs
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes.services.vector_db import VectorDB

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the knowledge base from documents"
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
        help="Size of document chunks (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".txt", ".md", ".pdf", ".html"],
        help="File types to index (default: .txt .md .pdf .html)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )

    return parser.parse_args()


def read_text_file(path: Path) -> str:
    """Read a text file.

    Args:
        path: Path to file.

    Returns:
        File contents.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("failed_to_read_file", path=str(path), error=str(e))
        return ""


def read_pdf_file(path: Path) -> str:
    """Read a PDF file.

    Args:
        path: Path to PDF file.

    Returns:
        Extracted text.
    """
    try:
        import pypdf

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except ImportError:
        logger.error("pypdf_not_installed", path=str(path))
        return ""
    except Exception as e:
        logger.error("failed_to_read_pdf", path=str(path), error=str(e))
        return ""


def read_document(path: Path) -> str:
    """Read a document based on file extension.

    Args:
        path: Path to document.

    Returns:
        Document contents.
    """
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md", ".html", ".htm", ".rst", ".json"]:
        return read_text_file(path)
    elif suffix == ".pdf":
        return read_pdf_file(path)
    else:
        logger.warning("unsupported_file_type", path=str(path), suffix=suffix)
        return ""


def chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split document into chunks.

    Args:
        text: Document text.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            for i in range(min(end, len(text) - 1), start, -1):
                if text[i] in ".!?":
                    end = i + 1
                    break

        chunks.append(text[start:end].strip())
        start = end - chunk_overlap

    return chunks


async def seed_knowledge_base(args: argparse.Namespace) -> dict[str, Any]:
    """Seed the knowledge base.

    Args:
        args: Parsed command line arguments.

    Returns:
        Statistics about the operation.
    """
    documents_dir = Path(args.documents_dir)

    if not documents_dir.exists():
        logger.error("documents_directory_not_found", path=str(documents_dir))
        return {"error": "Directory not found"}

    # Find all documents
    documents: list[tuple[Path, str]] = []

    for file_type in args.file_types:
        for path in documents_dir.rglob(f"*{file_type}"):
            content = read_document(path)
            if content:
                documents.append((path, content))

    logger.info(
        "found_documents",
        count=len(documents),
        directory=str(documents_dir),
    )

    if args.dry_run:
        logger.info("dry_run_mode", documents=len(documents))
        for path, content in documents:
            chunks = chunk_document(content, args.chunk_size, args.chunk_overlap)
            logger.info(
                "would_index",
                path=str(path),
                characters=len(content),
                chunks=len(chunks),
            )
        return {"documents_found": len(documents), "mode": "dry_run"}

    # Connect to vector DB
    vector_db = VectorDB()
    await vector_db.connect()

    total_chunks = 0
    total_documents = 0

    try:
        for path, content in documents:
            # Chunk the document
            chunks = chunk_document(content, args.chunk_size, args.chunk_overlap)

            if not chunks:
                continue

            # Prepare documents for indexing
            docs = chunks
            metadatas = [
                {
                    "source": str(path.relative_to(documents_dir)),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

            # Index documents
            await vector_db.add(documents=docs, metadatas=metadatas)

            total_documents += 1
            total_chunks += len(chunks)

            logger.info(
                "indexed_document",
                path=str(path),
                chunks=len(chunks),
            )

    finally:
        await vector_db.disconnect()

    stats = {
        "documents_indexed": total_documents,
        "chunks_indexed": total_chunks,
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

        print(f"\nKnowledge base seeded successfully!")
        print(f"Documents indexed: {stats['documents_indexed']}")
        print(f"Chunks indexed: {stats['chunks_indexed']}")

        return 0

    except Exception as e:
        logger.exception("seed_failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
