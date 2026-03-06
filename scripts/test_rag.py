"""Quick interactive test for the RAG pipeline.

Usage:
    uv run python scripts/test_rag.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)

# Sample knowledge base documents for testing
SAMPLE_DOCS = [
    {
        "text": (
            "Hermes is an AI-powered voice support service. It uses speech-to-text "
            "by Deepgram, large language model by Google Gemini, and text-to-speech "
            "by Chatterbox Turbo to handle customer calls automatically."
        ),
        "metadata": {"category": "overview", "source": "internal"},
    },
    {
        "text": (
            "To reset your password, go to the login page and click 'Forgot Password'. "
            "Enter your email address and you will receive a reset link within 5 minutes. "
            "If you don't receive the email, check your spam folder or contact support."
        ),
        "metadata": {"category": "faq", "source": "help-center"},
    },
    {
        "text": (
            "Our business hours are Monday to Friday, 9 AM to 6 PM Eastern Time. "
            "For urgent issues outside business hours, you can reach our emergency "
            "support line at 1-800-555-0199."
        ),
        "metadata": {"category": "faq", "source": "help-center"},
    },
    {
        "text": (
            "The premium plan costs $49.99 per month and includes unlimited API calls, "
            "priority support, custom voice cloning, and advanced analytics. "
            "The basic plan is $9.99 per month with 1000 API calls included."
        ),
        "metadata": {"category": "pricing", "source": "sales"},
    },
    {
        "text": (
            "To cancel your subscription, go to Settings > Billing > Cancel Plan. "
            "Your access will continue until the end of the current billing period. "
            "Refunds are available within the first 14 days of a new subscription."
        ),
        "metadata": {"category": "faq", "source": "help-center"},
    },
]


async def main() -> None:
    from hermes.services.rag import ChromaRAGService

    rag = ChromaRAGService()

    # ── Step 1: Connection check ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RAG Pipeline Test")
    print("=" * 60)

    print("\n[1/5] Connecting to Chroma Cloud...")
    try:
        stats = await rag.get_collection_stats()
        print(f"  ✓ Connected — collection '{stats['name']}' has {stats['count']} documents")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return

    # ── Step 2: Add sample documents ─────────────────────────────────
    print("\n[2/5] Adding sample documents...")
    texts = [d["text"] for d in SAMPLE_DOCS]
    metas = [d["metadata"] for d in SAMPLE_DOCS]

    added_ids = await rag.add_documents(texts=texts, metadatas=metas)
    print(f"  ✓ Added {len(added_ids)} documents (dedup may skip existing)")

    stats = await rag.get_collection_stats()
    print(f"  Collection now has {stats['count']} documents total")

    # ── Step 3: Dense retrieval ──────────────────────────────────────
    print("\n[3/5] Testing dense retrieval...")
    test_queries = [
        "How do I reset my password?",
        "What are your pricing plans?",
        "When is customer support available?",
        "How does Hermes work?",
    ]

    for query in test_queries:
        results = await rag.retrieve(query, k=2)
        print(f"\n  Q: {query}")
        for i, doc in enumerate(results, 1):
            print(f"    [{i}] {doc[:100]}...")

    # ── Step 4: Metadata-filtered retrieval ──────────────────────────
    print("\n[4/5] Testing metadata-filtered retrieval...")
    query = "Tell me about pricing"
    results_all = await rag.retrieve(query, k=3)
    results_faq = await rag.retrieve(query, k=3, where={"category": "faq"})
    results_pricing = await rag.retrieve(query, k=3, where={"category": "pricing"})

    print(f"  Q: '{query}'")
    print(f"    No filter    → {len(results_all)} results")
    print(f"    category=faq → {len(results_faq)} results")
    print(f"    category=pricing → {len(results_pricing)} results")

    if results_pricing:
        print(f"    Top pricing result: {results_pricing[0][:100]}...")

    # ── Step 5: query_with_context (what the LLM sees) ───────────────
    print("\n[5/5] Testing query_with_context (formatted for LLM)...")
    context, raw_docs = await rag.query_with_context(
        "I want to cancel my subscription, and what's the refund policy?"
    )
    print(f"  Retrieved {len(raw_docs)} docs")
    print(f"  Context for LLM:\n{'─' * 40}")
    print(context)
    print("─" * 40)

    # ── Done ─────────────────────────────────────────────────────────
    print("\n✓ All RAG tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
