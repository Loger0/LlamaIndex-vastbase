#!/usr/bin/env python3
"""Demo: LlamaIndex + Vastbase RAG 完整工作流.

真实用户场景：创建知识库 → 摄入文档 → 自然语言问答 → 清理.

This standalone script validates that VastbaseVectorStore works as a
drop-in replacement for PGVectorStore through LlamaIndex's standard
RAG pipeline (VectorStoreIndex, StorageContext, QueryEngine).

Exit codes:
    0 — all demo scenarios passed
    1 — one or more scenarios failed
"""

import os
import sys
import traceback

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.vastbase import VastbaseVectorStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VASTBASE_HOST = os.environ.get("VASTBASE_HOST", "172.16.105.107")
VASTBASE_PORT = int(os.environ.get("VASTBASE_PORT", "15432"))
VASTBASE_DATABASE = os.environ.get("VASTBASE_DATABASE", "vastbase")
VASTBASE_USER = os.environ.get("VASTBASE_USER", "aidev")
VASTBASE_PASSWORD = os.environ.get("VASTBASE_PASSWORD", "Vbase_123456")

DEMO_TABLE_NAME = "demo_llamaindex_rag"
DEMO_EMBED_DIM = 2

# Use mock models for demo (no API keys needed)
Settings.embed_model = MockEmbedding(embed_dim=DEMO_EMBED_DIM)
Settings.llm = MockLLM()


def _make_store(table_name: str = DEMO_TABLE_NAME, **kwargs) -> VastbaseVectorStore:
    """Create a VastbaseVectorStore for the demo."""
    return VastbaseVectorStore(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=table_name,
        embed_dim=DEMO_EMBED_DIM,
        **kwargs,
    )


# ============================================================================
# Scenario 1: RAG Workflow — Ingest + Query
# ============================================================================


def demo_rag_workflow():
    """Documents → VectorStoreIndex → QueryEngine → response."""
    print("\n📥 Scenario 1: RAG Workflow (Ingest + Query)")
    print("-" * 50)

    store = _make_store()
    try:
        # Step 1: Create storage context
        print("   📡 Connecting to Vastbase...")
        storage_ctx = StorageContext.from_defaults(vector_store=store)
        print(f"   ✅ Connected, table: data_{DEMO_TABLE_NAME}")

        # Step 2: Ingest documents
        print("   📥 Ingesting documents...")
        docs = [
            Document(
                text="Vastbase G100 is a commercial database by VastData, based on PostgreSQL kernel.",
                metadata={"category": "product", "version": "3.0"},
            ),
            Document(
                text="Vastbase supports native vector search with HNSW and IVFFlat indexes.",
                metadata={"category": "feature", "version": "3.0"},
            ),
            Document(
                text="pyvastbase is the Python SDK for Vastbase, providing Collection API for vector operations.",
                metadata={"category": "sdk", "version": "0.2.7"},
            ),
        ]
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
        print(f"   ✅ Ingested {len(docs)} documents")

        # Step 3: Natural language queries
        print("   🔍 Running queries...")
        query_engine = index.as_query_engine(similarity_top_k=3)
        questions = [
            "What is Vastbase G100?",
            "What vector indexes does Vastbase support?",
            "What is pyvastbase?",
        ]
        for q in questions:
            response = query_engine.query(q)
            assert response is not None and len(str(response)) > 0, (
                f"Query '{q}' returned empty result"
            )
            print(f"   Q: {q}")
            print(f"   A: {str(response)[:80]}...")
        print("   ✅ All queries returned valid results")

        # Step 4: Cleanup
        print("   🧹 Cleaning up...")
        store.clear()
        print("   ✅ Cleaned up")

        return True
    finally:
        store.close()


# ============================================================================
# Scenario 2: Metadata Filtering
# ============================================================================


def demo_metadata_filtering():
    """Filter documents by metadata through the LlamaIndex pipeline."""
    print("\n🔍 Scenario 2: Metadata Filtering")
    print("-" * 50)

    store = _make_store(table_name="demo_filter")
    try:
        storage_ctx = StorageContext.from_defaults(vector_store=store)
        docs = [
            Document(text="Python is a programming language.", metadata={"lang": "python"}),
            Document(text="JavaScript runs in browsers.", metadata={"lang": "javascript"}),
            Document(text="Rust is memory-safe.", metadata={"lang": "rust"}),
        ]
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
        print(f"   ✅ Ingested {len(docs)} documents with metadata")

        # Filter by metadata
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="lang", value="python", operator=FilterOperator.EQ),
            ]
        )
        query_engine = index.as_query_engine(similarity_top_k=10, filters=filters)
        response = query_engine.query("programming language")
        assert response is not None
        print(f"   ✅ Filtered query (lang=python): {str(response)[:80]}...")

        store.clear()
        return True
    finally:
        store.close()


# ============================================================================
# Scenario 3: Retriever Integration
# ============================================================================


def demo_retriever():
    """Use LlamaIndex's retriever interface for direct node retrieval."""
    print("\n🔎 Scenario 3: Retriever Integration")
    print("-" * 50)

    store = _make_store(table_name="demo_retriever")
    try:
        storage_ctx = StorageContext.from_defaults(vector_store=store)
        docs = [
            Document(text="Vastbase provides vector search capabilities."),
            Document(text="The database supports both sync and async APIs."),
        ]
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
        print(f"   ✅ Built index from {len(docs)} documents")

        # Use retriever
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve("vector search")
        assert len(nodes) > 0, "Retriever returned no nodes"
        print(f"   ✅ Retriever returned {len(nodes)} node(s)")
        for i, node in enumerate(nodes):
            print(f"   [{i+1}] score={node.score:.4f} text={node.node.text[:60]}...")

        store.clear()
        return True
    finally:
        store.close()


# ============================================================================
# Scenario 4: Incremental Node Insertion
# ============================================================================


def demo_incremental_insert():
    """Insert nodes incrementally after index creation."""
    print("\n📝 Scenario 4: Incremental Node Insertion")
    print("-" * 50)

    store = _make_store(table_name="demo_incremental")
    try:
        # Create empty index from vector store
        index = VectorStoreIndex.from_vector_store(store)
        print("   ✅ Created empty index")

        # Insert nodes in batches
        batch1 = [
            TextNode(text="Batch 1 node A", embedding=[0.5, 0.5], metadata={"batch": "1"}),
            TextNode(text="Batch 1 node B", embedding=[0.3, 0.7], metadata={"batch": "1"}),
        ]
        index.insert_nodes(batch1)
        print(f"   ✅ Inserted batch 1 ({len(batch1)} nodes)")

        batch2 = [
            TextNode(text="Batch 2 node C", embedding=[0.8, 0.2], metadata={"batch": "2"}),
        ]
        index.insert_nodes(batch2)
        print(f"   ✅ Inserted batch 2 ({len(batch2)} nodes)")

        # Verify all nodes are retrievable
        result = store.query(
            VectorStoreQuery(
                query_embedding=[0.5, 0.5],
                similarity_top_k=10,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )
        assert len(result.nodes) >= 3, f"Expected 3+ nodes, got {len(result.nodes)}"
        print(f"   ✅ Verified: {len(result.nodes)} nodes retrievable")

        store.clear()
        return True
    finally:
        store.close()


# ============================================================================
# Scenario 5: Multi-Store Data Isolation
# ============================================================================


def demo_multi_store_isolation():
    """Verify data isolation between different collection instances."""
    print("\n🏪 Scenario 5: Multi-Store Data Isolation")
    print("-" * 50)

    store_a = _make_store(table_name="demo_store_a")
    store_b = _make_store(table_name="demo_store_b")
    try:
        ctx_a = StorageContext.from_defaults(vector_store=store_a)
        ctx_b = StorageContext.from_defaults(vector_store=store_b)

        # Ingest different docs to each store
        docs_a = [Document(text="Store A document")]
        docs_b = [Document(text="Store B document"), Document(text="Store B second doc")]

        index_a = VectorStoreIndex.from_documents(docs_a, storage_context=ctx_a)
        index_b = VectorStoreIndex.from_documents(docs_b, storage_context=ctx_b)
        print("   ✅ Ingested to both stores")

        # Verify isolation
        result_a = store_a.query(
            VectorStoreQuery(query_embedding=[0.5, 0.5], similarity_top_k=10, mode=VectorStoreQueryMode.DEFAULT)
        )
        result_b = store_b.query(
            VectorStoreQuery(query_embedding=[0.5, 0.5], similarity_top_k=10, mode=VectorStoreQueryMode.DEFAULT)
        )

        assert len(result_a.nodes) == 1, f"Store A should have 1 node, got {len(result_a.nodes)}"
        assert len(result_b.nodes) == 2, f"Store B should have 2 nodes, got {len(result_b.nodes)}"
        print(f"   ✅ Store A: {len(result_a.nodes)} node(s), Store B: {len(result_b.nodes)} node(s)")
        print("   ✅ Data isolation verified")

        store_a.clear()
        store_b.clear()
        return True
    finally:
        store_a.close()
        store_b.close()


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 60)
    print("🚀 Demo: LlamaIndex + Vastbase RAG 完整工作流")
    print("=" * 60)
    print(f"   Vastbase: {VASTBASE_HOST}:{VASTBASE_PORT}/{VASTBASE_DATABASE}")
    print(f"   Embed dim: {DEMO_EMBED_DIM} (mock)")
    print(f"   LLM: MockLLM")

    scenarios = [
        ("RAG Workflow", demo_rag_workflow),
        ("Metadata Filtering", demo_metadata_filtering),
        ("Retriever Integration", demo_retriever),
        ("Incremental Insert", demo_incremental_insert),
        ("Multi-Store Isolation", demo_multi_store_isolation),
    ]

    results = []
    for name, fn in scenarios:
        try:
            ok = fn()
            results.append((name, ok, None))
        except Exception as e:
            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Results Summary")
    print("=" * 60)
    all_pass = True
    for i, (name, ok, err) in enumerate(results, 1):
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"   {i}. {name}: {status}")
        if err:
            print(f"      Error: {err[:100]}")
            all_pass = False

    if all_pass:
        print("\n✅ All demo scenarios passed!")
    else:
        print("\n❌ Some demo scenarios failed.")

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
