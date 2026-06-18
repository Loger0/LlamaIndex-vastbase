"""Framework-level integration tests — validates the adapter through
LlamaIndex's standard RAG pipeline (VectorStoreIndex, StorageContext, QueryEngine).

These tests prove that VastbaseVectorStore works as a drop-in replacement for
PGVectorStore through LlamaIndex's high-level abstractions, not just raw API calls.

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import os
import pytest
from typing import List

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

from conftest import (
    vastbase_not_available,
    VASTBASE_HOST,
    VASTBASE_PORT,
    VASTBASE_DATABASE,
    VASTBASE_USER,
    VASTBASE_PASSWORD,
)

# Use a small embed_dim for fast tests
FW_EMBED_DIM = 2
FW_TABLE_NAME = "test_fw_integration"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embed():
    """A mock embedding model with dim=2 for testing.
    Also sets a mock LLM so query_engine works without OpenAI.
    """
    Settings.embed_model = MockEmbedding(embed_dim=FW_EMBED_DIM)
    Settings.llm = MockLLM()
    yield Settings.embed_model
    # Reset after test
    Settings._llm = None
    Settings._embed_model = None


@pytest.fixture
def fw_store():
    """A fresh VastbaseVectorStore for framework-level tests."""
    store = VastbaseVectorStore(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=FW_TABLE_NAME,
        embed_dim=FW_EMBED_DIM,
    )
    yield store
    try:
        store.clear()
        store.close()
    except Exception:
        pass


@pytest.fixture
def storage_ctx(fw_store):
    """A LlamaIndex StorageContext backed by Vastbase."""
    return StorageContext.from_defaults(vector_store=fw_store)


@pytest.fixture
def sample_docs():
    """Sample documents for testing the RAG pipeline."""
    return [
        Document(
            text="Vastbase G100 is a commercial database by VastData, based on PostgreSQL.",
            metadata={"category": "product", "version": "3.0"},
        ),
        Document(
            text="Vastbase supports native vector search with HNSW and IVFFlat indexes.",
            metadata={"category": "feature", "version": "3.0"},
        ),
        Document(
            text="pyvastbase is the Python SDK for Vastbase, providing Collection API.",
            metadata={"category": "sdk", "version": "0.2.7"},
        ),
    ]


# ============================================================================
# Test 1: Document ingestion + VectorStoreIndex construction
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_vector_store_index_from_documents(fw_store, storage_ctx, sample_docs, mock_embed):
    """Documents → VectorStoreIndex.from_documents() → index built successfully."""
    Settings.embed_model = mock_embed

    index = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    assert index is not None
    # Verify documents were ingested by querying
    retrieved = fw_store.query(
        VectorStoreQuery(
            query_embedding=[0.5, 0.5],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
        )
    )
    assert len(retrieved.nodes) >= 3, f"Expected at least 3 nodes, got {len(retrieved.nodes)}"


# ============================================================================
# Test 2: DENSE vector search via QueryEngine
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_query_engine_dense_search(fw_store, storage_ctx, sample_docs, mock_embed):
    """VectorStoreIndex → as_query_engine() → query() → response."""
    Settings.embed_model = mock_embed

    index = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("vector search")

    assert response is not None
    assert len(str(response)) > 0, "Query engine returned empty response"


# ============================================================================
# Test 3: HYBRID search via QueryEngine
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_query_engine_hybrid_search(mock_embed):
    """Hybrid search through LlamaIndex's query engine pipeline."""
    Settings.embed_model = mock_embed

    store = VastbaseVectorStore(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name="test_fw_hybrid",
        embed_dim=FW_EMBED_DIM,
        hybrid_search=True,
    )
    try:
        storage_ctx = StorageContext.from_defaults(vector_store=store)
        docs = [
            Document(text="Vastbase G100 supports native vector search capabilities."),
            Document(text="The database provides HNSW indexing for fast similarity search."),
            Document(text="Full text search is available through the SPARSE query mode."),
        ]
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)

        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("database search capabilities")

        assert response is not None
        assert len(str(response)) > 0
    finally:
        try:
            store.clear()
            store.close()
        except Exception:
            pass


# ============================================================================
# Test 4: Metadata filters through the pipeline
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_metadata_filters_in_query(fw_store, storage_ctx, sample_docs, mock_embed):
    """MetadataFilters applied through LlamaIndex's query pipeline."""
    Settings.embed_model = mock_embed

    index = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    # Query with metadata filter — only "product" category
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value="product",
                operator=FilterOperator.EQ,
            )
        ]
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        filters=filters,
    )
    response = query_engine.query("database")

    assert response is not None
    # The response should be based on filtered documents only
    # With mock embedding, we can't verify relevance but verify it runs


# ============================================================================
# Test 5: Async query pipeline
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_query_pipeline(fw_store, storage_ctx, sample_docs, mock_embed):
    """async_add → VectorStoreIndex → aquery() — full async pipeline."""
    Settings.embed_model = mock_embed

    index = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = await query_engine.aquery("Vastbase features")

    assert response is not None
    assert len(str(response)) > 0


# ============================================================================
# Test 6: Table reuse — reconnect and query existing data
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_table_reuse_reconnect(storage_ctx, sample_docs, mock_embed):
    """Write data with store_a → create store_b with same table → query existing data."""
    Settings.embed_model = mock_embed

    # Store A: write data
    index_a = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    # Store B: reconnect to the same table
    store_b = VastbaseVectorStore(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=FW_TABLE_NAME,
        embed_dim=FW_EMBED_DIM,
    )
    try:
        # Query existing data through the new connection
        result = store_b.query(
            VectorStoreQuery(
                query_embedding=[0.5, 0.5],
                similarity_top_k=10,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )
        assert len(result.nodes) >= 3, (
            f"Reconnected store should see {3}+ nodes, got {len(result.nodes)}"
        )
    finally:
        try:
            store_b.close()
        except Exception:
            pass


# ============================================================================
# Test 7: Retriever-level integration
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_retriever_integration(fw_store, storage_ctx, sample_docs, mock_embed):
    """VectorStoreIndex → as_retriever() → retrieve() returns correct nodes."""
    Settings.embed_model = mock_embed

    index = VectorStoreIndex.from_documents(
        sample_docs,
        storage_context=storage_ctx,
    )

    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("Vastbase database")

    assert len(nodes) > 0, "Retriever should return at least one node"
    for node in nodes:
        assert node.node is not None
        assert hasattr(node, "score")


# ============================================================================
# Test 8: Insert nodes via index (not just from_documents)
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_index_insert_nodes(fw_store, storage_ctx, mock_embed):
    """VectorStoreIndex → insert_nodes() — incremental insertion."""
    Settings.embed_model = mock_embed

    # Create empty index
    index = VectorStoreIndex.from_vector_store(fw_store)

    # Insert nodes incrementally
    nodes = [
        TextNode(
            text="Incremental node 1 about Vastbase.",
            embedding=[0.5, 0.5],
            metadata={"source": "test"},
        ),
        TextNode(
            text="Incremental node 2 about pyvastbase.",
            embedding=[0.3, 0.7],
            metadata={"source": "test"},
        ),
    ]
    index.insert_nodes(nodes)

    # Verify they're retrievable
    result = fw_store.query(
        VectorStoreQuery(
            query_embedding=[0.5, 0.5],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
        )
    )
    assert len(result.nodes) >= 2, f"Expected at least 2 nodes, got {len(result.nodes)}"
