"""Tests for VastbaseVectorStore async operations.

Tests async counterparts of all CRUD + query methods:
- async_add, adelete, adelete_nodes, aget_nodes, aclear, aquery

Adapted from PGVectorStore tests (parametrized use_async=True paths).

All tests use pyvastbase API exclusively — no SQLAlchemy, no asyncpg, no raw SQL.
"""

import pytest
import asyncio
from typing import List

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.vastbase import VastbaseVectorStore

from conftest import (
    vastbase_not_available,
    _get_sample_vector,
)


# ============================================================================
# async_add
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_add(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async_add accepts nodes and returns their IDs."""
    ids = await vb.async_add(node_embeddings)
    assert ids == ["aaa", "bbb", "ccc", "ddd"]


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_add_empty(
    vb: VastbaseVectorStore,
) -> None:
    """Verify async_add with empty list returns empty list."""
    ids = await vb.async_add([])
    assert ids == []


# ============================================================================
# aquery — DEFAULT
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_query_default(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async aquery in DEFAULT mode."""
    await vb.async_add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=1
    )
    res = await vb.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_query_default_with_filter(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async aquery in DEFAULT mode with metadata filter."""
    await vb.async_add(node_embeddings)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = await vb.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


# ============================================================================
# aquery — SPARSE
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_query_sparse(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify async aquery in SPARSE mode."""
    await vb_hybrid.async_add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )
    res = await vb_hybrid.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "ccc"
    assert res.nodes[1].node_id == "ddd"


# ============================================================================
# aquery — HYBRID
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_query_hybrid(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify async aquery in HYBRID mode."""
    await vb_hybrid.async_add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )
    res = await vb_hybrid.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 3


# ============================================================================
# adelete
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_delete(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async adelete by ref_doc_id."""
    await vb.async_add(node_embeddings)
    await vb.adelete(ref_doc_id="aaa")

    remaining = await vb.aget_nodes(node_ids=["aaa", "bbb"])
    remaining_ids = {n.node_id for n in remaining}
    assert "aaa" not in remaining_ids
    assert "bbb" in remaining_ids


# ============================================================================
# adelete_nodes
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_delete_nodes(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async adelete_nodes by node_ids."""
    await vb.async_add(node_embeddings)

    # Delete nothing
    await vb.adelete_nodes()
    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 4

    # Delete nonexistent
    await vb.adelete_nodes(["asdf"])
    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 4

    # Delete list
    await vb.adelete_nodes(["aaa", "bbb"])
    nodes = await vb.aget_nodes(node_ids=["ccc", "ddd"])
    assert len(nodes) == 2
    remaining = {n.node_id for n in nodes}
    assert remaining == {"ccc", "ddd"}


# ============================================================================
# aget_nodes
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_get_nodes(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async aget_nodes by node_ids."""
    await vb.async_add(node_embeddings)

    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb"])
    retrieved = {n.node_id for n in nodes}
    assert retrieved == {"aaa", "bbb"}


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_get_nodes_with_filters(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async aget_nodes with metadata filters."""
    await vb.async_add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="test_key", value="test_value", operator=MetadataFilter.EQ)
        ]
    )
    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb"], filters=filters)
    retrieved = {n.node_id for n in nodes}
    assert retrieved == {"bbb"}


# ============================================================================
# aclear
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_async_clear(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify async aclear removes all nodes."""
    await vb.async_add(node_embeddings)

    # Verify data exists
    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 4

    await vb.aclear()

    nodes = await vb.aget_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 0


# ============================================================================
# Mixed sync/async — add sync, query async
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_sync_add_async_query(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify data added via sync add() is visible via async aquery()."""
    vb.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=1
    )
    res = await vb.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


# ============================================================================
# Close connection
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.asyncio
async def test_close_connection():
    """Verify close() sets _is_initialized to False."""
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="test_async_close",
    )
    await store.close()
    assert store._is_initialized is False
    assert store.client is None
