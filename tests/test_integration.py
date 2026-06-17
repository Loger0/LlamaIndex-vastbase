"""End-to-end integration tests for VastbaseVectorStore.

Full lifecycle: create → add → search → get_nodes → delete → clear.

Also covers:
- Multiple store instances (index isolation)
- IndexNode round-trip
- Hybrid search E2E
- Filter + search E2E
- customize_search_fn integration

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest
import asyncio
from typing import List

from llama_index.core.schema import BaseNode, IndexNode, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
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
    _get_sample_vector,
)


# ============================================================================
# Full lifecycle: create → add → search → get → delete → clear
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_full_crud_lifecycle():
    """End-to-end: create store → add nodes → search → get_nodes → delete → clear.

    This test creates its own store instance to avoid interference.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name="test_full_lifecycle",
        schema_name="public",
        embed_dim=2,
    )

    try:
        # 1. Add nodes
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        nodes = [
            TextNode(
                text="document one about machine learning",
                id_="doc1",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc1")},
                extra_info={"topic": "ml", "year": 2024},
                embedding=_get_sample_vector(1.0),
            ),
            TextNode(
                text="document two about deep learning",
                id_="doc2",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc2")},
                extra_info={"topic": "dl", "year": 2025},
                embedding=_get_sample_vector(0.5),
            ),
            TextNode(
                text="document three about reinforcement learning",
                id_="doc3",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc3")},
                extra_info={"topic": "rl", "year": 2024},
                embedding=_get_sample_vector(0.1),
            ),
        ]
        ids = store.add(nodes)
        assert ids == ["doc1", "doc2", "doc3"]

        # 2. Search — DEFAULT
        q = VectorStoreQuery(
            query_embedding=_get_sample_vector(1.0), similarity_top_k=2
        )
        res = store.query(q)
        assert len(res.nodes) == 2
        assert res.nodes[0].node_id == "doc1"  # Most similar

        # 3. Search with metadata filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="topic", value="ml", operator=FilterOperator.EQ),
            ]
        )
        q2 = VectorStoreQuery(
            query_embedding=_get_sample_vector(1.0), similarity_top_k=5, filters=filters
        )
        res2 = store.query(q2)
        assert len(res2.nodes) == 1
        assert res2.nodes[0].node_id == "doc1"

        # 4. get_nodes
        retrieved = store.get_nodes(node_ids=["doc1", "doc3"])
        retrieved_ids = {n.node_id for n in retrieved}
        assert retrieved_ids == {"doc1", "doc3"}

        # 5. Delete by ref_doc_id
        store.delete(ref_doc_id="doc1")
        remaining = store.get_nodes(node_ids=["doc1", "doc2", "doc3"])
        remaining_ids = {n.node_id for n in remaining}
        assert "doc1" not in remaining_ids
        assert "doc2" in remaining_ids
        assert "doc3" in remaining_ids

        # 6. delete_nodes by filter
        filters_del = MetadataFilters(
            filters=[
                MetadataFilter(key="year", value=2024, operator=FilterOperator.EQ),
            ]
        )
        store.delete_nodes(filters=filters_del)
        remaining = store.get_nodes(node_ids=["doc2", "doc3"])
        remaining_ids = {n.node_id for n in remaining}
        assert "doc3" not in remaining_ids  # year=2024
        assert "doc2" in remaining_ids  # year=2025

        # 7. Clear
        store.clear()
        final = store.get_nodes(node_ids=["doc2"])
        assert len(final) == 0

    finally:
        asyncio.run(store.close())


# ============================================================================
# Hybrid search E2E
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_search_e2e():
    """E2E hybrid search: add English text → search with query_str + embedding."""
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name="test_hybrid_e2e",
        schema_name="public",
        embed_dim=2,
        hybrid_search=True,
    )

    try:
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        nodes = [
            TextNode(
                text="The quick brown fox jumps over the lazy dog",
                id_="fox1",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="fox1")},
                embedding=_get_sample_vector(0.1),
            ),
            TextNode(
                text="Machine learning is transforming industries",
                id_="ml1",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ml1")},
                extra_info={"topic": "ml"},
                embedding=_get_sample_vector(1.0),
            ),
            TextNode(
                text="The fox and the hound went hunting",
                id_="fox2",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="fox2")},
                extra_info={"topic": "story"},
                embedding=_get_sample_vector(0.5),
            ),
        ]
        store.add(nodes)

        # Hybrid search for "fox"
        q = VectorStoreQuery(
            query_embedding=_get_sample_vector(0.3),
            query_str="fox",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
        res = store.query(q)
        assert res.nodes
        # fox1 and fox2 should appear (text contains "fox"), plus maybe ml1
        node_ids = {n.node_id for n in res.nodes}
        assert "fox1" in node_ids
        assert "fox2" in node_ids

    finally:
        asyncio.run(store.close())


# ============================================================================
# IndexNode round-trip
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_index_node_roundtrip():
    """Verify IndexNode preserves index_id through add → query round-trip."""
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name="test_index_node_rt",
        schema_name="public",
        embed_dim=2,
    )

    try:
        nodes = [
            TextNode(
                text="original document",
                id_="original_1",
                embedding=_get_sample_vector(0.1),
            ),
            IndexNode(
                text="reference to original",
                id_="ref_1",
                index_id="original_1",
                embedding=_get_sample_vector(5.0),
            ),
        ]
        store.add(nodes)

        q = VectorStoreQuery(
            query_embedding=_get_sample_vector(5.0), similarity_top_k=2
        )
        res = store.query(q)
        assert res.nodes
        assert len(res.nodes) == 2
        assert res.nodes[0].node_id == "ref_1"
        assert isinstance(res.nodes[0], IndexNode)
        assert res.nodes[0].index_id == "original_1"
        assert isinstance(res.nodes[1], TextNode)

    finally:
        asyncio.run(store.close())


# ============================================================================
# customize_search_fn integration
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_customize_search_fn_integration():
    """Verify customize_search_fn callback is invoked during query."""
    call_log = []

    def log_calls(params: dict, **kwargs) -> dict:
        call_log.append(params.copy())
        return params

    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name="test_custom_fn",
        schema_name="public",
        embed_dim=2,
        customize_search_fn=log_calls,
    )

    try:
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        node = TextNode(
            text="test",
            id_="test1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test1")},
            embedding=_get_sample_vector(0.5),
        )
        store.add([node])

        q = VectorStoreQuery(
            query_embedding=_get_sample_vector(1.0), similarity_top_k=1
        )
        store.query(q)

        # Verify callback was called at least once with expected keys
        assert len(call_log) >= 1
        assert "limit" in call_log[0] or "expr" in call_log[0]

    finally:
        asyncio.run(store.close())


# ============================================================================
# Multiple store instances (index isolation)
# ============================================================================

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_multiple_stores_isolation():
    """Verify data isolation between two VastbaseVectorStore instances on different tables.

    Mirrors upstream test_hnsw_index_creation (multi-instance pattern).
    """
    store_a = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST, port=VASTBASE_PORT,
        database=VASTBASE_DATABASE, user=VASTBASE_USER, password=VASTBASE_PASSWORD,
        table_name="test_isolation_a", schema_name="public", embed_dim=2,
    )
    store_b = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST, port=VASTBASE_PORT,
        database=VASTBASE_DATABASE, user=VASTBASE_USER, password=VASTBASE_PASSWORD,
        table_name="test_isolation_b", schema_name="public", embed_dim=2,
    )

    try:
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        node_a = TextNode(
            text="data in store A",
            id_="node_a",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node_a")},
            embedding=_get_sample_vector(1.0),
        )
        node_b = TextNode(
            text="data in store B",
            id_="node_b",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node_b")},
            embedding=_get_sample_vector(0.1),
        )

        store_a.add([node_a])
        store_b.add([node_b])

        # store_a should only see node_a
        res_a = store_a.get_nodes(node_ids=["node_a", "node_b"])
        ids_a = {n.node_id for n in res_a}
        assert ids_a == {"node_a"}

        # store_b should only see node_b
        res_b = store_b.get_nodes(node_ids=["node_a", "node_b"])
        ids_b = {n.node_id for n in res_b}
        assert ids_b == {"node_b"}

        # Cleanup
        store_a.clear()
        store_b.clear()

    finally:
        asyncio.run(store_a.close())
        asyncio.run(store_b.close())
