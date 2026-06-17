"""Tests for VastbaseVectorStore CRUD operations: add, delete, delete_nodes, get_nodes, clear.

Adapted from PGVectorStore tests:
- test_add_to_db_and_query                 → test_add_nodes
- test_add_to_db_query_and_delete           → test_add_and_delete_by_ref_doc_id
- test_delete_nodes                         → test_delete_nodes_by_ids
- test_delete_nodes_metadata                → test_delete_nodes_by_ids_and_filters
- test_get_nodes_parametrized               → test_get_nodes_parametrized
- test_clear                                → test_clear_collection
- test_add_to_db_and_query_index_nodes      → test_add_index_nodes

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest
from typing import List, Optional

from llama_index.core.schema import BaseNode, IndexNode, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.vastbase import VastbaseVectorStore

from conftest import (
    vastbase_not_available,
    _get_sample_vector,
)


# ---------------------------------------------------------------------------
# add — node insertion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_add_nodes(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify add() accepts nodes and returns their IDs.

    Mirrors upstream test_add_to_db_and_query (add portion).
    """
    ids = vb.add(node_embeddings)
    assert ids == ["aaa", "bbb", "ccc", "ddd"]
    assert len(ids) == 4


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_add_nodes_empty_list(vb: VastbaseVectorStore) -> None:
    """Verify add() with an empty list returns an empty list."""
    ids = vb.add([])
    assert ids == []


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_add_single_node(vb: VastbaseVectorStore) -> None:
    """Verify add() with a single node."""
    node = TextNode(
        text="single node",
        id_="single_1",
        embedding=_get_sample_vector(0.5),
    )
    ids = vb.add([node])
    assert ids == ["single_1"]


# ---------------------------------------------------------------------------
# delete — by ref_doc_id
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_delete_by_ref_doc_id(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify delete() removes nodes by ref_doc_id.

    Mirrors upstream test_add_to_db_query_and_delete pattern — add, delete ref_doc_id, verify.
    """
    vb.add(node_embeddings)

    # Delete nodes whose ref_doc_id is "aaa"
    vb.delete(ref_doc_id="aaa")

    # Verify nodes with SOURCE "aaa" are removed (aaa itself)
    # node "aaa" has ref_doc_id "aaa" in its relationship
    remaining = vb.get_nodes(node_ids=["aaa", "bbb"])
    # aaa should be deleted, bbb should remain
    remaining_ids = {n.node_id for n in remaining}
    assert "aaa" not in remaining_ids
    assert "bbb" in remaining_ids


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_delete_nonexistent_ref_doc_id(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify delete() with a nonexistent ref_doc_id does not raise."""
    vb.add(node_embeddings)
    vb.delete(ref_doc_id="nonexistent_id")
    # Should not raise, should not remove anything
    remaining = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(remaining) == 4


# ---------------------------------------------------------------------------
# delete_nodes — by node_ids and/or filters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_delete_nodes_by_ids(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify delete_nodes() by node_ids.

    Mirrors upstream test_delete_nodes.
    """
    vb.add(node_embeddings)

    # delete nothing — both args None → no-op
    vb.delete_nodes()
    # All should still be present
    all_nodes = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(all_nodes) == 4

    # delete nonexistent ID — no-op
    vb.delete_nodes(["asdf"])
    all_nodes = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(all_nodes) == 4

    # delete list
    vb.delete_nodes(["aaa", "bbb"])
    remaining = vb.get_nodes(node_ids=["ccc", "ddd"])
    assert len(remaining) == 2
    remaining_ids = {n.node_id for n in remaining}
    assert remaining_ids == {"ccc", "ddd"}


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_delete_nodes_by_filters(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify delete_nodes() by metadata filters.

    Mirrors upstream test_delete_nodes_metadata.
    """
    vb.add(node_embeddings)

    # delete purely by filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_2",
                value="test_val_2",
                operator=FilterOperator.EQ,
            )
        ]
    )
    vb.delete_nodes(filters=filters)
    # Only ddd has test_key_2 = test_val_2, should be removed
    remaining = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    remaining_ids = {n.node_id for n in remaining}
    assert "ddd" not in remaining_ids
    assert remaining_ids == {"aaa", "bbb", "ccc"}


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_delete_nodes_by_ids_and_filters(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify delete_nodes() with both node_ids AND filters (intersection).

    Mirrors upstream test_delete_nodes_metadata (combined case).
    """
    vb.add(node_embeddings)

    # delete IDs that also match filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    vb.delete_nodes(["aaa", "bbb"], filters=filters)
    # bbb has test_key = test_value → matches and gets deleted
    # aaa does NOT have test_key → does NOT match filter → stays
    remaining = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    remaining_ids = {n.node_id for n in remaining}
    assert "bbb" not in remaining_ids
    assert "aaa" in remaining_ids
    assert "ccc" in remaining_ids
    assert "ddd" in remaining_ids


# ---------------------------------------------------------------------------
# get_nodes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
@pytest.mark.parametrize(
    ("node_ids", "filter_key", "filter_value", "filter_op", "expected_ids"),
    [
        (["aaa", "bbb"], None, None, None, ["aaa", "bbb"]),
        (None, "test_num", 1, FilterOperator.EQ, ["aaa"]),
        (["bbb", "ccc"], "test_key", "test_value", FilterOperator.EQ, ["bbb"]),
        (["ccc"], "test_key", "test_value", FilterOperator.EQ, []),
        (["aaa", "bbb"], "test_num", 999, FilterOperator.EQ, []),
    ],
)
def test_get_nodes_parametrized(
    vb: VastbaseVectorStore,
    node_embeddings: List[TextNode],
    node_ids: Optional[List[str]],
    filter_key: Optional[str],
    filter_value,
    filter_op,
    expected_ids: List[str],
) -> None:
    """Test get_nodes with various combinations of node_ids and filters.

    Mirrors upstream test_get_nodes_parametrized.
    """
    vb.add(node_embeddings)

    filters = None
    if filter_key is not None:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key=filter_key, value=filter_value, operator=filter_op)
            ]
        )

    nodes = vb.get_nodes(node_ids=node_ids, filters=filters)
    retrieved_ids = [node.node_id for node in nodes]
    assert set(retrieved_ids) == set(expected_ids)
    assert len(retrieved_ids) == len(expected_ids)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_clear_collection(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify clear() removes all nodes from the collection.

    Mirrors upstream test_clear.
    """
    vb.add(node_embeddings)

    # Verify data exists
    nodes = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 4

    # Clear
    vb.clear()

    # Verify empty
    nodes = vb.get_nodes(node_ids=["aaa", "bbb", "ccc", "ddd"])
    assert len(nodes) == 0


# ---------------------------------------------------------------------------
# IndexNode handling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_add_index_nodes(
    vb: VastbaseVectorStore, index_node_embeddings: List[BaseNode]
) -> None:
    """Verify IndexNode instances are stored and can be queried.

    Mirrors upstream test_add_to_db_and_query_index_nodes.
    """
    ids = vb.add(index_node_embeddings)
    assert ids == ["aaa", "bbb", "aaa_ref"]

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(5.0), similarity_top_k=2
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "aaa_ref"
    assert isinstance(res.nodes[0], IndexNode)
    assert hasattr(res.nodes[0], "index_id")
    assert res.nodes[1].node_id == "bbb"
    assert isinstance(res.nodes[1], TextNode)
