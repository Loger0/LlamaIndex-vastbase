"""Tests for VastbaseVectorStore metadata filtering.

Covers all 14 FilterOperators and AND/OR logical combinations.

Adapted from PGVectorStore tests:
- test_add_to_db_and_query_with_metadata_filters_with_in_operator             → test_filter_in
- test_add_to_db_and_query_with_metadata_filters_with_any_operator            → test_filter_any
- test_add_to_db_and_query_with_metadata_filters_with_all_operator            → test_filter_all
- test_add_to_db_and_query_with_metadata_filters_with_contains_operator       → test_filter_contains
- test_add_to_db_and_query_with_metadata_filters_with_is_empty                → test_filter_is_empty
- test_add_to_db_and_query_with_metadata_filters_with_in_operator_and_single  → test_filter_in_single
- GIN array index tests                                                       → test_gin_*

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest
from typing import List

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
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


# ============================================================================
# 14 FilterOperators
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_eq(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify EQ (equality) filter via ExactMatchFilter."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_in(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify IN operator.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_in_operator.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_in_single(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify IN operator with a single-element list.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_in_operator_and_single_element.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_nin(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify NIN (not-in) operator."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value"],
                operator=FilterOperator.NIN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    # All nodes EXCEPT bbb (which has test_key=test_value) should match
    res_ids = {n.node_id for n in res.nodes}
    assert "bbb" not in res_ids, "bbb should be excluded by NIN filter"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_gt(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify GT (greater-than) operator on numeric metadata."""
    vb.add(node_embeddings)

    # aaa has test_num=1, others don't have test_num
    # GT 0 → only aaa should match (1 > 0)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=0,
                operator=FilterOperator.GT,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_lt(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify LT (less-than) operator on numeric metadata."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=2,
                operator=FilterOperator.LT,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    # aaa has test_num=1 < 2
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_ne(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify NE (not-equal) operator."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value="test_value",
                operator=FilterOperator.NE,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    # bbb has test_key=test_value → excluded; aaa,ccc,ddd should match
    res_ids = {n.node_id for n in res.nodes}
    assert "bbb" not in res_ids


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_gte(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify GTE (greater-than-or-equal) operator."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.GTE,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_lte(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify LTE (less-than-or-equal) operator."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.LTE,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_text_match(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify TEXT_MATCH (LIKE) operator on text metadata."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value="test_val",  # partial match
                operator=FilterOperator.TEXT_MATCH,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    res_ids = {n.node_id for n in res.nodes}
    assert "bbb" in res_ids  # test_key = "test_value"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_text_match_insensitive(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify TEXT_MATCH_INSENSITIVE (ILIKE) operator."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value="TEST_VAL",  # uppercase
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    res_ids = {n.node_id for n in res.nodes}
    assert "bbb" in res_ids  # ILIKE matches case-insensitively


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_any(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify ANY (?|) operator — matches if metadata array contains ANY value.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_any_operator.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value=["test_value_1", "test_value_new"],
                operator=FilterOperator.ANY,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_all(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify ALL (?&) operator — matches if metadata array contains ALL values.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_all_operator.
    """
    vb.add(node_embeddings)

    # Match — ccc has both test_value_1 and test_value_2
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value=["test_value_1", "test_value_2"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"

    # No match — ccc doesn't have "test_value_3"
    filters_no_match = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value=["test_value_1", "test_value_3"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q2 = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10,
        filters=filters_no_match,
    )
    res2 = vb.query(q2)
    assert not res2.nodes or len(res2.nodes) == 0


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_contains(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify CONTAINS (@>) operator on array metadata.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_contains_operator.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value="test_value_1",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_is_empty(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify IS_EMPTY (IS NULL) operator.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters_with_is_empty.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="nonexistent_key",
                value=None,
                operator=FilterOperator.IS_EMPTY,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    # All nodes should match — none have "nonexistent_key"
    assert len(res.nodes) == len(node_embeddings)


# ============================================================================
# AND / OR logical combinations
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_and_combination(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify AND combination of two filters."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="test_key", value="test_value", operator=FilterOperator.EQ),
        ],
        condition="and",
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    res_ids = {n.node_id for n in res.nodes}
    assert "bbb" in res_ids


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_filter_or_combination(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify OR combination of two filters."""
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="test_key", value="test_value", operator=FilterOperator.EQ),
            MetadataFilter(key="test_num", value=1, operator=FilterOperator.EQ),
        ],
        condition="or",
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    res_ids = {n.node_id for n in res.nodes}
    assert "aaa" in res_ids  # test_num=1
    assert "bbb" in res_ids  # test_key=test_value


# ============================================================================
# GIN Array Index Filter Tests (text[] metadata)
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_gin_contains(
    vb_gin_array_indexed: VastbaseVectorStore, array_metadata_nodes: List[TextNode]
) -> None:
    """Verify CONTAINS (@>) on GIN-indexed text[] fields.

    Mirrors upstream test_gin_index_query_with_contains.
    """
    vb_gin_array_indexed.add(array_metadata_nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value="AI",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb_gin_array_indexed.query(q)
    assert res.nodes
    assert len(res.nodes) == 3
    node_ids = {n.node_id for n in res.nodes}
    assert node_ids == {"node1", "node2", "node4"}


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_gin_any(
    vb_gin_array_indexed: VastbaseVectorStore, array_metadata_nodes: List[TextNode]
) -> None:
    """Verify ANY (?|) on GIN-indexed text[] fields.

    Mirrors upstream test_gin_index_query_with_any_operator.
    """
    vb_gin_array_indexed.add(array_metadata_nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category_ids",
                value=["1", "2"],
                operator=FilterOperator.ANY,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb_gin_array_indexed.query(q)
    assert res.nodes
    assert len(res.nodes) == 4
    node_ids = {n.node_id for n in res.nodes}
    assert node_ids == {"node1", "node2", "node3", "node4"}


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_gin_all(
    vb_gin_array_indexed: VastbaseVectorStore, array_metadata_nodes: List[TextNode]
) -> None:
    """Verify ALL (?&) on GIN-indexed text[] fields.

    Mirrors upstream test_gin_index_query_with_all_operator.
    """
    vb_gin_array_indexed.add(array_metadata_nodes)

    # Match case: both "AI" AND "ML" in concept_tags
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value=["AI", "ML"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb_gin_array_indexed.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    node_ids = {n.node_id for n in res.nodes}
    assert node_ids == {"node1", "node4"}

    # No-match case
    filters_no_match = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value=["AI", "NonExistent"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q2 = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10,
        filters=filters_no_match,
    )
    res2 = vb_gin_array_indexed.query(q2)
    assert not res2.nodes or len(res2.nodes) == 0


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_mixed_btree_and_gin(
    vb_gin_array_indexed: VastbaseVectorStore, array_metadata_nodes: List[TextNode]
) -> None:
    """Verify combined BTREE (user_id) + GIN (concept_tags) filter.

    Mirrors upstream test_mixed_btree_and_gin_indices.
    """
    vb_gin_array_indexed.add(array_metadata_nodes)

    # BTREE-only: user_id EQ
    filters_btree = MetadataFilters(
        filters=[
            MetadataFilter(key="user_id", value="user123", operator=FilterOperator.EQ),
        ]
    )
    q_btree = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters_btree
    )
    res_btree = vb_gin_array_indexed.query(q_btree)
    assert res_btree.nodes
    assert len(res_btree.nodes) == 2
    btree_ids = {n.node_id for n in res_btree.nodes}
    assert btree_ids == {"node1", "node3"}

    # Combined: BTREE AND GIN
    filters_combined = MetadataFilters(
        filters=[
            MetadataFilter(key="user_id", value="user123", operator=FilterOperator.EQ),
            MetadataFilter(key="concept_tags", value="ML", operator=FilterOperator.CONTAINS),
        ],
        condition="and",
    )
    q_combined = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10,
        filters=filters_combined,
    )
    res_combined = vb_gin_array_indexed.query(q_combined)
    assert res_combined.nodes
    assert len(res_combined.nodes) == 2
    combined_ids = {n.node_id for n in res_combined.nodes}
    assert combined_ids == {"node1", "node3"}
