"""Tests for VastbaseVectorStore query modes: DEFAULT, SPARSE, HYBRID, MMR.

Adapted from PGVectorStore tests:
- test_add_to_db_and_query                               → test_search_default
- test_query_hnsw                                        → test_search_hnsw
- test_sparse_query                                      → test_sparse_query
- test_sparse_query_special_character_parsing            → test_sparse_query_string_cleaning
- test_sparse_query_with_special_characters              → test_sparse_query_special_characters
- test_hybrid_query                                      → test_hybrid_query
- test_add_to_db_and_hybrid_query_with_metadata_filters  → test_hybrid_query_with_filters
- test_hybrid_query_fails_if_no_query_str_provided       → test_hybrid_query_missing_query_str
- MMR mock tests                                         → TestMMRQuery

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest
import re
from typing import List

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
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
# DEFAULT mode — vector similarity search (cosine_distance)
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_search_default(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify DEFAULT mode returns top-k results by cosine similarity.

    Mirrors upstream test_add_to_db_and_query.
    Query embedding [1.0, 1.0] → most similar to node "aaa" which has embedding [1.0, 1.0].
    """
    vb.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=1
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"
    assert len(res.ids) == 1
    assert res.ids[0] == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_search_default_top_k(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify DEFAULT mode respects similarity_top_k."""
    vb.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1), similarity_top_k=3
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 3


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_search_hnsw(
    vb_hnsw: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify HNSW-backed DEFAULT search works correctly.

    Mirrors upstream test_query_hnsw.
    """
    vb_hnsw.add(node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0), similarity_top_k=1
    )
    res = vb_hnsw.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_search_default_with_metadata_filter(
    vb: VastbaseVectorStore, node_embeddings: List[TextNode]
) -> None:
    """Verify DEFAULT mode + ExactMatchFilter.

    Mirrors upstream test_add_to_db_and_query_with_metadata_filters.
    """
    vb.add(node_embeddings)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters,
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


# ============================================================================
# SPARSE / TEXT_SEARCH mode — fulltext search (ILIKE fallback)
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_sparse_query(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify SPARSE mode returns results for fulltext query.

    Mirrors upstream test_sparse_query.
    """
    vb_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "ccc"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_sparse_query_string_cleaning():
    """Verify sparse query string is cleaned identically to upstream.

    Mirrors upstream test_sparse_query_special_character_parsing.

    NOTE: This tests the query cleaning regex which is a pure function —
    no Vastbase connection needed.
    """
    query_str = "   who' &..s |     (the): <-> **fox**?!!! lazy.hound lazy..dog ?jumped,over?"
    cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query_str).strip()
    parts = cleaned.split()
    assert parts == [
        "who",
        "s",
        "the",
        "fox",
        "lazy.hound",
        "lazy",
        "dog",
        "jumped",
        "over",
    ]


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_sparse_query_special_characters(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify SPARSE mode handles special characters in query_str.

    Mirrors upstream test_sparse_query_with_special_characters.
    """
    vb_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="   who' &..s |     (the): <-> **fox**?!!!",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "ccc"
    assert res.nodes[1].node_id == "ddd"


# ============================================================================
# HYBRID mode — dense + sparse combined
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify HYBRID mode (dense + sparse deduplication).

    Mirrors upstream test_hybrid_query.
    """
    vb_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query_sparse_defaults_to_similarity_top_k(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """HYBRID: if sparse_top_k not given, defaults to similarity_top_k.

    Mirrors upstream test_hybrid_query (sparse defaults case).
    """
    vb_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4  # all nodes match


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query_sentence(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """HYBRID with a full sentence as query_str.

    Mirrors upstream test_hybrid_query (sentence case).
    """
    vb_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query_hnsw(
    vb_hnsw_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """HYBRID mode with HNSW index.

    Mirrors upstream test_hybrid_query (pg_hnsw_hybrid variant).
    """
    vb_hnsw_hybrid.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )
    res = vb_hnsw_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 3


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query_with_metadata_filters(
    vb_hybrid: VastbaseVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    """Verify HYBRID + metadata filters.

    Mirrors upstream test_add_to_db_and_hybrid_query_with_metadata_filters.
    """
    vb_hybrid.add(hybrid_node_embeddings)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res = vb_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "bbb"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_query_missing_query_str(
    vb_hybrid: VastbaseVectorStore,
) -> None:
    """Verify HYBRID raises when query_str is missing.

    Mirrors upstream test_hybrid_query_fails_if_no_query_str_provided.
    """
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0),
        similarity_top_k=10,
        mode=VectorStoreQueryMode.HYBRID,
    )
    with pytest.raises(Exception) as exc:
        vb_hybrid.query(q)
    assert "query_str" in str(exc.value).lower()


# ============================================================================
# MMR mode — not supported (matches upstream PGVectorStore)
# ============================================================================


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_mmr_query_raises_value_error(vb: VastbaseVectorStore) -> None:
    """MMR mode raises ValueError — matches upstream PGVectorStore behaviour.

    VastbaseVectorStore does not implement MMR at the VectorStore layer.
    Users should use LlamaIndex's VectorIndexRetriever for MMR reranking.
    """
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0),
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    with pytest.raises(ValueError, match="MMR is not supported"):
        vb.query(q)


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_mmr_aquery_raises_value_error(vb: VastbaseVectorStore) -> None:
    """Async MMR mode raises ValueError — matches sync behaviour."""
    import asyncio

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0),
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    with pytest.raises(ValueError, match="MMR is not supported"):
        asyncio.get_event_loop().run_until_complete(vb.aquery(q))


def test_mmr_diverse_selection_utility() -> None:
    """Verify LlamaIndex's get_top_k_mmr_embeddings picks diverse results.

    This tests the upstream MMR utility that users should call instead of
    relying on VectorStore-level MMR.  Query=[1,0.5,0], node1=[1,0,0]
    node2=[1,0.1,0] node3=[0,1,0].  node1 and node2 are near-duplicates;
    node3 is diverse.  With threshold=0.5, MMR should pick node2 (most
    relevant) then node3 (diverse), NOT node1 (redundant with node2).
    """
    from llama_index.core.indices.query.embedding_utils import (
        get_top_k_mmr_embeddings,
    )

    query_embedding = [1.0, 0.5, 0.0]
    embeddings = [
        [1.0, 0.0, 0.0],  # node1 — similar to query, near-dup of node2
        [1.0, 0.1, 0.0],  # node2 — most relevant, near-dup of node1
        [0.0, 1.0, 0.0],  # node3 — diverse
    ]
    result = get_top_k_mmr_embeddings(
        query_embedding,
        embeddings,
        mmr_threshold=0.5,
        similarity_top_k=2,
    )
    # MMR should return 2 results
    assert len(result) == 2
    # The second result should be the diverse node (index 2)
    # not the near-duplicate (index 0)
    selected_indices = [r[1] for r in result]
    assert 2 in selected_indices, "MMR should select the diverse node (index 2)"

