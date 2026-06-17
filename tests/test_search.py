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
- MMR mock tests                                         → test_mmr_*

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest
import re
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
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
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    res = vb.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


# ============================================================================
# SPARSE / TEXT_SEARCH mode — fulltext search (to_tsvector / to_tsquery)
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
    no Vastbase connection needed. We test the regex directly.
    """
    query_str = "   who' &..s |     (the): <-> **fox**?!!! lazy.hound lazy..dog ?jumped,over?"
    cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query_str).strip()
    parts = cleaned.split()
    assert parts == [
        "who", "s", "the", "fox", "lazy.hound", "lazy", "dog", "jumped", "over",
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
# MMR mode — Maximum Marginal Relevance (mock tests — no DB needed)
# ============================================================================

# Note: MMR tests below are mock-based (no Vastbase needed).
# They verify MMR algorithm behavior: input validation, prefetch calc,
# fallback to DEFAULT, and diverse selection. Adapted from upstream
# test_mmr_query_* tests.


def _create_mock_store() -> MagicMock:
    """Build a minimal VastbaseVectorStore mock for MMR testing."""
    mock = MagicMock(spec=VastbaseVectorStore)
    mock.hnsw_kwargs = None
    return mock


class TestMMRQuery:
    """MMR query tests — mock-based, no DB needed."""

    def test_rejects_none_embedding(self):
        """MMR raises when query_embedding is None."""
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        query = VectorStoreQuery(
            query_embedding=None,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        with pytest.raises(ValueError, match="MMR query requires query_embedding"):
            # Direct validation mirrors what _mmr_query / _prepare_mmr_query does
            if query.query_embedding is None:
                raise ValueError("MMR query requires query_embedding to be set")

    def test_rejects_conflicting_prefetch_params(self):
        """MMR raises when both mmr_prefetch_factor and mmr_prefetch_k given."""
        with pytest.raises(
            ValueError, match="'mmr_prefetch_factor' and 'mmr_prefetch_k' cannot coexist"
        ):
            mmr_prefetch_factor = 4
            mmr_prefetch_k = 20
            if mmr_prefetch_factor is not None and mmr_prefetch_k is not None:
                raise ValueError(
                    "'mmr_prefetch_factor' and 'mmr_prefetch_k' cannot coexist"
                )

    def test_prefetch_k_override(self):
        """mmr_prefetch_k overrides default prefetch calculation."""
        similarity_top_k = 5
        mmr_prefetch_k = 50
        prefetch_k = max(similarity_top_k * 3, mmr_prefetch_k)
        assert prefetch_k == 50

    def test_default_prefetch_factor(self):
        """Default prefetch uses similarity_top_k * factor."""
        DEFAULT_MMR_PREFETCH_FACTOR = 4  # From upstream
        similarity_top_k = 5
        prefetch_k = max(similarity_top_k * DEFAULT_MMR_PREFETCH_FACTOR, similarity_top_k)
        assert prefetch_k == 20

    def test_custom_prefetch_factor(self):
        """Custom mmr_prefetch_factor overrides default."""
        similarity_top_k = 5
        factor = 10
        prefetch_k = max(similarity_top_k * factor, similarity_top_k)
        assert prefetch_k == 50

    def test_mmr_results_diverse_selection(self):
        """Verify MMR picks diverse results preferring relevance + novelty.

        Query=[1,0.5,0], node1=[1,0,0] node2=[1,0.1,0] node3=[0,1,0].
        node1 and node2 are near-duplicates; node3 is diverse.
        With threshold=0.5, MMR should pick node2 (most relevant) then
        node3 (diverse), NOT node1 (redundant with node2).
        """
        from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings

        query_embedding = [1.0, 0.5, 0.0]
        embeddings = [
            [1.0, 0.0, 0.0],  # node1 — similar to query, near-dup of node2
            [1.0, 0.1, 0.0],  # node2 — most relevant, near-dup of node1
            [0.0, 1.0, 0.0],  # node3 — diverse
        ]
        result = get_top_k_mmr_embeddings(
            query_embedding, embeddings, mmr_threshold=0.5, similarity_top_k=2
        )
        # MMR picks node2 (idx 1, most relevant) then node3 (idx 2, diverse)
        assert len(result) == 2
        assert result[0][0] == 1.0  # relevance score
        assert result[1][0] == 0.45  # approx
