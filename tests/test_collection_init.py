"""Tests for VastbaseVectorStore initialization and collection setup.

Adapted from PGVectorStore tests:
- test_instance_creation  → test_vastbase_instance_creation
- test_hnsw_index_creation → test_hnsw_collection_initialization
- test_custom_engines      → removed (SQLAlchemy-specific)
- test_indexed_metadata    → test_indexed_metadata_collection_init

All tests use pyvastbase API exclusively — no SQLAlchemy, no psycopg2, no raw SQL.
"""

import pytest

from llama_index.vector_stores.vastbase import VastbaseVectorStore

from conftest import vastbase_not_available


# ---------------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_vastbase_instance_creation():
    """Verify VastbaseVectorStore can be instantiated and is lazy (client is None initially).

    Mirrors upstream test_instance_creation.
    """
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="test_llamaindex",
        schema_name="public",
    )
    assert isinstance(store, VastbaseVectorStore)
    # Client should be None before first operation (lazy init)
    assert store.client is None
    store.close()


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_from_params_defaults():
    """Verify from_params factory applies all defaults correctly."""
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
    )
    assert store.table_name == "llamaindex"
    assert store.schema_name == "public"
    assert store.embed_dim == 1536
    assert store.hybrid_search is False
    assert store.use_halfvec is False
    assert store.use_jsonb is False
    assert store.perform_setup is True
    assert store.debug is False
    store.close()


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_from_params_custom():
    """Verify from_params accepts custom values for all parameters."""
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="my_docs",
        schema_name="custom_schema",
        embed_dim=768,
        hybrid_search=True,
        text_search_config="simple",
        use_jsonb=True,
        use_halfvec=True,
        hnsw_kwargs={"hnsw_m": 32, "hnsw_ef_construction": 128, "hnsw_ef_search": 80},
        perform_setup=False,
        debug=True,
        initialization_fail_on_error=True,
    )
    assert store.table_name == "my_docs"
    assert store.schema_name == "custom_schema"
    assert store.embed_dim == 768
    assert store.hybrid_search is True
    assert store.use_halfvec is True
    assert store.use_jsonb is True
    assert store.perform_setup is False
    assert store.debug is True
    assert store.initialization_fail_on_error is True
    assert store.hnsw_kwargs == {
        "hnsw_m": 32,
        "hnsw_ef_construction": 128,
        "hnsw_ef_search": 80,
    }
    store.close()


# ---------------------------------------------------------------------------
# HNSW collection initialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hnsw_collection_initialization(vb_hnsw: VastbaseVectorStore):
    """Verify HNSW config is accepted and persisted.

    Mirrors upstream test_hnsw_index_creation but adapted for Vastbase —
    HNSW index is created via IndexParams.graph_index() rather than raw DDL.
    """
    assert isinstance(vb_hnsw, VastbaseVectorStore)
    assert vb_hnsw.hnsw_kwargs is not None
    assert vb_hnsw.hnsw_kwargs["hnsw_m"] == 16
    assert vb_hnsw.hnsw_kwargs["hnsw_ef_construction"] == 64
    assert vb_hnsw.hnsw_kwargs["hnsw_ef_search"] == 40


# ---------------------------------------------------------------------------
# Hybrid fulltext index initialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_hybrid_collection_initialization(vb_hybrid: VastbaseVectorStore):
    """Verify hybrid store is created with correct text_search_config.

    Mirrors upstream pg_hybrid fixture setup validation.
    """
    assert isinstance(vb_hybrid, VastbaseVectorStore)
    assert vb_hybrid.hybrid_search is True
    assert vb_hybrid.text_search_config == "english"


# ---------------------------------------------------------------------------
# Halfvec (FLOAT16_VECTOR) initialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_halfvec_collection_initialization(vb_halfvec: VastbaseVectorStore):
    """Verify halfvec store accepts use_halfvec=True and track table_name.

    Mirrors upstream pg_halfvec fixture setup.
    """
    assert isinstance(vb_halfvec, VastbaseVectorStore)
    assert vb_halfvec.use_halfvec is True


# ---------------------------------------------------------------------------
# Indexed metadata keys validation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_indexed_metadata_initialization(
    vb_indexed_metadata: VastbaseVectorStore,
):
    """Verify indexed_metadata_keys are stored correctly.

    Mirrors upstream test_indexed_metadata fixture validation.
    """
    assert vb_indexed_metadata.indexed_metadata_keys is not None
    assert ("test_text", "text") in vb_indexed_metadata.indexed_metadata_keys
    assert ("test_int", "int") in vb_indexed_metadata.indexed_metadata_keys


# ---------------------------------------------------------------------------
# GIN array indexed metadata initialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_gin_array_indexed_initialization(
    vb_gin_array_indexed: VastbaseVectorStore,
):
    """Verify GIN-indexed text[] metadata keys are stored.

    Mirrors upstream pg_gin_array_indexed fixture.
    """
    assert vb_gin_array_indexed.indexed_metadata_keys is not None
    keys = vb_gin_array_indexed.indexed_metadata_keys
    assert ("concept_tags", "text[]") in keys
    assert ("category_ids", "text[]") in keys
    assert ("user_id", "text") in keys


# ---------------------------------------------------------------------------
# Collection name generation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vastbase_not_available, reason="Vastbase is not available")
def test_collection_name_format():
    """Verify internal _collection_name follows the expected naming convention."""
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="my_table",
        schema_name="my_schema",
    )
    # Collection name should be deterministic based on table_name
    assert store._collection_name == "data_my_table"
    store.close()
