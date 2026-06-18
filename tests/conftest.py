"""Shared test fixtures for VastbaseVectorStore tests.

All database operations use pyvastbase — no SQLAlchemy, no psycopg2, no raw SQL.
"""

from typing import Any, Dict, Generator, List, Optional, Union

import pytest

from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)

from llama_index.vector_stores.vastbase import VastbaseVectorStore

# ---------------------------------------------------------------------------
# Vastbase connection params
# ---------------------------------------------------------------------------

VASTBASE_HOST = "172.16.105.107"
VASTBASE_PORT = 15432
VASTBASE_DATABASE = "vastbase"
VASTBASE_USER = "aidev"
VASTBASE_PASSWORD = "Vbase_123456"

TEST_TABLE_NAME = "test_llamaindex"
TEST_SCHEMA_NAME = "public"
TEST_EMBED_DIM = 2

# Detect whether Vastbase is reachable
try:
    from pyvastbase import VastbaseClient

    _client = VastbaseClient(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
    )
    _client.close()
    vastbase_not_available = False
except Exception:
    vastbase_not_available = True


# ---------------------------------------------------------------------------
# Helper — build sample embedding vectors
# ---------------------------------------------------------------------------


def _get_sample_vector(num: float) -> List[float]:
    """Return a sample embedding vector [num, 1, 1, ..., 1] of length TEST_EMBED_DIM."""
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


# ---------------------------------------------------------------------------
# VastbaseVectorStore fixtures — basic
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb() -> Generator[VastbaseVectorStore, None, None]:
    """Basic VastbaseVectorStore with default params.

    Mirrors upstream ``pg`` fixture: minimal setup, cosine distance, flat metadata.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )
    # Clear stale data from previous runs to prevent data-residue failures
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — hybrid search
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_hybrid() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with hybrid_search enabled.

    Mirrors upstream ``pg_hybrid`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_hybrid",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — indexed metadata keys
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_indexed_metadata() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with indexed_metadata_keys.

    Mirrors upstream ``pg_indexed_metadata`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_idx_meta",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
        indexed_metadata_keys={("test_text", "text"), ("test_int", "int")},
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — HNSW index
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_hnsw() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with HNSW graph index.

    Mirrors upstream ``pg_hnsw`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_hnsw",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — HNSW + hybrid
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_hnsw_hybrid() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with HNSW index and hybrid_search enabled.

    Mirrors upstream ``pg_hnsw_hybrid`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_hnsw_hybrid",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — halfvec (FLOAT16_VECTOR)
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_halfvec() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with use_halfvec=True (FLOAT16_VECTOR).

    Mirrors upstream ``pg_halfvec`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_halfvec",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        use_halfvec=True,
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


@pytest.fixture()
def vb_halfvec_hybrid() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with hybrid_search + use_halfvec.

    Mirrors upstream ``pg_halfvec_hybrid`` fixture.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_halfvec_hybrid",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        use_halfvec=True,
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — GIN array indexed metadata
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_gin_array_indexed() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with text[] GIN index metadata.

    Indexes concept_tags (text[]) and category_ids (text[]) with GIN,
    and user_id (text) with BTREE. Mirrors upstream ``pg_gin_array_indexed``.
    """
    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_gin",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        indexed_metadata_keys={
            ("concept_tags", "text[]"),
            ("category_ids", "text[]"),
            ("user_id", "text"),
        },
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VastbaseVectorStore — customize_search_fn
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_custom_search_fn() -> Generator[VastbaseVectorStore, None, None]:
    """VastbaseVectorStore with a customize_search_fn callback.

    Mirrors upstream ``pg_custom_query`` by adding a custom EXPR to every search.
    """

    def add_custom_expr(params: Dict, **kwargs: Any) -> Dict:
        params["expr"] = f"({params.get('expr', 'true')}) AND true"
        return params

    store = VastbaseVectorStore.from_params(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=TEST_TABLE_NAME + "_custom",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
        customize_search_fn=add_custom_expr,
    )
    try:
        store.clear()
    except Exception:
        pass
    yield store
    try:
        store.clear()
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parametrized fixture — dispatches to vb or vb_halfvec
# ---------------------------------------------------------------------------


@pytest.fixture()
def vb_fixture(request: pytest.FixtureRequest) -> VastbaseVectorStore:
    """Parametrized fixture returning vb or vb_halfvec.

    Usage: @pytest.mark.parametrize("vb_fixture", ["vb", "vb_halfvec"], indirect=True)
    """
    if request.param == "vb":
        return request.getfixturevalue("vb")
    elif request.param == "vb_halfvec":
        return request.getfixturevalue("vb_halfvec")
    else:
        raise ValueError(f"Unknown vb_fixture param: {request.param}")


# ---------------------------------------------------------------------------
# Node embedding fixtures — reused from upstream with minor adaptations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    """Four text nodes with controlled embeddings for deterministic search tests."""
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            extra_info={"test_num": 1},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="consectetur adipiscing elit",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_list": ["test_value_1", "test_value_2"]},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="sed do eiusmod tempor",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_2": "test_val_2"},
            embedding=_get_sample_vector(0.1),
        ),
    ]


@pytest.fixture(scope="session")
def hybrid_node_embeddings() -> List[TextNode]:
    """Nodes with English text for sparse/hybrid search tests."""
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            embedding=_get_sample_vector(5.0),
        ),
        TextNode(
            text="The fox and the hound",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ddd")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(10.0),
        ),
    ]


@pytest.fixture(scope="session")
def index_node_embeddings() -> List[BaseNode]:
    """Nodes including an IndexNode for index-node handling tests."""
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        IndexNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="aaa_ref",
            index_id="aaa",
            embedding=_get_sample_vector(5.0),
        ),
    ]


@pytest.fixture(scope="session")
def array_metadata_nodes() -> List[TextNode]:
    """Nodes with text array metadata for GIN index testing.

    Distribution:
    - node1: tags=["AI", "ML"], categories=["1", "2"], user="user123"
    - node2: tags=["AI", "NLP"], categories=["2", "3"], user="user456"
    - node3: tags=["Computer Vision", "ML"], categories=["1", "3"], user="user123"
    - node4: tags=["Deep Learning", "AI", "ML"], categories=["1", "2", "3"], user="user789"
    """
    return [
        TextNode(
            text="Artificial Intelligence and Machine Learning",
            id_="node1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node1")},
            extra_info={
                "concept_tags": ["AI", "ML"],
                "category_ids": ["1", "2"],
                "user_id": "user123",
            },
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="Natural Language Processing with AI",
            id_="node2",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node2")},
            extra_info={
                "concept_tags": ["AI", "NLP"],
                "category_ids": ["2", "3"],
                "user_id": "user456",
            },
            embedding=_get_sample_vector(0.5),
        ),
        TextNode(
            text="Computer Vision and Image Recognition",
            id_="node3",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node3")},
            extra_info={
                "concept_tags": ["Computer Vision", "ML"],
                "category_ids": ["1", "3"],
                "user_id": "user123",
            },
            embedding=_get_sample_vector(0.3),
        ),
        TextNode(
            text="Deep Learning Neural Networks",
            id_="node4",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node4")},
            extra_info={
                "concept_tags": ["Deep Learning", "AI", "ML"],
                "category_ids": ["1", "2", "3"],
                "user_id": "user789",
            },
            embedding=_get_sample_vector(0.2),
        ),
    ]
