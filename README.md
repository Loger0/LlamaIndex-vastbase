# LlamaIndex Vastbase Vector Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Vastbase vector store adapter for LlamaIndex, providing a drop-in replacement for `PGVectorStore` using [pyvastbase](https://pypi.org/project/pyvastbase/) as the backend. All vector operations go through the pyvastbase Collection API — no raw SQL, no SQLAlchemy, no psycopg2.

## Features

- **Full feature parity** with LlamaIndex's `PGVectorStore` (llama-index-vector-stores-postgres v0.8.1)
- **Four query modes**: DEFAULT (cosine similarity), SPARSE/TEXT_SEARCH (ILIKE fallback), HYBRID (dense + sparse merge), MMR (raises `ValueError` — use LlamaIndex's `VectorIndexRetriever` instead)
- **Sync + async** dual implementation with full API parity
- **14 metadata filter operators** (EQ, GT, LT, NE, GTE, LTE, IN, NIN, CONTAINS, TEXT_MATCH, TEXT_MATCH_INSENSITIVE, IS_EMPTY, ANY, ALL)
- **HNSW graph index** via pyvastbase `IndexParams.graph_index()`
- **FLOAT_VECTOR + FLOAT16_VECTOR** (half-precision) embedding support
- **`customize_search_fn` callback** for advanced search customization
- **IndexNode round-trip** support for index-based retrieval

## Installation

```bash
pip install llama-index-vector-stores-vastbase
```

### Requirements

- Python >= 3.10
- Vastbase V3 (3.0.8+) with vector support enabled
- `pyvastbase >= 0.2.7`
- `llama-index-core >= 0.13.0, < 0.15`

## Quick Start

```python
from llama_index.vector_stores.vastbase import VastbaseVectorStore
from llama_index.core.schema import TextNode

# Create a vector store
store = VastbaseVectorStore.from_params(
    host="localhost",
    port=15432,
    database="vastbase",
    user="your_user",
    password="your_password",
    table_name="my_documents",
    embed_dim=1536,
)

# Add nodes
nodes = [
    TextNode(text="Hello world", id_="node1", embedding=[0.1] * 1536),
    TextNode(text="Goodbye world", id_="node2", embedding=[0.2] * 1536),
]
store.add(nodes)

# Query
from llama_index.core.vector_stores.types import VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=[0.1] * 1536,
    similarity_top_k=2,
)
result = store.query(query)
for node in result.nodes:
    print(f"{node.node_id}: {node.text}")

# Cleanup
store.close()
```

## Usage

### Constructor

```python
# Using from_params (recommended)
store = VastbaseVectorStore.from_params(
    host="172.16.105.107",
    port=15432,
    database="vastbase",
    user="aidev",
    password="secret",
    table_name="llamaindex",         # collection name
    schema_name="public",
    embed_dim=1536,                   # embedding dimension
    hybrid_search=False,              # enable HYBRID query mode
    hnsw_kwargs={                     # HNSW index configuration
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 100,
    },
    use_halfvec=False,                # use FLOAT16_VECTOR
    perform_setup=True,               # auto-create collection on first use
    debug=False,
    customize_search_fn=None,         # optional search callback
)
```

### Adding Nodes

```python
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

nodes = [
    TextNode(
        text="Document about machine learning",
        id_="doc1",
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc1")},
        extra_info={"topic": "ml", "year": 2024},
        embedding=[0.1] * 1536,
    ),
]
ids = store.add(nodes)
```

### Query Modes

**DEFAULT** — Cosine similarity vector search:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

q = VectorStoreQuery(
    query_embedding=[0.1] * 1536,
    similarity_top_k=5,
    mode=VectorStoreQueryMode.DEFAULT,
)
result = store.query(q)
```

**SPARSE / TEXT_SEARCH** — Keyword text search (ILIKE fallback):

```python
q = VectorStoreQuery(
    query_str="machine learning",
    similarity_top_k=5,
    mode=VectorStoreQueryMode.SPARSE,
)
result = store.query(q)
```

**HYBRID** — Dense vector + sparse text search with deduplication:

```python
q = VectorStoreQuery(
    query_embedding=[0.1] * 1536,
    query_str="machine learning",
    similarity_top_k=5,
    mode=VectorStoreQueryMode.HYBRID,
)
result = store.query(q)
```

**MMR** — Not supported (raises `ValueError`). Use LlamaIndex's `VectorIndexRetriever` for MMR reranking.

### Metadata Filtering

```python
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="topic", value="ml", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)

# With query
q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=5, filters=filters)
result = store.query(q)

# With get_nodes
nodes = store.get_nodes(filters=filters)
```

### CRUD Operations

```python
# Get nodes by ID
nodes = store.get_nodes(node_ids=["doc1", "doc2"])

# Delete by ref_doc_id
store.delete(ref_doc_id="doc1")

# Delete specific nodes
store.delete_nodes(node_ids=["doc1", "doc2"])

# Delete with filters
store.delete_nodes(filters=my_filters)

# Clear all data
store.clear()

# Close connection
store.close()
```

### Async API

All methods have async equivalents with an `a` prefix:

```python
# Async add
ids = await store.async_add(nodes)

# Async query
result = await store.aquery(q)

# Async delete
await store.adelete(ref_doc_id="doc1")
await store.adelete_nodes(node_ids=["doc1"])

# Async get / clear
nodes = await store.aget_nodes(node_ids=["doc1"])
await store.aclear()

# Async close
await store.aclose()
```

### customize_search_fn

Inject custom logic into every search call:

```python
def add_custom_expr(params: dict, **kwargs) -> dict:
    """Add a custom filter expression to every search."""
    params["expr"] = f"({params.get('expr', 'true')}) AND metadata_->>'status' = 'active'"
    return params

store = VastbaseVectorStore.from_params(
    host="localhost",
    port=15432,
    database="vastbase",
    user="user",
    password="pass",
    customize_search_fn=add_custom_expr,
)
```

## API Reference

| Method | Description |
|--------|-------------|
| `from_params(...)` | Factory method to create a store from connection parameters |
| `add(nodes, **kwargs)` | Insert nodes, returns list of node IDs |
| `async_add(nodes, **kwargs)` | Async version of `add()` |
| `query(query, **kwargs)` | Query the store (dispatches by mode) |
| `aquery(query, **kwargs)` | Async version of `query()` |
| `delete(ref_doc_id, **kwargs)` | Delete nodes by ref_doc_id |
| `adelete(ref_doc_id, **kwargs)` | Async version of `delete()` |
| `delete_nodes(node_ids, filters, **kwargs)` | Delete nodes by IDs and/or filters |
| `adelete_nodes(node_ids, filters, **kwargs)` | Async version of `delete_nodes()` |
| `get_nodes(node_ids, filters)` | Retrieve nodes by IDs and/or filters |
| `aget_nodes(node_ids, filters)` | Async version of `get_nodes()` |
| `clear()` | Remove all data from the collection |
| `aclear()` | Async version of `clear()` |
| `close()` | Close the Vastbase client connection |
| `aclose()` | Async version of `close()` |

## Differences from PGVectorStore

| Feature | PGVectorStore | VastbaseVectorStore |
|---------|--------------|-------------------|
| **Database driver** | SQLAlchemy + psycopg2/asyncpg + pgvector | pyvastbase (VastbaseClient + Collection API) |
| **Vector type** | pgvector `Vector(N)` / `HalfVec(N)` | Vastbase native `FLOAT_VECTOR` / `FLOAT16_VECTOR` |
| **Table creation** | SQLAlchemy `declarative_base` + `type()` | pyvastbase `CollectionSchema` + `FieldSchema` |
| **HNSW index** | Raw DDL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef_construction)` |
| **Full-text search** | `to_tsvector()` / `to_tsquery()` / `ts_rank()` | Client-side ILIKE fallback with word-boundary scoring |
| **HYBRID search** | Parallel dense + sparse, dedup by node_id | Same pattern (dense first, then unseen sparse) |
| **MMR mode** | Delegates to VectorIndexRetriever | `ValueError` (same behavior) |
| **Array operators** | PG JSONB `?|`, `?&`, `@>` | Client-side in-memory filtering (ANY, ALL, CONTAINS) |
| **Connection** | PostgreSQL connection string URL | Individual host/port/database/user/password params |
| **Async** | asyncpg + SQLAlchemy asyncio | pyvastbase `AsyncCollection` + `AsyncConnections` |
| **Session params** | `SET ivfflat.probes`, `SET hnsw.ef_search` | Passed directly via search API params |

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"
pip install pytest pytest-asyncio

# Unit tests (no database required)
python -m pytest tests/test_collection_init.py tests/test_crud.py tests/test_filter.py tests/test_search.py tests/test_async.py -v

# Integration tests (requires Vastbase at 172.16.105.107:15432)
python -m pytest tests/test_integration.py -v

# All tests
python -m pytest tests/ -v
```

### Project Structure

```
llama-index-vector-stores-vastbase/
├── pyproject.toml
├── README.md
├── llama_index/
│   └── vector_stores/
│       └── vastbase/
│           ├── __init__.py       # exports VastbaseVectorStore
│           └── base.py           # full implementation (~1950 lines)
└── tests/
    ├── conftest.py               # shared fixtures + Vastbase connection
    ├── test_collection_init.py   # 9 tests — collection creation/schema
    ├── test_crud.py              # 15 tests — add/delete/get_nodes/clear
    ├── test_filter.py            # 21 tests — metadata filter translation
    ├── test_search.py            # 20 tests — query modes (DEFAULT/SPARSE/HYBRID/MMR)
    ├── test_async.py             # 13 tests — async API parity
    ├── test_integration.py       # 6 tests — E2E against real Vastbase
    ├── TEST_PLAN.md              # test delivery report
    └── NYQUIST_MAP.md            # Nyquist verification mapping
```

## License

MIT
