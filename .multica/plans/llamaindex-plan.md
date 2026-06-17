# LlamaIndex VastbaseVectorStore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llama-index-vector-stores-vastbase` — a standalone pip package implementing `VastbaseVectorStore(BasePydanticVectorStore)` that replaces PGVectorStore's SQLAlchemy+pgvector stack with pyvastbase Collection API.

**Architecture:** Faithful Adapter — single-file `base.py` (~1200 lines) mirroring upstream PGVectorStore's internal structure. VastbaseClient for connection, CollectionSchema+FieldSchema for table creation, expr strings for filters, client.search/query/hybrid for the 4 query modes.

**Tech Stack:** Python >=3.9, llama-index-core>=0.13.0<0.15, pyvastbase>=0.2.7, pytest, pytest-asyncio

## Global Constraints

- pyvastbase>=0.2.7 (no psycopg2, asyncpg, sqlalchemy, pgvector)
- llama-index-core>=0.13.0,<0.15
- Class name: `VastbaseVectorStore(BasePydanticVectorStore)`
- Package name: `llama-index-vector-stores-vastbase`
- Package version: 0.1.0
- Single-file implementation in `llama_index/vector_stores/vastbase/base.py`
- All 13 abstract methods from BasePydanticVectorStore must be implemented
- Sync+async method pairs for all operations
- Filter translation uses PG-compatible JSON operators (Q2 decision)
- Full-text search uses to_tsvector/to_tsquery (Q3 decision)
- customize_search_fn callback hook for query customization (Q4 decision)
- use_halfvec parameter maps to FLOAT16_VECTOR (Q5 decision)
- Error handling follows PGVectorStore pattern: _logger.warning + initialization_fail_on_error

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Create | Package metadata, deps: llama-index-core + pyvastbase |
| `llama_index/vector_stores/vastbase/__init__.py` | Create | Export VastbaseVectorStore |
| `llama_index/vector_stores/vastbase/base.py` | Create | Full implementation (~1200 lines) |
| `tests/test_vastbase_vector_store.py` | Create | Unit tests (mock pyvastbase) |
| `tests/test_vastbase_integration.py` | Create | Integration tests (real Vastbase) |

---

### Task 1: Package scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `llama_index/vector_stores/vastbase/__init__.py`

**Produces:** Importable (but empty) package structure

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p llama_index/vector_stores/vastbase
mkdir -p tests
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
description = "Vastbase vector store integration for LlamaIndex"
readme = "README.md"
requires-python = ">=3.9,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[tool.setuptools.packages.find]
include = ["llama_index*"]
```

- [ ] **Step 3: Write __init__.py**

```python
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

__all__ = ["VastbaseVectorStore"]
```

- [ ] **Step 4: Verify package is importable**

Run: `python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"`
Expected: `ImportError` (base module doesn't exist yet) — confirms package structure is correct but implementation pending

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llama_index/vector_stores/vastbase/__init__.py
git commit -m "feat: add package scaffolding for llama-index-vector-stores-vastbase"
```

---

### Task 2: VastbaseVectorStore class skeleton with __init__ and from_params

**Files:**
- Create: `llama_index/vector_stores/vastbase/base.py`
- Create: `tests/test_vastbase_vector_store.py`

**Produces:**
- `VastbaseVectorStore.__init__(host, port, database, user, password, table_name, schema_name, embed_dim, hybrid_search, text_search_config, use_jsonb, use_halfvec, hnsw_kwargs, perform_setup, debug, initialization_fail_on_error, indexed_metadata_keys, customize_search_fn) -> None`
- `VastbaseVectorStore.from_params(...) -> VastbaseVectorStore`
- `VastbaseVectorStore.client -> Any` (property, returns None before init)

- [ ] **Step 1: Write failing test for instantiation**

File: `tests/test_vastbase_vector_store.py`
```python
import pytest
from llama_index.vector_stores.vastbase import VastbaseVectorStore

def test_create_store_with_defaults():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    assert store.host == "localhost"
    assert store.port == 15432
    assert store.table_name == "llamaindex"
    assert store.schema_name == "public"
    assert store.embed_dim == 1536
    assert store.hybrid_search == False
    assert store.use_halfvec == False

def test_from_params_factory():
    store = VastbaseVectorStore.from_params(
        host="db.example.com", port=5432, database="mydb",
        user="admin", password="secret", table_name="docs",
        embed_dim=768, use_jsonb=True,
    )
    assert store.host == "db.example.com"
    assert store.table_name == "docs"
    assert store.embed_dim == 768
    assert store.use_jsonb == True

def test_client_returns_none_before_init():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    assert store.client is None

def test_stores_text_and_flat_metadata():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    assert store.stores_text == True
    assert store.flat_metadata == False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: FAIL — `ImportError` or `ModuleNotFoundError` (base.py doesn't exist)

- [ ] **Step 3: Write class skeleton with __init__ and from_params**

File: `llama_index/vector_stores/vastbase/base.py`
```python
"""Vastbase vector store for LlamaIndex."""
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.vector_stores.types import BasePydanticVectorStore

_logger = logging.getLogger(__name__)


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase Vector Store for LlamaIndex.

    Uses pyvastbase Collection API to replace PGVectorStore's
    SQLAlchemy + pgvector driver stack.

    Examples:
        ```python
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        vector_store = VastbaseVectorStore.from_params(
            host="172.16.105.107",
            port=15432,
            database="vastbase",
            user="aidev",
            password="Vbase_123456",
            table_name="my_docs",
            embed_dim=1536,
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    # Connection parameters
    host: str
    port: int = 15432
    database: str = "vastbase"
    user: str = "aidev"
    password: str = ""

    # Collection parameters
    table_name: str = "llamaindex"
    schema_name: str = "public"

    # Vector configuration
    embed_dim: int = 1536
    use_halfvec: bool = False

    # Full-text search configuration
    hybrid_search: bool = False
    text_search_config: str = "english"

    # Metadata configuration
    use_jsonb: bool = False

    # Index configuration
    hnsw_kwargs: Optional[Dict[str, Any]] = None

    # Behavior configuration
    perform_setup: bool = True
    debug: bool = False
    initialization_fail_on_error: bool = False

    # Private attributes
    _client: Any = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _collection_name: str = PrivateAttr(default=None)
    _customize_search_fn: Optional[Callable] = PrivateAttr(default=None)

    def __init__(
        self,
        host: str,
        port: int = 15432,
        database: str = "vastbase",
        user: str = "aidev",
        password: str = "",
        table_name: str = "llamaindex",
        schema_name: str = "public",
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        use_jsonb: bool = False,
        use_halfvec: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        perform_setup: bool = True,
        debug: bool = False,
        initialization_fail_on_error: bool = False,
        indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None,
        customize_search_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            use_jsonb=use_jsonb,
            use_halfvec=use_halfvec,
            hnsw_kwargs=hnsw_kwargs,
            perform_setup=perform_setup,
            debug=debug,
            initialization_fail_on_error=initialization_fail_on_error,
        )
        self._customize_search_fn = customize_search_fn

    @classmethod
    def class_name(cls) -> str:
        return "VastbaseVectorStore"

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int = 15432,
        database: str = "vastbase",
        user: str = "aidev",
        password: str = "",
        table_name: str = "llamaindex",
        schema_name: str = "public",
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        use_jsonb: bool = False,
        use_halfvec: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        perform_setup: bool = True,
        debug: bool = False,
        initialization_fail_on_error: bool = False,
        indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None,
        customize_search_fn: Optional[Callable] = None,
    ) -> "VastbaseVectorStore":
        return cls(
            host=host, port=port, database=database, user=user, password=password,
            table_name=table_name, schema_name=schema_name,
            hybrid_search=hybrid_search, text_search_config=text_search_config,
            embed_dim=embed_dim, use_jsonb=use_jsonb, use_halfvec=use_halfvec,
            hnsw_kwargs=hnsw_kwargs, perform_setup=perform_setup, debug=debug,
            initialization_fail_on_error=initialization_fail_on_error,
            indexed_metadata_keys=indexed_metadata_keys,
            customize_search_fn=customize_search_fn,
        )

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._client
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add VastbaseVectorStore class skeleton with __init__ and from_params"
```

---

### Task 3: DBEmbeddingRow, _node_to_row_dict, _db_rows_to_query_result

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `DBEmbeddingRow(NamedTuple)` — internal result struct
- `_node_to_row_dict(self, node: BaseNode) -> Dict[str, Any]` — convert BaseNode to insert dict
- `_db_rows_to_query_result(self, rows: List[DBEmbeddingRow]) -> VectorStoreQueryResult` — convert query results

- [ ] **Step 1: Write failing test for _node_to_row_dict**

Append to `tests/test_vastbase_vector_store.py`:
```python
from llama_index.core.schema import TextNode
from llama_index.vector_stores.vastbase.base import DBEmbeddingRow


def test_node_to_row_dict():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    node = TextNode(id_="node-1", text="hello world", embedding=[0.1, 0.2, 0.3])
    node.metadata = {"ref_doc_id": "doc-1", "page": 1}
    row = store._node_to_row_dict(node)
    assert row["node_id"] == "node-1"
    assert row["text"] == "hello world"
    assert row["embedding"] == [0.1, 0.2, 0.3]
    assert row["metadata_"]["ref_doc_id"] == "doc-1"


def test_db_rows_to_query_result():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    rows = [
        DBEmbeddingRow(node_id="n1", text="t1", metadata={"ref_doc_id": "d1"}, similarity=0.95),
        DBEmbeddingRow(node_id="n2", text="t2", metadata={"ref_doc_id": "d2"}, similarity=0.80),
    ]
    result = store._db_rows_to_query_result(rows)
    assert len(result.nodes) == 2
    assert result.ids == ["n1", "n2"]
    assert result.similarities == [0.95, 0.80]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_node_to_row_dict tests/test_vastbase_vector_store.py::test_db_rows_to_query_result -v`
Expected: FAIL — `AttributeError` (methods not defined)

- [ ] **Step 3: Implement DBEmbeddingRow, _node_to_row_dict, _db_rows_to_query_result**

Append to `llama_index/vector_stores/vastbase/base.py` (before class definition):
```python
from typing import NamedTuple

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    custom_fields: dict = {}
    similarity: float = 0.0
```

Append inside VastbaseVectorStore class:
```python
    def _node_to_row_dict(self, node: BaseNode) -> Dict[str, Any]:
        metadata = node_to_metadata_dict(
            node, remove_text=True, flat_metadata=self.flat_metadata
        )
        return {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "metadata_": metadata,
            "embedding": node.get_embedding(),
        }

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for row in rows:
            try:
                node = metadata_dict_to_node(row.metadata)
                node.set_content(str(row.text))
            except Exception:
                node = TextNode(
                    id_=row.node_id, text=row.text, metadata=row.metadata
                )
            if row.custom_fields:
                node.metadata["custom_fields"] = row.custom_fields
            similarities.append(row.similarity)
            ids.append(row.node_id)
            nodes.append(node)
        return VectorStoreQueryResult(
            nodes=nodes, similarities=similarities, ids=ids
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add DBEmbeddingRow, _node_to_row_dict, _db_rows_to_query_result"
```

---

### Task 4: _initialize, _create_collection_if_not_exists, close

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `_initialize(self) -> None` — create VastbaseClient + optionally create collection/index
- `_create_collection_if_not_exists(self) -> None` — CollectionSchema + has_collection check
- `close(self) -> None` — dispose client

- [ ] **Step 1: Write failing test for _initialize with mock**

Append to `tests/test_vastbase_vector_store.py`:
```python
from unittest.mock import patch, MagicMock


def test_initialize_creates_client():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    assert not store._is_initialized

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True  # collection already exists
        mock_client_cls.return_value = mock_client

        store._initialize()

        mock_client_cls.assert_called_once_with(
            host="localhost", port=15432, database="test", user="test", password="test"
        )
        assert store._is_initialized
        assert store._client is mock_client


def test_initialize_creates_collection_when_not_exists():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test", perform_setup=True)

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False  # collection doesn't exist
        mock_client_cls.return_value = mock_client

        store._initialize()

        mock_client.has_collection.assert_called_once()
        mock_client.create_collection_with_schema.assert_called_once()


def test_close_disposes_client():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client_cls.return_value = mock_client
        store._initialize()

    store.close()
    mock_client.close.assert_called_once()
    assert not store._is_initialized
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_initialize_creates_client -v`
Expected: FAIL

- [ ] **Step 3: Implement _initialize, _create_collection_if_not_exists, close**

Append to imports at top of `base.py`:
```python
from pyvastbase import VastbaseClient
from pyvastbase import DataType, CollectionSchema, FieldSchema
```

Append inside VastbaseVectorStore class:
```python
    def _initialize(self) -> None:
        if self._is_initialized:
            return
        fail_on_error = self.initialization_fail_on_error

        self._client = VastbaseClient(
            host=self.host, port=self.port,
            database=self.database, user=self.user, password=self.password,
        )
        self._collection_name = f"{self.schema_name}.data_{self.table_name}"

        if self.perform_setup:
            try:
                self._create_collection_if_not_exists()
            except Exception as e:
                _logger.warning(f"Vastbase Setup: Error creating collection: {e}")
                if fail_on_error:
                    raise
            if self.hnsw_kwargs is not None:
                try:
                    self._create_hnsw_index()
                except Exception as e:
                    _logger.warning(f"Vastbase Setup: Error creating HNSW index: {e}")
                    if fail_on_error:
                        raise
        self._is_initialized = True

    def _create_collection_if_not_exists(self) -> None:
        if self._client.has_collection(self._collection_name):
            return
        vector_dtype = DataType.FLOAT16_VECTOR if self.use_halfvec else DataType.FLOAT_VECTOR
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.TEXT),
            FieldSchema(name="metadata_", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=vector_dtype, dim=self.embed_dim),
        ]
        if self.hybrid_search:
            fields.append(FieldSchema(name="text_search_tsv", dtype=DataType.TEXT))
        schema = CollectionSchema(name=self._collection_name, fields=fields)
        self._client.create_collection_with_schema(schema)

    def close(self) -> None:
        if not self._is_initialized:
            return
        self._client.close()
        self._is_initialized = False
```

Stub for `_create_hnsw_index` (will be fully implemented in Task 14):
```python
    def _create_hnsw_index(self) -> None:
        """Create HNSW/Graph index on embedding field. Stub — full implementation in Task 14."""
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add _initialize, _create_collection_if_not_exists, close"
```

---

### Task 5: add and async_add

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `add(self, nodes: List[BaseNode], **add_kwargs) -> List[str]`
- `async_add(self, nodes: List[BaseNode], **kwargs) -> List[str]`

- [ ] **Step 1: Write failing tests for add and async_add**

Append to `tests/test_vastbase_vector_store.py`:
```python
import pytest


def test_add_inserts_nodes():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    node1 = TextNode(id_="n1", text="hello", embedding=[0.1, 0.2])
    node2 = TextNode(id_="n2", text="world", embedding=[0.3, 0.4])

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.insert.return_value = MagicMock(insert_count=2)
        mock_cls.return_value = mock_client

        ids = store.add([node1, node2])

        assert ids == ["n1", "n2"]
        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args[0][0] == store._collection_name
        assert len(call_args[0][1]) == 2


@pytest.mark.asyncio
async def test_async_add_inserts_nodes():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    node1 = TextNode(id_="n1", text="hello", embedding=[0.1, 0.2])

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        with patch("llama_index.vector_stores.vastbase.base.AsyncCollection") as mock_async:
            mock_async_col = MagicMock()
            mock_async_col.insert = MagicMock()
            mock_async.return_value = mock_async_col

            ids = await store.async_add([node1])

            assert ids == ["n1"]
            mock_async_col.insert.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_add_inserts_nodes -v`
Expected: FAIL — `AttributeError: 'VastbaseVectorStore' object has no attribute 'add'`

- [ ] **Step 3: Implement add and async_add**

Append inside VastbaseVectorStore class:
```python
    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        self._initialize()
        rows = [self._node_to_row_dict(node) for node in nodes]
        self._client.insert(self._collection_name, rows)
        return [node.node_id for node in nodes]

    async def async_add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        from pyvastbase import AsyncCollection

        self._initialize()
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        rows = [self._node_to_row_dict(node) for node in nodes]
        await self._async_collection.insert(rows)
        return [node.node_id for node in nodes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: 11 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add add and async_add methods"
```

---

### Task 6: Filter translation — OPERATOR_MAP, _build_filter_clause, _recursively_apply_filters

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `OPERATOR_MAP: Dict[FilterOperator, str]` — 14 operators mapped to SQL
- `_build_filter_clause(self, filter_: MetadataFilter) -> str` — single filter to expr substring
- `_recursively_apply_filters(self, filters: MetadataFilters) -> str` — AND/OR nesting

- [ ] **Step 1: Write failing tests for filter translation**

Append to `tests/test_vastbase_vector_store.py`:
```python
from llama_index.core.vector_stores.types import (
    FilterOperator, FilterCondition, MetadataFilter, MetadataFilters,
)


def test_build_filter_clause_eq():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    f = MetadataFilter(key="page", value=5, operator=FilterOperator.EQ)
    clause = store._build_filter_clause(f)
    assert "metadata_->>'page'" in clause
    assert "=" in clause
    assert "5" in clause


def test_build_filter_clause_in():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    f = MetadataFilter(key="tag", value=["a", "b"], operator=FilterOperator.IN)
    clause = store._build_filter_clause(f)
    assert "IN" in clause
    assert "'a'" in clause
    assert "'b'" in clause


def test_build_filter_clause_contains():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    f = MetadataFilter(key="tags", value="ml", operator=FilterOperator.CONTAINS)
    clause = store._build_filter_clause(f)
    assert "@>" in clause


def test_build_filter_clause_is_empty():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    f = MetadataFilter(key="deleted", value=None, operator=FilterOperator.IS_EMPTY)
    clause = store._build_filter_clause(f)
    assert "IS NULL" in clause


def test_recursively_apply_filters_and():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="page", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="tag", value=["a"], operator=FilterOperator.IN),
        ],
        condition=FilterCondition.AND,
    )
    clause = store._recursively_apply_filters(filters)
    assert " AND " in clause


def test_recursively_apply_filters_nested_or():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    inner = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    outer = MetadataFilters(
        filters=[inner, MetadataFilter(key="c", value=3, operator=FilterOperator.GT)],
        condition=FilterCondition.AND,
    )
    clause = store._recursively_apply_filters(outer)
    assert " AND " in clause
    assert " OR " in clause
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_build_filter_clause_eq -v`
Expected: FAIL

- [ ] **Step 3: Implement filter translation**

Append to imports at top of `base.py`:
```python
from llama_index.core.vector_stores.types import (
    FilterOperator, MetadataFilters, MetadataFilter,
)
```

Append before class definition:
```python
OPERATOR_MAP = {
    FilterOperator.EQ: "=",
    FilterOperator.GT: ">",
    FilterOperator.LT: "<",
    FilterOperator.NE: "!=",
    FilterOperator.GTE: ">=",
    FilterOperator.LTE: "<=",
    FilterOperator.IN: "IN",
    FilterOperator.NIN: "NOT IN",
    FilterOperator.CONTAINS: "@>",
    FilterOperator.TEXT_MATCH: "LIKE",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "ILIKE",
    FilterOperator.IS_EMPTY: "IS NULL",
    FilterOperator.ANY: "?|",
    FilterOperator.ALL: "?&",
}
```

Append inside VastbaseVectorStore class:
```python
    def _build_filter_clause(self, filter_: MetadataFilter) -> str:
        op = OPERATOR_MAP.get(filter_.operator, "=")
        if filter_.operator in (FilterOperator.IN, FilterOperator.NIN):
            vals = ", ".join(f"'{e}'" for e in filter_.value)
            return f"metadata_->>'{filter_.key}' {op} ({vals})"
        elif filter_.operator in (FilterOperator.ANY, FilterOperator.ALL):
            vals = ", ".join(f"'{e}'" for e in filter_.value)
            return f"metadata_::jsonb->'{filter_.key}' {op} array[{vals}]"
        elif filter_.operator == FilterOperator.CONTAINS:
            return f"metadata_::jsonb->'{filter_.key}' {op} '[\"{filter_.value}\"]'"
        elif filter_.operator in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
            return f"metadata_->>'{filter_.key}' {op} '%{filter_.value}%'"
        elif filter_.operator == FilterOperator.IS_EMPTY:
            return f"metadata_->>'{filter_.key}' {op}"
        else:
            try:
                float(filter_.value)
                return f"(metadata_->>'{filter_.key}')::float {op} {filter_.value}"
            except (ValueError, TypeError):
                return f"metadata_->>'{filter_.key}' {op} '{filter_.value}'"

    def _recursively_apply_filters(self, filters: MetadataFilters) -> str:
        if not filters.filters:
            return "true"
        if filters.condition not in ("and", "or"):
            raise ValueError(f"Invalid condition: {filters.condition}")
        joiner = " AND " if filters.condition == "and" else " OR "
        parts = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                parts.append(f"({self._recursively_apply_filters(f)})")
            else:
                parts.append(f"({self._build_filter_clause(f)})")
        return joiner.join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "filter"`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add filter translation — OPERATOR_MAP, _build_filter_clause, _recursively_apply_filters"
```

---

### Task 7: _query_with_score and _aquery_with_score (DEFAULT mode)

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `_query_with_score(self, embedding, limit, metadata_filters, **kwargs) -> List[DBEmbeddingRow]`
- `_aquery_with_score(self, embedding, limit, metadata_filters, **kwargs) -> List[DBEmbeddingRow]`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_query_with_score_calls_client_search():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    embedding = [0.1, 0.2, 0.3]

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        # Mock search to return results
        mock_result = MagicMock()
        mock_result.id = "n1"
        mock_result.distance = 0.05
        mock_result.data = {"node_id": "n1", "text": "hello", "metadata_": {"ref_doc_id": "d1"}}
        mock_client.search.return_value = [[mock_result]]
        mock_cls.return_value = mock_client

        rows = store._query_with_score(embedding, limit=5)

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 5
        assert call_kwargs["param"]["metric_type"] == "COSINE"
        assert len(rows) == 1
        assert rows[0].node_id == "n1"
        assert rows[0].similarity == pytest.approx(0.95)  # 1 - 0.05


def test_query_with_score_with_customize_fn():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    store._customize_search_fn = lambda params, **kw: {**params, "limit": 20}
    embedding = [0.1, 0.2, 0.3]

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[]]  # empty results
        mock_cls.return_value = mock_client

        store._query_with_score(embedding, limit=10)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 20  # customized by hook
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_query_with_score_calls_client_search -v`
Expected: FAIL

- [ ] **Step 3: Implement _query_with_score and _aquery_with_score**

Append inside VastbaseVectorStore class:
```python
    def _query_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        search_params = {"metric_type": "COSINE"}
        if self.hnsw_kwargs:
            ef = kwargs.get("hnsw_ef_search") or self.hnsw_kwargs.get("hnsw_ef_search")
            if ef:
                search_params["ef"] = int(ef)

        expr = None
        if metadata_filters:
            expr = self._recursively_apply_filters(metadata_filters)

        if self._customize_search_fn:
            params = {"expr": expr, "limit": limit, "param": search_params}
            params = self._customize_search_fn(params, **kwargs)
            expr = params.get("expr")
            limit = params.get("limit", limit)
            search_params = params.get("param", search_params)

        results = self._client.search(
            self._collection_name,
            data=[embedding],
            limit=limit,
            expr=expr,
            param=search_params,
            output_fields=["node_id", "text", "metadata_"],
        )
        rows = []
        for r in results[0]:
            data = r.data if hasattr(r, "data") else {}
            similarity = 1.0 - r.distance if r.distance is not None else 0.0
            rows.append(DBEmbeddingRow(
                node_id=data.get("node_id", getattr(r, "id", "")),
                text=data.get("text", ""),
                metadata=data.get("metadata_", {}),
                similarity=similarity,
            ))
        return rows

    async def _aquery_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        from pyvastbase import AsyncCollection

        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)

        search_params = {"metric_type": "COSINE"}
        if self.hnsw_kwargs:
            ef = kwargs.get("hnsw_ef_search") or self.hnsw_kwargs.get("hnsw_ef_search")
            if ef:
                search_params["ef"] = int(ef)

        expr = None
        if metadata_filters:
            expr = self._recursively_apply_filters(metadata_filters)

        if self._customize_search_fn:
            params = {"expr": expr, "limit": limit, "param": search_params}
            params = self._customize_search_fn(params, **kwargs)
            expr = params.get("expr")
            limit = params.get("limit", limit)
            search_params = params.get("param", search_params)

        results = await self._async_collection.search(
            data=[embedding],
            limit=limit,
            expr=expr,
            param=search_params,
            output_fields=["node_id", "text", "metadata_"],
        )
        rows = []
        for r in results[0]:
            data = r.data if hasattr(r, "data") else {}
            similarity = 1.0 - r.distance if r.distance is not None else 0.0
            rows.append(DBEmbeddingRow(
                node_id=data.get("node_id", getattr(r, "id", "")),
                text=data.get("text", ""),
                metadata=data.get("metadata_", {}),
                similarity=similarity,
            ))
        return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "query_with_score"`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add _query_with_score and _aquery_with_score (DEFAULT mode)"
```

---

### Task 8: delete, adelete, delete_nodes, adelete_nodes

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `delete(self, ref_doc_id, **delete_kwargs) -> None`
- `adelete(self, ref_doc_id, **delete_kwargs) -> None`
- `delete_nodes(self, node_ids, filters, **delete_kwargs) -> None`
- `adelete_nodes(self, node_ids, filters, **delete_kwargs) -> None`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_delete_by_ref_doc_id():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store.delete("doc-123")

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert "metadata_->>'ref_doc_id' = 'doc-123'" in str(call_args)


def test_delete_nodes_by_ids():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store.delete_nodes(node_ids=["n1", "n2"])

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert "node_id IN (" in str(call_args)
        assert "'n1'" in str(call_args)
        assert "'n2'" in str(call_args)


def test_delete_nodes_with_filters():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    filters = MetadataFilters(
        filters=[MetadataFilter(key="status", value="deleted", operator=FilterOperator.EQ)]
    )

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store.delete_nodes(filters=filters)

        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert "metadata_->>'status'" in str(call_args)


def test_delete_nodes_noop_when_empty():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store.delete_nodes(node_ids=None, filters=None)

        mock_client.delete.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_delete_by_ref_doc_id -v`
Expected: FAIL

- [ ] **Step 3: Implement delete methods**

Append inside VastbaseVectorStore class:
```python
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._initialize()
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        self._client.delete(self._collection_name, expr=expr)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from pyvastbase import AsyncCollection

        self._initialize()
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        await self._async_collection.delete(expr=expr)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        if not node_ids and not filters:
            return
        self._initialize()
        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            expr_parts.append(self._recursively_apply_filters(filters))
        expr = " AND ".join(expr_parts)
        self._client.delete(self._collection_name, expr=expr)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        from pyvastbase import AsyncCollection

        if not node_ids and not filters:
            return
        self._initialize()
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            expr_parts.append(self._recursively_apply_filters(filters))
        expr = " AND ".join(expr_parts)
        await self._async_collection.delete(expr=expr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "delete"`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add delete, adelete, delete_nodes, adelete_nodes"
```

---

### Task 9: get_nodes, aget_nodes, clear, aclear

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `get_nodes(self, node_ids, filters) -> List[BaseNode]`
- `aget_nodes(self, node_ids, filters) -> List[BaseNode]`
- `clear(self) -> None`
- `aclear(self) -> None`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_get_nodes_by_ids():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.query.return_value = [
            {"node_id": "n1", "text": "hello", "metadata_": {"ref_doc_id": "d1"}, "embedding": [0.1, 0.2]}
        ]
        mock_cls.return_value = mock_client

        nodes = store.get_nodes(node_ids=["n1"])

        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"
        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args.kwargs
        assert "node_id IN (" in call_kwargs["expr"]


def test_get_nodes_requires_ids_or_filters():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        with pytest.raises(AssertionError):
            store.get_nodes(node_ids=None, filters=None)


def test_clear_truncates_collection():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store.clear()

        mock_client.truncate_collection.assert_called_once_with(store._collection_name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_get_nodes_by_ids -v`
Expected: FAIL

- [ ] **Step 3: Implement get_nodes, clear and async variants**

Append inside VastbaseVectorStore class:
```python
    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        assert node_ids is not None or filters is not None, (
            "Either node_ids or filters must be provided"
        )
        self._initialize()
        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            expr_parts.append(self._recursively_apply_filters(filters))
        expr = " AND ".join(expr_parts) if expr_parts else "true"

        results = self._client.query(
            self._collection_name,
            expr=expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )
        nodes = []
        for item in results:
            node_id = item.get("node_id", "")
            text = item.get("text", "")
            metadata = item.get("metadata_", {})
            embedding = item.get("embedding", None)
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(str(text))
                node.embedding = embedding
            except Exception:
                node = TextNode(id_=node_id, text=text, metadata=metadata, embedding=embedding)
            nodes.append(node)
        return nodes

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        from pyvastbase import AsyncCollection

        assert node_ids is not None or filters is not None, (
            "Either node_ids or filters must be provided"
        )
        self._initialize()
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            expr_parts.append(self._recursively_apply_filters(filters))
        expr = " AND ".join(expr_parts) if expr_parts else "true"

        results = await self._async_collection.query(
            expr=expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )
        nodes = []
        for item in results:
            node_id = item.get("node_id", "")
            text = item.get("text", "")
            metadata = item.get("metadata_", {})
            embedding = item.get("embedding", None)
            try:
                node = metadata_dict_to_node(metadata)
                node.set_content(str(text))
                node.embedding = embedding
            except Exception:
                node = TextNode(id_=node_id, text=text, metadata=metadata, embedding=embedding)
            nodes.append(node)
        return nodes

    def clear(self) -> None:
        self._initialize()
        self._client.truncate_collection(self._collection_name)

    async def aclear(self) -> None:
        from pyvastbase import AsyncCollection

        self._initialize()
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        await self._async_collection.truncate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "get_nodes or clear"`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add get_nodes, aget_nodes, clear, aclear"
```

---

### Task 10: _sparse_query_with_rank and async variant (SPARSE/TEXT_SEARCH mode)

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `_sparse_query_with_rank(self, query_str, limit, metadata_filters) -> List[DBEmbeddingRow]`
- `_async_sparse_query_with_rank(self, query_str, limit, metadata_filters) -> List[DBEmbeddingRow]`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_sparse_query_with_rank():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.query.return_value = [
            {"node_id": "n1", "text": "hello world", "metadata_": {}}
        ]
        mock_cls.return_value = mock_client

        rows = store._sparse_query_with_rank("hello world", limit=5)

        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args.kwargs
        assert "@@ to_tsquery" in call_kwargs["expr"]
        assert len(rows) == 1
        assert rows[0].node_id == "n1"


def test_sparse_query_requires_query_str():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        with pytest.raises(ValueError, match="query_str must be specified"):
            store._sparse_query_with_rank(None, limit=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_sparse_query_with_rank -v`
Expected: FAIL

- [ ] **Step 3: Implement _sparse_query_with_rank and async variant**

Append to imports at top of `base.py`:
```python
import re
```

Append inside VastbaseVectorStore class:
```python
    def _sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        if query_str is None:
            raise ValueError("query_str must be specified for a sparse vector query.")
        query_str = re.sub(r"(?!\b\.\b)\W+", " ", query_str).strip()
        query_str = query_str.replace(" ", "|")
        expr = f"text_search_tsv @@ to_tsquery('{self.text_search_config}', '{query_str}')"
        if metadata_filters:
            filter_expr = self._recursively_apply_filters(metadata_filters)
            expr = f"({expr}) AND ({filter_expr})"
        results = self._client.query(
            self._collection_name,
            expr=expr,
            output_fields=["node_id", "text", "metadata_"],
            limit=limit,
        )
        rows = []
        for item in results:
            rows.append(DBEmbeddingRow(
                node_id=item.get("node_id", ""),
                text=item.get("text", ""),
                metadata=item.get("metadata_", {}),
                similarity=1.0,
            ))
        return rows

    async def _async_sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> List[DBEmbeddingRow]:
        from pyvastbase import AsyncCollection

        if query_str is None:
            raise ValueError("query_str must be specified for a sparse vector query.")
        if self._async_collection is None:
            self._async_collection = AsyncCollection(self._collection_name)
        query_str = re.sub(r"(?!\b\.\b)\W+", " ", query_str).strip()
        query_str = query_str.replace(" ", "|")
        expr = f"text_search_tsv @@ to_tsquery('{self.text_search_config}', '{query_str}')"
        if metadata_filters:
            filter_expr = self._recursively_apply_filters(metadata_filters)
            expr = f"({expr}) AND ({filter_expr})"
        results = await self._async_collection.query(
            expr=expr,
            output_fields=["node_id", "text", "metadata_"],
            limit=limit,
        )
        rows = []
        for item in results:
            rows.append(DBEmbeddingRow(
                node_id=item.get("node_id", ""),
                text=item.get("text", ""),
                metadata=item.get("metadata_", {}),
                similarity=1.0,
            ))
        return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "sparse"`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add _sparse_query_with_rank and async variant (SPARSE/TEXT_SEARCH mode)"
```

---

### Task 11: _hybrid_query and _async_hybrid_query (HYBRID mode)

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `_hybrid_query(self, query: VectorStoreQuery, **kwargs) -> List[DBEmbeddingRow]`
- `_async_hybrid_query(self, query: VectorStoreQuery, **kwargs) -> List[DBEmbeddingRow]`
- `_dedup_results(results: List[DBEmbeddingRow]) -> List[DBEmbeddingRow]` (module-level)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
from llama_index.vector_stores.vastbase.base import _dedup_results


def test_dedup_results():
    rows = [
        DBEmbeddingRow(node_id="n1", text="a", metadata={}, similarity=0.9),
        DBEmbeddingRow(node_id="n2", text="b", metadata={}, similarity=0.8),
        DBEmbeddingRow(node_id="n1", text="a", metadata={}, similarity=0.7),  # duplicate
    ]
    deduped = _dedup_results(rows)
    assert len(deduped) == 2
    assert {r.node_id for r in deduped} == {"n1", "n2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_dedup_results -v`
Expected: FAIL (function not defined)

- [ ] **Step 3: Implement _hybrid_query, _async_hybrid_query, _dedup_results**

Append at module level (after class definition):
```python
def _dedup_results(results: List[DBEmbeddingRow]) -> List[DBEmbeddingRow]:
    seen_ids = set()
    deduped = []
    for result in results:
        if result.node_id not in seen_ids:
            deduped.append(result)
            seen_ids.add(result.node_id)
    return deduped
```

Append inside VastbaseVectorStore class:
```python
    def _hybrid_query(
        self, query: "VectorStoreQuery", **kwargs: Any
    ) -> List[DBEmbeddingRow]:
        if query.alpha is not None:
            _logger.warning("vastbase hybrid search does not support alpha parameter.")
        sparse_top_k = query.sparse_top_k or query.similarity_top_k
        dense_results = self._query_with_score(
            query.query_embedding, query.similarity_top_k, query.filters, **kwargs
        )
        sparse_results = self._sparse_query_with_rank(
            query.query_str, sparse_top_k, query.filters
        )
        return _dedup_results(dense_results + sparse_results)

    async def _async_hybrid_query(
        self, query: "VectorStoreQuery", **kwargs: Any
    ) -> List[DBEmbeddingRow]:
        import asyncio

        if query.alpha is not None:
            _logger.warning("vastbase hybrid search does not support alpha parameter.")
        sparse_top_k = query.sparse_top_k or query.similarity_top_k
        dense_results, sparse_results = await asyncio.gather(
            self._aquery_with_score(
                query.query_embedding, query.similarity_top_k, query.filters, **kwargs
            ),
            self._async_sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            ),
        )
        return _dedup_results(list(dense_results) + list(sparse_results))
```

Add import for VectorStoreQuery at top:
```python
from llama_index.core.vector_stores.types import (
    FilterOperator, MetadataFilters, MetadataFilter,
    VectorStoreQuery, VectorStoreQueryMode,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py::test_dedup_results -v`
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add _hybrid_query, _async_hybrid_query, _dedup_results (HYBRID mode)"
```

---

### Task 12: _query_with_embedding, _mmr_query, _async_mmr_query (MMR mode)

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `_query_with_embedding(self, embedding, limit, metadata_filters, **kwargs) -> List[Tuple[DBEmbeddingRow, List[float]]]`
- `_prepare_mmr_query(self, query, **kwargs) -> Tuple[int, Optional[float]]`
- `_mmr_rerank_results(self, query, results, mmr_threshold) -> Optional[VectorStoreQueryResult]`
- `_mmr_query(self, query, **kwargs) -> VectorStoreQueryResult`
- `_async_mmr_query(self, query, **kwargs) -> VectorStoreQueryResult`

- [ ] **Step 1: Write failing test for _prepare_mmr_query**

Append to `tests/test_vastbase_vector_store.py`:
```python
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

def test_prepare_mmr_query():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
    )
    prefetch_k, mmr_threshold = store._prepare_mmr_query(query)
    assert prefetch_k == 20  # 5 * DEFAULT_MMR_PREFETCH_FACTOR (4.0)
    assert mmr_threshold is None


def test_mmr_query_requires_embedding():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    query = VectorStoreQuery(similarity_top_k=5, mode=VectorStoreQueryMode.MMR)

    with pytest.raises(ValueError, match="MMR query requires query_embedding"):
        store._prepare_mmr_query(query)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_prepare_mmr_query -v`
Expected: FAIL

- [ ] **Step 3: Implement MMR methods**

Append to imports:
```python
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
```

Append constant before class:
```python
DEFAULT_MMR_PREFETCH_FACTOR = 4.0
```

Append inside VastbaseVectorStore class:
```python
    def _prepare_mmr_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[int, Optional[float]]:
        if query.query_embedding is None:
            raise ValueError("MMR query requires query_embedding")
        if (
            kwargs.get("mmr_prefetch_factor") is not None
            and kwargs.get("mmr_prefetch_k") is not None
        ):
            raise ValueError(
                "'mmr_prefetch_factor' and 'mmr_prefetch_k' "
                "cannot coexist in a call to query()"
            )
        mmr_prefetch_k = kwargs.get("mmr_prefetch_k")
        if mmr_prefetch_k is not None:
            prefetch_k = int(mmr_prefetch_k)
        else:
            prefetch_k = int(
                query.similarity_top_k
                * kwargs.get("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)
            )
        prefetch_k = max(prefetch_k, query.similarity_top_k)
        mmr_threshold = (
            query.mmr_threshold
            if query.mmr_threshold is not None
            else kwargs.get("mmr_threshold")
        )
        if mmr_threshold is not None and not (0 <= mmr_threshold <= 1):
            raise ValueError(
                f"mmr_threshold must be between 0 and 1, got {mmr_threshold}"
            )
        return prefetch_k, mmr_threshold

    def _query_with_embedding(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[Tuple[DBEmbeddingRow, List[float]]]:
        search_params = {"metric_type": "COSINE"}
        if self.hnsw_kwargs:
            ef = kwargs.get("hnsw_ef_search") or self.hnsw_kwargs.get("hnsw_ef_search")
            if ef:
                search_params["ef"] = int(ef)
        expr = None
        if metadata_filters:
            expr = self._recursively_apply_filters(metadata_filters)
        results = self._client.search(
            self._collection_name,
            data=[embedding],
            limit=limit,
            expr=expr,
            param=search_params,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )
        rows = []
        for r in results[0]:
            data = r.data if hasattr(r, "data") else {}
            similarity = 1.0 - r.distance if r.distance is not None else 0.0
            emb = data.get("embedding", [])
            rows.append((
                DBEmbeddingRow(
                    node_id=data.get("node_id", ""),
                    text=data.get("text", ""),
                    metadata=data.get("metadata_", {}),
                    similarity=similarity,
                ),
                list(emb) if emb else [],
            ))
        return rows

    def _mmr_rerank_results(
        self,
        query: VectorStoreQuery,
        results_with_embeddings: List[Tuple[DBEmbeddingRow, List[float]]],
        mmr_threshold: Optional[float],
    ) -> Optional[VectorStoreQueryResult]:
        if not results_with_embeddings:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        embeddings = [emb for _, emb in results_with_embeddings]
        node_ids = [row.node_id for row, _ in results_with_embeddings]
        valid_indices = [i for i, emb in enumerate(embeddings) if emb]
        if not valid_indices:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        valid_embeddings = [embeddings[i] for i in valid_indices]
        valid_node_ids = [node_ids[i] for i in valid_indices]
        if len(valid_embeddings) < query.similarity_top_k:
            _logger.warning(
                f"Not enough valid embeddings for MMR: "
                f"{len(valid_embeddings)} < {query.similarity_top_k}. "
                f"Falling back to regular search."
            )
            return None
        mmr_similarities, mmr_ids = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=valid_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=valid_node_ids,
            mmr_threshold=mmr_threshold,
        )
        result_map = {row.node_id: row for row, _ in results_with_embeddings}
        ordered_rows = []
        for sim, nid in zip(mmr_similarities, mmr_ids):
            if nid in result_map:
                row = result_map[nid]
                ordered_rows.append(DBEmbeddingRow(
                    node_id=row.node_id, text=row.text,
                    metadata=row.metadata, custom_fields=row.custom_fields,
                    similarity=sim,
                ))
        return self._db_rows_to_query_result(ordered_rows)

    def _mmr_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        prefetch_k, mmr_threshold = self._prepare_mmr_query(query, **kwargs)
        db_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("mmr_prefetch_factor", "mmr_prefetch_k", "mmr_threshold")
        }
        results_with_embeddings = self._query_with_embedding(
            query.query_embedding, prefetch_k, query.filters, **db_kwargs
        )
        result = self._mmr_rerank_results(query, results_with_embeddings, mmr_threshold)
        if result is not None:
            return result
        rows = self._query_with_score(
            query.query_embedding, query.similarity_top_k, query.filters, **db_kwargs
        )
        return self._db_rows_to_query_result(rows)

    async def _async_mmr_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        prefetch_k, mmr_threshold = self._prepare_mmr_query(query, **kwargs)
        db_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("mmr_prefetch_factor", "mmr_prefetch_k", "mmr_threshold")
        }
        # Async MMR: use sync _query_with_embedding for now (simplification)
        # Full async coverage via AsyncCollection can be added in a later iteration
        results_with_embeddings = self._query_with_embedding(
            query.query_embedding, prefetch_k, query.filters, **db_kwargs
        )
        result = self._mmr_rerank_results(query, results_with_embeddings, mmr_threshold)
        if result is not None:
            return result
        rows = await self._aquery_with_score(
            query.query_embedding, query.similarity_top_k, query.filters, **db_kwargs
        )
        return self._db_rows_to_query_result(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "mmr"`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add _mmr_query, _async_mmr_query, _query_with_embedding (MMR mode)"
```

---

### Task 13: query and aquery dispatcher

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:**
- `query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult`
- `aquery(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_query_default_mode():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    query = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=5, mode=VectorStoreQueryMode.DEFAULT)

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.search.return_value = [[]]
        mock_cls.return_value = mock_client

        result = store.query(query)
        assert isinstance(result, VectorStoreQueryResult)
        mock_client.search.assert_called_once()


def test_query_invalid_mode_raises():
    store = VastbaseVectorStore(host="localhost", port=15432, database="test", user="test", password="test")
    query = VectorStoreQuery(query_embedding=[0.1], similarity_top_k=5, mode="invalid_mode")

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        with pytest.raises(ValueError, match="Invalid query mode"):
            store.query(query)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vastbase_vector_store.py::test_query_default_mode -v`
Expected: FAIL

- [ ] **Step 3: Implement query and aquery dispatchers**

Append inside VastbaseVectorStore class:
```python
    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.HYBRID:
            rows = self._hybrid_query(query, **kwargs)
        elif query.mode in [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH]:
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            rows = self._sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            )
        elif query.mode == VectorStoreQueryMode.MMR:
            return self._mmr_query(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            rows = self._query_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")
        return self._db_rows_to_query_result(rows)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.HYBRID:
            rows = await self._async_hybrid_query(query, **kwargs)
        elif query.mode in [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH]:
            sparse_top_k = query.sparse_top_k or query.similarity_top_k
            rows = await self._async_sparse_query_with_rank(
                query.query_str, sparse_top_k, query.filters
            )
        elif query.mode == VectorStoreQueryMode.MMR:
            return await self._async_mmr_query(query, **kwargs)
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            rows = await self._aquery_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")
        return self._db_rows_to_query_result(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vastbase_vector_store.py -v -k "query"` 
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: add query and aquery dispatchers"
```

---

### Task 14: _create_hnsw_index full implementation

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_vastbase_vector_store.py`

**Produces:** Full `_create_hnsw_index` using IndexParams.graph_index()

- [ ] **Step 1: Write failing test for _create_hnsw_index**

Append to `tests/test_vastbase_vector_store.py`:
```python
def test_create_hnsw_index_calls_client():
    store = VastbaseVectorStore(
        host="localhost", port=15432, database="test", user="test", password="test",
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 100},
    )

    with patch("llama_index.vector_stores.vastbase.base.VastbaseClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_cls.return_value = mock_client

        store._create_hnsw_index()

        mock_client.create_index.assert_called_once()
        call_kwargs = mock_client.create_index.call_args.kwargs
        assert call_kwargs["field_name"] == "embedding"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vastbase_vector_store.py::test_create_hnsw_index_calls_client -v`
Expected: PASS (stub exists, but verify create_index is actually called with right params)

- [ ] **Step 3: Replace stub with full implementation**

Replace the stub `_create_hnsw_index` method in base.py:
```python
    def _create_hnsw_index(self) -> None:
        from pyvastbase import IndexParams

        if "hnsw_m" not in self.hnsw_kwargs or "hnsw_ef_construction" not in self.hnsw_kwargs:
            raise ValueError(
                "hnsw_kwargs must contain 'hnsw_m' and 'hnsw_ef_construction'"
            )
        hnsw_m = self.hnsw_kwargs["hnsw_m"]
        hnsw_ef_construction = self.hnsw_kwargs["hnsw_ef_construction"]
        params = IndexParams.graph_index(
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
        )
        self._client.create_index(
            collection_name=self._collection_name,
            field_name="embedding",
            index_params=params,
        )
```

- [ ] **Step 4: Run all tests to verify**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_vastbase_vector_store.py
git commit -m "feat: implement _create_hnsw_index with IndexParams.graph_index()"
```

---

### Task 15: Final integration test, conftest, and push

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_vastbase_integration.py`

**Produces:** Integration test skeleton with real Vastbase connection

- [ ] **Step 1: Write conftest.py with Vastbase fixture**

File: `tests/conftest.py`
```python
import os
import pytest
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def vastbase_store():
    """Create a VastbaseVectorStore connected to the test Vastbase instance."""
    store = VastbaseVectorStore.from_params(
        host=os.environ.get("VASTBASE_HOST", "172.16.105.107"),
        port=int(os.environ.get("VASTBASE_PORT", "15432")),
        database=os.environ.get("VASTBASE_DATABASE", "vastbase"),
        user=os.environ.get("VASTBASE_USER", "aidev"),
        password=os.environ.get("VASTBASE_PASSWORD", "Vbase_123456"),
        table_name="test_llamaindex_integration",
        embed_dim=128,
        perform_setup=True,
        initialization_fail_on_error=True,
    )
    yield store
    # Cleanup
    try:
        store.clear()
    except Exception:
        pass
    store.close()
```

- [ ] **Step 2: Write integration test**

File: `tests/test_vastbase_integration.py`
```python
"""Integration tests for VastbaseVectorStore — requires a running Vastbase instance."""
import pytest
from llama_index.core.schema import TextNode


@pytest.mark.integration
def test_full_crud_lifecycle(vastbase_store):
    """Test add → get_nodes → delete → clear lifecycle."""
    store = vastbase_store

    # Add nodes
    node1 = TextNode(id_="int-n1", text="Integration test node one", embedding=[0.1] * 128)
    node1.metadata = {"ref_doc_id": "doc-int-1", "page": 1}
    node2 = TextNode(id_="int-n2", text="Integration test node two", embedding=[0.2] * 128)
    node2.metadata = {"ref_doc_id": "doc-int-1", "page": 2}

    ids = store.add([node1, node2])
    assert ids == ["int-n1", "int-n2"]

    # Get nodes by ID
    nodes = store.get_nodes(node_ids=["int-n1"])
    assert len(nodes) == 1
    assert nodes[0].node_id == "int-n1"
    assert "Integration test node one" in nodes[0].get_content()

    # Delete by ref_doc_id
    store.delete(ref_doc_id="doc-int-1")

    # Verify deletion
    nodes = store.get_nodes(node_ids=["int-n1"])
    assert len(nodes) == 0


@pytest.mark.integration
def test_vector_search_default_mode(vastbase_store):
    """Test DEFAULT mode vector search."""
    store = vastbase_store

    node = TextNode(id_="int-search", text="Search test", embedding=[0.5] * 128)
    store.add([node])

    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    query = VectorStoreQuery(
        query_embedding=[0.5] * 128,
        similarity_top_k=3,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = store.query(query)
    assert len(result.nodes) > 0
    assert len(result.similarities) > 0


@pytest.mark.integration
def test_clear_collection(vastbase_store):
    """Test clearing the collection."""
    store = vastbase_store

    node = TextNode(id_="int-clear", text="Clear test", embedding=[0.3] * 128)
    store.add([node])

    store.clear()

    nodes = store.get_nodes(node_ids=["int-clear"])
    assert len(nodes) == 0
```

- [ ] **Step 3: Verify unit tests still pass**

Run: `pytest tests/test_vastbase_vector_store.py -v`
Expected: All unit tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_vastbase_integration.py
git commit -m "test: add integration tests and conftest for VastbaseVectorStore"
```

---

### Task 16: Update STATE.md, set label, notify

- [ ] **Step 1: Update .multica/STATE.md**

Replace Phase 1 section with:
```markdown
### Phase 0: 框架诊断 ✅ 完成 — 2026-06-17T09:30:00Z
### Phase 1: eco-issue-analyst ✅ 完成 — 2026-06-17T10:36:00Z
- Phase 2a Plan-Check: pending
- Phase 2b Implement: pending
- Phase 3 Framework Test: pending
```

Add to 产出物索引:
```markdown
- `.multica/specs/llamaindex-spec.md` — Design Spec (14 节, 419 行)
- `.multica/plans/llamaindex-plan.md` — Implementation Plan (16 tasks)
```

- [ ] **Step 2: Commit STATE.md and plan**

```bash
git add .multica/STATE.md .multica/plans/llamaindex-plan.md
git commit -m "state: llamaindex — 需求分析完成，进入规划审查"
```

- [ ] **Step 3: Push**

```bash
git push origin feature/llamaindex-vastbase-adapter
```

- [ ] **Step 4: Switch Multica issue label**

```bash
multica issue label remove d6ea4899-e5dd-47a9-945e-2eb23514c0a9 --label "需求分析中"
multica issue label add d6ea4899-e5dd-47a9-945e-2eb23514c0a9 --label "规划审查中"
```

- [ ] **Step 5: Reassign to task-dispatcher**

```bash
multica issue assign d6ea4899-e5dd-47a9-945e-2eb23514c0a9 --to task-dispatcher
```
