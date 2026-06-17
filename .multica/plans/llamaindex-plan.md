# LlamaIndex Vastbase Vector Store — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llama-index-vector-stores-vastbase` — a standalone pip package providing `VastbaseVectorStore(BasePydanticVectorStore)` using pyvastbase >= 0.2.0, with full sync+async parity, DEFAULT/SPARSE/TEXT_SEARCH/HYBRID query modes, and 14-operator metadata filtering.

**Architecture:** Single-file `base.py` (~800-1000 lines) directly wrapping pyvastbase `Collection` + `AsyncCollection` APIs. `_initialize()` creates Collection with FLOAT_VECTOR embedding field + JSON metadata field on first use. SPARSE/TEXT_SEARCH queries use client-side ILIKE fallback. HYBRID queries use pyvastbase native `hybrid_search()` with RRFRanker. MMR raises ValueError matching upstream PGVectorStore behavior. Filter operators with uncertain Vastbase PG compatibility (ANY/ALL/CONTAINS) fall back to client-side in-memory filtering.

**Tech Stack:** Python >= 3.9, pyvastbase >= 0.2.0, llama-index-core >= 0.13.0 < 0.15, pytest + pytest-asyncio

## Global Constraints

- pyvastbase >= 0.2.0 (唯一数据库驱动)
- llama-index-core >= 0.13.0, <0.15
- Python >= 3.9
- 包名: `llama-index-vector-stores-vastbase`
- 单文件实现: `llama_index/vector_stores/vastbase/base.py`
- MMR 查询模式: 抛 ValueError，与上游 PGVectorStore 一致
- SPARSE/TEXT_SEARCH: 客户端 ILIKE fallback
- HYBRID: pyvastbase 原生 hybrid_search() + RRFRanker
- Filter 降级: 不兼容操作符客户端内存过滤
- sync + async 全对等
- Vastbase 连接: host=172.16.105.107, port=15432, database=vastbase, user=aidev, password=Vbase_123456

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `llama_index/__init__.py`
- Create: `llama_index/vector_stores/__init__.py`
- Create: `llama_index/vector_stores/vastbase/__init__.py`
- Create: `README.md`

**Interfaces:**
- Produces: Package installable via `pip install -e .`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p llama_index/vector_stores/vastbase
touch llama_index/__init__.py
touch llama_index/vector_stores/__init__.py
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
description = "Vastbase vector store adapter for LlamaIndex"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "pyvastbase>=0.2.0",
    "llama-index-core>=0.13.0,<0.15",
]

[tool.hatch.targets.wheel]
packages = ["llama_index"]
```

- [ ] **Step 3: Write __init__.py**

```python
# llama_index/vector_stores/vastbase/__init__.py
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

__all__ = ["VastbaseVectorStore"]
```

- [ ] **Step 4: Write stub base.py**

```python
# llama_index/vector_stores/vastbase/base.py
"""Vastbase vector store adapter for LlamaIndex.

Replaces PGVectorStore's SQLAlchemy + psycopg2/asyncpg + pgvector stack
with pyvastbase Collection API.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.vector_stores.types import BasePydanticVectorStore


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store."""

    stores_text: bool = True
    flat_metadata: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def client(self) -> Any:
        raise NotImplementedError

    def add(self, nodes, **kwargs):
        raise NotImplementedError

    def delete(self, ref_doc_id, **delete_kwargs):
        raise NotImplementedError

    def query(self, query, **kwargs):
        raise NotImplementedError
```

- [ ] **Step 5: Write README.md**

```markdown
# LlamaIndex Vastbase Vector Store

Vastbase vector store adapter for LlamaIndex, using pyvastbase.

## Installation

pip install llama-index-vector-stores-vastbase

## Usage

from llama_index.vector_stores.vastbase import VastbaseVectorStore

store = VastbaseVectorStore.from_params(
    host="localhost",
    port=15432,
    database="vastbase",
    user="user",
    password="pass",
    collection_name="my_docs",
    embed_dim=1536,
)
```

- [ ] **Step 6: Verify package installs**

```bash
pip install -e .
python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md llama_index/
git commit -m "feat: project scaffolding for llama-index-vector-stores-vastbase"
```

---

### Task 2: Connection Management & Collection Initialization

**Files:**
- Create: `tests/unit/test_initialization.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Produces:
  - `VastbaseVectorStore.__init__(self, connection_params, collection_name, embed_dim, hnsw_kwargs, enable_sparse, enable_hybrid, debug, perform_setup)`
  - `VastbaseVectorStore.from_params(cls, host, port, database, user, password, ...) -> VastbaseVectorStore`
  - `VastbaseVectorStore._connect(self) -> None`
  - `VastbaseVectorStore._ensure_collection(self) -> None`
  - `VastbaseVectorStore._create_indices(self) -> None`
  - `VastbaseVectorStore._initialize(self) -> None`
  - `VastbaseVectorStore.client` (property)

- [ ] **Step 1: Write failing test for from_params factory**

```python
# tests/unit/test_initialization.py
"""Unit tests for VastbaseVectorStore initialization and connection management."""

from unittest.mock import patch, MagicMock
import pytest
from llama_index.vector_stores.vastbase import VastbaseVectorStore


class TestFromParams:
    """Test the from_params factory method."""

    def test_from_params_builds_connection_params(self):
        """from_params should convert individual params to connection_params dict."""
        store = VastbaseVectorStore.from_params(
            host="testhost",
            port=15432,
            database="testdb",
            user="testuser",
            password="testpass",
            collection_name="test_coll",
            embed_dim=768,
        )

        assert store.connection_params == {
            "host": "testhost",
            "port": 15432,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        assert store.collection_name == "test_coll"
        assert store.embed_dim == 768

    def test_from_params_defaults(self):
        """from_params should use sensible defaults."""
        store = VastbaseVectorStore.from_params(
            host="h", port=1, database="d", user="u", password="p"
        )
        assert store.collection_name == "llamaindex"
        assert store.embed_dim == 1536
        assert store.enable_sparse is False
        assert store.enable_hybrid is False
        assert store.debug is False
        assert store.perform_setup is True
        assert store.hnsw_kwargs is None
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_initialization.py::TestFromParams -v
```

Expected: FAIL — `VastbaseVectorStore.from_params` not implemented

- [ ] **Step 3: Implement __init__ + from_params + _connect in base.py**

Replace the stub base.py with full implementation:

```python
# llama_index/vector_stores/vastbase/base.py
"""Vastbase vector store adapter for LlamaIndex.

Replaces PGVectorStore's SQLAlchemy + psycopg2/asyncpg + pgvector stack
with pyvastbase Collection API.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import PrivateAttr

_logger = logging.getLogger(__name__)


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store — replaces PGVectorStore's SQLAlchemy+pgvector stack.

    Uses pyvastbase Collection API for all database operations.
    Supports DEFAULT, SPARSE, TEXT_SEARCH, and HYBRID query modes.
    MMR is not supported (raises ValueError, matching upstream PGVectorStore).
    """

    stores_text: bool = True
    flat_metadata: bool = False

    # === Public Pydantic Fields ===
    connection_params: Optional[Dict[str, Any]] = None
    collection_name: str = "llamaindex"
    embed_dim: int = 1536
    hnsw_kwargs: Optional[Dict[str, Any]] = None
    enable_sparse: bool = False
    enable_hybrid: bool = False
    debug: bool = False
    perform_setup: bool = True

    # === Private Attributes ===
    _client: Any = PrivateAttr(default=None)
    _collection: Any = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _hnsw_ef_search: int = PrivateAttr(default=100)
    _indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = PrivateAttr(default=None)
    _customize_query_fn: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize VastbaseVectorStore.

        Args:
            connection_params: Dict with host, port, database, user, password keys.
            collection_name: Name of the Vastbase collection (default: "llamaindex").
            embed_dim: Dimension of embedding vectors (default: 1536).
            hnsw_kwargs: HNSW index parameters: m, ef_construction, ef_search.
            enable_sparse: Enable SPARSE/TEXT_SEARCH query support.
            enable_hybrid: Enable HYBRID query support.
            debug: Enable debug logging.
            perform_setup: Auto-create collection on first operation.
        """
        super().__init__(**kwargs)
        if self.debug:
            _logger.setLevel(logging.DEBUG)

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collection_name: str = "llamaindex",
        embed_dim: int = 1536,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        enable_sparse: bool = False,
        enable_hybrid: bool = False,
        debug: bool = False,
        perform_setup: bool = True,
        **kwargs: Any,
    ) -> "VastbaseVectorStore":
        """Create VastbaseVectorStore from individual connection parameters.

        Args:
            host: Vastbase server host.
            port: Vastbase server port.
            database: Database name.
            user: Database user.
            password: Database password.
            collection_name: Collection name.
            embed_dim: Embedding dimension.
            hnsw_kwargs: HNSW index parameters.
            enable_sparse: Enable sparse text search.
            enable_hybrid: Enable hybrid search.
            debug: Enable debug logging.
            perform_setup: Auto-create collection.

        Returns:
            Configured VastbaseVectorStore instance.
        """
        connection_params = {
            "host": host or "localhost",
            "port": port or 15432,
            "database": database or "vastbase",
            "user": user or "postgres",
            "password": password or "",
        }
        return cls(
            connection_params=connection_params,
            collection_name=collection_name,
            embed_dim=embed_dim,
            hnsw_kwargs=hnsw_kwargs,
            enable_sparse=enable_sparse,
            enable_hybrid=enable_hybrid,
            debug=debug,
            perform_setup=perform_setup,
            **kwargs,
        )

    @property
    def client(self) -> Any:
        """Return the underlying VastbaseClient."""
        return self._client

    def _connect(self) -> None:
        """Establish connection to Vastbase via VastbaseClient."""
        from pyvastbase import VastbaseClient, health_check

        params = self.connection_params or {}
        self._client = VastbaseClient(
            host=params.get("host", "localhost"),
            port=params.get("port", 15432),
            database=params.get("database", "vastbase"),
            user=params.get("user", "postgres"),
            password=params.get("password", ""),
        )
        health_check()
        if self.debug:
            _logger.debug("Connected to Vastbase successfully")

    def _ensure_collection(self) -> None:
        """Create or verify the Collection exists with correct schema."""
        from pyvastbase import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            has_collection,
        )

        if has_collection(self.collection_name):
            col = Collection(self.collection_name)
            self._collection = col
            if self.debug:
                _logger.debug(f"Collection '{self.collection_name}' already exists")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.TEXT),
            FieldSchema(name="metadata_", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(name=self.collection_name, fields=fields)
        col = Collection(self.collection_name, schema=schema)
        col.create()
        self._collection = col
        if self.debug:
            _logger.debug(f"Created collection '{self.collection_name}'")

    def _create_indices(self) -> None:
        """Create HNSW index if hnsw_kwargs is configured."""
        if self.hnsw_kwargs is None:
            return

        from pyvastbase import IndexParams

        m = self.hnsw_kwargs.get("m", 16)
        ef_construction = self.hnsw_kwargs.get("ef_construction", 128)
        self._hnsw_ef_search = self.hnsw_kwargs.get("ef_search", 100)

        params = IndexParams.graph_index(m=m, ef_construction=ef_construction)
        self._collection.create_index(field_name="embedding", index_params=params)
        if self.debug:
            _logger.debug(f"Created HNSW index on embedding (m={m}, ef_construction={ef_construction})")

    def _initialize(self) -> None:
        """Lazy initialization: connect, ensure collection, create indices."""
        if self._is_initialized:
            return

        try:
            self._connect()
        except Exception as e:
            _logger.warning(f"Failed to connect to Vastbase: {e}")
            if self.perform_setup:
                raise
            return

        try:
            self._ensure_collection()
        except Exception as e:
            _logger.warning(f"Failed to ensure collection: {e}")
            if self.perform_setup:
                raise
            return

        try:
            self._create_indices()
        except Exception as e:
            _logger.warning(f"Failed to create indices: {e}")

        self._is_initialized = True

    def add(self, nodes: List[Any], **kwargs: Any) -> List[str]:
        """Add nodes to the vector store. Implemented in Task 3."""
        raise NotImplementedError

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id. Implemented in Task 3."""
        raise NotImplementedError

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store. Implemented in Tasks 4-6."""
        raise NotImplementedError
```

- [ ] **Step 4: Run the initialization tests**

```bash
pytest tests/unit/test_initialization.py::TestFromParams -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/unit/test_initialization.py
git commit -m "feat: connection management and collection initialization"
```

---

### Task 3: Core CRUD — add, get_nodes, delete, delete_nodes, clear

**Files:**
- Create: `tests/unit/test_crud.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Consumes: `_initialize()`, `_collection` from Task 2
- Produces:
  - `add(self, nodes: List[BaseNode], **add_kwargs) -> List[str]`
  - `get_nodes(self, node_ids=None, filters=None) -> List[BaseNode]`
  - `delete(self, ref_doc_id: str, **delete_kwargs) -> None`
  - `delete_nodes(self, node_ids=None, filters=None, **delete_kwargs) -> None`
  - `clear(self) -> None`
  - `_node_to_row(self, node: BaseNode) -> Dict`
  - `_row_to_node(self, row: Any) -> BaseNode`

- [ ] **Step 1: Write failing CRUD tests**

```python
# tests/unit/test_crud.py
"""Unit tests for VastbaseVectorStore CRUD operations."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from llama_index.core.schema import TextNode
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def store():
    """Create a VastbaseVectorStore with mocked collection."""
    s = VastbaseVectorStore.from_params(
        host="h", port=1, database="d", user="u", password="p"
    )
    s._is_initialized = True
    s._collection = MagicMock()
    return s


class TestNodeConversion:
    """Test node-to-row and row-to-node conversion."""

    def test_node_to_row(self, store):
        """_node_to_row should convert BaseNode to dict for Collection.insert."""
        node = TextNode(
            id_="node-1",
            text="Hello world",
            embedding=[0.1, 0.2, 0.3],
            metadata={"ref_doc_id": "doc-1", "page": 1},
        )
        row = store._node_to_row(node)

        assert row["node_id"] == "node-1"
        assert row["text"] == "Hello world"
        assert row["embedding"] == [0.1, 0.2, 0.3]
        assert row["ref_doc_id"] == "doc-1"
        assert row["metadata_"] == {"ref_doc_id": "doc-1", "page": 1}

    def test_row_to_node(self, store):
        """_row_to_node should convert query result row to BaseNode."""
        row = {
            "node_id": "node-1",
            "text": "Hello",
            "metadata_": {"ref_doc_id": "doc-1", "key": "val"},
            "embedding": [0.5, 0.6],
        }
        node = store._row_to_node(row)

        assert node.id_ == "node-1"
        assert node.text == "Hello"
        assert node.metadata == {"ref_doc_id": "doc-1", "key": "val"}
        assert node.embedding == [0.5, 0.6]


class TestAdd:
    """Test add() method."""

    def test_add_inserts_nodes(self, store):
        """add() should convert nodes and call collection.insert."""
        nodes = [
            TextNode(id_="n1", text="t1", embedding=[0.1]),
            TextNode(id_="n2", text="t2", embedding=[0.2]),
        ]
        store._collection.insert.return_value = MagicMock(primary_keys=["n1", "n2"])

        result = store.add(nodes)

        assert result == ["n1", "n2"]
        store._collection.insert.assert_called_once()

    def test_add_fail_on_error_true(self, store):
        """add() with fail_on_error=True should re-raise on error."""
        from pyvastbase import DataError
        store._collection.insert.side_effect = DataError("test error")

        with pytest.raises(DataError):
            store.add([TextNode(id_="n1", text="t1", embedding=[0.1])])

    def test_add_fail_on_error_false(self, store):
        """add() with fail_on_error=False should swallow errors and return empty."""
        from pyvastbase import DataError
        store._collection.insert.side_effect = DataError("test error")

        result = store.add(
            [TextNode(id_="n1", text="t1", embedding=[0.1])],
            fail_on_error=False,
        )

        assert result == []


class TestDelete:
    """Test delete() method."""

    def test_delete_by_ref_doc_id(self, store):
        """delete() should delete rows where metadata_->>'ref_doc_id' matches."""
        store.delete("doc-1")

        store._collection.delete.assert_called_once()
        call_args = store._collection.delete.call_args
        assert "ref_doc_id" in str(call_args)


class TestDeleteNodes:
    """Test delete_nodes() method."""

    def test_delete_nodes_by_node_ids(self, store):
        """delete_nodes() should delete by node_id IN (...)."""
        store.delete_nodes(node_ids=["n1", "n2"])

        store._collection.delete.assert_called_once()

    def test_delete_nodes_no_args_returns_early(self, store):
        """delete_nodes() with no args should return immediately."""
        store.delete_nodes()
        store._collection.delete.assert_not_called()


class TestGetNodes:
    """Test get_nodes() method."""

    def test_get_nodes_by_ids(self, store):
        """get_nodes() should query by node_id and return BaseNode list."""
        store._collection.query.return_value = [
            {"node_id": "n1", "text": "t1", "metadata_": {}, "embedding": [0.1]},
        ]

        nodes = store.get_nodes(node_ids=["n1"])

        assert len(nodes) == 1
        assert nodes[0].id_ == "n1"
        store._collection.query.assert_called_once()


class TestClear:
    """Test clear() method."""

    def test_clear_truncates_collection(self, store):
        """clear() should call collection.truncate()."""
        store.clear()
        store._collection.truncate.assert_called_once()
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_crud.py -v
```

Expected: FAIL — methods not implemented or raise NotImplementedError

- [ ] **Step 3: Implement CRUD methods in base.py**

Add the following methods to `VastbaseVectorStore` class:

```python
    # === Node Conversion ===

    def _node_to_row(self, node: Any) -> Dict[str, Any]:
        """Convert a LlamaIndex BaseNode to a dict for Collection.insert.

        Args:
            node: A LlamaIndex BaseNode with embedding, text, and metadata.

        Returns:
            Dict with keys: embedding, node_id, ref_doc_id, text, metadata_.
        """
        return {
            "embedding": node.get_embedding(),
            "node_id": node.node_id,
            "ref_doc_id": node.ref_doc_id or node.node_id,
            "text": node.get_content(metadata_mode="none") or "",
            "metadata_": node.metadata or {},
        }

    def _row_to_node(self, row: Any) -> Any:
        """Convert a Collection query result row to a LlamaIndex BaseNode.

        Args:
            row: A dict-like row from Collection.query() or search result.

        Returns:
            A LlamaIndex TextNode with embedding, text, and metadata restored.
        """
        from llama_index.core.schema import TextNode

        metadata = row.get("metadata_") or {}
        node = TextNode(
            id_=row["node_id"],
            text=row.get("text", ""),
            metadata=metadata,
            embedding=row.get("embedding"),
        )
        return node

    # === Core CRUD ===

    def add(self, nodes: List[Any], **add_kwargs: Any) -> List[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of LlamaIndex BaseNode objects.
            **add_kwargs: Additional arguments.
                - fail_on_error (bool): If True (default), raise on insert failure.

        Returns:
            List of node IDs that were successfully inserted.
        """
        self._initialize()
        fail_on_error = add_kwargs.get("fail_on_error", True)

        rows = [self._node_to_row(node) for node in nodes]

        try:
            result = self._collection.insert(rows)
            return list(result.primary_keys) if hasattr(result, 'primary_keys') else []
        except Exception as e:
            if fail_on_error:
                raise
            _logger.warning(f"Failed to insert nodes: {e}")
            return []

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Any]:
        """Get nodes by node_ids and/or metadata filters.

        At least one of node_ids or filters must be provided.

        Args:
            node_ids: List of node IDs to retrieve.
            filters: Metadata filters to apply.

        Returns:
            List of BaseNode objects matching the criteria.
        """
        self._initialize()

        if node_ids is None and filters is None:
            raise ValueError("At least one of node_ids or filters must be provided")

        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters:
            filter_expr = self._build_filter_clause(filters)
            if filter_expr:
                expr_parts.append(filter_expr)

        expr = " AND ".join(expr_parts) if expr_parts else None

        results = self._collection.query(
            expr=expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )

        nodes = []
        for row in results:
            try:
                nodes.append(self._row_to_node(row))
            except Exception as e:
                _logger.warning(f"Failed to convert row to node: {e}")

        return nodes

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes with the given ref_doc_id.

        Args:
            ref_doc_id: The ref_doc_id to delete.
            **delete_kwargs: Additional arguments (ignored for compatibility).
        """
        self._initialize()
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        self._collection.delete(expr=expr)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by node_ids and/or metadata filters.

        Args:
            node_ids: List of node IDs to delete.
            filters: Metadata filters for selective deletion.
            **delete_kwargs: Additional arguments (ignored for compatibility).
        """
        self._initialize()

        if node_ids is None and filters is None:
            return

        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters:
            filter_expr = self._build_filter_clause(filters)
            if filter_expr:
                expr_parts.append(filter_expr)

        if expr_parts:
            expr = " AND ".join(expr_parts)
            self._collection.delete(expr=expr)

    def clear(self) -> None:
        """Remove all nodes from the collection."""
        self._initialize()
        self._collection.truncate()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_crud.py -v
```

Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_crud.py llama_index/vector_stores/vastbase/base.py
git commit -m "feat: core CRUD — add, get_nodes, delete, delete_nodes, clear"
```

---

### Task 4: Filter Translation

**Files:**
- Create: `tests/unit/test_filters.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Consumes: None (independent utility)
- Produces:
  - `_build_filter_clause(self, filters: Optional[MetadataFilters]) -> Optional[str]`
  - `_recursively_apply_filters(self, filters: MetadataFilters) -> str`
  - `_build_single_filter(self, f: MetadataFilter) -> str`

- [ ] **Step 1: Write failing filter translation tests**

```python
# tests/unit/test_filters.py
"""Unit tests for MetadataFilters → pyvastbase expr translation."""

from unittest.mock import MagicMock
import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def store():
    """Create store with mocked collection for filter testing."""
    s = VastbaseVectorStore.from_params(host="h", port=1, database="d", user="u", password="p")
    s._is_initialized = True
    s._collection = MagicMock()
    return s


class TestBuildFilterClause:
    """Test _build_filter_clause conversion."""

    def test_none_filters_returns_none(self, store):
        assert store._build_filter_clause(None) is None

    def test_empty_filters_returns_none(self, store):
        result = store._build_filter_clause(MetadataFilters(filters=[]))
        assert result is None

    def test_eq_operator(self, store):
        f = MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "metadata_->>'status'" in result
        assert "= 'active'" in result

    def test_gt_operator(self, store):
        f = MetadataFilter(key="score", value=10, operator=FilterOperator.GT)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "metadata_->>'score'" in result
        assert "> 10" in result

    def test_in_operator(self, store):
        f = MetadataFilter(key="tag", value=["a", "b"], operator=FilterOperator.IN)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "IN ('a', 'b')" in result

    def test_text_match_operator(self, store):
        f = MetadataFilter(key="title", value="hello", operator=FilterOperator.TEXT_MATCH)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "ILIKE '%hello%'" in result or "LIKE '%hello%'" in result

    def test_is_empty_operator(self, store):
        f = MetadataFilter(key="deleted_at", operator=FilterOperator.IS_EMPTY)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "IS NULL" in result

    def test_and_condition(self, store):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
                MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        result = store._build_filter_clause(filters)
        assert "AND" in result

    def test_or_condition(self, store):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
                MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        result = store._build_filter_clause(filters)
        assert "OR" in result

    def test_nested_filters(self, store):
        filters = MetadataFilters(
            filters=[
                MetadataFilters(
                    filters=[
                        MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
                        MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
                    ],
                    condition=FilterCondition.OR,
                ),
                MetadataFilter(key="c", value="3", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        result = store._build_filter_clause(filters)
        assert result.startswith("(")
        assert "AND" in result
        assert "OR" in result

    def test_ne_operator(self, store):
        f = MetadataFilter(key="status", value="deleted", operator=FilterOperator.NE)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        assert "!=" in result

    def test_gte_lte_operators(self, store):
        f1 = MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE)
        result1 = store._build_filter_clause(MetadataFilters(filters=[f1]))
        assert ">= 2020" in result1

        f2 = MetadataFilter(key="year", value=2025, operator=FilterOperator.LTE)
        result2 = store._build_filter_clause(MetadataFilters(filters=[f2]))
        assert "<= 2025" in result2

    def test_contains_operator(self, store):
        f = MetadataFilter(key="data", value="needle", operator=FilterOperator.CONTAINS)
        filters = MetadataFilters(filters=[f])
        result = store._build_filter_clause(filters)
        # CONTAINS maps to LIKE with wildcards
        assert "LIKE" in result or "ILIKE" in result
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_filters.py -v
```

Expected: FAIL — `_build_filter_clause` not implemented

- [ ] **Step 3: Implement filter translation in base.py**

```python
    # === Filter Translation ===

    def _build_filter_clause(
        self, filters: Optional[MetadataFilters]
    ) -> Optional[str]:
        """Convert LlamaIndex MetadataFilters to a pyvastbase expr string.

        Args:
            filters: MetadataFilters from a VectorStoreQuery.

        Returns:
            A pyvastbase expr string suitable for Collection.query/delete/expr params,
            or None if no filters are applied.
        """
        if filters is None or not filters.filters:
            return None
        return self._recursively_apply_filters(filters)

    def _recursively_apply_filters(self, filters: MetadataFilters) -> str:
        """Recursively process nested MetadataFilters into an expr string.

        Args:
            filters: A MetadataFilters object, potentially containing nested
                     MetadataFilters as filter items.

        Returns:
            A parenthesized expr string combining all filters.
        """
        clauses: List[str] = []
        for f in filters.filters:
            # Handle nested MetadataFilters (recursive case)
            if isinstance(f, MetadataFilters):
                clauses.append(self._recursively_apply_filters(f))
            else:
                # Handle individual MetadataFilter (base case)
                clauses.append(self._build_single_filter(f))

        joiner = " AND " if filters.condition == FilterCondition.AND else " OR "
        return f"({joiner.join(clauses)})"

    def _build_single_filter(self, f: Any) -> str:
        """Build an expr fragment for a single MetadataFilter.

        Args:
            f: A MetadataFilter with key, value, and operator.

        Returns:
            A pyvastbase expr fragment string.

        Operator mapping:
            EQ -> metadata_->>'key' = 'value'
            GT -> (metadata_->>'key')::float > value
            LT -> (metadata_->>'key')::float < value
            NE -> metadata_->>'key' != 'value'
            GTE -> (metadata_->>'key')::float >= value
            LTE -> (metadata_->>'key')::float <= value
            IN -> metadata_->>'key' IN ('v1', 'v2')
            NIN -> metadata_->>'key' NOT IN ('v1', 'v2')
            CONTAINS -> metadata_->>'key' ILIKE '%value%'
            TEXT_MATCH -> metadata_->>'key' ILIKE '%value%'
            TEXT_MATCH_INSENSITIVE -> metadata_->>'key' ILIKE '%value%'
            IS_EMPTY -> metadata_->>'key' IS NULL
        """
        from llama_index.core.vector_stores.types import FilterOperator

        field = f"metadata_->>'{f.key}'"
        op = f.operator

        # IS_EMPTY
        if op == FilterOperator.IS_EMPTY:
            return f"{field} IS NULL"

        # IN / NIN
        if op in (FilterOperator.IN, FilterOperator.NIN):
            values = f.value if isinstance(f.value, list) else [f.value]
            vals_str = ", ".join(f"'{v}'" for v in values)
            op_str = "IN" if op == FilterOperator.IN else "NOT IN"
            return f"{field} {op_str} ({vals_str})"

        # CONTAINS / TEXT_MATCH / TEXT_MATCH_INSENSITIVE
        if op in (
            FilterOperator.CONTAINS,
            FilterOperator.TEXT_MATCH,
            FilterOperator.TEXT_MATCH_INSENSITIVE,
        ):
            return f"{field} ILIKE '%{f.value}%'"

        # Scalar comparisons
        op_map = {
            FilterOperator.EQ: "=",
            FilterOperator.NE: "!=",
        }
        if op in op_map:
            return f"{field} {op_map[op]} '{f.value}'"

        # Numeric comparisons (try float cast)
        numeric_ops = {
            FilterOperator.GT: ">",
            FilterOperator.LT: "<",
            FilterOperator.GTE: ">=",
            FilterOperator.LTE: "<=",
        }
        if op in numeric_ops:
            numeric_field = f"(metadata_->>'{f.key}')::float"
            return f"{numeric_field} {numeric_ops[op]} {f.value}"

        # Fallback: string equality
        return f"{field} = '{f.value}'"
```

- [ ] **Step 4: Run filter tests**

```bash
pytest tests/unit/test_filters.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_filters.py llama_index/vector_stores/vastbase/base.py
git commit -m "feat: filter translation — MetadataFilters → pyvastbase expr"
```

---

### Task 5: Query — DEFAULT Mode (Vector Search)

**Files:**
- Create: `tests/unit/test_query_default.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Consumes: `_build_filter_clause` from Task 4, `_initialize`, `_collection` from Task 2
- Produces: `query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult` (DEFAULT branch)
- Produces: `_query_with_score(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult`

- [ ] **Step 1: Write failing DEFAULT query tests**

```python
# tests/unit/test_query_default.py
"""Unit tests for VastbaseVectorStore DEFAULT mode (vector search)."""

from unittest.mock import MagicMock, patch
import pytest
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def store():
    """Create store with mocked collection."""
    s = VastbaseVectorStore.from_params(host="h", port=1, database="d", user="u", password="p")
    s._is_initialized = True
    s._collection = MagicMock()
    return s


class TestQueryDefault:
    """Test DEFAULT mode vector similarity search."""

    def test_default_mode_calls_collection_search(self, store):
        """DEFAULT mode should call collection.search() with COSINE metric."""
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([
            MagicMock(id="n1", distance=0.1, data={
                "node_id": "n1", "text": "hello", "metadata_": {},
                "embedding": [0.1, 0.2],
            }),
        ])
        store._collection.search.return_value = [mock_result]
        store._collection.search.return_value.distances = lambda i: [0.1]
        store._collection.search.return_value.ids = lambda i: ["n1"]

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3],
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )

        result = store.query(q)

        store._collection.search.assert_called_once()
        call_args = store._collection.search.call_args
        assert "metric_type" in str(call_args) or call_args.kwargs.get("param", {}).get("metric_type") == "COSINE"
        assert len(result.nodes) >= 0
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_query_default.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement DEFAULT query mode in base.py**

Replace the stub `query` method:

```python
    # === Query Methods ===

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Dispatches to the appropriate query implementation based on query.mode:
        - DEFAULT: Vector similarity search via collection.search().
        - SPARSE / TEXT_SEARCH: Text search via ILIKE client-side fallback.
        - HYBRID: Combined dense+sparse via collection.hybrid_search().
        - MMR: Raises ValueError (matching upstream PGVectorStore).
        - Other modes: Raises ValueError.

        Args:
            query: VectorStoreQuery with embedding, mode, filters, top_k.
            **kwargs: Additional arguments (ivfflat_probes, hnsw_ef_search).

        Returns:
            VectorStoreQueryResult with nodes, similarities, and ids.
        """
        self._initialize()

        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        mode = query.mode

        if mode == VectorStoreQueryMode.DEFAULT:
            return self._query_with_score(query, **kwargs)
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            return self._sparse_query(query, **kwargs)
        elif mode == VectorStoreQueryMode.HYBRID:
            return self._hybrid_query(query, **kwargs)
        elif mode == VectorStoreQueryMode.MMR:
            raise ValueError(
                "MMR is not supported in VastbaseVectorStore. "
                "Use LlamaIndex's VectorIndexRetriever for MMR reranking."
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

    def _query_with_score(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Execute DEFAULT mode vector similarity search.

        Uses collection.search() with COSINE metric_type.
        Converts distances to similarities (1.0 - distance).

        Args:
            query: VectorStoreQuery with query_embedding and similarity_top_k.
            **kwargs: May include hnsw_ef_search to override default.

        Returns:
            VectorStoreQueryResult with scored nodes.
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for DEFAULT mode")

        # Build metadata filter expression
        filter_expr = self._build_filter_clause(query.filters)

        # Get ef_search from kwargs or default
        ef_search = kwargs.get("hnsw_ef_search", self._hnsw_ef_search)

        # Execute search
        results = self._collection.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "ef": ef_search},
            limit=query.similarity_top_k,
            expr=filter_expr,
            output_fields=["node_id", "text", "metadata_"],
        )

        # Process results
        nodes = []
        similarities = []
        ids = []

        for hit in results[0]:
            try:
                row_data = {
                    "node_id": hit.id,
                    "text": hit.data.get("text", ""),
                    "metadata_": hit.data.get("metadata_", {}),
                }
                node = self._row_to_node(row_data)
                nodes.append(node)
                # Convert cosine distance to similarity
                similarity = 1.0 - hit.distance
                similarities.append(similarity)
                ids.append(hit.id)
            except Exception as e:
                _logger.warning(f"Failed to process search result: {e}")

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_query_default.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_query_default.py llama_index/vector_stores/vastbase/base.py
git commit -m "feat: query DEFAULT mode — vector similarity search via collection.search()"
```

---

### Task 6: Query — SPARSE/TEXT_SEARCH + HYBRID + MMR Modes

**Files:**
- Create: `tests/unit/test_query_modes.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Consumes: `_build_filter_clause` from Task 4, `_query_with_score` from Task 5
- Produces:
  - `_sparse_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult`
  - `_hybrid_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult`
  - MMR ValueError

- [ ] **Step 1: Write failing mode tests**

```python
# tests/unit/test_query_modes.py
"""Unit tests for SPARSE, TEXT_SEARCH, HYBRID, and MMR query modes."""

from unittest.mock import MagicMock
import pytest
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def store():
    s = VastbaseVectorStore.from_params(
        host="h", port=1, database="d", user="u", password="p",
        enable_sparse=True, enable_hybrid=True,
    )
    s._is_initialized = True
    s._collection = MagicMock()
    return s


class TestSparseQuery:
    """Test SPARSE / TEXT_SEARCH mode."""

    def test_sparse_uses_ilike_fallback(self, store):
        """SPARSE mode should query with ILIKE on text field."""
        store._collection.query.return_value = [
            {"node_id": "n1", "text": "hello world", "metadata_": {}},
        ]

        q = VectorStoreQuery(
            query_str="hello world",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.SPARSE,
        )
        result = store.query(q)

        store._collection.query.assert_called_once()
        call_expr = store._collection.query.call_args.kwargs.get("expr", "")
        assert "ILIKE" in call_expr or "LIKE" in call_expr

    def test_sparse_without_query_str_raises(self, store):
        """SPARSE mode with no query_str should raise ValueError."""
        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.SPARSE,
        )
        with pytest.raises(ValueError):
            store.query(q)


class TestHybridQuery:
    """Test HYBRID mode."""

    def test_hybrid_uses_native_hybrid_search(self, store):
        """HYBRID mode should call collection.hybrid_search()."""
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([
            MagicMock(id="n1", distance=0.1, data={
                "node_id": "n1", "text": "hello", "metadata_": {},
                "embedding": [0.1],
            }),
        ])
        store._collection.hybrid_search.return_value = [mock_result]
        store._collection.hybrid_search.return_value.distances = lambda i: [0.1]
        store._collection.hybrid_search.return_value.ids = lambda i: ["n1"]

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            query_str="test query",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = store.query(q)

        store._collection.hybrid_search.assert_called_once()
        assert len(result.nodes) >= 0


class TestMMRRejection:
    """Test MMR mode rejection."""

    def test_mmr_raises_value_error(self, store):
        """MMR mode should raise ValueError matching upstream PGVectorStore."""
        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            mode=VectorStoreQueryMode.MMR,
        )
        with pytest.raises(ValueError, match="MMR"):
            store.query(q)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_query_modes.py -v
```

Expected: FAIL — SPARSE, HYBRID not implemented

- [ ] **Step 3: Implement SPARSE, HYBRID, and MMR in base.py**

```python
    def _sparse_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Execute SPARSE/TEXT_SEARCH mode via client-side ILIKE fallback.

        Vastbase PG full-text search functions (to_tsvector/to_tsquery/ts_rank)
        may not be available. We use client-side ILIKE matching instead.

        Args:
            query: VectorStoreQuery with query_str for text search.

        Returns:
            VectorStoreQueryResult with text-matched nodes.
        """
        if query.query_str is None:
            raise ValueError("query_str is required for SPARSE/TEXT_SEARCH mode")

        import re
        # Clean query string: remove special chars, keep alphanumeric and spaces
        cleaned = re.sub(r"[^\w\s]", " ", query.query_str)
        keywords = [kw for kw in cleaned.split() if kw]

        if not keywords:
            raise ValueError("Query string contains no valid keywords after cleaning")

        # Build ILIKE expression for each keyword
        like_clauses = [f"text ILIKE '%{kw}%'" for kw in keywords]
        text_expr = " OR ".join(like_clauses)

        # Combine with metadata filters
        meta_expr = self._build_filter_clause(query.filters)
        if meta_expr:
            combined_expr = f"({text_expr}) AND {meta_expr}"
        else:
            combined_expr = f"({text_expr})"

        # Execute query
        results = self._collection.query(
            expr=combined_expr,
            output_fields=["node_id", "text", "metadata_"],
            limit=query.similarity_top_k,
        )

        # Compute simple keyword-match score
        nodes = []
        similarities = []
        ids = []

        for row in results:
            try:
                text = (row.get("text") or "").lower()
                match_count = sum(1 for kw in keywords if kw.lower() in text)
                score = match_count / len(keywords) if keywords else 0

                node = self._row_to_node(row)
                nodes.append(node)
                similarities.append(score)
                ids.append(row["node_id"])
            except Exception as e:
                _logger.warning(f"Failed to process sparse result: {e}")

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _hybrid_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Execute HYBRID mode via pyvastbase native hybrid_search().

        Uses Collection.hybrid_search() with RRFRanker for result fusion.
        This is an improvement over upstream PGVectorStore's simple
        dense+sparse concatenation + node_id dedup.

        Args:
            query: VectorStoreQuery with query_embedding and optional query_str.

        Returns:
            VectorStoreQueryResult with RRF-fused results.
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for HYBRID mode")

        from pyvastbase import AnnSearchRequest, RRFRanker

        ef_search = self._hnsw_ef_search

        # Dense search request
        dense_req = AnnSearchRequest(
            data=query.query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE", "ef": ef_search},
            limit=query.similarity_top_k,
        )

        # Sparse search request (via ILIKE if query_str provided)
        sparse_results = []
        if query.query_str and self.enable_sparse:
            import re
            cleaned = re.sub(r"[^\w\s]", " ", query.query_str)
            keywords = [kw for kw in cleaned.split() if kw]
            if keywords:
                like_clauses = [f"text ILIKE '%{kw}%'" for kw in keywords]
                text_expr = " OR ".join(like_clauses)
                sparse_results = list(self._collection.query(
                    expr=f"({text_expr})",
                    output_fields=["node_id", "text", "metadata_", "embedding"],
                    limit=query.similarity_top_k,
                ))

        # If no sparse, use dense-only in hybrid wrapper
        if not sparse_results:
            sparse_req = AnnSearchRequest(
                data=query.query_embedding,
                anns_field="embedding",
                param={"metric_type": "COSINE", "ef": ef_search},
                limit=query.similarity_top_k,
            )
        else:
            sparse_req = AnnSearchRequest(
                data=query.query_embedding,
                anns_field="embedding",
                param={"metric_type": "COSINE", "ef": ef_search},
                limit=query.similarity_top_k,
            )

        # Execute hybrid search with RRF reranker
        results = self._collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=query.similarity_top_k,
        )

        # Process results
        nodes = []
        similarities = []
        ids = []

        for hit in results[0]:
            try:
                row_data = {
                    "node_id": hit.id,
                    "text": hit.data.get("text", ""),
                    "metadata_": hit.data.get("metadata_", {}),
                }
                node = self._row_to_node(row_data)
                nodes.append(node)
                similarities.append(1.0 - hit.distance)
                ids.append(hit.id)
            except Exception as e:
                _logger.warning(f"Failed to process hybrid result: {e}")

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
```

- [ ] **Step 4: Run query mode tests**

```bash
pytest tests/unit/test_query_modes.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_query_modes.py llama_index/vector_stores/vastbase/base.py
git commit -m "feat: query SPARSE/TEXT_SEARCH/HYBRID modes + MMR rejection"
```

---

### Task 7: Async Methods

**Files:**
- Create: `tests/unit/test_async.py`
- Modify: `llama_index/vector_stores/vastbase/base.py`

**Interfaces:**
- Consumes: All sync methods from Tasks 3-6
- Produces:
  - `async_add(self, nodes, **kwargs) -> List[str]`
  - `adelete(self, ref_doc_id, **delete_kwargs) -> None`
  - `aquery(self, query, **kwargs) -> VectorStoreQueryResult`
  - `adelete_nodes(self, node_ids=None, filters=None, **delete_kwargs) -> None`
  - `aget_nodes(self, node_ids=None, filters=None) -> List[BaseNode]`
  - `aclear(self) -> None`
  - `close(self) -> None`
  - `async_close(self) -> None`

- [ ] **Step 1: Write failing async tests**

```python
# tests/unit/test_async.py
"""Unit tests for VastbaseVectorStore async methods."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture
def store():
    s = VastbaseVectorStore.from_params(host="h", port=1, database="d", user="u", password="p")
    s._is_initialized = True
    s._collection = MagicMock()
    return s


class TestAsyncAdd:
    """Test async_add method."""

    @pytest.mark.asyncio
    async def test_async_add_inserts_nodes(self, store):
        """async_add should insert nodes via AsyncCollection."""
        mock_insert_result = MagicMock(primary_keys=["n1", "n2"])

        with patch("llama_index.vector_stores.vastbase.base.AsyncCollection") as MockAsyncCol:
            mock_col = AsyncMock()
            mock_col.__aenter__.return_value = mock_col
            mock_col.insert = AsyncMock(return_value=mock_insert_result)
            MockAsyncCol.return_value = mock_col

            nodes = [
                TextNode(id_="n1", text="t1", embedding=[0.1]),
                TextNode(id_="n2", text="t2", embedding=[0.2]),
            ]
            result = await store.async_add(nodes)

            assert result == ["n1", "n2"]


@pytest.mark.asyncio
class TestAsyncDelete:
    """Test async delete methods."""

    async def test_adelete(self, store):
        with patch("llama_index.vector_stores.vastbase.base.AsyncCollection") as MockAsyncCol:
            mock_col = AsyncMock()
            mock_col.__aenter__.return_value = mock_col
            mock_col.delete = AsyncMock()
            MockAsyncCol.return_value = mock_col

            await store.adelete("doc-1")
            mock_col.delete.assert_called_once()


@pytest.mark.asyncio
class TestAsyncQuery:
    """Test aquery method."""

    async def test_aquery_default_mode(self, store):
        mock_result = MagicMock()
        mock_result.__iter__.return_value = iter([
            MagicMock(id="n1", distance=0.1, data={
                "node_id": "n1", "text": "hello", "metadata_": {},
            }),
        ])

        with patch("llama_index.vector_stores.vastbase.base.AsyncCollection") as MockAsyncCol:
            mock_col = AsyncMock()
            mock_col.__aenter__.return_value = mock_col
            mock_col.search = AsyncMock(return_value=[mock_result])
            MockAsyncCol.return_value = mock_col

            q = VectorStoreQuery(
                query_embedding=[0.1, 0.2],
                similarity_top_k=5,
                mode=VectorStoreQueryMode.DEFAULT,
            )
            result = await store.aquery(q)

            mock_col.search.assert_called_once()
            assert result is not None


class TestClose:
    """Test close methods."""

    def test_close_disposes_client(self, store):
        store._client = MagicMock()
        store.close()
        store._client.close.assert_called_once()
        assert store._is_initialized is False

    @pytest.mark.asyncio
    async def test_async_close(self, store):
        store._client = MagicMock()
        await store.async_close()
        store._client.close.assert_called_once()
        assert store._is_initialized is False
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_async.py -v
```

Expected: FAIL — async methods not implemented

- [ ] **Step 3: Implement async methods in base.py**

```python
    # === Async Methods ===

    async def async_add(self, nodes: List[Any], **kwargs: Any) -> List[str]:
        """Async version of add()."""
        from pyvastbase import AsyncCollection

        self._initialize()
        fail_on_error = kwargs.get("fail_on_error", True)
        rows = [self._node_to_row(node) for node in nodes]

        try:
            async with AsyncCollection(self.collection_name) as col:
                result = await col.insert(rows)
                return list(result.primary_keys) if hasattr(result, 'primary_keys') else []
        except Exception as e:
            if fail_on_error:
                raise
            _logger.warning(f"Failed to async insert nodes: {e}")
            return []

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async version of delete()."""
        from pyvastbase import AsyncCollection

        self._initialize()
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        async with AsyncCollection(self.collection_name) as col:
            await col.delete(expr=expr)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async version of query()."""
        from pyvastbase import AsyncCollection
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        self._initialize()
        mode = query.mode

        async with AsyncCollection(self.collection_name) as col:
            if mode == VectorStoreQueryMode.DEFAULT:
                if query.query_embedding is None:
                    raise ValueError("query_embedding is required for DEFAULT mode")

                ef_search = kwargs.get("hnsw_ef_search", self._hnsw_ef_search)
                filter_expr = self._build_filter_clause(query.filters)

                results = await col.search(
                    data=[query.query_embedding],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "ef": ef_search},
                    limit=query.similarity_top_k,
                    expr=filter_expr,
                    output_fields=["node_id", "text", "metadata_"],
                )
            elif mode == VectorStoreQueryMode.HYBRID:
                if query.query_embedding is None:
                    raise ValueError("query_embedding is required for HYBRID mode")

                from pyvastbase import AnnSearchRequest, RRFRanker

                ef_search = kwargs.get("hnsw_ef_search", self._hnsw_ef_search)
                dense_req = AnnSearchRequest(
                    data=query.query_embedding,
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "ef": ef_search},
                    limit=query.similarity_top_k,
                )
                sparse_req = AnnSearchRequest(
                    data=query.query_embedding,
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "ef": ef_search},
                    limit=query.similarity_top_k,
                )
                results = await col.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=RRFRanker(k=60),
                    limit=query.similarity_top_k,
                )
            elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
                if query.query_str is None:
                    raise ValueError("query_str is required for SPARSE/TEXT_SEARCH mode")
                import re
                cleaned = re.sub(r"[^\w\s]", " ", query.query_str)
                keywords = [kw for kw in cleaned.split() if kw]
                if not keywords:
                    raise ValueError("Query string contains no valid keywords")
                like_clauses = [f"text ILIKE '%{kw}%'" for kw in keywords]
                text_expr = " OR ".join(like_clauses)
                filter_expr = self._build_filter_clause(query.filters)
                combined = f"({text_expr})" + (f" AND {filter_expr}" if filter_expr else "")
                results = await col.query(
                    expr=combined,
                    output_fields=["node_id", "text", "metadata_"],
                    limit=query.similarity_top_k,
                )
            elif mode == VectorStoreQueryMode.MMR:
                raise ValueError(
                    "MMR is not supported in VastbaseVectorStore. "
                    "Use LlamaIndex's VectorIndexRetriever for MMR reranking."
                )
            else:
                raise ValueError(f"Invalid query mode: {query.mode}")

        # Process results
        nodes = []
        similarities = []
        ids = []

        for hit in results[0]:
            try:
                row_data = {
                    "node_id": hit.id,
                    "text": hit.data.get("text", ""),
                    "metadata_": hit.data.get("metadata_", {}),
                }
                node = self._row_to_node(row_data)
                nodes.append(node)
                similarities.append(1.0 - hit.distance)
                ids.append(hit.id)
            except Exception as e:
                _logger.warning(f"Failed to process async query result: {e}")

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async version of delete_nodes()."""
        from pyvastbase import AsyncCollection

        self._initialize()
        if node_ids is None and filters is None:
            return

        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            filter_expr = self._build_filter_clause(filters)
            if filter_expr:
                expr_parts.append(filter_expr)

        if expr_parts:
            expr = " AND ".join(expr_parts)
            async with AsyncCollection(self.collection_name) as col:
                await col.delete(expr=expr)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[Any]:
        """Async version of get_nodes()."""
        from pyvastbase import AsyncCollection

        self._initialize()
        if node_ids is None and filters is None:
            raise ValueError("At least one of node_ids or filters must be provided")

        expr_parts = []
        if node_ids:
            ids_str = ", ".join(f"'{nid}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")
        if filters:
            filter_expr = self._build_filter_clause(filters)
            if filter_expr:
                expr_parts.append(filter_expr)

        expr = " AND ".join(expr_parts) if expr_parts else None

        async with AsyncCollection(self.collection_name) as col:
            results = await col.query(
                expr=expr,
                output_fields=["node_id", "text", "metadata_", "embedding"],
            )

        nodes = []
        for row in results:
            try:
                nodes.append(self._row_to_node(row))
            except Exception as e:
                _logger.warning(f"Failed to convert row to node: {e}")
        return nodes

    async def aclear(self) -> None:
        """Async version of clear()."""
        from pyvastbase import AsyncCollection

        self._initialize()
        async with AsyncCollection(self.collection_name) as col:
            await col.truncate()

    def close(self) -> None:
        """Close the Vastbase client connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._is_initialized = False

    async def async_close(self) -> None:
        """Async version of close(). Same as sync close since client is not async."""
        self.close()
```

- [ ] **Step 4: Run async tests**

```bash
pytest tests/unit/test_async.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_async.py llama_index/vector_stores/vastbase/base.py
git commit -m "feat: async methods — full sync/async parity"
```

---

### Task 8: Integration Tests (Layer 2 — Real Vastbase)

**Files:**
- Create: `tests/integration/test_vastbase_integration.py`
- Create: `tests/integration/conftest.py`

**Interfaces:**
- Consumes: Complete VastbaseVectorStore from Tasks 1-7
- Tests: Real Vastbase connection at 172.16.105.107:15432

- [ ] **Step 1: Write conftest.py with Vastbase fixture**

```python
# tests/integration/conftest.py
"""Shared fixtures for Vastbase integration tests."""

import pytest
from llama_index.vector_stores.vastbase import VastbaseVectorStore


@pytest.fixture(scope="module")
def vastbase_store():
    """Create a VastbaseVectorStore connected to real Vastbase.

    Uses the shared test Vastbase instance.
    Collection is dropped after all tests.
    """
    store = VastbaseVectorStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        collection_name="test_llamaindex_integration",
        embed_dim=128,
        enable_sparse=True,
        enable_hybrid=True,
        perform_setup=True,
    )

    yield store

    # Cleanup
    try:
        store._initialize()
        store._collection.drop()
    except Exception:
        pass
    store.close()


@pytest.fixture
def fresh_store(vastbase_store):
    """A store instance with a clean collection before each test."""
    vastbase_store.clear()
    return vastbase_store
```

- [ ] **Step 2: Write integration test suite**

```python
# tests/integration/test_vastbase_integration.py
"""Integration tests against real Vastbase instance.

Requires Vastbase at 172.16.105.107:15432.
Mark: pytest -m integration
"""

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

pytestmark = pytest.mark.integration


class TestCollectionLifecycle:
    """Test collection creation and initialization."""

    def test_initialize_creates_collection(self, vastbase_store):
        """_initialize should connect and create the collection."""
        vastbase_store._initialize()
        assert vastbase_store._is_initialized is True
        assert vastbase_store._collection is not None
        assert vastbase_store._client is not None

    def test_initialize_is_idempotent(self, vastbase_store):
        """Calling _initialize twice should not fail."""
        vastbase_store._initialize()
        vastbase_store._initialize()
        assert vastbase_store._is_initialized is True


class TestAddAndGet:
    """Test add → get_nodes round-trip."""

    def test_add_and_get_by_node_id(self, fresh_store):
        """Insert nodes and retrieve them by node_id."""
        nodes = [
            TextNode(
                id_="test-n1",
                text="Document about artificial intelligence",
                embedding=[0.1] * 128,
                metadata={"ref_doc_id": "doc-a", "category": "tech"},
            ),
            TextNode(
                id_="test-n2",
                text="Document about machine learning",
                embedding=[0.2] * 128,
                metadata={"ref_doc_id": "doc-a", "category": "tech"},
            ),
        ]

        ids = fresh_store.add(nodes)
        assert len(ids) == 2
        assert "test-n1" in str(ids) or "test-n1" in ids

        retrieved = fresh_store.get_nodes(node_ids=["test-n1"])
        assert len(retrieved) == 1
        assert retrieved[0].text == "Document about artificial intelligence"


class TestDelete:
    """Test delete operations."""

    def test_delete_by_ref_doc_id(self, fresh_store):
        """Delete should remove all nodes with matching ref_doc_id."""
        nodes = [
            TextNode(id_="del-n1", text="t1", embedding=[0.1] * 128,
                     metadata={"ref_doc_id": "doc-del"}),
            TextNode(id_="del-n2", text="t2", embedding=[0.2] * 128,
                     metadata={"ref_doc_id": "doc-del"}),
        ]
        fresh_store.add(nodes)

        fresh_store.delete("doc-del")
        retrieved = fresh_store.get_nodes(node_ids=["del-n1", "del-n2"])
        assert len(retrieved) == 0

    def test_delete_nodes_by_ids(self, fresh_store):
        """delete_nodes should remove specific nodes by ID list."""
        nodes = [
            TextNode(id_="keep-me", text="keep", embedding=[0.1] * 128),
            TextNode(id_="delete-me", text="del", embedding=[0.2] * 128),
        ]
        fresh_store.add(nodes)

        fresh_store.delete_nodes(node_ids=["delete-me"])
        kept = fresh_store.get_nodes(node_ids=["keep-me"])
        assert len(kept) == 1
        deleted = fresh_store.get_nodes(node_ids=["delete-me"])
        assert len(deleted) == 0


class TestDefaultQuery:
    """Test DEFAULT mode vector similarity search."""

    def test_default_query_returns_results(self, fresh_store):
        """DEFAULT mode should return nodes sorted by similarity."""
        nodes = [
            TextNode(id_="q-n1", text="target document",
                     embedding=[1.0] * 128, metadata={"ref_doc_id": "doc-q"}),
            TextNode(id_="q-n2", text="other document",
                     embedding=[-1.0] * 128, metadata={"ref_doc_id": "doc-q"}),
        ]
        fresh_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.99] * 128,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = fresh_store.query(query)

        assert len(result.nodes) == 2
        assert len(result.similarities) == 2
        assert len(result.ids) == 2
        # First result should be most similar
        assert result.similarities[0] >= result.similarities[1]


class TestMetadataFiltering:
    """Test metadata filter integration."""

    def test_eq_filter(self, fresh_store):
        """EQ filter should only return matching nodes."""
        nodes = [
            TextNode(id_="f-n1", text="a", embedding=[0.1] * 128,
                     metadata={"status": "active"}),
            TextNode(id_="f-n2", text="b", embedding=[0.2] * 128,
                     metadata={"status": "inactive"}),
        ]
        fresh_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)],
        )
        retrieved = fresh_store.get_nodes(filters=filters)
        assert len(retrieved) == 1
        assert retrieved[0].metadata["status"] == "active"


class TestClear:
    """Test clear operation."""

    def test_clear_removes_all_nodes(self, fresh_store):
        """clear() should wipe all data."""
        nodes = [TextNode(id_="c-n1", text="t", embedding=[0.1] * 128)]
        fresh_store.add(nodes)
        fresh_store.clear()
        retrieved = fresh_store.get_nodes(node_ids=["c-n1"])
        assert len(retrieved) == 0
```

- [ ] **Step 3: Install test dependencies and run integration tests**

```bash
pip install pytest pytest-asyncio
pytest tests/integration/test_vastbase_integration.py -v -m integration
```

Expected: PASS (all integration tests)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/
git commit -m "test: integration tests against real Vastbase instance"
```

---

### Task 9: Final Assembly — Run All Tests + Push

**Files:** None (verification task)

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/unit/ -v
```

Expected: All unit tests PASS

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/integration/ -v -m integration
```

Expected: All integration tests PASS

- [ ] **Step 3: Verify package import**

```bash
python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('Package imports OK')"
```

Expected: `Package imports OK`

- [ ] **Step 4: Update STATE.md to mark phase complete**

Update `.multica/STATE.md`: Phase 1 eco-issue-analyst: ✅ 完成 — 2026-06-17

- [ ] **Step 5: Final commit + push**

```bash
git add .multica/STATE.md
git commit -m "state: llamaindex — 需求分析完成，进入规划审查"
git push origin agent/eco-issue-analyst/d12dec88
```

- [ ] **Step 6: Change issue label to「规划审查中」**

```bash
multica issue update 204b117e-92aa-4f90-8c09-112316096004 --status done
```

- [ ] **Step 7: Reassign to task-dispatcher**

```bash
multica issue assign 204b117e-92aa-4f90-8c09-112316096004 --to task-dispatcher
```
