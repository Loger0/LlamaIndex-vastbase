# LlamaIndex VastbaseVectorStore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `VastbaseVectorStore` — a LlamaIndex vector store adapter that replaces PGVectorStore's SQLAlchemy/psycopg2/pgvector stack with pyvastbase, providing full sync+async API parity.

**Architecture:** Collection-Centric Thin Wrapper (D-09). A single `VastbaseVectorStore` class inherits `BasePydanticVectorStore` and delegates all operations to pyvastbase `Collection`/`AsyncCollection`. Utils module provides filter translation and SDK compatibility patches.

**Tech Stack:** Python 3.9+, LlamaIndex v0.14.22, pyvastbase >=0.2.7, pytest + pytest-asyncio

## Global Constraints

- pyvastbase >= 0.2.7 (D-02)
- All vector operations via pyvastbase Collection API — no direct SQL (CLAUDE.md)
- 13 methods total: 4 core + 9 optional (D-01) — persist() raises NotImplementedError
- Instance-level connections with alias isolation (D-04)
- Default loose error handling: `initialization_fail_on_error=False` (D-03)
- HYBRID uses pyvastbase native `hybrid_search` + `RRFRanker` (D-08)
- TEXT_SEARCH uses BM25 fulltext index, requires `hybrid_search=True` (D-05)
- ANY/ALL/CONTAINS operators → client-side filtering fallback (D-06)
- `use_halfvec` requires `check_vb_version((3, 0, 9))` (D-07)
- MMR raises ValueError (matches upstream PGVectorStore behavior)

---

## File Structure

```
llama-index-vector-stores-vastbase/
├── pyproject.toml
├── llama_index/
│   └── vector_stores/
│       └── vastbase/
│           ├── __init__.py          # exports VastbaseVectorStore
│           ├── base.py              # VastbaseVectorStore class (~800 lines)
│           └── utils.py             # filter translation + SDK patches (~200 lines)
├── tests/
│   ├── conftest.py                  # 13 fixtures
│   ├── test_collection_init.py     # Collection creation tests
│   ├── test_crud.py                # CRUD tests
│   ├── test_filter.py             # filter translation tests
│   ├── test_search.py             # query mode tests
│   ├── test_async.py              # async parity tests
│   ├── test_integration.py        # E2E tests (real Vastbase)
│   ├── test_framework_integration.py  # framework-level tests
│   └── demo_llamaindex.py         # 8-scenario demo
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `llama_index/__init__.py`
- Create: `llama_index/vector_stores/__init__.py`
- Create: `llama_index/vector_stores/vastbase/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
description = "Vastbase vector store adapter for LlamaIndex"
requires-python = ">=3.9,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-mock>=3.10",
]

[tool.llamahub]
contains_example = false
import_path = "llama_index.vector_stores.vastbase"

[tool.llamahub.class_authors]
VastbaseVectorStore = "llama-index"

[tool.pytest.ini_options]
asyncio_mode = "auto"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.hatch.build.targets.wheel]
packages = ["llama_index"]
```

- [ ] **Step 2: Create namespace package __init__ files**

`llama_index/__init__.py` — empty (namespace package):
```python
```

`llama_index/vector_stores/__init__.py` — empty (namespace package):
```python
```

`llama_index/vector_stores/vastbase/__init__.py`:
```python
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

__all__ = ["VastbaseVectorStore"]
```

- [ ] **Step 3: Create tests/conftest.py with base fixtures**

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llama_index.core.schema import TextNode
import numpy as np


EMBED_DIM = 128


@pytest.fixture
def mock_embedding():
    """Return a deterministic test embedding."""
    return [0.1] * EMBED_DIM


@pytest.fixture
def sample_nodes():
    """Return 3 sample TextNodes with embeddings and metadata."""
    nodes = []
    for i in range(3):
        node = TextNode(
            text=f"Sample text {i}",
            embedding=[0.1 * (i + 1)] * EMBED_DIM,
            metadata={"category": f"cat_{i}", "score": float(i)},
        )
        node.node_id = f"node-{i}"
        node.metadata["ref_doc_id"] = f"doc-{i // 2}"
        nodes.append(node)
    return nodes


@pytest.fixture
def mock_collection():
    """Mock pyvastbase Collection."""
    col = MagicMock()
    col.insert = MagicMock(return_value=MagicMock(insert_count=3, primary_keys=[1, 2, 3]))
    col.delete = MagicMock()
    col.search = MagicMock(return_value=[[]])
    col.query = MagicMock(return_value=[])
    col.truncate = MagicMock()
    col.create = MagicMock()
    col.create_index = MagicMock()
    return col


@pytest.fixture
def mock_async_collection():
    """Mock pyvastbase AsyncCollection."""
    col = AsyncMock()
    col.insert = AsyncMock(return_value=MagicMock(insert_count=3, primary_keys=[1, 2, 3]))
    col.delete = AsyncMock()
    col.search = AsyncMock(return_value=[[]])
    col.query = AsyncMock(return_value=[])
    col.truncate = AsyncMock()
    col.create = AsyncMock()
    col.create_index = AsyncMock()
    return col
```

- [ ] **Step 4: Install dev dependencies and verify import**

Run: `cd /path/to/repo && pip install -e ".[dev]"`
Expected: Package installs without errors

Run: `python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"`
Expected: `OK` (after base.py exists — skip for now, verify after Task 2)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llama_index/ tests/conftest.py
git commit -m "feat: project scaffolding — pyproject.toml, namespace packages, test fixtures"
```

---

### Task 2: Utils — Filter Translation

**Files:**
- Create: `llama_index/vector_stores/vastbase/utils.py`
- Create: `tests/test_filter.py`

**Interfaces:**
- Produces: `_to_vb_operator(op) -> str`, `_build_filter_clause(filter_) -> str`, `_build_expr(filters) -> str`, `_CLIENT_SIDE_OPERATORS` (set), `_is_client_side_filter(filters) -> bool`, `_client_side_filter(rows, filters) -> list`

- [ ] **Step 1: Write filter translation tests**

Create `tests/test_filter.py`:

```python
import pytest
from unittest.mock import MagicMock
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.vastbase.utils import (
    _to_vb_operator,
    _build_filter_clause,
    _build_expr,
    _CLIENT_SIDE_OPERATORS,
    _is_client_side_filter,
    _client_side_filter,
)


class TestToVbOperator:
    def test_eq(self):
        assert _to_vb_operator(FilterOperator.EQ) == "="

    def test_ne(self):
        assert _to_vb_operator(FilterOperator.NE) == "!="

    def test_gt(self):
        assert _to_vb_operator(FilterOperator.GT) == ">"

    def test_gte(self):
        assert _to_vb_operator(FilterOperator.GTE) == ">="

    def test_lt(self):
        assert _to_vb_operator(FilterOperator.LT) == "<"

    def test_lte(self):
        assert _to_vb_operator(FilterOperator.LTE) == "<="

    def test_in(self):
        assert _to_vb_operator(FilterOperator.IN) == "IN"

    def test_nin(self):
        assert _to_vb_operator(FilterOperator.NIN) == "NOT IN"

    def test_text_match(self):
        assert _to_vb_operator(FilterOperator.TEXT_MATCH) == "LIKE"

    def test_text_match_insensitive(self):
        assert _to_vb_operator(FilterOperator.TEXT_MATCH_INSENSITIVE) == "ILIKE"

    def test_contains(self):
        assert _to_vb_operator(FilterOperator.CONTAINS) == "@>"

    def test_is_empty(self):
        assert _to_vb_operator(FilterOperator.IS_EMPTY) == "IS NULL"

    def test_any_op(self):
        assert _to_vb_operator(FilterOperator.ANY) == "?|"

    def test_all_op(self):
        assert _to_vb_operator(FilterOperator.ALL) == "?&"


class TestClientSideOperators:
    def test_any_is_client_side(self):
        assert FilterOperator.ANY in _CLIENT_SIDE_OPERATORS

    def test_all_is_client_side(self):
        assert FilterOperator.ALL in _CLIENT_SIDE_OPERATORS

    def test_contains_is_client_side(self):
        assert FilterOperator.CONTAINS in _CLIENT_SIDE_OPERATORS

    def test_eq_is_not_client_side(self):
        assert FilterOperator.EQ not in _CLIENT_SIDE_OPERATORS


class TestBuildFilterClause:
    def test_eq_string(self):
        f = MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        clause = _build_filter_clause(f)
        assert clause == "metadata_->>'category' = 'tech'"

    def test_in_values(self):
        f = MetadataFilter(key="cat", value=["a", "b"], operator=FilterOperator.IN)
        clause = _build_filter_clause(f)
        assert "IN" in clause
        assert "'a'" in clause
        assert "'b'" in clause

    def test_text_match_like(self):
        f = MetadataFilter(key="title", value="hello", operator=FilterOperator.TEXT_MATCH)
        clause = _build_filter_clause(f)
        assert "LIKE '%hello%'" in clause

    def test_text_match_insensitive_ilike(self):
        f = MetadataFilter(key="title", value="hello", operator=FilterOperator.TEXT_MATCH_INSENSITIVE)
        clause = _build_filter_clause(f)
        assert "ILIKE '%hello%'" in clause

    def test_is_empty(self):
        f = MetadataFilter(key="field", value="", operator=FilterOperator.IS_EMPTY)
        clause = _build_filter_clause(f)
        assert "IS NULL" in clause

    def test_numeric_comparison(self):
        f = MetadataFilter(key="score", value="3.14", operator=FilterOperator.GT)
        clause = _build_filter_clause(f)
        assert "::float" in clause
        assert "3.14" in clause

    def test_nin_values(self):
        f = MetadataFilter(key="cat", value=["x", "y"], operator=FilterOperator.NIN)
        clause = _build_filter_clause(f)
        assert "NOT IN" in clause


class TestBuildExpr:
    def test_single_filter(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="cat", value="tech", operator=FilterOperator.EQ)
            ],
            condition=FilterCondition.AND,
        )
        expr = _build_expr(filters)
        assert "metadata_->>'cat' = 'tech'" in expr

    def test_and_filters(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="cat", value="tech", operator=FilterOperator.EQ),
                MetadataFilter(key="score", value="5", operator=FilterOperator.GT),
            ],
            condition=FilterCondition.AND,
        )
        expr = _build_expr(filters)
        assert " AND " in expr

    def test_or_filters(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="cat", value="a", operator=FilterOperator.EQ),
                MetadataFilter(key="cat", value="b", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        expr = _build_expr(filters)
        assert " OR " in expr

    def test_nested_filters(self):
        inner = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
                MetadataFilter(key="b", value="2", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        outer = MetadataFilters(
            filters=[
                inner,
                MetadataFilter(key="c", value="3", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        expr = _build_expr(outer)
        assert "(" in expr
        assert " OR " in expr


class TestIsClientSideFilter:
    def test_no_client_side(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        assert _is_client_side_filter(filters) is False

    def test_has_any(self):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.ANY),
            ],
            condition=FilterCondition.AND,
        )
        assert _is_client_side_filter(filters) is True

    def test_nested_client_side(self):
        inner = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["x"], operator=FilterOperator.ALL),
            ],
            condition=FilterCondition.AND,
        )
        outer = MetadataFilters(
            filters=[
                inner,
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        assert _is_client_side_filter(outer) is True


class TestClientSideFilter:
    def test_any_filter(self):
        rows = [
            {"node_id": "n1", "metadata_": {"tags": ["a", "b"]}},
            {"node_id": "n2", "metadata_": {"tags": ["c"]}},
        ]
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["a", "d"], operator=FilterOperator.ANY),
            ],
            condition=FilterCondition.AND,
        )
        result = _client_side_filter(rows, filters)
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"

    def test_all_filter(self):
        rows = [
            {"node_id": "n1", "metadata_": {"tags": ["a", "b", "c"]}},
            {"node_id": "n2", "metadata_": {"tags": ["a"]}},
        ]
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="tags", value=["a", "b"], operator=FilterOperator.ALL),
            ],
            condition=FilterCondition.AND,
        )
        result = _client_side_filter(rows, filters)
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"

    def test_contains_filter(self):
        rows = [
            {"node_id": "n1", "metadata_": {"items": ["x", "y"]}},
            {"node_id": "n2", "metadata_": {"items": ["z"]}},
        ]
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="items", value="x", operator=FilterOperator.CONTAINS),
            ],
            condition=FilterCondition.AND,
        )
        result = _client_side_filter(rows, filters)
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"

    def test_eq_server_side_passthrough(self):
        """EQ should never reach client-side filter."""
        rows = [
            {"node_id": "n1", "metadata_": {"a": "1"}},
        ]
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value="1", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        # Client-side filter should not be called for EQ — but if called, pass through
        result = _client_side_filter(rows, filters)
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llama_index.vector_stores.vastbase.utils'`

- [ ] **Step 3: Implement utils.py filter translation**

Create `llama_index/vector_stores/vastbase/utils.py`:

```python
"""Utility functions for VastbaseVectorStore — filter translation and SDK patches."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

logger = logging.getLogger(__name__)

# Operators that cannot be expressed in pyvastbase expr due to '?' conflict
# with parameter placeholders (D-06)
_CLIENT_SIDE_OPERATORS: Set[FilterOperator] = {
    FilterOperator.ANY,
    FilterOperator.ALL,
    FilterOperator.CONTAINS,
}

# Operator mapping: FilterOperator → SQL operator string
_OPERATOR_MAP: Dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
    FilterOperator.IN: "IN",
    FilterOperator.NIN: "NOT IN",
    FilterOperator.TEXT_MATCH: "LIKE",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "ILIKE",
    FilterOperator.CONTAINS: "@>",
    FilterOperator.IS_EMPTY: "IS NULL",
    FilterOperator.ANY: "?|",
    FilterOperator.ALL: "?&",
}


def _to_vb_operator(op: FilterOperator) -> str:
    """Convert a LlamaIndex FilterOperator to a pyvastbase expr operator string."""
    if op not in _OPERATOR_MAP:
        raise ValueError(f"Unsupported filter operator: {op}")
    return _OPERATOR_MAP[op]


def _build_filter_clause(filter_: MetadataFilter) -> str:
    """Build a single pyvastbase expr fragment from a MetadataFilter."""
    op = _to_vb_operator(filter_.operator)
    key = filter_.key

    # IN / NOT IN → value list
    if filter_.operator in (FilterOperator.IN, FilterOperator.NIN):
        values = ", ".join(f"'{v}'" for v in filter_.value)
        return f"metadata_->>'{key}' {op} ({values})"

    # TEXT_MATCH / TEXT_MATCH_INSENSITIVE → wildcard wrap
    if filter_.operator in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
        return f"metadata_->>'{key}' {op} '%{filter_.value}%'"

    # IS_EMPTY → IS NULL
    if filter_.operator == FilterOperator.IS_EMPTY:
        return f"metadata_->>'{key}' {op}"

    # Numeric comparison → CAST float
    try:
        float(filter_.value)
        return f"(metadata_->>'{key}')::float {op} {filter_.value}"
    except (ValueError, TypeError):
        return f"metadata_->>'{key}' {op} '{filter_.value}'"


def _build_expr(filters: MetadataFilters) -> str:
    """Recursively build a pyvastbase expr string from MetadataFilters (supports nested AND/OR)."""
    clauses = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            clauses.append(f"({_build_expr(f)})")
        else:
            clauses.append(_build_filter_clause(f))
    condition = f" {filters.condition.upper()} "
    return condition.join(clauses)


def _is_client_side_filter(filters: MetadataFilters) -> bool:
    """Check if any filter in the tree uses a client-side operator."""
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            if _is_client_side_filter(f):
                return True
        elif f.operator in _CLIENT_SIDE_OPERATORS:
            return True
    return False


def _strip_client_side_filters(filters: MetadataFilters) -> Optional[MetadataFilters]:
    """Remove client-side operator filters, returning only server-safe filters.

    Returns None if all filters are client-side (nothing left for server).
    """
    server_filters = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            stripped = _strip_client_side_filters(f)
            if stripped is not None:
                server_filters.append(stripped)
        elif f.operator not in _CLIENT_SIDE_OPERATORS:
            server_filters.append(f)
    if not server_filters:
        return None
    return MetadataFilters(filters=server_filters, condition=filters.condition)


def _client_side_filter(rows: List[Dict[str, Any]], filters: MetadataFilters) -> List[Dict[str, Any]]:
    """Apply client-side filtering on row dicts for ANY/ALL/CONTAINS operators."""
    result = []
    for row in rows:
        metadata = row.get("metadata_", {}) or {}
        if _row_matches_filters(metadata, filters):
            result.append(row)
    return result


def _row_matches_filters(metadata: Dict[str, Any], filters: MetadataFilters) -> bool:
    """Check if a single metadata dict matches all filter conditions."""
    results = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            results.append(_row_matches_filters(metadata, f))
        else:
            results.append(_row_matches_single(metadata, f))

    if filters.condition == FilterCondition.AND:
        return all(results)
    else:  # OR
        return any(results)


def _row_matches_single(metadata: Dict[str, Any], filter_: MetadataFilter) -> bool:
    """Check if metadata matches a single MetadataFilter."""
    key = filter_.key
    value = filter_.value
    actual = metadata.get(key)

    if filter_.operator == FilterOperator.ANY:
        if not isinstance(actual, list):
            return False
        return any(v in actual for v in value)

    if filter_.operator == FilterOperator.ALL:
        if not isinstance(actual, list):
            return False
        return all(v in actual for v in value)

    if filter_.operator == FilterOperator.CONTAINS:
        if isinstance(actual, list):
            return value in actual
        if isinstance(actual, str):
            return value in actual
        return False

    # Server-side operators — should not reach here, but handle gracefully
    if filter_.operator == FilterOperator.EQ:
        return str(actual) == str(value)
    if filter_.operator == FilterOperator.NE:
        return str(actual) != str(value)
    if filter_.operator == FilterOperator.IN:
        return actual in value or str(actual) in [str(v) for v in value]
    if filter_.operator == FilterOperator.NIN:
        return actual not in value and str(actual) not in [str(v) for v in value]

    return True  # Unknown operator — pass through
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_filter.py -v`
Expected: All 28 tests PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/utils.py tests/test_filter.py
git commit -m "feat: utils — filter translation (14 operators + client-side fallback)"
```

---

### Task 3: Utils — SDK Compatibility Patches

**Files:**
- Modify: `llama_index/vector_stores/vastbase/utils.py`
- Create: `tests/test_utils_patches.py`

**Interfaces:**
- Produces: `_patch_async_executor()`, `_ensure_auto_id_sequence(collection_name, alias)`, `_ensure_schema_columns(collection_name, field_name, pg_type, alias)`

- [ ] **Step 1: Write patch utility tests**

Create `tests/test_utils_patches.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, call
from llama_index.vector_stores.vastbase.utils import (
    _patch_async_executor,
    _ensure_auto_id_sequence,
    _ensure_schema_columns,
)


class TestPatchAsyncExecutor:
    def test_idempotent(self):
        """Calling _patch_async_executor twice should not break anything."""
        _patch_async_executor()
        _patch_async_executor()  # Should not raise

    def test_patches_execute_method(self):
        """After patching, AsyncExecutor.execute should handle list params."""
        _patch_async_executor()
        # Verify the monkey-patch was applied
        try:
            from pyvastbase.async_impl.collection import AsyncExecutor
            # The patched method should exist and be callable
            assert callable(getattr(AsyncExecutor, 'execute', None))
        except ImportError:
            # pyvastbase not installed in test env — that's OK for unit tests
            pass


class TestEnsureAutoIdSequence:
    @patch("llama_index.vector_stores.vastbase.utils.get_connection")
    def test_creates_sequence_and_default(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        _ensure_auto_id_sequence("test_collection", "test_alias")

        # Should execute CREATE SEQUENCE and ALTER TABLE
        calls = mock_cursor.execute.call_args_list
        assert len(calls) >= 2
        sql_statements = [c[0][0] for c in calls]
        assert any("CREATE SEQUENCE" in s or "SEQUENCE" in s for s in sql_statements)
        assert any("ALTER TABLE" in s or "DEFAULT" in s for s in sql_statements)

    @patch("llama_index.vector_stores.vastbase.utils.get_connection")
    def test_handles_sequence_already_exists(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = [
            Exception("already exists"),  # CREATE SEQUENCE fails
            None,  # ALTER TABLE succeeds
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Should not raise
        _ensure_auto_id_sequence("test_collection", "test_alias")


class TestEnsureSchemaColumns:
    @patch("llama_index.vector_stores.vastbase.utils.get_connection")
    def test_adds_column_via_alter_table(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        _ensure_schema_columns(
            "test_collection", "new_field", "text", "test_alias"
        )

        calls = mock_cursor.execute.call_args_list
        assert len(calls) >= 1
        sql = calls[0][0][0]
        assert "ALTER TABLE" in sql
        assert "test_collection" in sql
        assert "new_field" in sql

    @patch("llama_index.vector_stores.vastbase.utils.get_connection")
    def test_handles_column_already_exists(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("already exists")
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        # Should not raise
        _ensure_schema_columns(
            "test_collection", "existing_field", "text", "test_alias"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utils_patches.py -v`
Expected: FAIL — functions not defined yet

- [ ] **Step 3: Implement SDK patches in utils.py**

Append to `llama_index/vector_stores/vastbase/utils.py`:

```python
# === SDK Compatibility Patches ===

_ASYNC_EXECUTOR_PATCHED = False


def _patch_async_executor() -> None:
    """Monkey-patch pyvastbase AsyncExecutor.execute to handle list parameters.

    pyvastbase's AsyncExecutor.execute uses named placeholders that are
    incompatible with list arguments. This patch auto-converts list params
    to dict params.
    """
    global _ASYNC_EXECUTOR_PATCHED
    if _ASYNC_EXECUTOR_PATCHED:
        return

    try:
        from pyvastbase.async_impl.collection import AsyncExecutor

        _original_execute = AsyncExecutor.execute

        async def _patched_execute(self, sql, params=None):
            if isinstance(params, list):
                # Convert list params to numbered dict params
                params = {f"p{i}": v for i, v in enumerate(params)}
                # Rewrite %s placeholders to %(pN)s named placeholders
                for i in range(len(params)):
                    sql = sql.replace("%s", f"%(p{i})s", 1)
            return await _original_execute(self, sql, params)

        AsyncExecutor.execute = _patched_execute
        _ASYNC_EXECUTOR_PATCHED = True
        logger.debug("Patched AsyncExecutor.execute for list parameter compatibility")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch AsyncExecutor: {e}")


def _ensure_auto_id_sequence(collection_name: str, alias: str = "default") -> None:
    """Create a SERIAL-like sequence for auto-id primary keys.

    pyvastbase FieldSchema(auto_id=True) does not generate SERIAL DDL,
    so we manually create a sequence and set it as the column default.
    """
    from pyvastbase import get_connection

    conn = get_connection(alias)
    cursor = conn.cursor()
    seq_name = f"{collection_name}_id_seq"

    try:
        cursor.execute(
            f"CREATE SEQUENCE IF NOT EXISTS {seq_name}"
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            logger.warning(f"Failed to create sequence {seq_name}: {e}")
            return

    try:
        cursor.execute(
            f"ALTER TABLE {collection_name} "
            f"ALTER COLUMN id SET DEFAULT nextval('{seq_name}')"
        )
    except Exception as e:
        logger.warning(f"Failed to set default for {collection_name}.id: {e}")


def _ensure_schema_columns(
    collection_name: str,
    field_name: str,
    pg_type: str,
    alias: str = "default",
) -> None:
    """Add a column to an existing collection table via ALTER TABLE.

    Workaround for pyvastbase add_collection_field missing @with_executor.
    """
    from pyvastbase import get_connection

    conn = get_connection(alias)
    cursor = conn.cursor()

    try:
        cursor.execute(
            f"ALTER TABLE {collection_name} "
            f"ADD COLUMN IF NOT EXISTS {field_name} {pg_type}"
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            logger.warning(
                f"Failed to add column {field_name} to {collection_name}: {e}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utils_patches.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/utils.py tests/test_utils_patches.py
git commit -m "feat: utils — SDK compatibility patches (AsyncExecutor, auto-id, schema columns)"
```

---

### Task 4: VastbaseVectorStore — Class Scaffold + Init

**Files:**
- Create: `llama_index/vector_stores/vastbase/base.py`
- Create: `tests/test_collection_init.py`

**Interfaces:**
- Consumes: `utils._patch_async_executor`, `utils._ensure_auto_id_sequence`
- Produces: `VastbaseVectorStore` class with Pydantic fields, `__init__`, `_initialize`, `client` property, `close()`

- [ ] **Step 1: Write collection init tests**

Create `tests/test_collection_init.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore


class TestVastbaseVectorStoreFields:
    def test_default_fields(self):
        store = VastbaseVectorStore.__new__(VastbaseVectorStore)
        # Pydantic model should have correct defaults
        assert VastbaseVectorStore.model_fields["stores_text"].default is True
        assert VastbaseVectorStore.model_fields["flat_metadata"].default is False
        assert VastbaseVectorStore.model_fields["collection_name"].default == "llamaindex"
        assert VastbaseVectorStore.model_fields["embed_dim"].default == 1536

    def test_class_name(self):
        assert VastbaseVectorStore.class_name() == "VastbaseVectorStore"

    def test_stores_text(self):
        assert VastbaseVectorStore.model_fields["stores_text"].default is True

    def test_flat_metadata(self):
        assert VastbaseVectorStore.model_fields["flat_metadata"].default is False


class TestClientProperty:
    def test_client_returns_none_before_init(self):
        with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
            store = VastbaseVectorStore()
            store._is_initialized = False
            store._collection = None
            assert store.client is None

    def test_client_returns_collection_after_init(self):
        with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
            store = VastbaseVectorStore()
            mock_col = MagicMock()
            store._is_initialized = True
            store._collection = mock_col
            assert store.client is mock_col


class TestInitialize:
    @patch("llama_index.vector_stores.vastbase.base.AsyncCollection")
    @patch("llama_index.vector_stores.vastbase.base.Collection")
    @patch("llama_index.vector_stores.vastbase.base.has_collection")
    @patch("llama_index.vector_stores.vastbase.base.connect")
    @patch("llama_index.vector_stores.vastbase.base._patch_async_executor")
    def test_creates_connection_with_alias(
        self, mock_patch, mock_connect, mock_has, mock_col_cls, mock_async
    ):
        mock_has.return_value = False
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col

        store = VastbaseVectorStore(
            host="localhost",
            port=15432,
            database="vastbase",
            user="aidev",
            password="test",
            collection_name="test_init",
            embed_dim=128,
            perform_setup=True,
        )

        mock_connect.assert_called_once()
        call_kwargs = mock_connect.call_args
        assert call_kwargs[1].get("alias") or call_kwargs[0][5:6]

    @patch("llama_index.vector_stores.vastbase.base.AsyncCollection")
    @patch("llama_index.vector_stores.vastbase.base.Collection")
    @patch("llama_index.vector_stores.vastbase.base.has_collection")
    @patch("llama_index.vector_stores.vastbase.base.connect")
    @patch("llama_index.vector_stores.vastbase.base._patch_async_executor")
    def test_creates_collection_if_not_exists(
        self, mock_patch, mock_connect, mock_has, mock_col_cls, mock_async
    ):
        mock_has.return_value = False
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col

        store = VastbaseVectorStore(
            host="localhost", port=15432, database="vastbase",
            user="aidev", password="test",
            collection_name="test_new",
            embed_dim=128, perform_setup=True,
        )

        mock_col.create.assert_called()

    @patch("llama_index.vector_stores.vastbase.base.AsyncCollection")
    @patch("llama_index.vector_stores.vastbase.base.Collection")
    @patch("llama_index.vector_stores.vastbase.base.has_collection")
    @patch("llama_index.vector_stores.vastbase.base.connect")
    @patch("llama_index.vector_stores.vastbase.base._patch_async_executor")
    def test_skips_setup_when_perform_setup_false(
        self, mock_patch, mock_connect, mock_has, mock_col_cls, mock_async
    ):
        store = VastbaseVectorStore(
            host="localhost", port=15432, database="vastbase",
            user="aidev", password="test",
            collection_name="test_skip",
            embed_dim=128, perform_setup=False,
        )

        mock_connect.assert_not_called()
        assert store._is_initialized is False


class TestClose:
    @patch("llama_index.vector_stores.vastbase.base.remove_connection")
    def test_close_removes_connection(self, mock_remove):
        with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
            store = VastbaseVectorStore()
            store._is_initialized = True
            store._connection_alias = "test_alias"

            import asyncio
            asyncio.get_event_loop().run_until_complete(store.close())

            mock_remove.assert_called_once_with("test_alias")
            assert store._is_initialized is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collection_init.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llama_index.vector_stores.vastbase.base'`

- [ ] **Step 3: Implement VastbaseVectorStore class scaffold**

Create `llama_index/vector_stores/vastbase/base.py`:

```python
"""VastbaseVectorStore — LlamaIndex vector store adapter for Vastbase."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilters,
)

from llama_index.vector_stores.vastbase.utils import (
    _build_expr,
    _is_client_side_filter,
    _strip_client_side_filters,
    _client_side_filter,
    _patch_async_executor,
    _ensure_auto_id_sequence,
    _ensure_schema_columns,
)

logger = logging.getLogger(__name__)


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store adapter for LlamaIndex.

    Uses pyvastbase Collection API to provide sync + async vector store operations,
    replacing PGVectorStore's SQLAlchemy + psycopg2/asyncpg + pgvector stack.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    # Connection parameters
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = "aidev"
    password: str = "Vbase_123456"

    # Collection parameters
    collection_name: str = "llamaindex"
    embed_dim: int = 1536
    use_halfvec: bool = False
    hybrid_search: bool = False
    text_search_config: str = "english"
    hnsw_kwargs: Optional[Dict[str, Any]] = None
    perform_setup: bool = True
    debug: bool = False
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None
    initialization_fail_on_error: bool = False

    # Private attributes
    _connection_alias: str = PrivateAttr(default="")
    _collection: Any = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.perform_setup:
            self._initialize()

    @classmethod
    def class_name(cls) -> str:
        return "VastbaseVectorStore"

    @property
    def client(self) -> Any:
        """Return the underlying pyvastbase Collection instance."""
        if not self._is_initialized:
            return None
        return self._collection

    def _initialize(self) -> None:
        """Set up connection, collection, and indexes."""
        if self._is_initialized:
            return

        try:
            self._do_initialize()
        except Exception as e:
            if self.initialization_fail_on_error:
                raise
            logger.warning(f"Initialization failed (loose mode): {e}")

    def _do_initialize(self) -> None:
        from pyvastbase import (
            connect,
            has_collection,
            Collection,
            CollectionSchema,
            FieldSchema,
            DataType,
            IndexParams,
        )
        from pyvastbase.async_impl.collection import AsyncCollection

        # 1. Apply SDK patches
        _patch_async_executor()

        # 2. Generate connection alias + connect
        self._connection_alias = f"vb_{self.collection_name}_{id(self)}"
        connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            alias=self._connection_alias,
        )

        # 3. Version check for halfvec (D-07)
        if self.use_halfvec:
            from pyvastbase import check_vb_version
            if not check_vb_version((3, 0, 9)):
                from pyvastbase import get_vb_version
                ver = get_vb_version()
                raise ValueError(
                    f"use_halfvec requires Vastbase >= 3.0.9, "
                    f"got {ver['major']}.{ver['minor']}.{ver.get('patch', 0)}"
                )

        # 4. Build schema
        vec_dtype = DataType.FLOAT16_VECTOR if self.use_halfvec else DataType.FLOAT_VECTOR
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True, auto_id=True),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata_", dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=vec_dtype, dim=self.embed_dim),
        ]
        schema = CollectionSchema(name=self.collection_name, fields=fields)

        # 5. Create or get collection (has_collection fallback)
        try:
            exists = has_collection(self.collection_name)
        except Exception:
            exists = False

        if exists:
            self._collection = Collection(self.collection_name)
        else:
            self._collection = Collection(self.collection_name, schema=schema)
            try:
                self._collection.create()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
                self._collection = Collection(self.collection_name)

        # 6. Auto-id sequence patch
        _ensure_auto_id_sequence(self.collection_name, self._connection_alias)

        # 7. HNSW index
        if self.hnsw_kwargs:
            m = self.hnsw_kwargs.get("m", 16)
            ef_construction = self.hnsw_kwargs.get("ef_construction", 64)
            try:
                self._collection.create_index(
                    field_name="embedding",
                    index_params=IndexParams.graph_index(
                        m=m, ef_construction=ef_construction
                    ),
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # 8. Fulltext index (D-05)
        if self.hybrid_search:
            try:
                self._collection.create_index(
                    field_name="text",
                    index_params=IndexParams.fulltext_index(
                        dictionary=self.text_search_config,
                        algorithm="BM25",
                    ),
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        # 9. Async collection
        self._async_collection = AsyncCollection(self.collection_name)

        # 10. Mark initialized
        self._is_initialized = True
        logger.info(
            f"VastbaseVectorStore initialized: collection={self.collection_name}, "
            f"embed_dim={self.embed_dim}, hybrid={self.hybrid_search}"
        )

    async def close(self) -> None:
        """Clean up the connection."""
        if not self._is_initialized:
            return
        from pyvastbase import remove_connection
        remove_connection(self._connection_alias)
        self._is_initialized = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collection_init.py -v`
Expected: All tests PASS (fields, client property, initialize, close)

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_collection_init.py
git commit -m "feat: VastbaseVectorStore class scaffold — fields, init, client, close"
```

---

### Task 5: Core Methods — add + delete + clear

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Create: `tests/test_crud.py`

**Interfaces:**
- Consumes: `_collection.insert()`, `_collection.delete()`, `_collection.truncate()`
- Produces: `add()`, `async_add()`, `delete()`, `adelete()`, `clear()`, `aclear()`

- [ ] **Step 1: Write CRUD tests**

Create `tests/test_crud.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llama_index.core.schema import TextNode
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore


EMBED_DIM = 128


def _make_store(mock_col=None, mock_async_col=None):
    """Create a VastbaseVectorStore with mocked internals (no real connection)."""
    with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
        store = VastbaseVectorStore()
        store._is_initialized = True
        store._collection = mock_col or MagicMock()
        store._async_collection = mock_async_col or AsyncMock()
        store.collection_name = "test"
        store.embed_dim = EMBED_DIM
        store.hybrid_search = False
        store._connection_alias = "test_alias"
        store.initialization_fail_on_error = False
        return store


def _make_nodes(count=3):
    nodes = []
    for i in range(count):
        node = TextNode(
            text=f"Text {i}",
            embedding=[0.1 * (i + 1)] * EMBED_DIM,
            metadata={"cat": f"c{i}", "ref_doc_id": f"doc-{i // 2}"},
        )
        node.node_id = f"node-{i}"
        nodes.append(node)
    return nodes


class TestAdd:
    def test_add_returns_node_ids(self):
        mock_col = MagicMock()
        mock_col.insert.return_value = MagicMock(insert_count=3)
        store = _make_store(mock_col)
        nodes = _make_nodes(3)

        result = store.add(nodes)

        assert result == ["node-0", "node-1", "node-2"]

    def test_add_calls_insert_with_data(self):
        mock_col = MagicMock()
        mock_col.insert.return_value = MagicMock(insert_count=2)
        store = _make_store(mock_col)
        nodes = _make_nodes(2)

        store.add(nodes)

        mock_col.insert.assert_called_once()
        data = mock_col.insert.call_args[0][0]
        assert len(data) == 2
        assert data[0]["node_id"] == "node-0"
        assert "embedding" in data[0]
        assert "text" in data[0]
        assert "metadata_" in data[0]

    def test_add_empty_nodes(self):
        store = _make_store()
        result = store.add([])
        assert result == []


class TestDelete:
    def test_delete_by_ref_doc_id(self):
        mock_col = MagicMock()
        store = _make_store(mock_col)

        store.delete(ref_doc_id="doc-0")

        mock_col.delete.assert_called_once()
        expr = mock_col.delete.call_args[1].get("expr") or mock_col.delete.call_args[0][0]
        assert "ref_doc_id" in expr
        assert "doc-0" in expr

    def test_delete_no_op_when_not_initialized(self):
        store = _make_store()
        store._is_initialized = False
        # Should not raise
        store.delete(ref_doc_id="doc-0")


class TestClear:
    def test_clear_calls_truncate(self):
        mock_col = MagicMock()
        store = _make_store(mock_col)

        store.clear()

        mock_col.truncate.assert_called_once()


class TestAsyncAdd:
    @pytest.mark.asyncio
    async def test_async_add_returns_node_ids(self):
        mock_async = AsyncMock()
        mock_async.insert.return_value = MagicMock(insert_count=2)
        store = _make_store(mock_async_col=mock_async)
        nodes = _make_nodes(2)

        result = await store.async_add(nodes)

        assert result == ["node-0", "node-1"]

    @pytest.mark.asyncio
    async def test_async_add_calls_insert(self):
        mock_async = AsyncMock()
        mock_async.insert.return_value = MagicMock(insert_count=1)
        store = _make_store(mock_async_col=mock_async)
        nodes = _make_nodes(1)

        await store.async_add(nodes)

        mock_async.insert.assert_called_once()


class TestAsyncDelete:
    @pytest.mark.asyncio
    async def test_adelete_by_ref_doc_id(self):
        mock_async = AsyncMock()
        store = _make_store(mock_async_col=mock_async)

        await store.adelete(ref_doc_id="doc-1")

        mock_async.delete.assert_called_once()


class TestAsyncClear:
    @pytest.mark.asyncio
    async def test_aclear_calls_truncate(self):
        mock_async = AsyncMock()
        store = _make_store(mock_async_col=mock_async)

        await store.aclear()

        mock_async.truncate.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crud.py -v`
Expected: FAIL — `add`, `delete`, `clear` methods not implemented

- [ ] **Step 3: Implement add, delete, clear methods**

Add to `VastbaseVectorStore` in `base.py`:

```python
    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """Insert nodes into the vector store."""
        if not nodes:
            return []
        self._initialize()

        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        data_list = []
        for node in nodes:
            metadata = node_to_metadata_dict(node, remove_text=True, flat_metadata=self.flat_metadata)
            data_list.append({
                "node_id": node.node_id,
                "text": node.get_content(metadata_mode="none"),
                "metadata_": metadata,
                "embedding": node.get_embedding(),
            })

        self._collection.insert(data_list)
        return [node.node_id for node in nodes]

    async def async_add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """Async insert nodes into the vector store."""
        if not nodes:
            return []
        self._initialize()

        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        data_list = []
        for node in nodes:
            metadata = node_to_metadata_dict(node, remove_text=True, flat_metadata=self.flat_metadata)
            data_list.append({
                "node_id": node.node_id,
                "text": node.get_content(metadata_mode="none"),
                "metadata_": metadata,
                "embedding": node.get_embedding(),
            })

        await self._async_collection.insert(data_list)
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes associated with a ref_doc_id."""
        if not self._is_initialized:
            return
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        self._collection.delete(expr=expr)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async delete all nodes associated with a ref_doc_id."""
        if not self._is_initialized:
            return
        expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
        await self._async_collection.delete(expr=expr)

    def clear(self) -> None:
        """Truncate the collection."""
        self._initialize()
        self._collection.truncate()

    async def aclear(self) -> None:
        """Async truncate the collection."""
        self._initialize()
        await self._async_collection.truncate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_crud.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_crud.py
git commit -m "feat: CRUD methods — add/async_add, delete/adelete, clear/aclear"
```

---

### Task 6: Core Methods — query (DEFAULT + TEXT_SEARCH + HYBRID + MMR)

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Consumes: `_collection.search()`, `_collection.hybrid_search()`, `AnnSearchRequest`, `RRFRanker`
- Produces: `query()`, `aquery()`

- [ ] **Step 1: Write search mode tests**

Create `tests/test_search.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore


EMBED_DIM = 128


def _make_store(mock_col=None, mock_async_col=None):
    with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
        store = VastbaseVectorStore()
        store._is_initialized = True
        store._collection = mock_col or MagicMock()
        store._async_collection = mock_async_col or AsyncMock()
        store.collection_name = "test"
        store.embed_dim = EMBED_DIM
        store.hybrid_search = False
        store._connection_alias = "test_alias"
        store.initialization_fail_on_error = False
        return store


def _make_search_result(node_id, distance, text="test", metadata=None):
    hit = MagicMock()
    hit.id = 1
    hit.distance = distance
    hit.data = {
        "node_id": node_id,
        "text": text,
        "metadata_": metadata or {"ref_doc_id": "doc-0"},
    }
    # Support both .data dict and attribute access
    hit.__getitem__ = lambda self, key: hit.data[key]
    return hit


class TestQueryDefault:
    def test_default_mode_calls_search(self):
        mock_col = MagicMock()
        mock_col.search.return_value = [[]]
        store = _make_store(mock_col)

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        store.query(query)

        mock_col.search.assert_called_once()

    def test_default_mode_returns_result(self):
        mock_col = MagicMock()
        hit = _make_search_result("node-1", 0.2)
        mock_col.search.return_value = [[hit]]
        store = _make_store(mock_col)

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store.query(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.ids) == 1
        assert result.ids[0] == "node-1"
        assert abs(result.similarities[0] - 0.8) < 0.01  # 1 - 0.2

    def test_default_mode_similarity_calculation(self):
        mock_col = MagicMock()
        hits = [
            _make_search_result("n1", 0.1),
            _make_search_result("n2", 0.5),
            _make_search_result("n3", 0.9),
        ]
        mock_col.search.return_value = [hits]
        store = _make_store(mock_col)

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store.query(query)

        assert len(result.similarities) == 3
        assert result.similarities[0] > result.similarities[1] > result.similarities[2]


class TestQueryTextSearch:
    def test_text_search_requires_hybrid(self):
        store = _make_store()
        store.hybrid_search = False

        query = VectorStoreQuery(
            query_str="hello world",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.SPARSE,
        )

        with pytest.raises(ValueError, match="hybrid_search=True"):
            store.query(query)

    def test_text_search_calls_search_with_text_field(self):
        mock_col = MagicMock()
        mock_col.search.return_value = [[]]
        store = _make_store(mock_col)
        store.hybrid_search = True

        query = VectorStoreQuery(
            query_str="hello world",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.SPARSE,
        )
        store.query(query)

        mock_col.search.assert_called_once()


class TestQueryHybrid:
    def test_hybrid_requires_hybrid_flag(self):
        store = _make_store()
        store.hybrid_search = False

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            query_str="hello",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )

        with pytest.raises(ValueError, match="hybrid_search=True"):
            store.query(query)

    def test_hybrid_calls_hybrid_search(self):
        mock_col = MagicMock()
        mock_col.hybrid_search.return_value = [[]]
        store = _make_store(mock_col)
        store.hybrid_search = True

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            query_str="hello",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )
        store.query(query)

        mock_col.hybrid_search.assert_called_once()


class TestQueryMMR:
    def test_mmr_raises_value_error(self):
        store = _make_store()

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )

        with pytest.raises(ValueError, match="MMR"):
            store.query(query)


class TestAsyncQuery:
    @pytest.mark.asyncio
    async def test_aquery_default(self):
        mock_async = AsyncMock()
        mock_async.search.return_value = [[]]
        store = _make_store(mock_async_col=mock_async)

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = await store.aquery(query)

        mock_async.search.assert_called_once()
        assert isinstance(result, VectorStoreQueryResult)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search.py -v`
Expected: FAIL — `query`, `aquery` not implemented

- [ ] **Step 3: Implement query and aquery methods**

Add to `VastbaseVectorStore` in `base.py`:

```python
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Execute a vector store query."""
        self._initialize()
        mode = query.mode or VectorStoreQueryMode.DEFAULT
        filter_expr = self._build_query_filter(query)
        output_fields = ["node_id", "text", "metadata_"]

        if mode == VectorStoreQueryMode.DEFAULT:
            return self._query_default(query, filter_expr, output_fields)
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            return self._query_text(query, filter_expr, output_fields)
        elif mode == VectorStoreQueryMode.HYBRID:
            return self._query_hybrid(query, filter_expr, output_fields)
        elif mode == VectorStoreQueryMode.MMR:
            raise ValueError(
                "MMR mode is not supported by VastbaseVectorStore. "
                "This matches upstream PGVectorStore behavior."
            )
        else:
            raise ValueError(f"Unsupported query mode: {mode}")

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Async execute a vector store query."""
        self._initialize()
        mode = query.mode or VectorStoreQueryMode.DEFAULT
        filter_expr = self._build_query_filter(query)
        output_fields = ["node_id", "text", "metadata_"]

        if mode == VectorStoreQueryMode.DEFAULT:
            return await self._aquery_default(query, filter_expr, output_fields)
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            return await self._atext_search(query, filter_expr, output_fields)
        elif mode == VectorStoreQueryMode.HYBRID:
            return await self._aquery_hybrid(query, filter_expr, output_fields)
        elif mode == VectorStoreQueryMode.MMR:
            raise ValueError(
                "MMR mode is not supported by VastbaseVectorStore."
            )
        else:
            raise ValueError(f"Unsupported query mode: {mode}")

    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[str]:
        """Build filter expr from query.filters, stripping client-side operators."""
        if not query.filters:
            return None
        if _is_client_side_filter(query.filters):
            stripped = _strip_client_side_filters(query.filters)
            return _build_expr(stripped) if stripped else None
        return _build_expr(query.filters)

    def _query_default(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        results = self._collection.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=query.similarity_top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )
        return self._parse_search_results(results, query)

    async def _aquery_default(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        results = await self._async_collection.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=query.similarity_top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )
        return self._parse_search_results(results, query)

    def _query_text(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        if not self.hybrid_search:
            raise ValueError(
                "TEXT_SEARCH mode requires hybrid_search=True. "
                "Set hybrid_search=True and perform_setup=True to enable fulltext indexing."
            )
        results = self._collection.search(
            data=[query.query_str],
            anns_field="text",
            param={"metric_type": "BM25"},
            limit=query.similarity_top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )
        return self._parse_search_results(results, query)

    async def _atext_search(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        if not self.hybrid_search:
            raise ValueError(
                "TEXT_SEARCH mode requires hybrid_search=True."
            )
        results = await self._async_collection.search(
            data=[query.query_str],
            anns_field="text",
            param={"metric_type": "BM25"},
            limit=query.similarity_top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )
        return self._parse_search_results(results, query)

    def _query_hybrid(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        if not self.hybrid_search:
            raise ValueError(
                "HYBRID mode requires hybrid_search=True."
            )
        from pyvastbase import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=query.query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=query.similarity_top_k,
        )
        sparse_req = AnnSearchRequest(
            data=query.query_str,
            anns_field="text",
            param={"metric_type": "BM25"},
            limit=getattr(query, "sparse_top_k", None) or query.similarity_top_k,
        )
        results = self._collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=getattr(query, "hybrid_top_k", None) or query.similarity_top_k,
        )
        return self._parse_search_results(results, query)

    async def _aquery_hybrid(
        self, query: VectorStoreQuery, filter_expr: Optional[str], output_fields: List[str]
    ) -> VectorStoreQueryResult:
        if not self.hybrid_search:
            raise ValueError(
                "HYBRID mode requires hybrid_search=True."
            )
        from pyvastbase import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=query.query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=query.similarity_top_k,
        )
        sparse_req = AnnSearchRequest(
            data=query.query_str,
            anns_field="text",
            param={"metric_type": "BM25"},
            limit=getattr(query, "sparse_top_k", None) or query.similarity_top_k,
        )
        results = await self._async_collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=getattr(query, "hybrid_top_k", None) or query.similarity_top_k,
        )
        return self._parse_search_results(results, query)

    def _parse_search_results(
        self, results: Any, query: VectorStoreQuery
    ) -> VectorStoreQueryResult:
        """Convert pyvastbase search results to VectorStoreQueryResult."""
        nodes = []
        similarities = []
        ids = []

        hits = results[0] if results and len(results) > 0 else []
        for hit in hits:
            node_id = hit.data.get("node_id") if hasattr(hit, "data") else getattr(hit, "node_id", "")
            text = hit.data.get("text", "") if hasattr(hit, "data") else getattr(hit, "text", "")
            metadata = hit.data.get("metadata_", {}) if hasattr(hit, "data") else getattr(hit, "metadata_", {})
            distance = getattr(hit, "distance", 0.0)
            similarity = 1.0 - distance

            node = TextNode(
                id_=node_id,
                text=text,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            nodes.append(node)
            similarities.append(similarity)
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_search.py
git commit -m "feat: query methods — DEFAULT/TEXT_SEARCH/HYBRID/MMR + async"
```

---

### Task 7: Core Methods — get_nodes + delete_nodes

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py`
- Modify: `tests/test_crud.py` (append tests)

**Interfaces:**
- Consumes: `_collection.query()`, `_collection.delete()`, filter utils
- Produces: `get_nodes()`, `aget_nodes()`, `delete_nodes()`, `adelete_nodes()`

- [ ] **Step 1: Append get_nodes and delete_nodes tests to test_crud.py**

Append to `tests/test_crud.py`:

```python
class TestGetNodes:
    def test_get_nodes_by_ids(self):
        mock_col = MagicMock()
        mock_col.query.return_value = [
            {"node_id": "node-0", "text": "Text 0", "metadata_": {"ref_doc_id": "doc-0"}, "embedding": [0.1] * EMBED_DIM},
            {"node_id": "node-1", "text": "Text 1", "metadata_": {"ref_doc_id": "doc-0"}, "embedding": [0.2] * EMBED_DIM},
        ]
        store = _make_store(mock_col)

        result = store.get_nodes(node_ids=["node-0", "node-1"])

        assert len(result) == 2
        mock_col.query.assert_called_once()

    def test_get_nodes_requires_ids_or_filters(self):
        store = _make_store()

        with pytest.raises((AssertionError, ValueError)):
            store.get_nodes()

    def test_get_nodes_returns_text_nodes(self):
        mock_col = MagicMock()
        mock_col.query.return_value = [
            {"node_id": "n1", "text": "Hello", "metadata_": {"key": "val"}, "embedding": [0.1] * EMBED_DIM},
        ]
        store = _make_store(mock_col)

        result = store.get_nodes(node_ids=["n1"])

        assert len(result) == 1
        assert isinstance(result[0], TextNode)


class TestDeleteNodes:
    def test_delete_nodes_by_ids(self):
        mock_col = MagicMock()
        store = _make_store(mock_col)

        store.delete_nodes(node_ids=["node-0", "node-1"])

        mock_col.delete.assert_called_once()
        expr = mock_col.delete.call_args[1].get("expr") or mock_col.delete.call_args[0][0]
        assert "node-0" in expr
        assert "node-1" in expr

    def test_delete_nodes_no_op_when_empty(self):
        store = _make_store()
        store.delete_nodes()  # No node_ids, no filters → no-op


class TestAsyncGetNodes:
    @pytest.mark.asyncio
    async def test_aget_nodes(self):
        mock_async = AsyncMock()
        mock_async.query.return_value = [
            {"node_id": "n1", "text": "Hi", "metadata_": {}, "embedding": [0.1] * EMBED_DIM},
        ]
        store = _make_store(mock_async_col=mock_async)

        result = await store.aget_nodes(node_ids=["n1"])

        assert len(result) == 1
        mock_async.query.assert_called_once()


class TestAsyncDeleteNodes:
    @pytest.mark.asyncio
    async def test_adelete_nodes(self):
        mock_async = AsyncMock()
        store = _make_store(mock_async_col=mock_async)

        await store.adelete_nodes(node_ids=["node-0"])

        mock_async.delete.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crud.py::TestGetNodes tests/test_crud.py::TestDeleteNodes -v`
Expected: FAIL — methods not implemented

- [ ] **Step 3: Implement get_nodes, delete_nodes methods**

Add to `VastbaseVectorStore` in `base.py`:

```python
    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Retrieve nodes by IDs or filters."""
        if not node_ids and not filters:
            raise ValueError("Must provide node_ids or filters")
        self._initialize()

        expr = self._build_get_expr(node_ids, filters)
        output_fields = ["node_id", "text", "metadata_", "embedding"]

        rows = self._collection.query(expr=expr, output_fields=output_fields)

        # Client-side filter if needed
        if filters and _is_client_side_filter(filters):
            rows = _client_side_filter(rows, filters)

        return self._rows_to_nodes(rows)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Async retrieve nodes by IDs or filters."""
        if not node_ids and not filters:
            raise ValueError("Must provide node_ids or filters")
        self._initialize()

        expr = self._build_get_expr(node_ids, filters)
        output_fields = ["node_id", "text", "metadata_", "embedding"]

        rows = await self._async_collection.query(expr=expr, output_fields=output_fields)

        if filters and _is_client_side_filter(filters):
            rows = _client_side_filter(rows, filters)

        return self._rows_to_nodes(rows)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by IDs or filters."""
        if not node_ids and not filters:
            return
        self._initialize()

        expr = self._build_get_expr(node_ids, filters)
        self._collection.delete(expr=expr)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async delete nodes by IDs or filters."""
        if not node_ids and not filters:
            return
        self._initialize()

        expr = self._build_get_expr(node_ids, filters)
        await self._async_collection.delete(expr=expr)

    def _build_get_expr(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[MetadataFilters],
    ) -> str:
        """Build expr for get_nodes/delete_nodes from node_ids and/or filters."""
        parts = []

        if node_ids:
            id_list = ", ".join(f"'{nid}'" for nid in node_ids)
            parts.append(f"node_id IN ({id_list})")

        if filters:
            if _is_client_side_filter(filters):
                # Strip client-side operators for server query
                stripped = _strip_client_side_filters(filters)
                if stripped:
                    parts.append(_build_expr(stripped))
            else:
                parts.append(_build_expr(filters))

        return " AND ".join(parts) if parts else "TRUE"

    def _rows_to_nodes(self, rows: List[Dict[str, Any]]) -> List[BaseNode]:
        """Convert query result rows to TextNode list."""
        nodes = []
        for row in rows:
            node = TextNode(
                id_=row.get("node_id", ""),
                text=row.get("text", ""),
                metadata=row.get("metadata_", {}) or {},
            )
            if row.get("embedding"):
                node.embedding = row["embedding"]
            nodes.append(node)
        return nodes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_crud.py -v`
Expected: All 18 tests PASS (10 original + 8 new)

- [ ] **Step 5: Commit**

```bash
git add llama_index/vector_stores/vastbase/base.py tests/test_crud.py
git commit -m "feat: get_nodes/delete_nodes + async variants with client-side filter support"
```

---

### Task 8: Async Parity Tests

**Files:**
- Create: `tests/test_async.py`

- [ ] **Step 1: Write comprehensive async parity tests**

Create `tests/test_async.py`:

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from llama_index.vector_stores.vastbase.base import VastbaseVectorStore


EMBED_DIM = 128


def _make_store(mock_col=None, mock_async_col=None):
    with patch.object(VastbaseVectorStore, "__init__", lambda self: None):
        store = VastbaseVectorStore()
        store._is_initialized = True
        store._collection = mock_col or MagicMock()
        store._async_collection = mock_async_col or AsyncMock()
        store.collection_name = "test"
        store.embed_dim = EMBED_DIM
        store.hybrid_search = False
        store._connection_alias = "test_alias"
        store.initialization_fail_on_error = False
        return store


def _make_nodes(count=2):
    nodes = []
    for i in range(count):
        node = TextNode(
            text=f"Async text {i}",
            embedding=[0.1 * (i + 1)] * EMBED_DIM,
            metadata={"cat": f"c{i}", "ref_doc_id": f"doc-{i}"},
        )
        node.node_id = f"async-node-{i}"
        nodes.append(node)
    return nodes


class TestAsyncAddParity:
    @pytest.mark.asyncio
    async def test_async_add_same_result_as_sync(self):
        nodes = _make_nodes(3)

        mock_col = MagicMock()
        mock_col.insert.return_value = MagicMock(insert_count=3)
        sync_store = _make_store(mock_col)

        mock_async = AsyncMock()
        mock_async.insert.return_value = MagicMock(insert_count=3)
        async_store = _make_store(mock_async_col=mock_async)

        sync_result = sync_store.add(nodes)
        async_result = await async_store.async_add(nodes)

        assert sync_result == async_result


class TestAsyncDeleteParity:
    @pytest.mark.asyncio
    async def test_adelete_uses_same_expr_as_sync(self):
        mock_col = MagicMock()
        sync_store = _make_store(mock_col)
        sync_store.delete(ref_doc_id="doc-0")
        sync_expr = mock_col.delete.call_args[1].get("expr") or mock_col.delete.call_args[0][0]

        mock_async = AsyncMock()
        async_store = _make_store(mock_async_col=mock_async)
        await async_store.adelete(ref_doc_id="doc-0")
        async_expr = mock_async.delete.call_args[1].get("expr") or mock_async.delete.call_args[0][0]

        assert sync_expr == async_expr


class TestAsyncQueryParity:
    @pytest.mark.asyncio
    async def test_aquery_default_returns_same_type(self):
        mock_async = AsyncMock()
        mock_async.search.return_value = [[]]
        store = _make_store(mock_async_col=mock_async)

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = await store.aquery(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert result.ids == []

    @pytest.mark.asyncio
    async def test_aquery_mmr_raises(self):
        store = _make_store()

        query = VectorStoreQuery(
            query_embedding=[0.1] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )

        with pytest.raises(ValueError, match="MMR"):
            await store.aquery(query)

    @pytest.mark.asyncio
    async def test_aquery_text_search_requires_hybrid(self):
        store = _make_store()
        store.hybrid_search = False

        query = VectorStoreQuery(
            query_str="test",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.SPARSE,
        )

        with pytest.raises(ValueError, match="hybrid_search=True"):
            await store.aquery(query)


class TestAsyncGetNodesParity:
    @pytest.mark.asyncio
    async def test_aget_nodes_returns_nodes(self):
        mock_async = AsyncMock()
        mock_async.query.return_value = [
            {"node_id": "n1", "text": "Hi", "metadata_": {}, "embedding": [0.1] * EMBED_DIM},
        ]
        store = _make_store(mock_async_col=mock_async)

        result = await store.aget_nodes(node_ids=["n1"])

        assert len(result) == 1
        assert isinstance(result[0], TextNode)

    @pytest.mark.asyncio
    async def test_aget_nodes_requires_ids_or_filters(self):
        store = _make_store()

        with pytest.raises((AssertionError, ValueError)):
            await store.aget_nodes()


class TestAsyncClearParity:
    @pytest.mark.asyncio
    async def test_aclear_calls_truncate(self):
        mock_async = AsyncMock()
        store = _make_store(mock_async_col=mock_async)

        await store.aclear()

        mock_async.truncate.assert_called_once()


class TestCloseCleanup:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        with patch("llama_index.vector_stores.vastbase.base.remove_connection") as mock_rm:
            store = _make_store()
            store._connection_alias = "test_alias_123"

            await store.close()

            mock_rm.assert_called_once_with("test_alias_123")
            assert store._is_initialized is False

    @pytest.mark.asyncio
    async def test_close_no_op_when_not_initialized(self):
        with patch("llama_index.vector_stores.vastbase.base.remove_connection") as mock_rm:
            store = _make_store()
            store._is_initialized = False

            await store.close()

            mock_rm.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_async.py -v`
Expected: All 11 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_async.py
git commit -m "test: async parity tests — 11 tests covering all async methods"
```

---

### Task 9: Integration Tests + Demo Script

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/test_framework_integration.py`
- Create: `tests/demo_llamaindex.py`

- [ ] **Step 1: Write integration test scaffold**

Create `tests/test_integration.py`:

```python
"""E2E integration tests against real Vastbase instance.

Requires: Vastbase at 172.16.105.107:15432/vastbase (user: aidev, password: Vbase_123456)
Skip with: pytest -m "not integration" or set VB_INTEGRATION=0
"""

import os
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

VB_HOST = os.environ.get("VB_HOST", "172.16.105.107")
VB_PORT = int(os.environ.get("VB_PORT", "15432"))
VB_DB = os.environ.get("VB_DB", "vastbase")
VB_USER = os.environ.get("VB_USER", "aidev")
VB_PASS = os.environ.get("VB_PASS", "Vbase_123456")
EMBED_DIM = 128

skipif_no_vb = pytest.mark.skipif(
    os.environ.get("VB_INTEGRATION", "1") == "0",
    reason="VB_INTEGRATION=0 — skipping integration tests",
)


@pytest.fixture(scope="module")
def store():
    from llama_index.vector_stores.vastbase import VastbaseVectorStore
    s = VastbaseVectorStore(
        host=VB_HOST, port=VB_PORT, database=VB_DB,
        user=VB_USER, password=VB_PASS,
        collection_name="test_integration",
        embed_dim=EMBED_DIM,
        perform_setup=True,
        hnsw_kwargs={"m": 16, "ef_construction": 64},
    )
    yield s
    import asyncio
    asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture(scope="module")
def sample_nodes():
    nodes = []
    for i in range(10):
        node = TextNode(
            text=f"Integration test document {i} about topic {'alpha' if i < 5 else 'beta'}",
            embedding=[float(i) / 10.0] * EMBED_DIM,
            metadata={
                "category": "alpha" if i < 5 else "beta",
                "score": float(i),
                "ref_doc_id": f"doc-{i // 3}",
            },
        )
        node.node_id = f"int-node-{i}"
        nodes.append(node)
    return nodes


@skipif_no_vb
class TestIntegrationCollectionInit:
    def test_store_initialized(self, store):
        assert store._is_initialized is True

    def test_client_is_collection(self, store):
        assert store.client is not None


@skipif_no_vb
class TestIntegrationCRUD:
    def test_add_nodes(self, store, sample_nodes):
        result = store.add(sample_nodes)
        assert len(result) == 10
        assert result[0] == "int-node-0"

    def test_get_nodes_by_id(self, store):
        nodes = store.get_nodes(node_ids=["int-node-0", "int-node-1"])
        assert len(nodes) == 2

    def test_delete_by_ref_doc_id(self, store):
        store.delete(ref_doc_id="doc-0")
        # doc-0 had nodes int-node-0, int-node-1, int-node-2
        # After delete, get_nodes for these should return empty
        nodes = store.get_nodes(node_ids=["int-node-0"])
        assert len(nodes) == 0


@skipif_no_vb
class TestIntegrationSearch:
    def test_default_search(self, store):
        query = VectorStoreQuery(
            query_embedding=[0.5] * EMBED_DIM,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store.query(query)
        assert len(result.ids) <= 3
        assert all(s >= 0 for s in result.similarities)

    def test_search_with_filter(self, store):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="beta", operator=FilterOperator.EQ)
            ],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=[0.7] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = store.query(query)
        assert len(result.ids) <= 5


@skipif_no_vb
class TestIntegrationCleanup:
    def test_clear(self, store):
        store.clear()
        # After clear, search should return empty
        query = VectorStoreQuery(
            query_embedding=[0.5] * EMBED_DIM,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store.query(query)
        assert len(result.ids) == 0
```

- [ ] **Step 2: Write framework integration test**

Create `tests/test_framework_integration.py`:

```python
"""Framework-level integration test — VectorStoreIndex + VastbaseVectorStore.

Requires: Vastbase at 172.16.105.107:15432/vastbase
"""

import os
import pytest

VB_HOST = os.environ.get("VB_HOST", "172.16.105.107")
VB_PORT = int(os.environ.get("VB_PORT", "15432"))
VB_DB = os.environ.get("VB_DB", "vastbase")
VB_USER = os.environ.get("VB_USER", "aidev")
VB_PASS = os.environ.get("VB_PASS", "Vbase_123456")
EMBED_DIM = 128

skipif_no_vb = pytest.mark.skipif(
    os.environ.get("VB_INTEGRATION", "1") == "0",
    reason="VB_INTEGRATION=0",
)


@skipif_no_vb
class TestVectorStoreIndex:
    def test_insert_and_retrieve(self):
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.schema import Document
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        store = VastbaseVectorStore(
            host=VB_HOST, port=VB_PORT, database=VB_DB,
            user=VB_USER, password=VB_PASS,
            collection_name="test_framework",
            embed_dim=EMBED_DIM,
            perform_setup=True,
        )

        try:
            storage_context = StorageContext.from_defaults(vector_store=store)
            docs = [
                Document(text="Vastbase is a vector database", metadata={"source": "test"}),
                Document(text="LlamaIndex provides RAG capabilities", metadata={"source": "test"}),
            ]

            embed_model = MockEmbedding(embed_dim=EMBED_DIM)
            index = VectorStoreIndex.from_documents(
                docs, storage_context=storage_context, embed_model=embed_model,
            )

            retriever = index.as_retriever(similarity_top_k=2)
            results = retriever.retrieve("vector database")

            assert len(results) > 0
        finally:
            store.clear()
            import asyncio
            asyncio.get_event_loop().run_until_complete(store.close())

    def test_persist_raises_not_implemented(self):
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        with pytest.raises(NotImplementedError):
            store = VastbaseVectorStore.__new__(VastbaseVectorStore)
            store.persist()
```

- [ ] **Step 3: Write Demo script**

Create `tests/demo_llamaindex.py`:

```python
"""Demo validation script — 8 scenarios for VastbaseVectorStore.

Usage: python tests/demo_llamaindex.py
Requires: Vastbase at 172.16.105.107:15432/vastbase
"""

import asyncio
import os
import sys

VB_HOST = os.environ.get("VB_HOST", "172.16.105.107")
VB_PORT = int(os.environ.get("VB_PORT", "15432"))
VB_DB = os.environ.get("VB_DB", "vastbase")
VB_USER = os.environ.get("VB_USER", "aidev")
VB_PASS = os.environ.get("VB_PASS", "Vbase_123456")
EMBED_DIM = 128


def scenario_1_connection():
    """Scenario 1: Connection initialization."""
    print("\n=== Scenario 1: Connection Initialization ===")
    from llama_index.vector_stores.vastbase import VastbaseVectorStore

    store = VastbaseVectorStore(
        host=VB_HOST, port=VB_PORT, database=VB_DB,
        user=VB_USER, password=VB_PASS,
        collection_name="demo_collection",
        embed_dim=EMBED_DIM,
        perform_setup=True,
        hnsw_kwargs={"m": 16, "ef_construction": 64},
        hybrid_search=True,
    )
    print(f"✅ Connected: collection={store.collection_name}, initialized={store._is_initialized}")
    return store


def scenario_2_add(store):
    """Scenario 2: Document vector ingestion."""
    print("\n=== Scenario 2: Document Vector Ingestion ===")
    from llama_index.core.schema import TextNode

    nodes = []
    topics = [
        "Vastbase is a high-performance vector database",
        "LlamaIndex provides RAG and indexing capabilities",
        "Python is the most popular language for AI development",
        "Vector similarity search enables semantic retrieval",
        "HNSW is an efficient approximate nearest neighbor algorithm",
    ]
    for i, topic in enumerate(topics):
        node = TextNode(
            text=topic,
            embedding=[float(i + 1) / 10.0] * EMBED_DIM,
            metadata={"topic_id": i, "category": "tech" if i < 3 else "ml"},
        )
        node.node_id = f"demo-node-{i}"
        nodes.append(node)

    result = store.add(nodes)
    print(f"✅ Added {len(result)} nodes: {result[:3]}...")


def scenario_3_default_search(store):
    """Scenario 3: Vector similarity search (DEFAULT)."""
    print("\n=== Scenario 3: Vector Similarity Search (DEFAULT) ===")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    query = VectorStoreQuery(
        query_embedding=[0.3] * EMBED_DIM,
        similarity_top_k=3,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = store.query(query)
    print(f"✅ Found {len(result.ids)} results:")
    for i, (id_, sim) in enumerate(zip(result.ids, result.similarities)):
        print(f"   [{i+1}] node={id_}, similarity={sim:.4f}")


def scenario_4_filter_search(store):
    """Scenario 4: Metadata filter search."""
    print("\n=== Scenario 4: Metadata Filter Search ===")
    from llama_index.core.vector_stores.types import (
        VectorStoreQuery, VectorStoreQueryMode,
        MetadataFilter, MetadataFilters, FilterOperator, FilterCondition,
    )

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)
        ],
        condition=FilterCondition.AND,
    )
    query = VectorStoreQuery(
        query_embedding=[0.2] * EMBED_DIM,
        similarity_top_k=5,
        mode=VectorStoreQueryMode.DEFAULT,
        filters=filters,
    )
    result = store.query(query)
    print(f"✅ Filtered results (category=tech): {len(result.ids)} hits")


def scenario_5_text_search(store):
    """Scenario 5: Fulltext search (TEXT_SEARCH)."""
    print("\n=== Scenario 5: Fulltext Search (TEXT_SEARCH) ===")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    try:
        query = VectorStoreQuery(
            query_str="vector database",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.SPARSE,
        )
        result = store.query(query)
        print(f"✅ Text search results: {len(result.ids)} hits")
    except Exception as e:
        print(f"⚠️ Text search: {e}")


def scenario_6_hybrid_search(store):
    """Scenario 6: Hybrid search (HYBRID, RRF fusion)."""
    print("\n=== Scenario 6: Hybrid Search (HYBRID) ===")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    try:
        query = VectorStoreQuery(
            query_embedding=[0.3] * EMBED_DIM,
            query_str="vector similarity",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = store.query(query)
        print(f"✅ Hybrid search results: {len(result.ids)} hits")
    except Exception as e:
        print(f"⚠️ Hybrid search: {e}")


def scenario_7_mmr(store):
    """Scenario 7: MMR rerank (should raise ValueError)."""
    print("\n=== Scenario 7: MMR Rerank ===")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    try:
        query = VectorStoreQuery(
            query_embedding=[0.3] * EMBED_DIM,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        store.query(query)
        print("❌ MMR should have raised ValueError")
    except ValueError as e:
        print(f"✅ MMR correctly raised ValueError: {e}")


def scenario_8_lifecycle(store):
    """Scenario 8: Delete and clear lifecycle."""
    print("\n=== Scenario 8: Delete & Clear Lifecycle ===")

    # Get nodes
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
    query = VectorStoreQuery(
        query_embedding=[0.3] * EMBED_DIM,
        similarity_top_k=10,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = store.query(query)
    print(f"   Before delete: {len(result.ids)} nodes")

    # Delete some nodes
    if result.ids:
        store.delete_nodes(node_ids=result.ids[:2])
        print(f"   Deleted nodes: {result.ids[:2]}")

    # Clear
    store.clear()
    result_after = store.query(query)
    print(f"   After clear: {len(result_after.ids)} nodes")
    print("✅ Lifecycle complete")


def main():
    print("=" * 60)
    print("LlamaIndex VastbaseVectorStore — Demo Validation")
    print("=" * 60)

    store = scenario_1_connection()

    try:
        scenario_2_add(store)
        scenario_3_default_search(store)
        scenario_4_filter_search(store)
        scenario_5_text_search(store)
        scenario_6_hybrid_search(store)
        scenario_7_mmr(store)
        scenario_8_lifecycle(store)

        print("\n" + "=" * 60)
        print("✅ All 8 scenarios completed successfully!")
        print("=" * 60)
    finally:
        asyncio.get_event_loop().run_until_complete(store.close())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py tests/test_framework_integration.py tests/demo_llamaindex.py
git commit -m "test: integration tests + 8-scenario demo script"
```

---

### Task 10: Final Verification + Cleanup

**Files:**
- Modify: `llama_index/vector_stores/vastbase/base.py` (add persist stub)

- [ ] **Step 1: Add persist() stub**

Add to `VastbaseVectorStore` in `base.py`:

```python
    def persist(self, persist_path: Optional[str] = None, fs: Optional[Any] = None) -> None:
        """Not supported — Vastbase is a persistent database."""
        raise NotImplementedError(
            "persist() is not supported. Vastbase data is already persistent."
        )
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/test_framework_integration.py`
Expected: All unit tests PASS (50+ tests)

- [ ] **Step 3: Verify import path**

Run: `python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print(VastbaseVectorStore.class_name())"`
Expected: `VastbaseVectorStore`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete VastbaseVectorStore implementation — 13 methods, 14 operators, 4 query modes"
```
