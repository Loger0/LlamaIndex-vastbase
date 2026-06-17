# LlamaIndex ChatStore Vastbase 适配 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `VastbaseChatStore(BaseChatStore)` — a drop-in replacement for PostgresChatStore using pyvastbase Collection API, with 14 methods (7 sync + 7 native async).

**Architecture:** Standalone pip package (`llama-index-storage-chat-store-vastbase`) with 2 Python source files. Uses pyvastbase `Collection` + `AsyncCollection` to manage chat message storage as `{id, key, value}` rows, with all array operations (append, delete by index, delete last) performed in Python layer on JSON-serialized `List[ChatMessage]`.

**Tech Stack:** Python >=3.10, pyvastbase >=0.2.7, llama-index-core >=0.13.0,<0.15, pytest + pytest-asyncio

## Global Constraints

- pyproject.toml: `requires-python = ">=3.10,<4.0"`, dependencies: `llama-index-core>=0.13.0,<0.15`, `pyvastbase>=0.2.7`
- Zero direct SQL — all DB operations through pyvastbase Collection API
- All 14 methods explicitly implemented (no inherited `asyncio.to_thread()` defaults)
- Error handling: silent None/empty-list, matching upstream PostgresChatStore semantics
- Package namespace: `llama_index.storage.chat_store.vastbase`
- Class name: `VastbaseChatStore`
- Vastbase connection: host=172.16.105.107, port=15432, database=vastbase, user=aidev, password=Vbase_123456

---

## File Structure

```
llama-index-storage-chat-store-vastbase/
├── pyproject.toml                              # Task 1: Package config
├── README.md                                   # Task 1: Package readme
├── tests/
│   ├── __init__.py                             # Task 6: Test package
│   ├── conftest.py                             # Task 6: Vastbase fixtures
│   ├── test_vastbase_chatstore.py              # Tasks 7-13: Sync tests (RED)
│   └── test_vastbase_chatstore_async.py        # Tasks 14-20: Async tests (RED)
└── llama_index/
    └── storage/
        └── chat_store/
            └── vastbase/
                ├── __init__.py                 # Task 2: Package exports
                └── base.py                     # Tasks 3-5, 14-20: Implementation
```

---

### Task 1: Package Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`

**Interfaces:**
- Produces: Package metadata for `pip install -e .`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llama-index-storage-chat-store-vastbase"
version = "0.1.0"
description = "llama-index storage-chat-store vastbase integration"
authors = [{name = "Vastbase", email = "dev@vastdata.com.cn"}]
requires-python = ">=3.10,<4.0"
readme = "README.md"
license = "MIT"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.2.1",
    "pytest-asyncio>=0.21.0",
]

[tool.llamahub]
contains_example = false
import_path = "llama_index.storage.chat_store.vastbase"

[tool.llamahub.class_authors]
VastbaseChatStore = "vastbase"

[tool.hatch.build.targets.wheel]
include = ["llama_index/"]
exclude = ["**/BUILD"]
```

- [ ] **Step 2: Write README.md**

```markdown
# LlamaIndex ChatStore — Vastbase Integration

Vastbase-backed chat history storage for LlamaIndex. Drop-in replacement
for `llama-index-storage-chat-store-postgres`.

## Installation

```bash
pip install llama-index-storage-chat-store-vastbase
```

## Usage

```python
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

# From URI
chat_store = VastbaseChatStore.from_uri(
    "vastbase://user:pass@host:5432/db"
)

# From params
chat_store = VastbaseChatStore.from_params(
    host="localhost", port=5432, database="vastbase",
    user="user", password="pass",
)

chat_store.set_messages("session_1", messages)
msgs = chat_store.get_messages("session_1")
```
```

- [ ] **Step 3: Verify package is installable**

Run: `cd llama-index-storage-chat-store-vastbase && pip install -e . 2>&1 | tail -5`
Expected: `Successfully installed llama-index-storage-chat-store-vastbase-0.1.0`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml README.md
git commit -m "chore: scaffold llama-index-storage-chat-store-vastbase package"
```

---

### Task 2: Package __init__.py

**Files:**
- Create: `llama_index/storage/chat_store/vastbase/__init__.py`

**Interfaces:**
- Produces: `VastbaseChatStore` export

- [ ] **Step 1: Write __init__.py**

```python
"""LlamaIndex ChatStore — Vastbase Integration."""

from llama_index.storage.chat_store.vastbase.base import VastbaseChatStore

__all__ = ["VastbaseChatStore"]
```

- [ ] **Step 2: Verify import works (will fail on class not yet implemented, but module should load)**

Run: `python -c "import llama_index.storage.chat_store.vastbase" 2>&1`
Expected: ImportError about VastbaseChatStore (not ModuleNotFoundError)

- [ ] **Step 3: Create directory structure**

Run: `mkdir -p llama_index/storage/chat_store/vastbase`

- [ ] **Step 4: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/__init__.py
git commit -m "feat: add vastbase chat_store package init"
```

---

### Task 3: VastbaseChatStore — Class Skeleton + Connection Management

**Files:**
- Create: `llama_index/storage/chat_store/vastbase/base.py`

**Interfaces:**
- Consumes: `BaseChatStore` from `llama_index.core.storage.chat_store.base`
- Produces: `VastbaseChatStore` class with `__init__`, `from_params`, `from_uri`, `_parse_uri`, `_ensure_initialized`, all 14 method stubs raising `NotImplementedError`

- [ ] **Step 1: Write class skeleton with connection management**

```python
"""Vastbase ChatStore implementation using pyvastbase Collection API."""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pyvastbase import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    connect,
    has_collection,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore


class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store using pyvastbase Collection API.

    Drop-in replacement for PostgresChatStore. Stores chat messages
    as JSON-serialized lists keyed by session identifier.
    """

    # === Pydantic Fields ===
    table_name: str = Field(
        default="chatstore",
        description="Collection name for storing chat messages.",
    )
    schema_name: str = Field(
        default="public",
        description="Database schema name (retained for API compatibility).",
    )
    use_jsonb: bool = Field(
        default=False,
        description="Retained for API compatibility. Vastbase stores values as TEXT.",
    )

    # === Private Attributes ===
    _host: str = PrivateAttr(default="localhost")
    _port: int = PrivateAttr(default=5432)
    _database: str = PrivateAttr(default="vastbase")
    _user: str = PrivateAttr(default="")
    _password: str = PrivateAttr(default="")
    _connection_alias: str = PrivateAttr(default="")
    _coll: Optional[Collection] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "vastbase",
        user: str = "",
        password: str = "",
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ):
        """Initialize VastbaseChatStore with connection parameters.

        Connection is lazy — established on first method call via _ensure_initialized().
        """
        super().__init__(
            table_name=table_name.lower(),
            schema_name=schema_name.lower(),
        )
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._connection_alias = f"chatstore_{table_name}_{id(self)}"

    @classmethod
    def from_params(
        cls,
        host: str = "localhost",
        port: int = 5432,
        database: str = "vastbase",
        user: str = "",
        password: str = "",
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ) -> "VastbaseChatStore":
        """Create instance from individual connection parameters."""
        return cls(
            host=host, port=port, database=database,
            user=user, password=password,
            table_name=table_name, schema_name=schema_name,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ) -> "VastbaseChatStore":
        """Create instance from vastbase://user:pass@host:port/db URI."""
        params = cls._parse_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            schema_name=schema_name,
            use_jsonb=use_jsonb,
        )

    @staticmethod
    def _parse_uri(uri: str) -> dict:
        """Parse vastbase://user:pass@host:port/db URI."""
        result = urlparse(uri)
        if result.scheme not in ("vastbase", "postgresql", "postgres"):
            raise ValueError(
                f"Unsupported URI scheme: {result.scheme}. "
                f"Expected vastbase:// or postgresql://"
            )
        return {
            "host": result.hostname or "localhost",
            "port": result.port or 5432,
            "database": result.path.lstrip("/") or "vastbase",
            "user": result.username or "",
            "password": result.password or "",
        }

    @staticmethod
    def _escape_key(key: str) -> str:
        """Escape double-quote characters in key for query expressions."""
        return key.replace('"', '\\"')

    def _ensure_initialized(self) -> None:
        """Lazy-init: connect + create/load collection on first use."""
        if self._initialized:
            return
        connect(
            host=self._host, port=self._port,
            database=self._database,
            user=self._user, password=self._password,
            alias=self._connection_alias,
        )
        legacy_name = f"data_{self.table_name}"
        actual_name = legacy_name if has_collection(legacy_name) else self.table_name
        if not has_collection(actual_name):
            schema = CollectionSchema(name=actual_name, fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
                FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="value", dtype=DataType.TEXT),
            ])
            col = Collection(actual_name, schema=schema)
            col.create()
        else:
            col = Collection(actual_name)
        self.table_name = actual_name
        self._coll = col
        self._initialized = True

    def _get_existing(self, key: str) -> Optional[tuple]:
        """Query collection for key. Returns (row_id, list_of_ChatMessage) or None."""
        escaped_key = self._escape_key(key)
        results = self._coll.query(
            expr=f'key == "{escaped_key}"',
            limit=1,
            output_fields=["id", "value"],
        )
        if not results:
            return None
        row = results[0]
        raw_value = row.get("value", "[]")
        try:
            messages = [ChatMessage.model_validate(m) for m in json.loads(raw_value)]
        except (json.JSONDecodeError, Exception):
            messages = []
        return (row["id"], messages)

    @staticmethod
    def _serialize_messages(messages: List[ChatMessage]) -> str:
        """Serialize a list of ChatMessage objects to a JSON string."""
        return json.dumps([m.model_dump() for m in messages])

    # === 7 Synchronous Methods (stubs — implemented in Tasks 4-13) ===

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        raise NotImplementedError

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        raise NotImplementedError

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        raise NotImplementedError

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        raise NotImplementedError

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        raise NotImplementedError

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        raise NotImplementedError

    def get_keys(self) -> List[str]:
        """Get all keys."""
        raise NotImplementedError

    # === 7 Asynchronous Methods (stubs — implemented in Tasks 14-20) ===

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Async version of set_messages."""
        raise NotImplementedError

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Async version of get_messages."""
        raise NotImplementedError

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of add_message."""
        raise NotImplementedError

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Async version of delete_messages."""
        raise NotImplementedError

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of delete_message."""
        raise NotImplementedError

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of delete_last_message."""
        raise NotImplementedError

    async def aget_keys(self) -> List[str]:
        """Async version of get_keys."""
        raise NotImplementedError
```

- [ ] **Step 2: Verify import works (stubs, no connection yet)**

Run: `python -c "from llama_index.storage.chat_store.vastbase import VastbaseChatStore; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify from_uri parsing**

Run:
```python
python -c "
from llama_index.storage.chat_store.vastbase import VastbaseChatStore
params = VastbaseChatStore._parse_uri('vastbase://user:pass@host:5432/db')
assert params == {'host': 'host', 'port': 5432, 'database': 'db', 'user': 'user', 'password': 'pass'}, f'Got: {params}'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: add VastbaseChatStore skeleton with connection management"
```

---

### Task 4: Implement set_messages + get_messages (Sync)

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — replace `set_messages` and `get_messages` stubs

**Interfaces:**
- Consumes: `_ensure_initialized()`, `_get_existing(key)`, `_serialize_messages(messages)`
- Produces: Working `set_messages(key, messages) -> None`, `get_messages(key) -> List[ChatMessage]`

- [ ] **Step 1: Replace set_messages stub**

```python
def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
    """Set messages for a key."""
    self._ensure_initialized()
    serialized = self._serialize_messages(messages)
    existing = self._get_existing(key)
    if existing:
        row_id, _ = existing
        self._coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    else:
        self._coll.insert([{"key": key, "value": serialized}])
```

- [ ] **Step 2: Replace get_messages stub**

```python
def get_messages(self, key: str) -> List[ChatMessage]:
    """Get messages for a key."""
    self._ensure_initialized()
    existing = self._get_existing(key)
    if existing:
        _, messages = existing
        return messages
    return []
```

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement set_messages and get_messages"
```

---

### Task 5: Implement add_message (Sync)

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — replace `add_message` stub

**Interfaces:**
- Consumes: `_ensure_initialized()`, `_get_existing(key)`, `_serialize_messages(messages)`
- Produces: Working `add_message(key, message) -> None`

- [ ] **Step 1: Replace add_message stub**

```python
def add_message(self, key: str, message: ChatMessage) -> None:
    """Add a message for a key."""
    self._ensure_initialized()
    new_msg_dict = message.model_dump()
    existing = self._get_existing(key)
    if existing:
        row_id, messages = existing
        messages.append(message)
        serialized = self._serialize_messages(messages)
        self._coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    else:
        serialized = json.dumps([new_msg_dict])
        self._coll.insert([{"key": key, "value": serialized}])
```

- [ ] **Step 2: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement add_message"
```

---

### Task 6: Test Infrastructure — conftest.py

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `chat_store` pytest fixture, `SYNC_TABLE_NAME` constant, `VASTBASE_CONFIG` dict

- [ ] **Step 1: Write tests/__init__.py (empty)**

```python
```

- [ ] **Step 2: Write tests/conftest.py**

```python
"""Shared test fixtures for VastbaseChatStore tests."""

import pytest
from pyvastbase import close_all, has_collection

from llama_index.storage.chat_store.vastbase import VastbaseChatStore

VASTBASE_CONFIG = {
    "host": "172.16.105.107",
    "port": 15432,
    "database": "vastbase",
    "user": "aidev",
    "password": "Vbase_123456",
}

SYNC_TABLE_NAME = "test_chatstore_sync"
ASYNC_TABLE_NAME = "test_chatstore_async"


@pytest.fixture
def chat_store():
    """Create a VastbaseChatStore for sync testing. Cleans up after."""
    store = VastbaseChatStore(
        **VASTBASE_CONFIG,
        table_name=SYNC_TABLE_NAME,
    )
    yield store
    # Cleanup: drop test collection and close connections
    try:
        if has_collection(SYNC_TABLE_NAME):
            from pyvastbase import Collection
            Collection(SYNC_TABLE_NAME).drop()
    except Exception:
        pass
    close_all()
```

- [ ] **Step 3: Verify conftest loads**

Run: `cd llama-index-storage-chat-store-vastbase && python -c "import tests.conftest; print('OK')" 2>&1`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add conftest with Vastbase fixtures"
```

---

### Task 7: Test set_messages (RED)

**Files:**
- Create: `tests/test_vastbase_chatstore.py`

**Interfaces:**
- Consumes: `chat_store` fixture from conftest

- [ ] **Step 1: Write failing tests for set_messages**

```python
"""Synchronous method tests for VastbaseChatStore."""

from llama_index.core.llms import ChatMessage


class TestSetMessages:
    """Tests for set_messages method."""

    def test_set_messages_new_key(self, chat_store):
        """set_messages creates a new key with messages."""
        msgs = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
        ]
        chat_store.set_messages("test_key_1", msgs)
        result = chat_store.get_messages("test_key_1")
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "hello"
        assert result[1].role == "assistant"
        assert result[1].content == "hi there"

    def test_set_messages_overwrite(self, chat_store):
        """set_messages overwrites existing messages for a key."""
        msgs_a = [ChatMessage(role="user", content="first")]
        msgs_b = [ChatMessage(role="user", content="second")]
        chat_store.set_messages("test_key_2", msgs_a)
        chat_store.set_messages("test_key_2", msgs_b)
        result = chat_store.get_messages("test_key_2")
        assert len(result) == 1
        assert result[0].content == "second"

    def test_set_messages_empty_list(self, chat_store):
        """set_messages with empty list stores empty array."""
        chat_store.set_messages("test_key_3", [])
        result = chat_store.get_messages("test_key_3")
        assert result == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore.py::TestSetMessages -v 2>&1`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore.py
git commit -m "test: add set_messages tests"
```

---

### Task 8: Test get_messages (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore.py` — add `TestGetMessages` class

- [ ] **Step 1: Add get_messages tests**

```python
class TestGetMessages:
    """Tests for get_messages method."""

    def test_get_messages_existing(self, chat_store):
        """get_messages returns stored messages."""
        msgs = [ChatMessage(role="user", content="hello")]
        chat_store.set_messages("gkey_1", msgs)
        result = chat_store.get_messages("gkey_1")
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_get_messages_nonexistent(self, chat_store):
        """get_messages returns empty list for nonexistent key."""
        result = chat_store.get_messages("nonexistent_key")
        assert result == []

    def test_get_messages_fidelity(self, chat_store):
        """get_messages preserves all ChatMessage fields."""
        msg = ChatMessage(
            role="assistant",
            content="I can help",
            additional_kwargs={"tool_calls": [{"name": "search"}]},
        )
        chat_store.set_messages("gkey_2", [msg])
        result = chat_store.get_messages("gkey_2")
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "I can help"
        assert result[0].additional_kwargs == {"tool_calls": [{"name": "search"}]}
```

- [ ] **Step 2: Run tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore.py::TestGetMessages -v 2>&1`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore.py
git commit -m "test: add get_messages tests"
```

---

### Task 9: Test add_message (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore.py` — add `TestAddMessage` class

- [ ] **Step 1: Add add_message tests**

```python
class TestAddMessage:
    """Tests for add_message method."""

    def test_add_message_existing_key(self, chat_store):
        """add_message appends to existing key."""
        chat_store.set_messages("akey_1", [ChatMessage(role="user", content="first")])
        chat_store.add_message("akey_1", ChatMessage(role="assistant", content="second"))
        result = chat_store.get_messages("akey_1")
        assert len(result) == 2
        assert result[0].content == "first"
        assert result[1].content == "second"

    def test_add_message_new_key(self, chat_store):
        """add_message creates new key if not exists."""
        chat_store.add_message("akey_2", ChatMessage(role="user", content="hello"))
        result = chat_store.get_messages("akey_2")
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_add_message_multiple(self, chat_store):
        """add_message preserves order over multiple calls."""
        chat_store.add_message("akey_3", ChatMessage(role="user", content="a"))
        chat_store.add_message("akey_3", ChatMessage(role="user", content="b"))
        chat_store.add_message("akey_3", ChatMessage(role="user", content="c"))
        result = chat_store.get_messages("akey_3")
        assert len(result) == 3
        assert [m.content for m in result] == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore.py::TestAddMessage -v 2>&1`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore.py
git commit -m "test: add add_message tests"
```

---

### Task 10: Implement delete_messages + delete_message (Sync)

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — replace `delete_messages` and `delete_message` stubs

- [ ] **Step 1: Replace delete_messages stub**

```python
def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
    """Delete messages for a key."""
    self._ensure_initialized()
    existing = self._get_existing(key)
    if existing:
        row_id, _ = existing
        self._coll.delete(pks=[row_id])
    return None
```

- [ ] **Step 2: Replace delete_message stub**

```python
def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
    """Delete specific message for a key."""
    self._ensure_initialized()
    existing = self._get_existing(key)
    if existing is None:
        return None
    row_id, messages = existing
    if idx < 0 or idx >= len(messages):
        return None
    removed = messages.pop(idx)
    serialized = self._serialize_messages(messages)
    self._coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    return removed
```

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement delete_messages and delete_message"
```

---

### Task 11: Implement delete_last_message + get_keys (Sync)

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — replace `delete_last_message` and `get_keys` stubs

- [ ] **Step 1: Replace delete_last_message stub**

```python
def delete_last_message(self, key: str) -> Optional[ChatMessage]:
    """Delete last message for a key."""
    self._ensure_initialized()
    existing = self._get_existing(key)
    if existing is None:
        return None
    row_id, messages = existing
    if len(messages) == 0:
        return None
    removed = messages.pop()
    serialized = self._serialize_messages(messages)
    self._coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    return removed
```

- [ ] **Step 2: Replace get_keys stub**

```python
def get_keys(self) -> List[str]:
    """Get all keys."""
    self._ensure_initialized()
    results = self._coll.query(
        expr="id >= 0",
        output_fields=["key"],
        limit=10000,
    )
    return [r["key"] for r in results]
```

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement delete_last_message and get_keys"
```

---

### Task 12: Test delete_messages (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore.py` — add `TestDeleteMessages` class

- [ ] **Step 1: Add delete_messages tests**

```python
class TestDeleteMessages:
    """Tests for delete_messages method."""

    def test_delete_messages_existing(self, chat_store):
        """delete_messages removes the key and returns None."""
        chat_store.set_messages("dkey_1", [ChatMessage(role="user", content="hi")])
        result = chat_store.delete_messages("dkey_1")
        assert result is None
        assert chat_store.get_messages("dkey_1") == []

    def test_delete_messages_nonexistent(self, chat_store):
        """delete_messages on nonexistent key returns None without error."""
        result = chat_store.delete_messages("nonexistent_key")
        assert result is None

    def test_delete_messages_removes_from_keys(self, chat_store):
        """delete_messages removes key from get_keys."""
        chat_store.set_messages("dkey_2", [ChatMessage(role="user", content="x")])
        assert "dkey_2" in chat_store.get_keys()
        chat_store.delete_messages("dkey_2")
        assert "dkey_2" not in chat_store.get_keys()
```

- [ ] **Step 2: Run tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore.py::TestDeleteMessages -v 2>&1`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore.py
git commit -m "test: add delete_messages tests"
```

---

### Task 13: Test delete_message + delete_last_message + get_keys (RED → GREEN → Commit)

**Files:**
- Modify: `tests/test_vastbase_chatstore.py` — add `TestDeleteMessage`, `TestDeleteLastMessage`, `TestGetKeys` classes

- [ ] **Step 1: Add remaining sync tests**

```python
class TestDeleteMessage:
    """Tests for delete_message method."""

    def test_delete_message_middle(self, chat_store):
        """delete_message removes correct message by index."""
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="user", content="c"),
        ]
        chat_store.set_messages("dmkey_1", msgs)
        removed = chat_store.delete_message("dmkey_1", 1)
        assert removed is not None
        assert removed.content == "b"
        remaining = chat_store.get_messages("dmkey_1")
        assert [m.content for m in remaining] == ["a", "c"]

    def test_delete_message_first(self, chat_store):
        """delete_message with idx=0 removes first message."""
        chat_store.set_messages("dmkey_2", [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="user", content="second"),
        ])
        removed = chat_store.delete_message("dmkey_2", 0)
        assert removed.content == "first"
        remaining = chat_store.get_messages("dmkey_2")
        assert [m.content for m in remaining] == ["second"]

    def test_delete_message_last(self, chat_store):
        """delete_message with idx=-1 on last element."""
        chat_store.set_messages("dmkey_3", [
            ChatMessage(role="user", content="x"),
            ChatMessage(role="user", content="y"),
        ])
        removed = chat_store.delete_message("dmkey_3", 1)
        assert removed.content == "y"
        remaining = chat_store.get_messages("dmkey_3")
        assert [m.content for m in remaining] == ["x"]

    def test_delete_message_out_of_bounds_high(self, chat_store):
        """delete_message with out-of-bounds idx returns None."""
        chat_store.set_messages("dmkey_4", [ChatMessage(role="user", content="only")])
        result = chat_store.delete_message("dmkey_4", 5)
        assert result is None

    def test_delete_message_negative_idx(self, chat_store):
        """delete_message with negative idx returns None."""
        chat_store.set_messages("dmkey_5", [ChatMessage(role="user", content="only")])
        result = chat_store.delete_message("dmkey_5", -1)
        assert result is None

    def test_delete_message_nonexistent_key(self, chat_store):
        """delete_message on nonexistent key returns None."""
        result = chat_store.delete_message("no_such_key", 0)
        assert result is None


class TestDeleteLastMessage:
    """Tests for delete_last_message method."""

    def test_delete_last_message(self, chat_store):
        """delete_last_message removes and returns last message."""
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="user", content="c"),
        ]
        chat_store.set_messages("dlkey_1", msgs)
        removed = chat_store.delete_last_message("dlkey_1")
        assert removed.content == "c"
        remaining = chat_store.get_messages("dlkey_1")
        assert [m.content for m in remaining] == ["a", "b"]

    def test_delete_last_message_single(self, chat_store):
        """delete_last_message with single message returns it and leaves empty."""
        chat_store.set_messages("dlkey_2", [ChatMessage(role="user", content="only")])
        removed = chat_store.delete_last_message("dlkey_2")
        assert removed.content == "only"
        assert chat_store.get_messages("dlkey_2") == []

    def test_delete_last_message_nonexistent(self, chat_store):
        """delete_last_message on nonexistent key returns None."""
        result = chat_store.delete_last_message("no_such_key")
        assert result is None

    def test_delete_last_message_empty_array(self, chat_store):
        """delete_last_message on empty array returns None."""
        chat_store.set_messages("dlkey_3", [])
        result = chat_store.delete_last_message("dlkey_3")
        assert result is None


class TestGetKeys:
    """Tests for get_keys method."""

    def test_get_keys(self, chat_store):
        """get_keys returns all stored keys."""
        chat_store.set_messages("gk_1", [ChatMessage(role="user", content="a")])
        chat_store.set_messages("gk_2", [ChatMessage(role="user", content="b")])
        keys = chat_store.get_keys()
        assert "gk_1" in keys
        assert "gk_2" in keys

    def test_get_keys_empty(self, chat_store):
        """get_keys returns empty list when no keys stored."""
        keys = chat_store.get_keys()
        assert keys == []

    def test_get_keys_consistency(self, chat_store):
        """get_keys reflects set_messages and delete_messages."""
        chat_store.set_messages("gk_3", [ChatMessage(role="user", content="x")])
        assert "gk_3" in chat_store.get_keys()
        chat_store.delete_messages("gk_3")
        assert "gk_3" not in chat_store.get_keys()
```

- [ ] **Step 2: Run all sync tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore.py -v 2>&1`
Expected: ALL 22 tests passed (3 set + 3 get + 3 add + 3 delete_messages + 6 delete_message + 4 delete_last + 3 get_keys)

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore.py
git commit -m "test: add delete_message, delete_last_message, get_keys tests — sync tests complete"
```

---

### Task 14: Async Test Infrastructure — conftest.py update

**Files:**
- Modify: `tests/conftest.py` — add async fixture

- [ ] **Step 1: Add async fixture to conftest.py**

```python
import asyncio


@pytest.fixture
async def async_chat_store():
    """Create a VastbaseChatStore for async testing. Cleans up after."""
    from pyvastbase import AsyncConnections
    store = VastbaseChatStore(
        **VASTBASE_CONFIG,
        table_name=ASYNC_TABLE_NAME,
    )
    yield store
    # Cleanup
    try:
        if has_collection(ASYNC_TABLE_NAME):
            from pyvastbase import Collection
            Collection(ASYNC_TABLE_NAME).drop()
    except Exception:
        pass
    await AsyncConnections.remove_connection(store._connection_alias)
    close_all()
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add async test fixture for VastbaseChatStore"
```

---

### Task 15: Implement Async Methods (GREEN — 7 methods)

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — replace all 7 async stubs with native AsyncCollection implementations

- [ ] **Step 1: Add async helper and replace all async stubs**

```python
async def _ensure_async_coll(self):
    """Lazy-init async collection with native AsyncConnections."""
    from pyvastbase import AsyncCollection, AsyncConnections
    self._ensure_initialized()  # ensure connection alias exists
    await AsyncConnections.connect(
        host=self._host, port=self._port,
        database=self._database,
        user=self._user, password=self._password,
        alias=self._connection_alias,
    )
    return AsyncCollection(self.table_name)


async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
    """Async version of set_messages."""
    coll = await self._ensure_async_coll()
    serialized = self._serialize_messages(messages)
    escaped_key = self._escape_key(key)
    existing = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id"],
    )
    if existing:
        await coll.upsert([{"id": existing[0]["id"], "key": key, "value": serialized}])
    else:
        await coll.insert([{"key": key, "value": serialized}])


async def aget_messages(self, key: str) -> List[ChatMessage]:
    """Async version of get_messages."""
    coll = await self._ensure_async_coll()
    escaped_key = self._escape_key(key)
    results = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id", "value"],
    )
    if not results:
        return []
    raw_value = results[0].get("value", "[]")
    try:
        return [ChatMessage.model_validate(m) for m in json.loads(raw_value)]
    except (json.JSONDecodeError, Exception):
        return []


async def async_add_message(self, key: str, message: ChatMessage) -> None:
    """Async version of add_message."""
    coll = await self._ensure_async_coll()
    new_msg_dict = message.model_dump()
    escaped_key = self._escape_key(key)
    existing = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id", "value"],
    )
    if existing:
        row = existing[0]
        raw_value = row.get("value", "[]")
        try:
            messages = [ChatMessage.model_validate(m) for m in json.loads(raw_value)]
        except (json.JSONDecodeError, Exception):
            messages = []
        messages.append(message)
        serialized = self._serialize_messages(messages)
        await coll.upsert([{"id": row["id"], "key": key, "value": serialized}])
    else:
        await coll.insert([{"key": key, "value": json.dumps([new_msg_dict])}])


async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
    """Async version of delete_messages."""
    coll = await self._ensure_async_coll()
    escaped_key = self._escape_key(key)
    existing = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id"],
    )
    if existing:
        await coll.delete(pks=[existing[0]["id"]])
    return None


async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
    """Async version of delete_message."""
    coll = await self._ensure_async_coll()
    escaped_key = self._escape_key(key)
    existing = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id", "value"],
    )
    if not existing:
        return None
    row = existing[0]
    raw_value = row.get("value", "[]")
    try:
        messages = [ChatMessage.model_validate(m) for m in json.loads(raw_value)]
    except (json.JSONDecodeError, Exception):
        return None
    if idx < 0 or idx >= len(messages):
        return None
    removed = messages.pop(idx)
    serialized = self._serialize_messages(messages)
    await coll.upsert([{"id": row["id"], "key": key, "value": serialized}])
    return removed


async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
    """Async version of delete_last_message."""
    coll = await self._ensure_async_coll()
    escaped_key = self._escape_key(key)
    existing = await coll.query(
        expr=f'key == "{escaped_key}"',
        limit=1,
        output_fields=["id", "value"],
    )
    if not existing:
        return None
    row = existing[0]
    raw_value = row.get("value", "[]")
    try:
        messages = [ChatMessage.model_validate(m) for m in json.loads(raw_value)]
    except (json.JSONDecodeError, Exception):
        return None
    if len(messages) == 0:
        return None
    removed = messages.pop()
    serialized = self._serialize_messages(messages)
    await coll.upsert([{"id": row["id"], "key": key, "value": serialized}])
    return removed


async def aget_keys(self) -> List[str]:
    """Async version of get_keys."""
    coll = await self._ensure_async_coll()
    results = await coll.query(
        expr="id >= 0",
        output_fields=["key"],
        limit=10000,
    )
    return [r["key"] for r in results]
```

- [ ] **Step 2: Verify async import works**

Run:
```python
python -c "
import asyncio
from llama_index.storage.chat_store.vastbase import VastbaseChatStore
print('Async imports OK')
"
```
Expected: `Async imports OK`

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement all 7 native async methods with AsyncCollection"
```

---

### Task 16: Test aset_messages + aget_messages (RED → GREEN)

**Files:**
- Create: `tests/test_vastbase_chatstore_async.py`

- [ ] **Step 1: Write async tests for set/get**

```python
"""Asynchronous method tests for VastbaseChatStore."""

import pytest
from llama_index.core.llms import ChatMessage


class TestAsyncSetMessages:
    """Tests for aset_messages method."""

    @pytest.mark.asyncio
    async def test_aset_messages_new_key(self, async_chat_store):
        """aset_messages creates a new key with messages."""
        msgs = [
            ChatMessage(role="user", content="async hello"),
            ChatMessage(role="assistant", content="async hi"),
        ]
        await async_chat_store.aset_messages("async_key_1", msgs)
        result = await async_chat_store.aget_messages("async_key_1")
        assert len(result) == 2
        assert result[0].content == "async hello"
        assert result[1].content == "async hi"

    @pytest.mark.asyncio
    async def test_aset_messages_overwrite(self, async_chat_store):
        """aset_messages overwrites existing messages."""
        await async_chat_store.aset_messages(
            "async_key_2", [ChatMessage(role="user", content="v1")]
        )
        await async_chat_store.aset_messages(
            "async_key_2", [ChatMessage(role="user", content="v2")]
        )
        result = await async_chat_store.aget_messages("async_key_2")
        assert len(result) == 1
        assert result[0].content == "v2"

    @pytest.mark.asyncio
    async def test_aset_messages_empty_list(self, async_chat_store):
        """aset_messages with empty list stores empty array."""
        await async_chat_store.aset_messages("async_key_3", [])
        result = await async_chat_store.aget_messages("async_key_3")
        assert result == []


class TestAsyncGetMessages:
    """Tests for aget_messages method."""

    @pytest.mark.asyncio
    async def test_aget_messages_nonexistent(self, async_chat_store):
        """aget_messages returns empty list for nonexistent key."""
        result = await async_chat_store.aget_messages("no_async_key")
        assert result == []

    @pytest.mark.asyncio
    async def test_aget_messages_fidelity(self, async_chat_store):
        """aget_messages preserves ChatMessage fields."""
        msg = ChatMessage(
            role="assistant",
            content="async fidelity",
            additional_kwargs={"key": "value"},
        )
        await async_chat_store.aset_messages("async_key_4", [msg])
        result = await async_chat_store.aget_messages("async_key_4")
        assert result[0].role == "assistant"
        assert result[0].additional_kwargs == {"key": "value"}
```

- [ ] **Step 2: Run async tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore_async.py -v 2>&1`
Expected: 5 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore_async.py
git commit -m "test: add aset_messages and aget_messages async tests"
```

---

### Task 17: Test async_add_message (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore_async.py` — add `TestAsyncAddMessage` class

- [ ] **Step 1: Add async add_message tests**

```python
class TestAsyncAddMessage:
    """Tests for async_add_message method."""

    @pytest.mark.asyncio
    async def test_async_add_message_existing_key(self, async_chat_store):
        """async_add_message appends to existing key."""
        await async_chat_store.aset_messages(
            "aakey_1", [ChatMessage(role="user", content="first")]
        )
        await async_chat_store.async_add_message(
            "aakey_1", ChatMessage(role="assistant", content="second")
        )
        result = await async_chat_store.aget_messages("aakey_1")
        assert len(result) == 2
        assert result[0].content == "first"
        assert result[1].content == "second"

    @pytest.mark.asyncio
    async def test_async_add_message_new_key(self, async_chat_store):
        """async_add_message creates new key."""
        await async_chat_store.async_add_message(
            "aakey_2", ChatMessage(role="user", content="new")
        )
        result = await async_chat_store.aget_messages("aakey_2")
        assert len(result) == 1
        assert result[0].content == "new"

    @pytest.mark.asyncio
    async def test_async_add_message_multiple(self, async_chat_store):
        """async_add_message preserves order."""
        for content in ["a", "b", "c"]:
            await async_chat_store.async_add_message(
                "aakey_3", ChatMessage(role="user", content=content)
            )
        result = await async_chat_store.aget_messages("aakey_3")
        assert [m.content for m in result] == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore_async.py::TestAsyncAddMessage -v 2>&1`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore_async.py
git commit -m "test: add async_add_message tests"
```

---

### Task 18: Test adelete_messages + adelete_message (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore_async.py` — add `TestAsyncDeleteMessages`, `TestAsyncDeleteMessage` classes

- [ ] **Step 1: Add async delete tests**

```python
class TestAsyncDeleteMessages:
    """Tests for adelete_messages method."""

    @pytest.mark.asyncio
    async def test_adelete_messages_existing(self, async_chat_store):
        """adelete_messages removes key and returns None."""
        await async_chat_store.aset_messages(
            "adkey_1", [ChatMessage(role="user", content="hi")]
        )
        result = await async_chat_store.adelete_messages("adkey_1")
        assert result is None
        remaining = await async_chat_store.aget_messages("adkey_1")
        assert remaining == []

    @pytest.mark.asyncio
    async def test_adelete_messages_nonexistent(self, async_chat_store):
        """adelete_messages on nonexistent key returns None."""
        result = await async_chat_store.adelete_messages("no_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_adelete_messages_removes_from_keys(self, async_chat_store):
        """adelete_messages removes key from aget_keys."""
        await async_chat_store.aset_messages(
            "adkey_2", [ChatMessage(role="user", content="x")]
        )
        assert "adkey_2" in await async_chat_store.aget_keys()
        await async_chat_store.adelete_messages("adkey_2")
        assert "adkey_2" not in await async_chat_store.aget_keys()


class TestAsyncDeleteMessage:
    """Tests for adelete_message method."""

    @pytest.mark.asyncio
    async def test_adelete_message_middle(self, async_chat_store):
        """adelete_message removes correct message by index."""
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="user", content="c"),
        ]
        await async_chat_store.aset_messages("admkey_1", msgs)
        removed = await async_chat_store.adelete_message("admkey_1", 1)
        assert removed.content == "b"
        remaining = await async_chat_store.aget_messages("admkey_1")
        assert [m.content for m in remaining] == ["a", "c"]

    @pytest.mark.asyncio
    async def test_adelete_message_out_of_bounds(self, async_chat_store):
        """adelete_message with out-of-bounds idx returns None."""
        await async_chat_store.aset_messages(
            "admkey_2", [ChatMessage(role="user", content="only")]
        )
        result = await async_chat_store.adelete_message("admkey_2", 5)
        assert result is None

    @pytest.mark.asyncio
    async def test_adelete_message_nonexistent_key(self, async_chat_store):
        """adelete_message on nonexistent key returns None."""
        result = await async_chat_store.adelete_message("ghost_key", 0)
        assert result is None
```

- [ ] **Step 2: Run tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore_async.py::TestAsyncDeleteMessages tests/test_vastbase_chatstore_async.py::TestAsyncDeleteMessage -v 2>&1`
Expected: 6 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chatstore_async.py
git commit -m "test: add adelete_messages and adelete_message async tests"
```

---

### Task 19: Test adelete_last_message + aget_keys (RED → GREEN)

**Files:**
- Modify: `tests/test_vastbase_chatstore_async.py` — add `TestAsyncDeleteLastMessage`, `TestAsyncGetKeys` classes

- [ ] **Step 1: Add remaining async tests**

```python
class TestAsyncDeleteLastMessage:
    """Tests for adelete_last_message method."""

    @pytest.mark.asyncio
    async def test_adelete_last_message(self, async_chat_store):
        """adelete_last_message removes and returns last message."""
        msgs = [
            ChatMessage(role="user", content="a"),
            ChatMessage(role="user", content="b"),
            ChatMessage(role="user", content="c"),
        ]
        await async_chat_store.aset_messages("adlkey_1", msgs)
        removed = await async_chat_store.adelete_last_message("adlkey_1")
        assert removed.content == "c"
        remaining = await async_chat_store.aget_messages("adlkey_1")
        assert [m.content for m in remaining] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_adelete_last_message_single(self, async_chat_store):
        """adelete_last_message with single message leaves empty."""
        await async_chat_store.aset_messages(
            "adlkey_2", [ChatMessage(role="user", content="only")]
        )
        removed = await async_chat_store.adelete_last_message("adlkey_2")
        assert removed.content == "only"
        remaining = await async_chat_store.aget_messages("adlkey_2")
        assert remaining == []

    @pytest.mark.asyncio
    async def test_adelete_last_message_nonexistent(self, async_chat_store):
        """adelete_last_message on nonexistent key returns None."""
        result = await async_chat_store.adelete_last_message("no_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_adelete_last_message_empty(self, async_chat_store):
        """adelete_last_message on empty array returns None."""
        await async_chat_store.aset_messages("adlkey_3", [])
        result = await async_chat_store.adelete_last_message("adlkey_3")
        assert result is None


class TestAsyncGetKeys:
    """Tests for aget_keys method."""

    @pytest.mark.asyncio
    async def test_aget_keys(self, async_chat_store):
        """aget_keys returns all keys."""
        await async_chat_store.aset_messages(
            "agk_1", [ChatMessage(role="user", content="a")]
        )
        await async_chat_store.aset_messages(
            "agk_2", [ChatMessage(role="user", content="b")]
        )
        keys = await async_chat_store.aget_keys()
        assert "agk_1" in keys
        assert "agk_2" in keys

    @pytest.mark.asyncio
    async def test_aget_keys_empty(self, async_chat_store):
        """aget_keys returns empty list when no keys."""
        keys = await async_chat_store.aget_keys()
        assert keys == []
```

- [ ] **Step 2: Run ALL async tests**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/test_vastbase_chatstore_async.py -v 2>&1`
Expected: ALL ~20 async tests passed

- [ ] **Step 3: Run ALL tests (sync + async)**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/ -v 2>&1`
Expected: ALL ~42 tests passed (22 sync + 20 async)

- [ ] **Step 4: Commit**

```bash
git add tests/test_vastbase_chatstore_async.py
git commit -m "test: add adelete_last_message and aget_keys async tests — async tests complete"
```

---

### Task 20: Final Verification + Cleanup

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` — final review for edge cases
- Verify: All tests pass, imports clean

- [ ] **Step 1: Run complete test suite**

Run: `cd llama-index-storage-chat-store-vastbase && python -m pytest tests/ -v --tb=short 2>&1`
Expected: ALL tests pass, 0 failures, 0 errors.

- [ ] **Step 2: Verify package exports**

Run:
```python
python -c "
from llama_index.storage.chat_store.vastbase import VastbaseChatStore
from llama_index.core.storage.chat_store.base import BaseChatStore
assert issubclass(VastbaseChatStore, BaseChatStore), 'Not a BaseChatStore subclass'
# Count methods
sync_methods = ['set_messages', 'get_messages', 'add_message', 'delete_messages', 'delete_message', 'delete_last_message', 'get_keys']
async_methods = ['aset_messages', 'aget_messages', 'async_add_message', 'adelete_messages', 'adelete_message', 'adelete_last_message', 'aget_keys']
for m in sync_methods + async_methods:
    assert hasattr(VastbaseChatStore, m), f'Missing method: {m}'
    # Verify it's not the base class default (for async methods)
    store_method = getattr(VastbaseChatStore, m)
    base_method = getattr(BaseChatStore, m, None)
    if base_method:
        assert store_method is not base_method, f'{m} is inherited from BaseChatStore, must be overridden'
print('All 14 methods explicitly overridden: OK')
"
```
Expected: `All 14 methods explicitly overridden: OK`

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: final verification — all 14 methods implemented, 42+ tests passing"
```

---

## Self-Review Checklist

Before handing off:

1. **Spec coverage:** Each spec section maps to tasks:
   - §2 Package Structure → Tasks 1-2
   - §3 pyvastbase API Mapping → Task 3 (schema), Tasks 4-5 (CRUD)
   - §4 Class Design → Task 3 (full skeleton)
   - §5 Collection Init Flow → Task 3 (_ensure_initialized)
   - §6 Core Methods → Tasks 4-5, 10-11 (7 sync methods + tests)
   - §7 Filter Translation → Task 3 (_escape_key)
   - §8 Async API → Tasks 14-15 (7 native async methods + tests)
   - §9 Error Handling → Test cases cover all edge cases (None, [], out-of-bounds)
   - §10 Testing Strategy → Tasks 6-13, 16-19 (42 test cases)
   - §11 Differences from Reference → Covered by architecture
   - §12 Future Work → Not implemented (out of scope)

2. **Placeholder scan:** No TBD, TODO, "implement later" found.

3. **Type consistency:** All method signatures match `BaseChatStore` abstract definitions.
   - Sync: `set_messages(key: str, messages: List[ChatMessage]) -> None`, etc.
   - Async: `aset_messages(key: str, messages: List[ChatMessage]) -> None`, etc.
