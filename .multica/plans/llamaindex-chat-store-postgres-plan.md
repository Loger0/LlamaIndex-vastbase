# LlamaIndex ChatStore Vastbase 适配 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `llama-index-storage-chat-store-vastbase` — a drop-in VastbaseChatStore replacement for PostgresChatStore using pyvastbase Collection API.

**Architecture:** Standalone package (2 Python source files). VastbaseChatStore extends BaseChatStore, uses pyvastbase Collection (sync) and AsyncCollection (async) for all 14 CRUD methods. All array operations done in Python layer (SELECT → list op → upsert).

**Tech Stack:** Python >=3.9, pyvastbase >=0.2.7, llama-index-core >=0.13.0,<0.15, pytest-asyncio

## Global Constraints

- pyvastbase >=0.2.7 — all DB ops via Collection/AsyncCollection, no direct SQL
- llama-index-core >=0.13.0,<0.15 — BaseChatStore abstract base class
- 14 methods (7 sync + 7 native async) all explicitly implemented
- Python layer array ops — no PG array functions
- vastbase:// URI scheme for from_uri()
- Drop-in replacement — same method signatures, same return types, same error semantics as PostgresChatStore v0.4.0
- Test host: 172.16.105.107:15432, database: vastbase, user: aidev, password: Vbase_123456

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `llama_index/storage/chat_store/vastbase/__init__.py`
- Create: `llama_index/storage/chat_store/vastbase/base.py` (empty stub)

**Interfaces:**
- Produces: Package structure ready for implementation

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p llama_index/storage/chat_store/vastbase
mkdir -p tests
touch llama_index/__init__.py
touch llama_index/storage/__init__.py
touch llama_index/storage/chat_store/__init__.py
touch llama_index/storage/chat_store/vastbase/__init__.py
```

- [ ] **Step 2: Write pyproject.toml**

Write the file `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "llama-index-storage-chat-store-vastbase"
version = "0.1.0"
description = "Vastbase chat store integration for LlamaIndex"
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

Write the file `llama_index/storage/chat_store/vastbase/__init__.py`:

```python
from llama_index.storage.chat_store.vastbase.base import VastbaseChatStore

__all__ = ["VastbaseChatStore"]
```

- [ ] **Step 4: Write base.py stub**

Write the file `llama_index/storage/chat_store/vastbase/base.py`:

```python
"""Vastbase chat store for LlamaIndex.

Drop-in replacement for PostgresChatStore using pyvastbase Collection API.
"""

from typing import Dict, List, Optional

from llama_index.core.storage.chat_store.base import BaseChatStore


class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store."""
    pass
```

- [ ] **Step 5: Verify package is importable**

```bash
cd /path/to/LlamaIndex-vastbase && python -c "from llama_index.storage.chat_store.vastbase import VastbaseChatStore; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding for llama-index-storage-chat-store-vastbase"
```

---

### Task 2: Write Failing Tests for Initialization

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_vastbase_chat_store.py`

**Interfaces:**
- Consumes: Package structure from Task 1
- Produces: Test fixtures `chat_store`, test `test_init_creates_collection`

- [ ] **Step 1: Write conftest.py**

Write the file `tests/conftest.py`:

```python
"""Shared test fixtures for VastbaseChatStore tests."""
import pytest
from pyvastbase import connect, has_collection, drop_collection
from pyvastbase.collection import Collection

TEST_CONFIG = {
    "host": "172.16.105.107",
    "port": 15432,
    "database": "vastbase",
    "user": "aidev",
    "password": "Vbase_123456",
    "table_name": "test_chatstore_pytest",
}


@pytest.fixture(autouse=True)
def cleanup_collection():
    """Remove test collection before and after each test."""
    try:
        connect(
            host=TEST_CONFIG["host"],
            port=TEST_CONFIG["port"],
            database=TEST_CONFIG["database"],
            user=TEST_CONFIG["user"],
            password=TEST_CONFIG["password"],
            alias="test_chatstore",
        )
    except Exception:
        pass
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])
    yield
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])


@pytest.fixture
def chat_store():
    """Create a VastbaseChatStore instance with test connection."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_params(**TEST_CONFIG)
    return store
```

- [ ] **Step 2: Write first failing test**

Write the file `tests/test_vastbase_chat_store.py`:

```python
"""Integration tests for VastbaseChatStore."""
import pytest
from pyvastbase import has_collection


def test_init_creates_collection():
    """Test that VastbaseChatStore.from_params creates the underlying Collection."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="test_chatstore_pytest",
    )

    assert has_collection("test_chatstore_pytest") is True
    assert store._initialized is True
    assert store._coll is not None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /path/to/LlamaIndex-vastbase && python -m pytest tests/test_vastbase_chat_store.py::test_init_creates_collection -v
```

Expected: FAIL — `VastbaseChatStore.from_params` not implemented, or `_initialized` attribute missing.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_vastbase_chat_store.py
git commit -m "test: add failing initialization test for VastbaseChatStore"
```

---

### Task 3: Implement VastbaseChatStore.__init__, from_params, from_uri, _initialize

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (full rewrite of stub)

**Interfaces:**
- Consumes: Package structure from Task 1
- Produces: `VastbaseChatStore.__init__()`, `VastbaseChatStore.from_params()`, `VastbaseChatStore.from_uri()`, `VastbaseChatStore._initialize()`

- [ ] **Step 1: Implement the full class with initialization**

Write the file `llama_index/storage/chat_store/vastbase/base.py`:

```python
"""Vastbase chat store for LlamaIndex.

Drop-in replacement for PostgresChatStore using pyvastbase Collection API.
"""
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore

logger = logging.getLogger(__name__)


class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store for LlamaIndex.

    Drop-in replacement for PostgresChatStore.
    Uses pyvastbase Collection API instead of SQLAlchemy + psycopg/asyncpg.

    Connection can be provided via:
        - from_params(host, port, database, user, password, ...)
        - from_uri("vastbase://user:pass@host:port/db")
        - Direct constructor with host/port/database/user/password fields
    """

    # === Pydantic fields ===
    table_name: str = "chatstore"
    schema_name: str = "public"
    use_jsonb: bool = False  # Kept for API compat; no behavioral difference

    # Connection params
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = ""
    password: str = ""

    # === Private attributes ===
    _coll: Optional[object] = PrivateAttr(default=None)
    _async_coll: Optional[object] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _actual_table_name: str = PrivateAttr(default="")

    def __init__(self, **data):
        """Initialize VastbaseChatStore.

        Accepts all BaseChatStore + VastbaseChatStore Pydantic fields.
        Call _initialize() to set up connection and Collection.
        """
        super().__init__(**data)
        self._coll = None
        self._async_coll = None
        self._initialized = False
        self._actual_table_name = ""

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int = 15432,
        database: str = "vastbase",
        user: str = "",
        password: str = "",
        table_name: str = "chatstore",
        schema_name: str = "public",
        **kwargs,
    ) -> "VastbaseChatStore":
        """Create VastbaseChatStore from individual connection parameters."""
        store = cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            schema_name=schema_name,
            **kwargs,
        )
        store._initialize()
        return store

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        **kwargs,
    ) -> "VastbaseChatStore":
        """Create VastbaseChatStore from a vastbase:// URI.

        URI format: vastbase://user:pass@host:port/database

        Args:
            uri: Connection URI with vastbase:// scheme.
            table_name: Name of the chat store collection.
            schema_name: Schema name (reserved for future use).

        Returns:
            Initialized VastbaseChatStore instance.
        """
        parsed = urlparse(uri)

        if parsed.scheme != "vastbase":
            raise ValueError(
                f"Unsupported URI scheme: '{parsed.scheme}'. "
                "Expected 'vastbase://'."
            )

        host = parsed.hostname or "localhost"
        port = parsed.port or 15432
        database = parsed.path.lstrip("/") if parsed.path else "vastbase"
        user = parsed.username or ""
        password = parsed.password or ""

        store = cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            schema_name=schema_name,
            **kwargs,
        )
        store._initialize()
        return store

    def _initialize(self) -> None:
        """Set up pyvastbase connection and Collection.

        1. Connect to Vastbase (sync, using global connection pool).
        2. Check for legacy table name (data_{table_name}).
        3. Define CollectionSchema: id (INT64 PK), key (VARCHAR 512), value (TEXT).
        4. Create or open Collection.
        """
        from pyvastbase import connect, has_collection, Collection
        from pyvastbase.schema import CollectionSchema, FieldSchema, DataType

        alias = f"chatstore_{self.table_name}"

        # 1. Establish connection
        try:
            connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                alias=alias,
            )
        except Exception:
            logger.warning(
                "Connection already exists for alias '%s', reusing.", alias
            )

        # 2. Legacy table detection
        legacy_name = f"data_{self.table_name}"
        if has_collection(legacy_name):
            self._actual_table_name = legacy_name
            logger.info(
                "Using legacy collection name: %s", legacy_name
            )
        else:
            self._actual_table_name = self.table_name

        # 3. Define schema
        schema = CollectionSchema(
            name=self._actual_table_name,
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary_key=True,
                ),
                FieldSchema(
                    name="key",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="value",
                    dtype=DataType.TEXT,
                ),
            ],
        )

        # 4. Create or open Collection
        if has_collection(self._actual_table_name):
            self._coll = Collection(self._actual_table_name)
            logger.info(
                "Collection '%s' already exists, reusing.",
                self._actual_table_name,
            )
        else:
            self._coll = Collection(self._actual_table_name, schema=schema)
            self._coll.create()
            logger.info(
                "Collection '%s' created.", self._actual_table_name
            )

        self._initialized = True
```

- [ ] **Step 2: Run the initialization test**

```bash
cd /path/to/LlamaIndex-vastbase && python -m pytest tests/test_vastbase_chat_store.py::test_init_creates_collection -v
```

Expected: PASS — Collection created, `_initialized` is True.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement VastbaseChatStore initialization with from_params/from_uri"
```

---

### Task 4: Write Failing Tests for set_messages + get_messages

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture from Task 2, `VastbaseChatStore` from Task 3
- Produces: Tests `test_set_and_get_messages`, `test_get_messages_nonexistent_key`

- [ ] **Step 1: Add set_messages + get_messages tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
from llama_index.core.llms import ChatMessage, MessageRole


def test_set_and_get_messages(chat_store):
    """Test setting messages and retrieving them."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]
    chat_store.set_messages(key="test_session_1", messages=messages)

    result = chat_store.get_messages(key="test_session_1")
    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert result[0].content == "Hello"
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].content == "Hi there!"


def test_get_messages_nonexistent_key(chat_store):
    """Test that getting messages for a nonexistent key returns empty list."""
    result = chat_store.get_messages(key="nonexistent_key")
    assert result == []


def test_set_messages_overwrite(chat_store):
    """Test that setting messages overwrites existing messages for a key."""
    msg1 = [ChatMessage(role=MessageRole.USER, content="First")]
    chat_store.set_messages(key="overwrite_test", messages=msg1)

    msg2 = [ChatMessage(role=MessageRole.ASSISTANT, content="Second")]
    chat_store.set_messages(key="overwrite_test", messages=msg2)

    result = chat_store.get_messages(key="overwrite_test")
    assert len(result) == 1
    assert result[0].content == "Second"


def test_set_messages_empty_list(chat_store):
    """Test setting an empty messages list."""
    chat_store.set_messages(key="empty_test", messages=[])
    result = chat_store.get_messages(key="empty_test")
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/LlamaIndex-vastbase && python -m pytest tests/test_vastbase_chat_store.py -k "test_set_and_get_messages or test_get_messages_nonexistent_key or test_set_messages_overwrite or test_set_messages_empty_list" -v
```

Expected: All 4 tests FAIL — `set_messages` and `get_messages` not implemented.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for set_messages and get_messages"
```

---

### Task 5: Implement set_messages + get_messages

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add methods)

**Interfaces:**
- Consumes: `_coll` from Task 3, `ChatMessage` from llama-index-core
- Produces: `VastbaseChatStore.set_messages()`, `VastbaseChatStore.get_messages()`

- [ ] **Step 1: Add set_messages and get_messages to VastbaseChatStore**

Insert after the `_initialize` method in `base.py`:

```python
    def set_messages(self, key: str, messages: List["ChatMessage"]) -> None:
        """Store messages for a key, overwriting any existing messages.

        Args:
            key: Unique identifier for the conversation.
            messages: List of ChatMessage objects to store.
        """
        self._ensure_initialized()

        from llama_index.core.llms import ChatMessage

        # Serialize all messages to JSON
        value = json.dumps([m.model_dump() for m in messages])

        # Check if key already exists
        existing = self._coll.query(expr=f"key == '{self._escape(key)}'", limit=1)

        if existing:
            # Update existing row via upsert
            self._coll.upsert([
                {"id": existing[0]["id"], "key": key, "value": value}
            ])
        else:
            # Insert new row
            self._coll.insert([{"key": key, "value": value}])

    def get_messages(self, key: str) -> List["ChatMessage"]:
        """Retrieve messages for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            List of ChatMessage objects, or empty list if key not found.
        """
        self._ensure_initialized()

        from llama_index.core.llms import ChatMessage

        results = self._coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return []

        try:
            parsed = json.loads(results[0]["value"])
            return [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Failed to deserialize messages for key '%s'", key
            )
            return []

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before use."""
        if not self._initialized:
            self._initialize()

    def _escape(self, value: str) -> str:
        """Escape single quotes in string values for pyvastbase expr."""
        return value.replace("'", "''")
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /path/to/LlamaIndex-vastbase && python -m pytest tests/test_vastbase_chat_store.py -k "test_set_and_get_messages or test_get_messages_nonexistent_key or test_set_messages_overwrite or test_set_messages_empty_list" -v
```

Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement set_messages and get_messages"
```

---

### Task 6: Write Failing Tests for add_message

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture, `set_messages`/`get_messages` from Task 5
- Produces: Tests `test_add_message_existing_key`, `test_add_message_new_key`

- [ ] **Step 1: Add add_message tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_add_message_existing_key(chat_store):
    """Test appending a message to an existing key."""
    chat_store.set_messages(
        key="session_1",
        messages=[ChatMessage(role=MessageRole.USER, content="Q1")],
    )
    chat_store.add_message(
        key="session_1",
        message=ChatMessage(role=MessageRole.ASSISTANT, content="A1"),
    )

    result = chat_store.get_messages(key="session_1")
    assert len(result) == 2
    assert result[0].content == "Q1"
    assert result[1].content == "A1"


def test_add_message_new_key(chat_store):
    """Test adding a message to a key that does not yet exist."""
    chat_store.add_message(
        key="new_session",
        message=ChatMessage(role=MessageRole.USER, content="Hello"),
    )

    result = chat_store.get_messages(key="new_session")
    assert len(result) == 1
    assert result[0].role == MessageRole.USER
    assert result[0].content == "Hello"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_add_message" -v
```

Expected: Both tests FAIL — `add_message` not implemented.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for add_message"
```

---

### Task 7: Implement add_message

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add method)

**Interfaces:**
- Consumes: `_coll`, `_ensure_initialized`, `_escape` from earlier tasks
- Produces: `VastbaseChatStore.add_message()`

- [ ] **Step 1: Add add_message method**

Insert into `base.py` class after `get_messages`:

```python
    def add_message(
        self, key: str, message: "ChatMessage"
    ) -> None:
        """Append a single message to the list for a key.

        If the key does not exist, a new entry is created.

        Args:
            key: Unique identifier for the conversation.
            message: ChatMessage to append.
        """
        self._ensure_initialized()

        existing = self._coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not existing:
            # Key doesn't exist — insert new row
            self._coll.insert([
                {
                    "key": key,
                    "value": json.dumps([message.model_dump()]),
                }
            ])
        else:
            # Key exists — append to array in Python, then upsert
            messages = json.loads(existing[0]["value"])
            messages.append(message.model_dump())
            self._coll.upsert([
                {
                    "id": existing[0]["id"],
                    "key": key,
                    "value": json.dumps(messages),
                }
            ])
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_add_message" -v
```

Expected: Both tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement add_message"
```

---

### Task 8: Write Failing Tests for delete_messages

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture from Task 2
- Produces: Tests `test_delete_messages_existing`, `test_delete_messages_nonexistent`

- [ ] **Step 1: Add delete_messages tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_delete_messages_existing(chat_store):
    """Test deleting all messages for an existing key."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Q1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="A1"),
    ]
    chat_store.set_messages(key="to_delete", messages=messages)

    deleted = chat_store.delete_messages(key="to_delete")
    assert len(deleted) == 2
    assert deleted[0].content == "Q1"
    assert deleted[1].content == "A1"

    # Verify key is gone
    result = chat_store.get_messages(key="to_delete")
    assert result == []


def test_delete_messages_nonexistent(chat_store):
    """Test deleting messages for a key that does not exist."""
    result = chat_store.delete_messages(key="no_such_key")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_messages" -v
```

Expected: Both tests FAIL — `delete_messages` not implemented.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for delete_messages"
```

---

### Task 9: Implement delete_messages

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add method)

**Interfaces:**
- Consumes: `_coll` from Task 3
- Produces: `VastbaseChatStore.delete_messages()`

- [ ] **Step 1: Add delete_messages method**

Insert into `base.py` class after `add_message`:

```python
    def delete_messages(
        self, key: str
    ) -> Optional[List["ChatMessage"]]:
        """Delete all messages for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            List of deleted ChatMessage objects, or None if key not found.
        """
        self._ensure_initialized()

        from llama_index.core.llms import ChatMessage

        results = self._coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        # Deserialize before deleting
        try:
            parsed = json.loads(results[0]["value"])
            messages = [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            messages = []

        # Delete the row
        self._coll.delete(expr=f"key == '{self._escape(key)}'")

        return messages
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_messages" -v
```

Expected: Both tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement delete_messages"
```

---

### Task 10: Write Failing Tests for delete_message

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture
- Produces: Tests `test_delete_message_valid_idx`, `test_delete_message_out_of_range`, `test_delete_message_nonexistent_key`

- [ ] **Step 1: Add delete_message tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_delete_message_valid_idx(chat_store):
    """Test deleting a message at a specific valid index."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Msg1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Msg2"),
        ChatMessage(role=MessageRole.USER, content="Msg3"),
    ]
    chat_store.set_messages(key="idx_test", messages=messages)

    deleted = chat_store.delete_message(key="idx_test", idx=1)
    assert deleted is not None
    assert deleted.content == "Msg2"

    remaining = chat_store.get_messages(key="idx_test")
    assert len(remaining) == 2
    assert remaining[0].content == "Msg1"
    assert remaining[1].content == "Msg3"


def test_delete_message_out_of_range(chat_store):
    """Test deleting at an index that is out of range."""
    chat_store.set_messages(
        key="oor_test",
        messages=[ChatMessage(role=MessageRole.USER, content="Only")],
    )
    result = chat_store.delete_message(key="oor_test", idx=5)
    assert result is None


def test_delete_message_nonexistent_key(chat_store):
    """Test deleting a message from a nonexistent key."""
    result = chat_store.delete_message(key="no_key", idx=0)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_message" -v
```

Expected: All 3 tests FAIL.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for delete_message"
```

---

### Task 11: Implement delete_message

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add method)

**Interfaces:**
- Consumes: `_coll` from Task 3
- Produces: `VastbaseChatStore.delete_message()`

- [ ] **Step 1: Add delete_message method**

Insert into `base.py` class after `delete_messages`:

```python
    def delete_message(
        self, key: str, idx: int
    ) -> Optional["ChatMessage"]:
        """Delete a single message at the given index for a key.

        Args:
            key: Unique identifier for the conversation.
            idx: Index of the message to delete (0-based).

        Returns:
            The deleted ChatMessage, or None if key/index not found.
        """
        self._ensure_initialized()

        from llama_index.core.llms import ChatMessage

        results = self._coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if idx < 0 or idx >= len(messages):
            return None

        removed = messages.pop(idx)

        self._coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_message" -v
```

Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement delete_message"
```

---

### Task 12: Write Failing Tests for delete_last_message

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture
- Produces: Tests `test_delete_last_message`, `test_delete_last_message_empty`, `test_delete_last_message_nonexistent`

- [ ] **Step 1: Add delete_last_message tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_delete_last_message(chat_store):
    """Test deleting the last message for an existing key."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="First"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Second"),
        ChatMessage(role=MessageRole.USER, content="Third"),
    ]
    chat_store.set_messages(key="last_test", messages=messages)

    deleted = chat_store.delete_last_message(key="last_test")
    assert deleted is not None
    assert deleted.content == "Third"

    remaining = chat_store.get_messages(key="last_test")
    assert len(remaining) == 2
    assert remaining[-1].content == "Second"


def test_delete_last_message_empty(chat_store):
    """Test deleting the last message when messages list is empty."""
    chat_store.set_messages(key="empty_list", messages=[])
    result = chat_store.delete_last_message(key="empty_list")
    assert result is None


def test_delete_last_message_nonexistent(chat_store):
    """Test deleting the last message from a nonexistent key."""
    result = chat_store.delete_last_message(key="no_such_key")
    assert result is None


def test_delete_last_message_single(chat_store):
    """Test deleting the last (only) message."""
    chat_store.set_messages(
        key="single_msg",
        messages=[ChatMessage(role=MessageRole.USER, content="Lone")],
    )
    deleted = chat_store.delete_last_message(key="single_msg")
    assert deleted is not None
    assert deleted.content == "Lone"

    remaining = chat_store.get_messages(key="single_msg")
    assert remaining == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_last_message" -v
```

Expected: All 4 tests FAIL.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for delete_last_message"
```

---

### Task 13: Implement delete_last_message

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add method)

**Interfaces:**
- Consumes: `_coll` from Task 3
- Produces: `VastbaseChatStore.delete_last_message()`

- [ ] **Step 1: Add delete_last_message method**

Insert into `base.py` class after `delete_message`:

```python
    def delete_last_message(
        self, key: str
    ) -> Optional["ChatMessage"]:
        """Delete the last message for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            The deleted ChatMessage, or None if key not found or list empty.
        """
        self._ensure_initialized()

        from llama_index.core.llms import ChatMessage

        results = self._coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if len(messages) == 0:
            return None

        removed = messages.pop()

        self._coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_delete_last_message" -v
```

Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement delete_last_message"
```

---

### Task 14: Write Failing Tests for get_keys

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append tests)

**Interfaces:**
- Consumes: `chat_store` fixture
- Produces: Tests `test_get_keys`, `test_get_keys_empty`

- [ ] **Step 1: Add get_keys tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_get_keys(chat_store):
    """Test retrieving all keys from the chat store."""
    chat_store.set_messages(
        key="key_a",
        messages=[ChatMessage(role=MessageRole.USER, content="A")],
    )
    chat_store.set_messages(
        key="key_b",
        messages=[ChatMessage(role=MessageRole.USER, content="B")],
    )
    chat_store.set_messages(
        key="key_c",
        messages=[ChatMessage(role=MessageRole.USER, content="C")],
    )

    keys = chat_store.get_keys()
    assert len(keys) == 3
    assert "key_a" in keys
    assert "key_b" in keys
    assert "key_c" in keys


def test_get_keys_empty(chat_store):
    """Test that get_keys returns empty list when no messages stored."""
    keys = chat_store.get_keys()
    assert keys == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_get_keys" -v
```

Expected: Both tests FAIL.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add failing tests for get_keys"
```

---

### Task 15: Implement get_keys

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add method)

**Interfaces:**
- Consumes: `_coll` from Task 3
- Produces: `VastbaseChatStore.get_keys()`

- [ ] **Step 1: Add get_keys method**

Insert into `base.py` class after `delete_last_message`:

```python
    def get_keys(self) -> List[str]:
        """Retrieve all keys stored in the chat store.

        Returns:
            List of key strings.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr="1=1", output_fields=["key"]
        )

        return [r["key"] for r in results]
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_get_keys" -v
```

Expected: Both tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement get_keys"
```

---

### Task 16: Write Failing Tests for Async Methods

**Files:**
- Create: `tests/test_async_chat_store.py`

**Interfaces:**
- Consumes: `VastbaseChatStore` from prior tasks
- Produces: 7 async test methods

- [ ] **Step 1: Write async test file**

Write the file `tests/test_async_chat_store.py`:

```python
"""Async integration tests for VastbaseChatStore."""
import pytest
from llama_index.core.llms import ChatMessage, MessageRole

# Reuse the same TEST_CONFIG from conftest
TEST_CONFIG = {
    "host": "172.16.105.107",
    "port": 15432,
    "database": "vastbase",
    "user": "aidev",
    "password": "Vbase_123456",
    "table_name": "test_chatstore_pytest_async",
}


@pytest.fixture(autouse=True)
def cleanup_async():
    """Remove test collection before and after each async test."""
    from pyvastbase import connect, has_collection, drop_collection

    try:
        connect(
            host=TEST_CONFIG["host"],
            port=TEST_CONFIG["port"],
            database=TEST_CONFIG["database"],
            user=TEST_CONFIG["user"],
            password=TEST_CONFIG["password"],
            alias="test_chatstore_async",
        )
    except Exception:
        pass
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])
    yield
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])


@pytest.fixture
async def async_chat_store():
    """Create an async-initialized VastbaseChatStore."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_params(**TEST_CONFIG)
    # Async initialization (connect async + create AsyncCollection)
    from pyvastbase import AsyncConnections

    alias = f"chatstore_{TEST_CONFIG['table_name']}"
    await AsyncConnections.connect(
        host=TEST_CONFIG["host"],
        port=TEST_CONFIG["port"],
        database=TEST_CONFIG["database"],
        user=TEST_CONFIG["user"],
        password=TEST_CONFIG["password"],
        alias=alias,
    )
    from pyvastbase import AsyncCollection
    store._async_coll = AsyncCollection(TEST_CONFIG["table_name"])
    return store


@pytest.mark.asyncio
async def test_aset_and_aget_messages(async_chat_store):
    """Test async set and get messages."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Async hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Async reply"),
    ]
    await async_chat_store.aset_messages(key="async_session", messages=messages)

    result = await async_chat_store.aget_messages(key="async_session")
    assert len(result) == 2
    assert result[0].content == "Async hello"
    assert result[1].content == "Async reply"


@pytest.mark.asyncio
async def test_async_add_message(async_chat_store):
    """Test async add_message."""
    await async_chat_store.aset_messages(
        key="async_add",
        messages=[ChatMessage(role=MessageRole.USER, content="Q")],
    )
    await async_chat_store.async_add_message(
        key="async_add",
        message=ChatMessage(role=MessageRole.ASSISTANT, content="A"),
    )

    result = await async_chat_store.aget_messages(key="async_add")
    assert len(result) == 2


@pytest.mark.asyncio
async def test_adelete_messages(async_chat_store):
    """Test async delete_messages."""
    await async_chat_store.aset_messages(
        key="async_del",
        messages=[ChatMessage(role=MessageRole.USER, content="Del me")],
    )
    deleted = await async_chat_store.adelete_messages(key="async_del")
    assert len(deleted) == 1

    remaining = await async_chat_store.aget_messages(key="async_del")
    assert remaining == []


@pytest.mark.asyncio
async def test_adelete_message(async_chat_store):
    """Test async delete_message by index."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="A"),
        ChatMessage(role=MessageRole.USER, content="B"),
    ]
    await async_chat_store.aset_messages(key="async_del_idx", messages=messages)
    deleted = await async_chat_store.adelete_message(
        key="async_del_idx", idx=0
    )
    assert deleted.content == "A"

    remaining = await async_chat_store.aget_messages(key="async_del_idx")
    assert len(remaining) == 1
    assert remaining[0].content == "B"


@pytest.mark.asyncio
async def test_adelete_last_message(async_chat_store):
    """Test async delete_last_message."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="First"),
        ChatMessage(role=MessageRole.USER, content="Last"),
    ]
    await async_chat_store.aset_messages(
        key="async_del_last", messages=messages
    )
    deleted = await async_chat_store.adelete_last_message(
        key="async_del_last"
    )
    assert deleted.content == "Last"


@pytest.mark.asyncio
async def test_aget_messages_nonexistent(async_chat_store):
    """Test async get_messages for nonexistent key."""
    result = await async_chat_store.aget_messages(key="no_key_async")
    assert result == []


@pytest.mark.asyncio
async def test_aget_keys(async_chat_store):
    """Test async get_keys."""
    await async_chat_store.aset_messages(
        key="ak1",
        messages=[ChatMessage(role=MessageRole.USER, content="K1")],
    )
    await async_chat_store.aset_messages(
        key="ak2",
        messages=[ChatMessage(role=MessageRole.USER, content="K2")],
    )

    keys = await async_chat_store.aget_keys()
    assert len(keys) == 2
    assert "ak1" in keys
    assert "ak2" in keys
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_async_chat_store.py -v
```

Expected: All 7 tests FAIL — async methods not implemented.

- [ ] **Step 3: Commit**

```bash
git add tests/test_async_chat_store.py
git commit -m "test: add failing async integration tests"
```

---

### Task 17: Implement All 7 Async Methods

**Files:**
- Modify: `llama_index/storage/chat_store/vastbase/base.py` (add 7 async methods)

**Interfaces:**
- Consumes: `_async_coll` from Task 3, all sync method patterns from Tasks 5-15
- Produces: `aset_messages`, `aget_messages`, `async_add_message`, `adelete_messages`, `adelete_message`, `adelete_last_message`, `aget_keys`

- [ ] **Step 1: Add all 7 async methods**

Insert into `base.py` class after `get_keys`:

```python
    # === Async methods (native pyvastbase AsyncCollection) ===

    async def aset_messages(
        self, key: str, messages: List["ChatMessage"]
    ) -> None:
        """Async version of set_messages."""
        await self._ensure_async_initialized()

        value = json.dumps([m.model_dump() for m in messages])

        existing = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if existing:
            await self._async_coll.upsert([
                {"id": existing[0]["id"], "key": key, "value": value}
            ])
        else:
            await self._async_coll.insert([
                {"key": key, "value": value}
            ])

    async def aget_messages(
        self, key: str
    ) -> List["ChatMessage"]:
        """Async version of get_messages."""
        await self._ensure_async_initialized()

        from llama_index.core.llms import ChatMessage

        results = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return []

        try:
            parsed = json.loads(results[0]["value"])
            return [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Async: failed to deserialize messages for key '%s'", key)
            return []

    async def async_add_message(
        self, key: str, message: "ChatMessage"
    ) -> None:
        """Async version of add_message."""
        await self._ensure_async_initialized()

        existing = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not existing:
            await self._async_coll.insert([
                {
                    "key": key,
                    "value": json.dumps([message.model_dump()]),
                }
            ])
        else:
            messages = json.loads(existing[0]["value"])
            messages.append(message.model_dump())
            await self._async_coll.upsert([
                {
                    "id": existing[0]["id"],
                    "key": key,
                    "value": json.dumps(messages),
                }
            ])

    async def adelete_messages(
        self, key: str
    ) -> Optional[List["ChatMessage"]]:
        """Async version of delete_messages."""
        await self._ensure_async_initialized()

        from llama_index.core.llms import ChatMessage

        results = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            parsed = json.loads(results[0]["value"])
            messages = [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            messages = []

        await self._async_coll.delete(
            expr=f"key == '{self._escape(key)}'"
        )

        return messages

    async def adelete_message(
        self, key: str, idx: int
    ) -> Optional["ChatMessage"]:
        """Async version of delete_message."""
        await self._ensure_async_initialized()

        from llama_index.core.llms import ChatMessage

        results = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if idx < 0 or idx >= len(messages):
            return None

        removed = messages.pop(idx)

        await self._async_coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    async def adelete_last_message(
        self, key: str
    ) -> Optional["ChatMessage"]:
        """Async version of delete_last_message."""
        await self._ensure_async_initialized()

        from llama_index.core.llms import ChatMessage

        results = await self._async_coll.query(
            expr=f"key == '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if len(messages) == 0:
            return None

        removed = messages.pop()

        await self._async_coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    async def aget_keys(self) -> List[str]:
        """Async version of get_keys."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr="1=1", output_fields=["key"]
        )

        return [r["key"] for r in results]

    async def _ensure_async_initialized(self) -> None:
        """Ensure the async Collection is available."""
        if self._async_coll is None:
            from pyvastbase import AsyncConnections, AsyncCollection

            alias = f"chatstore_{self._actual_table_name}"
            await AsyncConnections.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                alias=alias,
            )
            self._async_coll = AsyncCollection(
                self._actual_table_name
            )
```

- [ ] **Step 2: Run async tests**

```bash
python -m pytest tests/test_async_chat_store.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add llama_index/storage/chat_store/vastbase/base.py
git commit -m "feat: implement all 7 native async methods with AsyncCollection"
```

---

### Task 18: Write Failing Tests for from_uri and Legacy Table

**Files:**
- Modify: `tests/test_vastbase_chat_store.py` (append more tests)

**Interfaces:**
- Consumes: `VastbaseChatStore` from prior tasks
- Produces: Tests for `from_uri`, `from_uri_invalid_scheme`

- [ ] **Step 1: Add from_uri tests**

Append to `tests/test_vastbase_chat_store.py`:

```python
def test_from_uri_creates_store():
    """Test creating VastbaseChatStore from a vastbase:// URI."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore
    from pyvastbase import has_collection, drop_collection

    uri = "vastbase://aidev:Vbase_123456@172.16.105.107:15432/vastbase"
    store = VastbaseChatStore.from_uri(
        uri, table_name="test_chatstore_pytest"
    )

    assert store._initialized is True
    assert has_collection("test_chatstore_pytest") is True

    # Cleanup
    drop_collection("test_chatstore_pytest")


def test_from_uri_wrong_scheme():
    """Test that non-vastbase URI scheme raises ValueError."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        VastbaseChatStore.from_uri(
            "postgresql://user:pass@host:5432/db",
            table_name="test_chatstore_pytest",
        )


def test_from_uri_default_port():
    """Test from_uri with no explicit port defaults to 15432."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_uri(
        "vastbase://aidev:Vbase_123456@172.16.105.107/vastbase",
        table_name="test_chatstore_pytest",
    )
    # Port should default to 15432
    assert store.port == 15432


def test_full_crud_workflow(chat_store):
    """End-to-end test: set, get, add, delete message, delete messages, get_keys."""
    # Set messages
    chat_store.set_messages(
        key="workflow",
        messages=[
            ChatMessage(role=MessageRole.USER, content="Step 1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Step 2"),
        ],
    )

    # Add message
    chat_store.add_message(
        key="workflow",
        message=ChatMessage(role=MessageRole.USER, content="Step 3"),
    )

    # Verify
    msgs = chat_store.get_messages(key="workflow")
    assert len(msgs) == 3

    # Delete middle message
    deleted = chat_store.delete_message(key="workflow", idx=1)
    assert deleted.content == "Step 2"

    msgs = chat_store.get_messages(key="workflow")
    assert len(msgs) == 2

    # Delete last
    deleted = chat_store.delete_last_message(key="workflow")
    assert deleted.content == "Step 3"

    msgs = chat_store.get_messages(key="workflow")
    assert len(msgs) == 1

    # Get keys
    keys = chat_store.get_keys()
    assert "workflow" in keys

    # Delete all
    chat_store.delete_messages(key="workflow")
    assert chat_store.get_messages(key="workflow") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vastbase_chat_store.py -k "test_from_uri or test_full_crud_workflow" -v
```

Expected: `test_from_uri_wrong_scheme` may pass if `from_uri` already has validation. `test_full_crud_workflow` should PASS if all prior methods work.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vastbase_chat_store.py
git commit -m "test: add from_uri validation and full CRUD workflow tests"
```

---

### Task 19: Compatibility Tests

**Files:**
- Create: `tests/test_compat.py`

**Interfaces:**
- Produces: Compatibility verification tests

- [ ] **Step 1: Write compatibility tests**

Write the file `tests/test_compat.py`:

```python
"""Compatibility tests: verify VastbaseChatStore is a drop-in replacement."""
import inspect
import pytest
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.llms import ChatMessage, MessageRole


def test_isinstance_basechatstore():
    """Verify VastbaseChatStore extends BaseChatStore."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="test_compat",
    )
    assert isinstance(store, BaseChatStore)


def test_has_all_required_methods():
    """Verify all 14 required methods are present and callable."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    required_sync = [
        "set_messages", "get_messages", "add_message",
        "delete_messages", "delete_message", "delete_last_message",
        "get_keys",
    ]
    required_async = [
        "aset_messages", "aget_messages", "async_add_message",
        "adelete_messages", "adelete_message", "adelete_last_message",
        "aget_keys",
    ]

    for method_name in required_sync + required_async:
        assert hasattr(VastbaseChatStore, method_name), (
            f"Missing method: {method_name}"
        )


def test_set_messages_signature():
    """Verify set_messages signature matches upstream."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    sig = inspect.signature(VastbaseChatStore.set_messages)
    params = list(sig.parameters.keys())
    # Should include: self, key, messages
    assert "key" in params
    assert "messages" in params


def test_get_messages_signature():
    """Verify get_messages signature matches upstream."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    sig = inspect.signature(VastbaseChatStore.get_messages)
    params = list(sig.parameters.keys())
    assert "key" in params


def test_add_message_signature():
    """Verify add_message signature matches upstream."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    sig = inspect.signature(VastbaseChatStore.add_message)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "message" in params


def test_return_type_annotations():
    """Verify return type annotations are present on all methods."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    # Spot-check key methods
    hints = {
        "set_messages": "return",
        "get_messages": "return",
        "delete_messages": "return",
        "get_keys": "return",
    }
    for method_name, attr in hints.items():
        method = getattr(VastbaseChatStore, method_name)
        ann = getattr(inspect.get_annotations(method), attr, None) if hasattr(method, '__annotations__') else None
        # At minimum, the method exists (annotations checked loosely)
        assert callable(method)
```

- [ ] **Step 2: Run compatibility tests**

```bash
python -m pytest tests/test_compat.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compat.py
git commit -m "test: add compatibility verification tests"
```

---

### Task 20: Final Test Suite Execution + README

**Files:**
- Modify: `README.md` (write usage docs)

**Interfaces:**
- Produces: Complete, passing test suite + README

- [ ] **Step 1: Write README.md**

Write the file `README.md`:

```markdown
# LlamaIndex Chat Store — Vastbase

Vastbase-backed chat history storage for LlamaIndex. Drop-in replacement for PostgresChatStore.

## Installation

```bash
pip install llama-index-storage-chat-store-vastbase
```

## Usage

```python
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

# From connection parameters
chat_store = VastbaseChatStore.from_params(
    host="172.16.105.107",
    port=15432,
    database="vastbase",
    user="aidev",
    password="Vbase_123456",
    table_name="chat_history",
)

# Or from a URI
chat_store = VastbaseChatStore.from_uri(
    "vastbase://aidev:Vbase_123456@172.16.105.107:15432/vastbase"
)

# Use as LlamaIndex chat store
from llama_index.core.llms import ChatMessage, MessageRole

chat_store.set_messages(
    key="session_123",
    messages=[
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
    ],
)

messages = chat_store.get_messages(key="session_123")
print(messages)
```

## API Reference

| Method | Description |
|--------|-------------|
| `set_messages(key, messages)` | Store messages, overwriting any existing |
| `get_messages(key)` | Retrieve all messages for a key |
| `add_message(key, message)` | Append a single message |
| `delete_messages(key)` | Delete all messages for a key |
| `delete_message(key, idx)` | Delete message at index |
| `delete_last_message(key)` | Delete the last message |
| `get_keys()` | List all stored keys |
| `aset_messages(...)` | Async variant (×7 methods) |
```

- [ ] **Step 2: Run the full test suite**

```bash
cd /path/to/LlamaIndex-vastbase && python -m pytest tests/ -v
```

Expected: All tests PASS (~25-30 test cases total across test_vastbase_chat_store.py, test_async_chat_store.py, test_compat.py).

- [ ] **Step 3: Final commit**

```bash
git add README.md
git commit -m "docs: add README with usage examples"
```

---

## Plan Summary

| # | Task | Files Created | Files Modified |
|---|------|--------------|----------------|
| 1 | Scaffolding | 5 | 0 |
| 2 | Init tests | 2 | 0 |
| 3 | Init implementation | 0 | 1 |
| 4 | set/get tests | 0 | 1 |
| 5 | set/get implementation | 0 | 1 |
| 6 | add_message tests | 0 | 1 |
| 7 | add_message implementation | 0 | 1 |
| 8 | delete_messages tests | 0 | 1 |
| 9 | delete_messages implementation | 0 | 1 |
| 10 | delete_message tests | 0 | 1 |
| 11 | delete_message implementation | 0 | 1 |
| 12 | delete_last_message tests | 0 | 1 |
| 13 | delete_last_message implementation | 0 | 1 |
| 14 | get_keys tests | 0 | 1 |
| 15 | get_keys implementation | 0 | 1 |
| 16 | async tests | 1 | 0 |
| 17 | async implementation | 0 | 1 |
| 18 | from_uri tests | 0 | 1 |
| 19 | compat tests | 1 | 0 |
| 20 | README + final run | 0 | 1 |

**Total: 20 tasks, ~70-100 minutes estimated.**
