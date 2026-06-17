# LlamaIndex ChatStore Vastbase 适配 — Design Spec

**Date:** 2026-06-17
**Status:** Draft (Plan-Check Round 2)
**Integration Mode:** standalone
**pyvastbase Version:** >=0.2.7
**Target Framework:** llama-index-storage-chat-store-postgres v0.4.0
**Package:** llama-index-storage-chat-store-vastbase v0.1.0

---

## 1. Overview

### 目标

将 LlamaIndex PostgresChatStore 适配到 Vastbase，用 pyvastbase Collection API 完全替代上游的 SQLAlchemy + psycopg/asyncpg 驱动栈，实现功能全量对等的 `VastbaseChatStore`，作为 drop-in replacement。

### 非目标

- 不修改 LlamaIndex 框架源码（standalone 模式）
- 不实现超出上游 PostgresChatStore v0.4.0 功能范围的能力
- 不提供数据迁移工具（从 PG 到 Vastbase）
- 不使用直接 SQL（所有操作通过 pyvastbase Collection API）

### 核心差异

| 维度 | 上游 PostgresChatStore | VastbaseChatStore |
|------|----------------------|-------------------|
| 连接管理 | SQLAlchemy engine + sessionmaker（双引擎） | pyvastbase `connect()` + `Collection` / `AsyncCollection` |
| 数据模型 | ORM ARRAY(JSON/JSONB) | TEXT 字段存 JSON 序列化字符串 |
| upsert | `INSERT ON CONFLICT (key)` SQL | 先 query → 存在则 upsert(id)，不存在则 insert |
| 数组追加 | `array_cat(col, val)` SQL 函数 | SELECT value → Python list.append → upsert |
| 数组删除 | `value[: :idx] \|\| value[:idx+2:]` SQL 切片 | SELECT value → Python list.pop(idx) → upsert |
| 异步 | asyncpg 原生 SQL | pyvastbase `AsyncCollection` 原生异步 |
| URI | `postgresql+psycopg://` / `postgresql+asyncpg://` | `vastbase://` |

---

## 2. Package Structure

```
llama-index-storage-chat-store-vastbase/
├── pyproject.toml
├── README.md
├── tests/
│   ├── __init__.py
│   ├── conftest.py                          # Vastbase 连接 fixture
│   ├── test_vastbase_chatstore.py           # 同步方法测试（15 cases）
│   └── test_vastbase_chatstore_async.py     # 异步方法测试（15 cases）
└── llama_index/
    └── storage/
        └── chat_store/
            └── vastbase/
                ├── __init__.py              # 导出 VastbaseChatStore
                └── base.py                  # VastbaseChatStore 完整实现 (~400 行)
```

### pyproject.toml 依赖

```toml
[project]
name = "llama-index-storage-chat-store-vastbase"
version = "0.1.0"
description = "llama-index storage-chat-store vastbase integration"
requires-python = ">=3.10,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[tool.llamahub]
contains_example = false
import_path = "llama_index.storage.chat_store.vastbase"

[tool.llamahub.class_authors]
VastbaseChatStore = "vastbase"
```

---

## 3. pyvastbase API 映射

### Collection Schema

```python
from pyvastbase import Collection, CollectionSchema, FieldSchema, DataType

schema = CollectionSchema(name=table_name, fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
    FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="value", dtype=DataType.TEXT),
])
```

| 字段 | 类型 | 约束 | 用途 |
|------|------|------|------|
| `id` | INT64 | PK, auto_id | 自增主键，用于 upsert 定位行 |
| `key` | VARCHAR(512) | 业务唯一键 | LlamaIndex session key |
| `value` | TEXT | 可空 | 整个 `List[ChatMessage]` 序列化为 JSON 字符串 |

### CRUD 操作映射

| 操作 | pyvastbase API | 等价 SQL |
|------|---------------|----------|
| 插入新 key | `col.insert([{...}])` | `INSERT INTO ...` |
| 更新已有 key | `col.upsert([{...}])` | `INSERT ... ON CONFLICT (id) DO UPDATE` |
| 按 key 查询 | `col.query(expr="key == 'xxx'", limit=1)` | `SELECT ... WHERE key = ...` |
| 按 id 删除 | `col.delete(pks=[id])` | `DELETE ... WHERE id = ...` |
| 获取所有 key | `col.query(expr="1=1", output_fields=["key"])` | `SELECT key FROM ...` |
| 检测 collection | `has_collection(name)` | `inspect.get_table_names()` |

> **注意**: pyvastbase `upsert()` 只作用于 PK（id），不可按 key 字段 upsert。所有写入需「先查后写」。

---

## 4. Class Design

```python
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pyvastbase import (
    Collection, CollectionSchema, FieldSchema, DataType,
    AsyncCollection, AsyncConnections,
    connect, get_connection, remove_connection,
    has_collection, list_collections,
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
        description="Collection name for storing chat messages."
    )
    schema_name: str = Field(
        default="public",
        description="Database schema name (retained for API compatibility)."
    )
    use_jsonb: bool = Field(
        default=False,
        description="Retained for API compatibility with upstream. "
                    "Vastbase stores values as TEXT (JSON-serialized)."
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

        Does NOT connect on init — connection is lazy, established
        on first method call via _ensure_initialized().
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
        # Defer connection until first use

    # === Factory Methods ===

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
        """Create instance from vastbase:// URI.

        URI format: vastbase://user:password@host:port/database
        """
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
```

### 字段总览

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `table_name` | str | `"chatstore"` | Collection 名称 |
| `schema_name` | str | `"public"` | Schema 名称（兼容参数，pyvastbase 通过 search_path 管理） |
| `use_jsonb` | bool | `False` | 兼容参数，Vastbase 始终用 TEXT 存 JSON |
| `_host` | str | `"localhost"` | Vastbase 主机 |
| `_port` | int | `5432` | Vastbase 端口 |
| `_database` | str | `"vastbase"` | 数据库名 |
| `_user` | str | `""` | 用户名 |
| `_password` | str | `""` | 密码 |
| `_connection_alias` | str | auto | pyvastbase 连接别名 |
| `_coll` | Collection | None | pyvastbase Collection 实例（延迟初始化） |
| `_initialized` | bool | False | 初始化标记 |

---

## 5. Collection Initialization Flow

```
__init__() / from_params() / from_uri()
    │
    ▼
_ensure_initialized()   ← 首次方法调用时触发
    │
    ├─ 1. connect(host, port, database, user, password, alias)
    │
    ├─ 2. _check_legacy_collection_exists()
    │      └─ has_collection(f"data_{table_name}") → bool
    │
    ├─ 3. table_name = legacy_exists ? f"data_{table_name}" : table_name
    │
    ├─ 4. has_collection(table_name)?
    │      ├─ NO  → Collection(table_name, schema).create()
    │      └─ YES → Collection(table_name)  # 加载已有 collection
    │
    └─ 5. self._coll = col; self._initialized = True
```

### 旧表兼容

与上游 `_check_legacy_table_exists()` 行为一致：
- 调用 `has_collection(f"data_{table_name}")` 检查旧 collection 是否存在
- 若存在，使用 `data_{table_name}` 作为实际 collection 名称（向后兼容）
- 若不存在，使用用户指定的 `table_name`

### 延迟初始化

连接不在 `__init__` 时建立，而是在首次调用任何 CRUD 方法时通过 `_ensure_initialized()` 延迟初始化。这避免了：
- 导入时即建立数据库连接
- Pydantic 序列化/反序列化时的副作用

---

## 6. Core Methods

### 6.1 通用工具方法

```python
def _ensure_initialized(self):
    """Lazy-init: connect + create/load collection on first use."""
    if self._initialized:
        return
    # connect to Vastbase
    connect(
        host=self._host, port=self._port,
        database=self._database,
        user=self._user, password=self._password,
        alias=self._connection_alias,
    )
    # check legacy collection
    legacy_name = f"data_{self.table_name}"
    actual_name = legacy_name if has_collection(legacy_name) else self.table_name
    self.table_name = actual_name
    # create or load collection
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
    self._coll = col
    self._initialized = True
```

```python
def _get_existing(self, key: str) -> Optional[tuple]:
    """Query collection for key. Returns (id, parsed_messages) or None."""
    results = self._coll.query(expr=f'key == "{key}"', limit=1, output_fields=["id", "value"])
    if not results:
        return None
    row = results[0]
    raw = row.get("value", "[]")
    messages = [ChatMessage.model_validate(m) for m in json.loads(raw)]
    return (row["id"], messages)
```

### 6.2 set_messages

**功能**: 覆盖写入 key 下的全部消息。

**上游行为**: `INSERT ... ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value`

**Vastbase 实现**:
```
set_messages(key, messages):
    _ensure_initialized()
    serialized = json.dumps([m.model_dump() for m in messages])
    existing = _get_existing(key)           # query(key)
    if existing:
        _coll.upsert([{"id": existing[0], "key": key, "value": serialized}])
    else:
        _coll.insert([{"key": key, "value": serialized}])
    return None
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 新建 key | `set_messages("k1", [msg_a, msg_b])` | `get_messages("k1")` 返回 `[msg_a, msg_b]` |
| 2 | 覆盖已有 key | 先设 `[msg_a]`，再设 `[msg_c, msg_d]` | `get_messages("k1")` 返回 `[msg_c, msg_d]` |
| 3 | 空消息列表 | `set_messages("k1", [])` | `get_messages("k1")` 返回 `[]` |

### 6.3 get_messages

**功能**: 获取 key 下的全部 ChatMessage。

**上游行为**: `SELECT value WHERE key = :key` → JSON 反序列化

**Vastbase 实现**:
```
get_messages(key):
    _ensure_initialized()
    existing = _get_existing(key)
    if existing:
        return existing[1]   # parsed messages list
    return []
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | key 存在 | `get_messages("k1")` 已有 `[msg_a, msg_b]` | `[msg_a, msg_b]` |
| 2 | key 不存在 | `get_messages("nonexistent")` | `[]` |
| 3 | 空消息 | key 存在但 `value="[]"` | `[]` |
| 4 | 消息保真 | 存储 → 取回 | 所有 ChatMessage 字段完整保留（role, content, additional_kwargs） |

### 6.4 add_message

**功能**: 向 key 的消息数组追加一条消息。

**上游行为**: `INSERT ... ON CONFLICT (key) DO UPDATE SET value = array_cat(table.value, :value)`

**Vastbase 实现**:
```
add_message(key, message):
    _ensure_initialized()
    new_msg_json = message.model_dump()
    existing = _get_existing(key)
    if existing:
        # key 已存在：追加
        row_id, messages = existing
        messages.append(message)            # Python list.append
        serialized = json.dumps([m.model_dump() if hasattr(m, 'model_dump') else m for m in messages])
        _coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    else:
        # key 不存在：新建
        serialized = json.dumps([new_msg_json])
        _coll.insert([{"key": key, "value": serialized}])
    return None
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 追加到已有 key | key 有 `[msg_a]`，add `msg_b` | `get_messages` 返回 `[msg_a, msg_b]`（顺序保持） |
| 2 | 追加到新 key | key 不存在，add `msg_a` | `get_messages` 返回 `[msg_a]` |
| 3 | 多次追加 | 连续 add 3 条 | `get_messages` 返回 3 条（append 顺序） |

### 6.5 delete_messages

**功能**: 删除 key 对应的全部消息。

**上游行为**: `DELETE FROM table WHERE key = :key` → 返回 None（上游不返回被删内容）

**Vastbase 实现**:
```
delete_messages(key):
    _ensure_initialized()
    existing = _get_existing(key)
    if existing:
        _coll.delete(pks=[existing[0]])     # delete by id
    return None
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 删除已有 key | key 存在 | 返回 `None`，`get_messages` 返回 `[]` |
| 2 | 删除不存在 key | key 不存在 | 返回 `None`，不抛异常 |
| 3 | 删除后 keys | key 被删除后 | `get_keys()` 不包含该 key |

### 6.6 delete_message

**功能**: 删除 key 下指定索引的消息，返回被删消息。

**上游行为**: 先 `SELECT value` → Python list.pop(idx) → `UPDATE SET value = array_cat(value[: :idx], value[:idx+2:])`

**Vastbase 实现**:
```
delete_message(key, idx):
    _ensure_initialized()
    existing = _get_existing(key)
    if existing is None:
        return None
    row_id, messages = existing
    if idx < 0 or idx >= len(messages):
        return None
    removed = messages.pop(idx)            # Python list.pop
    serialized = json.dumps([m.model_dump() if hasattr(m, 'model_dump') else m for m in messages])
    _coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    return removed
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 删除中间消息 | key 有 `[a, b, c]`，`idx=1` | 返回 `b`，剩余 `[a, c]` |
| 2 | 删除第一条 | key 有 `[a, b]`，`idx=0` | 返回 `a`，剩余 `[b]` |
| 3 | 删除最后一条 | key 有 `[a, b]`，`idx=1` | 返回 `b`，剩余 `[a]` |
| 4 | idx 越界（负） | `idx=-1` | 返回 `None` |
| 5 | idx 越界（大） | key 有 `[a]`，`idx=5` | 返回 `None` |
| 6 | key 不存在 | key 不存在 | 返回 `None` |

### 6.7 delete_last_message

**功能**: 删除 key 下最后一条消息，返回被删消息。

**上游行为**: 先 `SELECT value` → Python list.pop() → `UPDATE SET value = value[1:array_length(value, 1) - 1]`

**Vastbase 实现**:
```
delete_last_message(key):
    _ensure_initialized()
    existing = _get_existing(key)
    if existing is None:
        return None
    row_id, messages = existing
    if len(messages) == 0:
        return None
    removed = messages.pop()               # Python list.pop (last)
    serialized = json.dumps([m.model_dump() if hasattr(m, 'model_dump') else m for m in messages])
    _coll.upsert([{"id": row_id, "key": key, "value": serialized}])
    return removed
```

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 删除最后一条 | key 有 `[a, b, c]` | 返回 `c`，剩余 `[a, b]` |
| 2 | 仅一条消息 | key 有 `[a]` | 返回 `a`，剩余 `[]` |
| 3 | key 不存在 | key 不存在 | 返回 `None` |
| 4 | 空数组 | key 存在但 value 为空数组 `[]` | 返回 `None` |

### 6.8 get_keys

**功能**: 获取所有 key 列表。

**上游行为**: `SELECT key FROM table`

**Vastbase 实现**:
```
get_keys():
    _ensure_initialized()
    results = _coll.query(expr="id >= 0", output_fields=["key"], limit=10000)
    return [r["key"] for r in results]
```

> **注意**: `limit=10000` 是 pyvastbase 的单次查询上限。ChatStore 场景下 key 数量通常远小于此值。

**验收标准**:
| # | 场景 | 输入 | 预期输出 |
|---|------|------|---------|
| 1 | 有 key | store 有 `"k1"`, `"k2"` | `["k1", "k2"]` |
| 2 | 空 store | 无任何 key | `[]` |
| 3 | 一致性 | `set_messages("k3", ...)` 后 | `get_keys()` 包含 `"k3"` |

---

## 7. Filter Translation

本适配器不使用 pyvastbase 的标量过滤（expr 仅用于精确 key 匹配），无需复杂的 filter 转换规则。

**key 查询表达式**:
```python
col.query(expr=f'key == "{key}"', limit=1, output_fields=["id", "value"])
```

### 转义规则

key 值可能包含双引号字符 `"`。需做转义：
```python
def _escape_key(key: str) -> str:
    return key.replace('"', '\\"')
```

---

## 8. Async API — 原生异步实现

### 策略

**不使用基类 `asyncio.to_thread()` 默认实现。** 7 个异步方法全部显式覆写，使用 pyvastbase `AsyncCollection` + `AsyncConnections` 实现原生异步 I/O。

### 异步连接管理

每个异步方法独立管理连接生命周期：
```
async def _ensure_async_coll(self) -> AsyncCollection:
    await AsyncConnections.connect(
        host=self._host, port=self._port,
        database=self._database,
        user=self._user, password=self._password,
        alias=self._connection_alias,
    )
    return AsyncCollection(self.table_name)
```

### 异步方法清单

| 方法 | 上游实现方式 | Vastbase 实现方式 |
|------|------------|------------------|
| `aset_messages` | asyncpg `await session.execute()` | `AsyncCollection.upsert()` / `insert()` |
| `aget_messages` | asyncpg `await session.execute()` | `AsyncCollection.query()` |
| `async_add_message` | asyncpg `await session.execute()` | `AsyncCollection.query()` + `upsert()` / `insert()` |
| `adelete_messages` | asyncpg `await session.execute()` | `AsyncCollection.delete(pks=[...])` |
| `adelete_message` | asyncpg `await session.execute()` | `AsyncCollection.query()` + `upsert()` |
| `adelete_last_message` | asyncpg `await session.execute()` | `AsyncCollection.query()` + `upsert()` |
| `aget_keys` | asyncpg `await session.execute()` | `AsyncCollection.query()` |

### 异步方法伪代码

```
async def aget_messages(self, key):
    coll = await _ensure_async_coll()
    results = await coll.query(expr=f'key == "{key}"', limit=1, output_fields=["id", "value"])
    if not results:
        return []
    return [ChatMessage.model_validate(m) for m in json.loads(results[0]["value"])]

async def aset_messages(self, key, messages):
    coll = await _ensure_async_coll()
    serialized = json.dumps([m.model_dump() for m in messages])
    existing = await coll.query(expr=f'key == "{key}"', limit=1, output_fields=["id"])
    if existing:
        await coll.upsert([{"id": existing[0]["id"], "key": key, "value": serialized}])
    else:
        await coll.insert([{"key": key, "value": serialized}])
```

### 异步验收标准

每个异步方法的验收标准与对应同步方法一致，额外要求：
- 在 `asyncio` 事件循环中调用不阻塞事件循环
- 并发调用不同 key 的方法互不干扰

---

## 9. Error Handling

### 逐方法异常行为矩阵

| 方法 | key 不存在 | idx 越界 | 数组为空 | 网络断开 | collection 不存在 |
|------|-----------|---------|---------|---------|-----------------|
| `set_messages` | 新建记录 | N/A | N/A | `ConnectionError` | auto-create |
| `get_messages` | 返回 `[]` | N/A | 返回 `[]` | `ConnectionError` | `CollectionNotExistsError` |
| `add_message` | 新建记录 | N/A | N/A | `ConnectionError` | auto-create |
| `delete_messages` | 返回 `None` | N/A | 返回 `None` | `ConnectionError` | `CollectionNotExistsError` |
| `delete_message` | 返回 `None` | 返回 `None` | 返回 `None` | `ConnectionError` | `CollectionNotExistsError` |
| `delete_last_message` | 返回 `None` | N/A | 返回 `None` | `ConnectionError` | `CollectionNotExistsError` |
| `get_keys` | N/A | N/A | 返回 `[]` | `ConnectionError` | `CollectionNotExistsError` |

### 异常传播策略

- pyvastbase 异常（`ConnectionError`, `CollectionNotExistsError`, `DataError`）直接向上传播，不包装
- 不与上游 PostgresChatStore 的 SQLAlchemy 异常类型兼容，但语义等价
- 反序列化失败（value 字段包含非法 JSON）→ 返回 `[]`（优雅降级，与上游行为对齐）

---

## 10. Testing Strategy

### 三层测试体系

| Gate | 阶段 | 范围 | 工具 |
|------|------|------|------|
| Gate 1 | Phase 2b（本子 Issue） | pyvastbase 集成测试 — 14 个方法功能验证 | pytest + Vastbase 连接 |
| Gate 2 | Phase 3（本子 Issue） | 框架兼容性测试 — LlamaIndex ChatStore 接口契约 | pytest + Vastbase |
| Gate 3 | 父 Issue Flow C | E2E 验收 — 与 LlamaIndex ChatEngine 集成 | 手动/自动化验收 |

### Gate 1 测试用例（15 个同步 + 15 个异步）

#### test_vastbase_chatstore.py

```python
class TestVastbaseChatStore:
    """同步方法测试"""

    def test_init_and_connect(self): ...
    def test_set_messages_new_key(self): ...
    def test_set_messages_overwrite(self): ...
    def test_set_messages_empty_list(self): ...
    def test_get_messages_existing(self): ...
    def test_get_messages_nonexistent(self): ...
    def test_get_messages_fidelity(self): ...
    def test_add_message_existing_key(self): ...
    def test_add_message_new_key(self): ...
    def test_add_message_multiple(self): ...
    def test_delete_messages_existing(self): ...
    def test_delete_messages_nonexistent(self): ...
    def test_delete_message_by_index(self): ...
    def test_delete_message_first(self): ...
    def test_delete_message_out_of_bounds(self): ...
    def test_delete_message_nonexistent_key(self): ...
    def test_delete_last_message(self): ...
    def test_delete_last_message_single(self): ...
    def test_delete_last_message_nonexistent(self): ...
    def test_delete_last_message_empty(self): ...
    def test_get_keys(self): ...
    def test_get_keys_empty(self): ...
    def test_from_params(self): ...
    def test_from_uri(self): ...
    def test_legacy_table_compatibility(self): ...
```

#### test_vastbase_chatstore_async.py

```python
class TestVastbaseChatStoreAsync:
    """异步方法测试 — 与同步测试结构一致"""

    @pytest.mark.asyncio
    async def test_aset_messages_new_key(self): ...

    @pytest.mark.asyncio
    async def test_aset_messages_overwrite(self): ...

    # ... 所有 7 个异步方法，每个至少 2-3 个 case
```

### conftest.py

```python
import pytest
from pyvastbase import connect, close_all

VASTBASE_CONFIG = {
    "host": "172.16.105.107",
    "port": 15432,
    "database": "vastbase",
    "user": "aidev",
    "password": "Vbase_123456",
}

@pytest.fixture
def chat_store():
    store = VastbaseChatStore(**VASTBASE_CONFIG, table_name="test_chatstore")
    yield store
    # cleanup: drop test collection
    if store._coll:
        store._coll.drop()
    close_all()
```

### Gate 2 框架兼容性测试

从上游 PostgresChatStore 测试套件提取适配：
- 替换 `PostgresChatStore.from_uri("postgresql://...")` → `VastbaseChatStore.from_uri("vastbase://...")`
- 替换 Docker PostgreSQL fixture → Vastbase 连接 fixture
- 保持所有 assert 逻辑不变

---

## 11. Differences from Reference (PostgresChatStore)

| # | 差异项 | PostgresChatStore (上游) | VastbaseChatStore (本适配) | 影响 |
|---|--------|------------------------|---------------------------|------|
| 1 | 驱动 | SQLAlchemy + psycopg + asyncpg | pyvastbase Collection + AsyncCollection | 无 SQLAlchemy 依赖 |
| 2 | 连接模型 | SyncEngine + AsyncEngine 双引擎 | pyvastbase connect() + AsyncConnections | 单连接 alias，非 pool |
| 3 | 数据存储 | ARRAY(JSON/JSONB) | TEXT (JSON 字符串) | 消息列表序列化为单 TEXT 字段 |
| 4 | upsert 方式 | `INSERT ON CONFLICT (key)` SQL | 先 query → Python 判断 → upsert(id) 或 insert | 非原子，存在 TOCTOU 竞态（概率极低） |
| 5 | 数组操作 | `array_cat()`, 切片 `[: :idx]` SQL 函数 | Python list.append/pop/slice | O(N) 全量读+写，但 N 很小 |
| 6 | 异步实现 | asyncpg 原生 SQL | pyvastbase AsyncCollection 原生异步 | 行为等价 |
| 7 | URI scheme | `postgresql+psycopg://` / `postgresql+asyncpg://` | `vastbase://` | 专属 scheme |
| 8 | Schema 管理 | `CREATE SCHEMA IF NOT EXISTS` SQL | pyvastbase 自动管理 | schema 概念由 search_path 替代 |
| 9 | 返回值类型 | SQLAlchemy Row | pyvastbase 查询结果 dict | 内部差异，对外透明 |
| 10 | 错误类型 | SQLAlchemyError / IntegrityError | pyvastbase VastbaseException 子类 | 异常类型不同，但异常传播语义一致 |

---

## 12. Future Work

1. **连接池管理**: 当前每个 VastbaseChatStore 实例一个连接 alias，未来可支持跨实例复用连接池
2. **批量操作优化**: `set_messages` 批量写入多条 key 时可用 `col.batch_insert()`
3. **消息压缩**: 大量历史消息可支持 gzip + base64 压缩存储
4. **分页支持**: `get_messages` 可扩展 `limit/offset` 参数支持消息分页
5. **数据迁移工具**: 从 PostgresChatStore → VastbaseChatStore 的数据导出/导入脚本
6. **use_jsonb 实际支持**: 当 Vastbase JSONB 类型通过 pyvastbase 可用时启用原生 JSONB 字段
