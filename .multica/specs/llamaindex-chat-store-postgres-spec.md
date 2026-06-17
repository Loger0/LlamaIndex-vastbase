# LlamaIndex ChatStore Vastbase 适配 — Design Spec

**Date:** 2026-06-17
**Status:** Draft
**Integration Mode:** standalone
**pyvastbase Version:** >=0.2.7
**Target Package:** llama-index-storage-chat-store-vastbase v0.1.0

---

## 1. Overview

### 目标

将 LlamaIndex 的 PostgresChatStore（PostgreSQL 聊天历史存储后端）适配到 Vastbase，用 pyvastbase 的 Collection API 替代上游的 SQLAlchemy + psycopg/asyncpg 驱动栈，实现全量功能对等的 `VastbaseChatStore`。

### 非目标

- 不修改 LlamaIndex 框架源码
- 不实现超出 `BaseChatStore` 接口的新功能
- 不支持分区/副本等 Vastbase 不支持特性
- 不使用直接 SQL — 所有操作通过 pyvastbase Collection API

### 参考实现

- 上游包: `llama-index-storage-chat-store-postgres` v0.4.0
- 核心文件: `llama_index/storage/chat_store/postgres/base.py` (480 行)
- 基类: `llama_index/core/storage/chat_store/base.py` — `BaseChatStore(BaseComponent)`

---

## 2. Package Structure

```
llama-index-storage-chat-store-vastbase/
├── pyproject.toml
├── README.md
└── llama_index/
    └── storage/
        └── chat_store/
            └── vastbase/
                ├── __init__.py
                └── base.py
```

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "llama-index-storage-chat-store-vastbase"
version = "0.1.0"
description = "Vastbase chat store integration for LlamaIndex"
requires-python = ">=3.9,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[tool.setuptools.packages.find]
include = ["llama_index*"]
```

### __init__.py

```python
from llama_index.storage.chat_store.vastbase.base import VastbaseChatStore

__all__ = ["VastbaseChatStore"]
```

---

## 3. pyvastbase API 映射

| 上游 PostgresChatStore 操作 | pyvastbase 等价操作 |
|---------------------------|-------------------|
| `CREATE SCHEMA IF NOT EXISTS` | `connect()` 连接时默认 schema，Collection 自动管理 |
| `CREATE TABLE IF NOT EXISTS` | `Collection(name, schema).create()` 或检查 `has_collection()` |
| `INSERT ... ON CONFLICT (key) DO UPDATE` | query → 存在则 `col.upsert([{id, key, value}])`，不存在则 `col.insert([{key, value}])` |
| `SELECT ... WHERE key = :key` | `col.query(expr=f"key == '{key}'", limit=1)` |
| `SELECT key FROM table` | `col.query(expr="1=1", output_fields=["key"])` |
| `DELETE FROM table WHERE key = :key` | `col.delete(expr=f"key == '{key}'")` |
| `array_cat(col, val)` | Python: `messages.append(new_msg)` → upsert |
| `value[: :idx] \|\| value[:idx+2:]` | Python: `messages.pop(idx)` → upsert |
| `value[1:array_length(value,1)-1]` | Python: `messages.pop()` → upsert |
| `metadata.create_all()` | `Collection(name, schema).create()` |
| `engine.dispose()` | GC + 连接池自动管理 |
| `ChatMessage.model_dump_json()` | 保持不变（与上游序列化兼容） |
| `ChatMessage.model_validate()` | 保持不变（与上游反序列化兼容） |

---

## 4. Class Design

### 字段表

```python
class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store for LlamaIndex.

    Drop-in replacement for PostgresChatStore using pyvastbase Collection API.
    """

    # === Pydantic 字段 ===
    table_name: str = "chatstore"
    schema_name: str = "public"
    use_jsonb: bool = False  # 保留参数兼容，实际用 TEXT 字段

    # === 连接参数（可选，二选一与 from_params/from_uri 配合） ===
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = ""
    password: str = ""

    # === 私有属性 ===
    _coll: Optional[Collection] = PrivateAttr(default=None)
    _async_coll: Optional[AsyncCollection] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _actual_table_name: str = PrivateAttr(default="")
```

### 连接管理

- **同步**：`connect(host, port, database, user, password, alias=...)` 建立全局连接
  - Collection 实例通过 `Collection(name, schema=...)` 自动复用默认连接
  - 使用别名避免与其他 Collection 实例冲突: `alias=f"chatstore_{table_name}"`
- **异步**：`AsyncConnections.connect(host, port, database, user, password, alias=...)` 建立异步连接
  - AsyncCollection 通过 `AsyncCollection(name, schema=...)` 复用

### 工厂方法

#### `from_params(...)`

```python
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
```

#### `from_uri(uri, ...)`

```python
@classmethod
def from_uri(
    cls,
    uri: str,  # vastbase://user:pass@host:port/db
    table_name: str = "chatstore",
    schema_name: str = "public",
    **kwargs,
) -> "VastbaseChatStore":
```

URI 解析：`urllib.parse.urlparse(uri)` → 提取 `scheme`（验证为 `vastbase`）、`username`、`password`、`hostname`、`port`、`path[1:]`（数据库名）。

---

## 5. Collection Initialization Flow

```
__init__ / from_params / from_uri
    │
    ▼
_initialize()
    │
    ├─ 1. 建立连接
    │   ├─ sync:  connect(host, port, database, user, password, alias=f"chatstore_{table_name}")
    │   └─ async: AsyncConnections.connect(host, port, database, user, password, alias=f"chatstore_{table_name}")
    │
    ├─ 2. 旧表兼容检测
    │   └─ has_collection(f"data_{schema_name}.{table_name}")?
    │       ├─ Yes → _actual_table_name = f"data_{table_name}"
    │       └─ No  → _actual_table_name = table_name
    │
    ├─ 3. 定义 Schema
    │   CollectionSchema(
    │       name=_actual_table_name,
    │       fields=[
    │           FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
    │           FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=512),
    │           FieldSchema(name="value", dtype=DataType.TEXT),
    │       ]
    │   )
    │
    ├─ 4. 创建/获取 Collection
    │   ├─ has_collection(_actual_table_name)?
    │   │   ├─ Yes → _coll = Collection(_actual_table_name)
    │   │   └─ No  → _coll = Collection(_actual_table_name, schema=schema)
    │   │            _coll.create()
    │   └─ _async_coll = AsyncCollection(_actual_table_name)
    │
    └─ 5. _initialized = True
```

**关键点：**
- Collection 名称格式：`{schema_name}.{actual_table_name}` 或直接用 `actual_table_name`（default schema 时）
- `FieldSchema` 的 `auto_id` 默认行为与 `id` 字段结合：pyvastbase INSERT 时不传 `id` 会自动生成
- `VARCHAR(512)` 与上游 `sa.String(512)` 对齐

---

## 6. Core Methods

### 验收标准说明

每个方法定义以下验收维度：
- **正常路径（Happy Path）**：标准输入 → 预期输出
- **边界条件**：空 key、key 不存在、空消息列表、大消息列表
- **错误场景**：连接断开、Collection 不存在

### 6.1 set_messages / aset_messages

**数据流：**

```
set_messages(key="user_abc", messages=[ChatMessage(...), ...])
    │
    ├─ value = json.dumps([m.model_dump() for m in messages])
    │
    ├─ existing = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if existing:
    │   _coll.upsert([{"id": existing[0]["id"], "key": key, "value": value}])
    │
    └─ else:
        _coll.insert([{"key": key, "value": value}])
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常 | key="abc", messages=[ChatMessage(role=USER, content="hi")] | 写入成功，返回 None |
| key已存在 | key="abc" 已存在，新 messages=[ChatMessage(role=ASSISTANT, content="bye")] | 覆盖旧值，value 更新为新 messages |
| 空消息列表 | key="abc", messages=[] | 写入 `[]` JSON，不抛异常 |
| 连接断开 | host unreachable → `pyvastbase.ConnectionError` | 异常向上传播 |
| 并发同key写入 | 两个线程同时 set_messages("abc", ...) | 后写入者覆盖（TOCTOU，接受） |

### 6.2 get_messages / aget_messages

**数据流：**

```
get_messages(key="user_abc")
    │
    ├─ results = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if not results:
    │   return []
    │
    └─ parsed = json.loads(results[0]["value"])
       return [ChatMessage.model_validate(m) for m in parsed]
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常 | key="abc" 存在且 value='[{...}]' | 返回 List[ChatMessage] |
| key不存在 | key="nonexistent" | 返回 `[]` |
| 空消息列表 | key 存在但 value='[]' | 返回 `[]` |
| JSON 损坏 | value 不是合法 JSON | 异常向上传播 |

### 6.3 add_message / async_add_message

**数据流：**

```
add_message(key="user_abc", message=ChatMessage(...))
    │
    ├─ results = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if not results:        # key 不存在 → 新建
    │   _coll.insert([{"key": key, "value": json.dumps([message.model_dump()])}])
    │   return None
    │
    └─ else:                   # key 存在 → 追加
        messages = json.loads(results[0]["value"])
        messages.append(message.model_dump())
        _coll.upsert([{"id": results[0]["id"], "key": key,
                       "value": json.dumps(messages)}])
        return None
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常追加 | key 已有 3 条 → add 第 4 条 | value 长度变为 4 |
| key不存在 | key 不存在 → add message | 新建 key，插入单条消息 |
| 空 key | key="" | 正常写入（与上游行为一致） |

### 6.4 delete_messages / adelete_messages

**数据流：**

```
delete_messages(key="user_abc")
    │
    ├─ results = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if not results:
    │   return None
    │
    ├─ messages = [ChatMessage.model_validate(m) for m in json.loads(results[0]["value"])]
    │
    ├─ _coll.delete(expr=f"key == '{key}'")
    │
    └─ return messages
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常删除 | key 存在且有 3 条消息 | 删除整行，返回被删的 3 条 ChatMessage |
| key不存在 | key="nonexistent" | 返回 `None` |
| 空消息列表 | key 存在但 value='[]' | 删除行，返回 `[]` |

### 6.5 delete_message / adelete_message

**数据流：**

```
delete_message(key="user_abc", idx=1)  # 删除索引 1 的消息
    │
    ├─ results = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if not results or not results[0]["value"]:
    │   return None
    │
    ├─ messages = json.loads(results[0]["value"])
    │
    ├─ if idx < 0 or idx >= len(messages):
    │   return None
    │
    ├─ removed = messages.pop(idx)
    │
    ├─ _coll.upsert([{"id": results[0]["id"], "key": key,
    │                  "value": json.dumps(messages)}])
    │
    └─ return ChatMessage.model_validate(removed)
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常 | key=已存在, idx=1 (valid) | 删除索引 1，返回被删 ChatMessage |
| idx 越界 | idx=100 > len(messages) | 返回 `None` |
| 负 idx | idx=-1 | 返回 `None` |
| key不存在 | key="nonexistent", idx=0 | 返回 `None` |
| 删除最后一条 | idx=-1 → 空列表 | value 更新为 '[]' |

### 6.6 delete_last_message / adelete_last_message

**数据流：**

```
delete_last_message(key="user_abc")
    │
    ├─ results = _coll.query(expr=f"key == '{key}'", limit=1)
    │
    ├─ if not results or not results[0]["value"]:
    │   return None
    │
    ├─ messages = json.loads(results[0]["value"])
    │
    ├─ if len(messages) == 0:
    │   return None
    │
    ├─ removed = messages.pop()
    │
    ├─ _coll.upsert([{"id": results[0]["id"], "key": key,
    │                  "value": json.dumps(messages)}])
    │
    └─ return ChatMessage.model_validate(removed)
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常 | key 有 3 条消息 | 删除最后一条，返回 ChatMessage |
| 空消息列表 | key 存在但 value='[]' | 返回 `None` |
| key不存在 | key="nonexistent" | 返回 `None` |
| 仅剩一条 | key 有 1 条消息 | 删除后 value='[]'，返回 ChatMessage |

### 6.7 get_keys / aget_keys

**数据流：**

```
get_keys()
    │
    ├─ results = _coll.query(expr="1=1", output_fields=["key"])
    │
    └─ return [r["key"] for r in results]
```

| 验收维度 | 输入 | 预期行为 |
|---------|------|---------|
| 正常 | Collection 有 5 条记录 | 返回 5 个 key 的列表 |
| 空 Collection | 无任何记录 | 返回 `[]` |

### 6.8 异步方法 (aset_* / aget_* / a* 7 个方法)

每个异步方法是其同步对应方法的原生异步版本，使用 `AsyncCollection`：

```
async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
    import json
    value = json.dumps([m.model_dump() for m in messages])
    existing = await self._async_coll.query(expr=f"key == '{key}'", limit=1)
    if existing:
        await self._async_coll.upsert([{
            "id": existing[0]["id"], "key": key, "value": value
        }])
    else:
        await self._async_coll.insert([{"key": key, "value": value}])
```

所有 7 个异步方法遵循相同模式：将同步方法中的 `_coll.*` 替换为 `await self._async_coll.*`。

---

## 7. Filter Translation

由于 pyvastbase `query()` 的 `expr` 参数接受 Python 表达式字符串（如 `"key == 'value'"`, `"id < 100"`），且 ChatStore 的查询模式极简单（仅按 key 等值查询），不需要复杂的 filter 转换层。

### 操作符映射

| 操作 | 上游 PostgresChatStore | VastbaseChatStore expr |
|------|----------------------|----------------------|
| key 等值查询 | `WHERE key = :key` | `f"key == '{key}'"` |
| key 全量查询 | `SELECT key FROM ...` | `"1=1" → output_fields=["key"]` |

### 特殊处理

- **key 值转义**：key 来自 LlamaIndex 内部（session id 等），不包含单引号，无需特殊转义。若未来出现含引号的 key，用 `key.replace("'", "''")` 处理。
- **SQL 注入防护**：pyvastbase `expr` 参数中的字符串值必须用单引号括起，key 值中的单引号进行双重转义。

---

## 8. Async API

### 策略：pyvastbase AsyncCollection 原生异步

不使用基类 `BaseChatStore` 的 `asyncio.to_thread()` 默认实现。所有 7 个异步方法用 pyvastbase `AsyncCollection` + `AsyncConnections` 完整覆写。

### 连接管理

```python
# 在 _initialize() 中：
from pyvastbase import AsyncConnections

await AsyncConnections.connect(
    host=self.host,
    port=self.port,
    database=self.database,
    user=self.user,
    password=self.password,
    alias=f"chatstore_{self._actual_table_name}",
)
self._async_coll = AsyncCollection(self._actual_table_name)
```

### 异步方法清单

| 异步方法 | 同步对应 | 差异 |
|---------|---------|------|
| `aset_messages` | `set_messages` | `await self._async_coll.query/upsert/insert` |
| `aget_messages` | `get_messages` | `await self._async_coll.query` |
| `async_add_message` | `add_message` | `await self._async_coll.query/insert/upsert` |
| `adelete_messages` | `delete_messages` | `await self._async_coll.query/delete` |
| `adelete_message` | `delete_message` | `await self._async_coll.query/upsert` |
| `adelete_last_message` | `delete_last_message` | `await self._async_coll.query/upsert` |
| `aget_keys` | `get_keys` | `await self._async_coll.query` |

---

## 9. Error Handling

### 行为矩阵

| 场景 | 方法 | 返回值 | 异常 |
|------|------|--------|------|
| key 不存在时 get_messages | get_messages | `[]` | 不抛 |
| key 不存在时 delete_messages | delete_messages | `None` | 不抛 |
| key 不存在时 delete_message | delete_message | `None` | 不抛 |
| key 不存在时 delete_last_message | delete_last_message | `None` | 不抛 |
| idx 越界 | delete_message | `None` | 不抛 |
| 空消息列表 get_messages | get_messages | `[]` | 不抛 |
| 空消息列表 delete_last_message | delete_last_message | `None` | 不抛 |
| 连接失败（任意方法） | 所有方法 | - | `pyvastbase.ConnectionError` 向上传播 |
| Collection 不存在 | 所有方法 | - | `pyvastbase.CollectionNotExistsError` 向上传播 |
| JSON 反序列化失败 | get_messages | - | `json.JSONDecodeError` 向上传播 |
| 向量维度错误 | N/A (ChatStore 不涉及) | - | 不使用 |

### 与上游差异

| 异常类型 | 上游行为 | VastbaseChatStore 行为 |
|---------|---------|----------------------|
| 连接失败 | `sqlalchemy.exc.OperationalError` | `pyvastbase.ConnectionError` |
| 表不存在 | `sqlalchemy.exc.ProgrammingError` | `pyvastbase.CollectionNotExistsError`（不应出现 — 初始化时已创建） |
| JSON 反序列化失败 | `json.JSONDecodeError` | 同上游 |

---

## 10. Testing Strategy

### 三层测试体系

```
tests/
├── conftest.py                    # Fixtures: vastbase_connect, chat_store, async_chat_store
├── test_vastbase_chat_store.py    # Layer 1: 集成测试（真实 Vastbase）
└── test_compat.py                 # Layer 2: 兼容性测试（drop-in replacement 验证）
```

### Layer 1: 集成测试（test_vastbase_chat_store.py）

使用真实 Vastbase 连接（172.16.105.107:15432）。

每个方法 ≥ 2 个测试用例：

| 测试方法 | 覆盖场景 |
|---------|---------|
| `test_set_messages_new_key` | 新 key 创建 → get_messages 验证 |
| `test_set_messages_overwrite` | 已有 key 覆盖 → 验证旧值被替换 |
| `test_get_messages_empty` | 不存在 key → 返回 [] |
| `test_add_message_existing` | 已有 key 追加 → 验证长度 +1 |
| `test_add_message_new_key` | 新 key 追加 → 验证自动创建 |
| `test_delete_messages_existing` | 存在 key 删除 → 验证返回值 + get 返回 [] |
| `test_delete_messages_nonexistent` | 不存在 key → 返回 None |
| `test_delete_message_valid_idx` | 有效索引删除 → 验证返回值 + 被删后列表 |
| `test_delete_message_out_of_range` | 越界 idx → 返回 None |
| `test_delete_last_message` | 删除最后一条 → 验证返回值 + 剩余列表 |
| `test_delete_last_message_empty` | 空列表 → 返回 None |
| `test_get_keys` | 多 key → 验证返回所有 key |
| `test_get_keys_empty` | 空 store → 返回 [] |

### Layer 1 Async: 异步集成测试

上述每个同步测试有对应的 `_async` 变体（使用 `pytest-asyncio`），如：
`test_aset_messages_new_key`、`test_aget_messages_empty`、`test_aget_keys` 等。

### Layer 2: 兼容性测试（test_compat.py）

验证 VastbaseChatStore 可作为 PostgresChatStore 的 drop-in replacement：

| 测试 | 验证项 |
|------|--------|
| `test_import_same_interface` | `from llama_index.storage.chat_store.vastbase import VastbaseChatStore` 成功 |
| `test_isinstance_basechatstore` | `isinstance(VastbaseChatStore(...), BaseChatStore)` |
| `test_same_return_types` | `get_messages` 返回 `List[ChatMessage]`、`delete_messages` 返回 `Optional[List[ChatMessage]]` 等 |
| `test_signature_compat` | 方法签名与上游完全一致 |

### conftest.py

```python
import pytest
from pyvastbase import connect

TEST_CONFIG = {
    "host": "172.16.105.107",
    "port": 15432,
    "database": "vastbase",
    "user": "aidev",
    "password": "Vbase_123456",
    "table_name": "test_chatstore",
}

@pytest.fixture(autouse=True)
def cleanup():
    """Each test gets a fresh collection."""
    from pyvastbase import has_collection, drop_collection
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])
    yield
    if has_collection(TEST_CONFIG["table_name"]):
        drop_collection(TEST_CONFIG["table_name"])

@pytest.fixture
def chat_store():
    connect(**{k: v for k, v in TEST_CONFIG.items() if k != "table_name"})
    store = VastbaseChatStore.from_params(**TEST_CONFIG)
    return store

@pytest.fixture
async def async_chat_store():
    from pyvastbase import AsyncConnections
    await AsyncConnections.connect(...)
    store = VastbaseChatStore.from_params(**TEST_CONFIG)
    await store._initialize_async()
    return store
```

---

## 11. Differences from Reference

| 维度 | 上游 PostgresChatStore (v0.4.0) | VastbaseChatStore (v0.1.0) |
|------|-------------------------------|--------------------------|
| **驱动层** | SQLAlchemy ORM + psycopg (同步) + asyncpg (异步) | pyvastbase Collection + AsyncCollection |
| **连接管理** | `create_engine` + `sessionmaker` (同步) + `create_async_engine` + `async_sessionmaker` (异步) | `connect()` (同步) + `AsyncConnections.connect()` (异步) |
| **数据模型** | `ARRAY(JSON)` 或 `ARRAY(JSONB)` 列 | TEXT 列存 JSON 字符串 |
| **Schema 管理** | `CREATE SCHEMA IF NOT EXISTS` + `metadata.create_all()` | `Collection(name, schema).create()` |
| **upsert 策略** | `INSERT ... ON CONFLICT (key) DO UPDATE`（数据库层原子） | query → exists? insert : upsert(id)（Python 层，TOCTOU） |
| **数组追加** | `UPDATE SET value = array_cat(value, :new)`（数据库层） | SELECT → Python list.append → upsert（Python 层） |
| **数组删除** | `value[: :idx] \|\| value[:idx+2:]`（数据库层） | SELECT → Python list.pop(idx) → upsert（Python 层） |
| **删最后一条** | `value[1:array_length(value,1)-1]`（数据库层） | SELECT → Python list.pop() → upsert（Python 层） |
| **异步实现** | asyncpg 原生 `await conn.execute()` | pyvastbase `AsyncCollection` 原生 `await col.query/upsert/insert` |
| **URI 格式** | `postgresql://` / `postgresql+psycopg://` / `postgresql+asyncpg://` | `vastbase://` |
| **旧表兼容** | `_check_legacy_table_exists()` SQL 查询 | `has_collection("data_{table_name}")` |
| **JSONB 模式** | `use_jsonb=True` → `ARRAY(JSONB)` + `jsonb_build_object()` | `use_jsonb` 保留参数兼容，实际无差异（TEXT） |
| **序列化** | `ChatMessage.model_dump_json()` + `ChatMessage.from_json()` / `model_validate()` | 同上游，保证数据格式兼容 |
| **异常类型** | `sqlalchemy.exc.*` + `psycopg.Error` / `asyncpg.Error` | `pyvastbase.VastbaseException` 及其子类 |

### 行为差异说明

1. **upsert 非原子性**：上游的 `ON CONFLICT (key) DO UPDATE` 是单条 SQL 原子操作。Vastbase 适配版的 query → upsert 两步间存在 TOCTOU 竞态窗口。对于 ChatStore 的使用场景（单用户 session），概率极低且后果无影响（后写入者覆盖），可接受。

2. **数组操作性能**：上游在数据库层完成追加/删除（单次 UPDATE），Vastbase 适配版需要 SELECT + Python 操作 + upsert（两次数据库往返）。ChatStore 的消息数通常很小（< 100 条），JSON 序列化/反序列化开销可忽略。

3. **TEXT vs JSON/JSONB**：上游用 `ARRAY(JSON)`/`ARRAY(JSONB)` 原生数组类型，Vastbase 用 TEXT 存序列化 JSON。查询和序列化行为等价，仅在数据库中存储格式不同。`use_jsonb` 参数保留但无实际效果。

---

## 12. Future Work

- **事务支持**：当前 pyvastbase 无事务 API，未来如支持可考虑将 SELECT + upsert 包裹在事务中消除 TOCTOU 竞态
- **批量操作优化**：`batch_insert` 支持批量 set_messages 多 key
- **连接池调优**：根据生产负载调整 pyvastbase 连接池大小
- **ORM 模型支持**：当 pyvastbase ORM 成熟后，可考虑用 `@Model` 装饰器替代手写 Collection 操作
- **监控集成**：添加操作延迟、错误率的指标收集
