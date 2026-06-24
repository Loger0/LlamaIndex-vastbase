# LlamaIndex VastbaseVectorStore 适配 — Design Spec

**Date:** 2026-06-24
**Status:** Draft
**Issue:** DRI-11
**Integration Mode:** standalone
**Coupling Level:** low
**pyvastbase Version:** >=0.2.7
**Framework:** LlamaIndex v0.14.22
**Reference Backend:** PGVectorStore (llama-index-vector-stores-postgres v0.8.1)

---

## 1. Overview

### 目标

为 LlamaIndex v0.14.22 实现 Vastbase 向量存储适配器 (`VastbaseVectorStore`)，以独立 pip 包 `llama-index-vector-stores-vastbase` 发布，对标 PGVectorStore (v0.8.1) 的全量功能，使用 pyvastbase 驱动栈替代 SQLAlchemy + psycopg2/asyncpg + pgvector 依赖。

### 非目标

- 不修改 `llama-index-core` 源码（standalone 模式，coupling_level: low）
- 不涉及 LlamaIndex 其他组件（ingestion pipeline、query engine 等）
- 不实现 PGVectorStore 未定义的额外功能
- 不支持 `customize_query_fn`（pyvastbase 无对应概念）
- 不支持 `persist()`（raise NotImplementedError，与上游一致）

### 包信息

| 项目 | 内容 |
|------|------|
| **包名** | `llama-index-vector-stores-vastbase` |
| **版本** | 0.1.0 |
| **类名** | `VastbaseVectorStore(BasePydanticVectorStore)` |
| **仓库** | `https://github.com/Loger0/LlamaIndex-vastbase.git` |
| **分支** | `feature/llama-index-vastbase` |

---

## 2. Package Structure

```
llama-index-vector-stores-vastbase/
├── pyproject.toml                              # 包配置、依赖、llamahub 注册
├── llama_index/
│   └── vector_stores/
│       └── vastbase/
│           ├── __init__.py                     # 导出 VastbaseVectorStore
│           ├── base.py                         # 核心实现 (~800 行)
│           └── utils.py                        # filter 转换、辅助工具 (~200 行)
├── tests/
│   ├── conftest.py                             # pytest fixtures (13 个)
│   ├── test_collection_init.py                 # Collection 创建/schema 测试
│   ├── test_crud.py                            # add/delete/get_nodes/clear
│   ├── test_filter.py                          # 14 种 FilterOperator 翻译
│   ├── test_search.py                          # 4 种查询模式
│   ├── test_async.py                           # 异步 API 对等测试
│   ├── test_integration.py                     # E2E 集成测试 (真实 Vastbase)
│   ├── test_framework_integration.py            # 框架级集成测试
│   └── demo_llamaindex.py                      # Demo 验收脚本 (8 场景)
```

### pyproject.toml

```toml
[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
requires-python = ">=3.9,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[tool.llamahub]
contains_example = false
import_path = "llama_index.vector_stores.vastbase"

[tool.llamahub.class_authors]
VastbaseVectorStore = "llama-index"

[tool.pytest.ini_options]
asyncio_mode = "auto"
filterwarnings = ["ignore::DeprecationWarning"]
```

### 依赖对比

| 替换前 (PGVectorStore) | 替换后 (VastbaseVectorStore) |
|----------------------|---------------------------|
| `psycopg2-binary>=2.9.9` | — |
| `asyncpg>=0.29.0` | — |
| `pgvector>=0.3.6` | — |
| `sqlalchemy[asyncio]>=1.4.49` | — |
| `llama-index-core>=0.13.0,<0.15` | `llama-index-core>=0.13.0,<0.15` |
| — | `pyvastbase>=0.2.7` |
| **5 个包** | **2 个包（减少 60%）** |

---

## 3. pyvastbase API 映射

### 3.1 核心 API 对应关系

| PGVectorStore 组件 | pyvastbase 替代 | API |
|---|---|---|
| SQLAlchemy `create_engine` + `sessionmaker` | `connect()` | 全局连接管理 |
| SQLAlchemy `create_async_engine` + `asyncpg` | `AsyncConnections` + `AsyncCollection` | 异步连接 |
| pgvector `Vector(N)` | `DataType.FLOAT_VECTOR` | 4字节浮点向量 |
| pgvector `HalfVec(N)` | `DataType.FLOAT16_VECTOR` | 2字节半精度向量 (需 3.0.9+) |
| `declarative_base()` + `type()` 动态建表 | `CollectionSchema` + `FieldSchema` | 声明式 schema |
| DDL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef_construction)` | HNSW 图索引 |
| `to_tsvector` + `to_tsquery` + `ts_rank` | `IndexParams.fulltext_index(algorithm="BM25")` | BM25 全文索引 (D-05) |
| `session.add()` + `session.commit()` | `col.insert(data_list)` | 批量插入 |
| `session.execute(delete(...))` | `col.delete(expr=...)` | 条件删除 |
| `session.execute(delete(full_table))` | `col.truncate()` | 清空表 |
| `cosine_distance` SQL | `col.search(metric_type="COSINE")` | 余弦相似度搜索 |
| `session.execute(text(raw_sql))` | `col.query(expr=...)` | 标量查询 |

### 3.2 已知 pyvastbase 兼容性问题及补丁

| 问题 | 影响 | 补丁方案 | 实现位置 |
|------|------|---------|---------|
| `has_collection` 偶发 `unexpected keyword argument 'using'` | Collection 存在性检查失败 | try/except + fall through 到 `create_collection` + 捕获 `already exists` 异常 | `_initialize()` |
| `AsyncExecutor.execute` 命名占位符不兼容 list 参数 | async 初始化/schema 加载失败 | `_patch_async_executor()` — 猴子补丁，自动将 list 转为 dict | `utils.py` |
| `FieldSchema(auto_id=True)` 不生成 `SERIAL` DDL | INSERT 时 NotNullViolation | `_ensure_auto_id_sequence()` — 手动创建 sequence + ALTER TABLE SET DEFAULT | `utils.py` |
| `add_collection_field` 缺少 `@with_executor` 装饰器 | 动态添加字段静默失败 | `_ensure_schema_columns()` — 直接执行 ALTER TABLE SQL | `utils.py` |

---

## 4. Class Design

### 4.1 Pydantic 字段（序列化字段）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stores_text` | `bool` | `True` | 声明支持文本存储 |
| `flat_metadata` | `bool` | `False` | 是否扁平化元数据 |
| `host` | `str` | `"localhost"` | Vastbase 主机 |
| `port` | `int` | `15432` | Vastbase 端口 |
| `database` | `str` | `"vastbase"` | 数据库名 |
| `user` | `str` | `"aidev"` | 用户名 |
| `password` | `str` | `"Vbase_123456"` | 密码 |
| `collection_name` | `str` | `"llamaindex"` | Collection 名 |
| `embed_dim` | `int` | `1536` | 向量维度 |
| `use_halfvec` | `bool` | `False` | 半精度向量 (需 Vastbase 3.0.9+) |
| `hybrid_search` | `bool` | `False` | 启用混合搜索 (D-08) |
| `text_search_config` | `str` | `"english"` | 全文分词器 |
| `hnsw_kwargs` | `Optional[Dict]` | `None` | HNSW 索引参数 |
| `perform_setup` | `bool` | `True` | 自动创建 Collection + 索引 |
| `debug` | `bool` | `False` | 调试模式 |
| `indexed_metadata_keys` | `Optional[Set]` | `None` | 需索引的元数据键 |
| `initialization_fail_on_error` | `bool` | `False` | 初始化失败是否抛出 (D-03) |

### 4.2 私有属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `_connection_alias` | `str` | pyvastbase 连接别名，格式 `vb_{collection_name}_{id(self)}` |
| `_collection` | `Collection` | pyvastbase 同步 Collection 实例 |
| `_async_collection` | `AsyncCollection` | pyvastbase 异步 Collection 实例 |
| `_is_initialized` | `bool` | 初始化标志 |

### 4.3 连接管理 (D-04)

- 实例级连接：每个 VastbaseVectorStore 实例在 `_initialize()` 时调用 `connect(alias=...)`
- alias 隔离：`_connection_alias = f"vb_{collection_name}_{id(self)}"`
- 清理：`close()` 调用 `remove_connection(alias)`

### 4.4 from_params 类方法

与上游 `PGVectorStore.from_params()` 签名一致，接受相同参数集，返回 `VastbaseVectorStore` 实例。

### 4.5 client 属性

```python
@property
def client(self) -> Any:
    """返回已初始化的 Collection 实例。"""
    return self._collection if self._is_initialized else None
```

---

## 5. Collection Initialization Flow

```
_initialize()
│
├── 1. 应用 pyvastbase 补丁
│   _patch_async_executor()          # 修复 AsyncExecutor list 参数兼容性
│
├── 2. 生成连接 alias + 建立连接
│   _connection_alias = f"vb_{collection_name}_{id(self)}"
│   connect(host, port, database, user, password, alias=_connection_alias)
│
├── 3. 版本检测（若 use_halfvec=True）(D-07)
│   if not check_vb_version((3, 0, 9)):
│       raise ValueError("use_halfvec requires Vastbase >= 3.0.9")
│
├── 4. 构建 CollectionSchema
│   fields = [
│       FieldSchema(name="id",        dtype=INT64, is_primary_key=True, auto_id=True),
│       FieldSchema(name="node_id",   dtype=VARCHAR, max_length=65535),
│       FieldSchema(name="text",      dtype=VARCHAR, max_length=65535),
│       FieldSchema(name="metadata_", dtype=JSON),
│       FieldSchema(name="embedding",
│           dtype=FLOAT16_VECTOR if use_halfvec else FLOAT_VECTOR, dim=embed_dim),
│   ]
│   schema = CollectionSchema(name=collection_name, fields=fields)
│
├── 5. 创建/获取 Collection（has_collection 容错）
│   try:
│       exists = has_collection(collection_name)
│   except Exception:
│       exists = False
│   if exists:
│       _collection = Collection(collection_name)
│   else:
│       _collection = Collection(collection_name, schema=schema)
│       try:
│           _collection.create()
│       except Exception as e:
│           if "already exists" not in str(e):
│               raise
│           _collection = Collection(collection_name)
│
├── 6. auto-id sequence 补丁
│   _ensure_auto_id_sequence(collection_name)
│   # 手动创建 SEQUENCE + ALTER TABLE SET DEFAULT
│
├── 7. 创建 HNSW 索引（若 hnsw_kwargs 非空）
│   m = hnsw_kwargs.get("m", 16)
│   ef_construction = hnsw_kwargs.get("ef_construction", 64)
│   _collection.create_index(
│       field_name="embedding",
│       index_params=IndexParams.graph_index(m=m, ef_construction=ef_construction)
│   )
│
├── 8. 创建全文索引（若 hybrid_search=True）(D-05)
│   _collection.create_index(
│       field_name="text",
│       index_params=IndexParams.fulltext_index(
│           dictionary=text_search_config, algorithm="BM25"
│       )
│   )
│
├── 9. 创建异步 Collection
│   _async_collection = AsyncCollection(collection_name)
│
└── 10. 标记完成
    _is_initialized = True
```

### 错误处理

初始化阶段的错误由 `initialization_fail_on_error` 控制 (D-03)：
- `True`（严格模式）：任何初始化失败直接抛出异常
- `False`（宽松模式，默认）：`_logger.warning()` 记录后继续，首次操作时可能失败

---

## 6. Core Methods

### 6.1 add / async_add — 批量写入节点

```
add(nodes: Sequence[BaseNode]) → List[str]
│
├── _initialize()（若未初始化）
├── 对每个 node 提取:
│   ├── node_id     → node.node_id
│   ├── embedding   → node.get_embedding()
│   ├── text        → node.get_content(metadata_mode=NONE)
│   └── metadata_   → node_to_metadata_dict(node, remove_text=True, flat_metadata=False)
│
├── data_list = [{"node_id": ..., "text": ..., "metadata_": ..., "embedding": [...]}, ...]
├── self._collection.insert(data_list)
└── return [node.node_id for node in nodes]
```

**async_add** 使用 `await self._async_collection.insert(data_list)`。

**与上游差异**: PGVectorStore 逐条 `session.add()` + 统一 `commit()` → pyvastbase `col.insert()` 一次请求批量写入。

### 6.2 delete / adelete — 按 ref_doc_id 删除

```
delete(ref_doc_id: str) → None
│
├── _initialize()
├── expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
├── self._collection.delete(expr=expr)
└── return None
```

**adelete** 使用 `await self._async_collection.delete(expr=expr)`。

### 6.3 delete_nodes / adelete_nodes — 按条件删除

```
delete_nodes(node_ids=None, filters=None) → None
│
├── if not node_ids and not filters: return
├── _initialize()
├── 构建 expr:
│   ├── node_ids 部分: "node_id IN ('id1', 'id2', ...)"
│   ├── filters 部分: _build_expr(filters)（见 Section 7）
│   └── 两者 AND 连接
│
├── 检查是否含 _CLIENT_SIDE_OPERATORS（ANY / ALL / CONTAINS）
│   ├── 若含客户端操作符:
│   │   ├── 先用服务端可处理的 expr 查询候选集
│   │   ├── 在 Python 内存中过滤
│   │   └── 按过滤后的 node_id 列表执行 delete
│   └── 若不含: 直接 col.delete(expr=expr)
```

### 6.4 get_nodes / aget_nodes — 读取节点

```
get_nodes(node_ids=None, filters=None) → List[BaseNode]
│
├── assert node_ids or filters, "Must provide node_ids or filters"
├── _initialize()
├── 构建 expr + output_fields=["node_id", "text", "metadata_", "embedding"]
│
├── 检查客户端操作符回退（同 6.3）
│
├── rows = col.query(expr=expr, output_fields=output_fields)
├── 对每行:
│   ├── metadata → TextNode 反序列化
│   ├── text → set_content
│   └── embedding → node.embedding
└── return List[TextNode]
```

### 6.5 clear / aclear — 清空集合

```
clear() → None
│
├── _initialize()
├── self._collection.truncate()
└── return None
```

### 6.6 query / aquery — 按模式分发

```
query(query: VectorStoreQuery) → VectorStoreQueryResult
│
├── _initialize()
├── filter_expr = _build_expr(query.filters) if query.filters else None
├── output_fields = ["node_id", "text", "metadata_"]
│
├── match query.mode:
│   │
│   ├── DEFAULT (向量余弦相似度):
│   │   results = col.search(
│   │       data=[query.query_embedding],
│   │       anns_field="embedding",
│   │       param={"metric_type": "COSINE"},
│   │       limit=query.similarity_top_k,
│   │       expr=filter_expr,
│   │       output_fields=output_fields,
│   │   )
│   │   similarities = [1.0 - hit.distance for hit in results[0]]
│   │
│   ├── SPARSE / TEXT_SEARCH (全文检索, D-05):
│   │   要求 hybrid_search=True
│   │   if not hybrid_search:
│   │       raise ValueError("TEXT_SEARCH requires hybrid_search=True")
│   │   results = _fulltext_search(query.query_str, limit, filter_expr)
│   │   similarities = [hit.score for hit in results]
│   │
│   ├── HYBRID (向量 + 全文, D-08):
│   │   dense_req = AnnSearchRequest(
│   │       data=query.query_embedding,
│   │       anns_field="embedding",
│   │       param={"metric_type": "COSINE"},
│   │       limit=query.similarity_top_k,
│   │   )
│   │   sparse_req = AnnSearchRequest(
│   │       data=query.query_str,
│   │       anns_field="text",
│   │       param={"metric_type": "BM25"},
│   │       limit=query.sparse_top_k or query.similarity_top_k,
│   │   )
│   │   results = col.hybrid_search(
│   │       reqs=[dense_req, sparse_req],
│   │       rerank=RRFRanker(k=60),
│   │       limit=query.hybrid_top_k or query.similarity_top_k,
│   │   )
│   │   similarities = [hit.score for hit in results[0]]
│   │
│   └── MMR (最大边际相关性重排):
│       raise ValueError(
│           "MMR mode is not supported. "
│           "This matches upstream PGVectorStore behavior."
│       )
│
├── 组装 nodes: 从 results 反序列化 TextNode
└── return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
```

### 6.7 close — 清理连接

```python
async def close(self) -> None:
    if not self._is_initialized:
        return
    remove_connection(self._connection_alias)
    self._is_initialized = False
```

---

## 7. Filter Translation

### 7.1 操作符映射表

| FilterOperator | pyvastbase expr 操作符 | 执行方式 |
|---------------|----------------------|---------|
| `EQ` | `=` | 服务端 |
| `NE` | `!=` | 服务端 |
| `GT` | `>` | 服务端 |
| `GTE` | `>=` | 服务端 |
| `LT` | `<` | 服务端 |
| `LTE` | `<=` | 服务端 |
| `IN` | `IN (...)` | 服务端 |
| `NIN` | `NOT IN (...)` | 服务端 |
| `TEXT_MATCH` | `LIKE '%value%'` | 服务端 |
| `TEXT_MATCH_INSENSITIVE` | `ILIKE '%value%'` | 服务端 |
| `IS_EMPTY` | `IS NULL` | 服务端 |
| `CONTAINS` | `@>` (JSONB) | 客户端回退 (D-06) |
| `ANY` | `?\|` (JSONB) | 客户端回退 (D-06) |
| `ALL` | `?&` (JSONB) | 客户端回退 (D-06) |

### 7.2 _CLIENT_SIDE_OPERATORS

ANY (`?\|`), ALL (`?&`), CONTAINS (`@>`) 标记为客户端操作符 (D-06)。原因：`?` 与 pyvastbase 命名参数占位符冲突，无法在 `expr` 字符串中安全使用。

回退策略：
1. 用服务端可处理的 expr 查询候选集（宽松过滤）
2. 在 Python 内存中精确过滤
3. 返回过滤后的结果

### 7.3 _build_filter_clause()

```python
def _build_filter_clause(filter_: MetadataFilter) -> str:
    op = _to_vb_operator(filter_.operator)
    key = filter_.key

    # IN / NOT IN → 值列表
    if filter_.operator in (IN, NIN):
        values = ", ".join(f"'{v}'" for v in filter_.value)
        return f"metadata_->>'{key}' {op} ({values})"

    # TEXT_MATCH / ILIKE → % 通配符包裹
    if filter_.operator in (TEXT_MATCH, TEXT_MATCH_INSENSITIVE):
        return f"metadata_->>'{key}' {op} '%{filter_.value}%'"

    # IS_EMPTY → IS NULL
    if filter_.operator == IS_EMPTY:
        return f"metadata_->>'{key}' {op}"

    # 数值比较 → CAST float
    try:
        float(filter_.value)
        return f"(metadata_->>'{key}')::float {op} {filter_.value}"
    except (ValueError, TypeError):
        return f"metadata_->>'{key}' {op} '{filter_.value}'"
```

### 7.4 _build_expr() — 递归 AND/OR

```python
def _build_expr(filters: MetadataFilters) -> str:
    clauses = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            clauses.append(f"({_build_expr(f)})")
        else:
            clauses.append(_build_filter_clause(f))
    condition = f" {filters.condition.upper()} "
    return condition.join(clauses)
```

**返回 `str`** — pyvastbase `expr` 参数接受原生 SQL 字符串，不需要 SQLAlchemy `text()` 对象。

---

## 8. Async API

每个同步方法对应一个 async 版本，通过 `AsyncCollection` 实现：

| 同步方法 | 异步方法 | pyvastbase API |
|---------|---------|---------------|
| `add()` | `async_add()` | `await async_col.insert()` |
| `delete()` | `adelete()` | `await async_col.delete()` |
| `delete_nodes()` | `adelete_nodes()` | `await async_col.delete()` |
| `get_nodes()` | `aget_nodes()` | `await async_col.query()` |
| `clear()` | `aclear()` | `await async_col.truncate()` |
| `query()` | `aquery()` | `await async_col.search()` / `hybrid_search()` |

### 与 PGVectorStore 异步差异

| 维度 | PGVectorStore (asyncpg) | VastbaseVectorStore (AsyncCollection) |
|------|------------------------|--------------------------------------|
| 底层驱动 | asyncpg + sqlalchemy AsyncEngine | psycopg 3 async 连接池 |
| HYBRID 并发 | `asyncio.gather(dense, sparse)` 手动并发 | `col.hybrid_search()` SDK 内部处理 |
| MMR | 异步预取 embedding → 同步 MMR 重排 | raise ValueError（不支持） |
| Session 参数 | `await session.execute(SET hnsw.ef_search)` | search `param={"ef": N}` 传递 |

---

## 9. Error Handling

### 9.1 异常传播策略

操作级异常（add/query/delete/get_nodes）不额外捕获 — pyvastbase 异常直接向上传播：

| pyvastbase 异常 | 触发场景 |
|----------------|---------|
| `CollectionNotExistsError` | 操作不存在的 Collection |
| `VectorDimensionError` | 向量维度不匹配 |
| `ConnectionError` | 网络/认证问题 |
| `SchemaError` | Schema 定义不合法 |
| `DataError` | 插入数据格式错误 |
| `SearchError` | 搜索参数非法 |

### 9.2 初始化错误处理 (D-03)

由 `initialization_fail_on_error` 参数控制：
- `True`（严格模式）：任何初始化失败直接抛出异常
- `False`（宽松模式，默认）：`_logger.warning()` 记录后继续

### 9.3 TEXT_SEARCH 无索引

当 `query.mode=TEXT_SEARCH` 但 `hybrid_search=False` 时：

```python
raise ValueError(
    "TEXT_SEARCH mode requires hybrid_search=True. "
    "Set hybrid_search=True and perform_setup=True to enable fulltext indexing."
)
```

### 9.4 MMR 不支持

当 `query.mode=MMR` 时：

```python
raise ValueError("MMR mode is not supported by VastbaseVectorStore.")
```

与上游 PGVectorStore 行为一致。

---

## 10. Testing Strategy

### 10.1 三层测试体系

| 层级 | Gate | 文件 | 覆盖范围 | 依赖 |
|------|------|------|---------|------|
| 单元测试 | Gate 1 | `test_collection_init.py`, `test_crud.py`, `test_filter.py`, `test_search.py`, `test_async.py` | 全部方法和过滤操作符 | mock pyvastbase |
| 集成测试 | Gate 2 | `test_integration.py`, `test_framework_integration.py` | E2E 全链路 | 真实 Vastbase |
| Demo 验收 | Gate 3 | `demo_llamaindex.py` | 8 场景端到端 | 真实 Vastbase |

### 10.2 单元测试 (Gate 1)

| 文件 | 测试数量 | 覆盖范围 |
|------|----------|---------|
| `test_collection_init.py` | ~9 | Collection 创建、schema 字段验证、HNSW 索引、halfvec 版本检测 |
| `test_crud.py` | ~15 | add/delete/get_nodes/clear 基础 CRUD |
| `test_filter.py` | ~21 | 14 种 FilterOperator 翻译 + 嵌套 AND/OR + 客户端过滤回退 |
| `test_search.py` | ~20 | 4 种查询模式 (DEFAULT/SPARSE/HYBRID/MMR) + 排序验证 |
| `test_async.py` | ~13 | async API 对等性 (async_add/aquery/adelete/aget_nodes/aclear) |

测试框架：pytest + pytest-asyncio (`asyncio_mode = "auto"`)

### 10.3 集成测试 (Gate 2)

连接 `172.16.105.107:15432/vastbase` (user: aidev, password: Vbase_123456)

验证点：
1. Collection 创建 + HNSW 索引创建
2. 向量写入 → node_id 返回
3. DEFAULT 向量搜索 → similarity 排序
4. TEXT_SEARCH 全文搜索 → BM25 排序
5. HYBRID 混合搜索 → RRF 融合
6. 元数据过滤 → 14 种操作符
7. delete / delete_nodes → 精确删除
8. clear → 全部清空
9. 异步方法功能等价
10. halfvec 支持 + auto-id sequence 补丁

### 10.4 Demo 验收 (Gate 3) — 8 场景

| # | 场景 | 验证目标 |
|---|------|---------|
| 1 | 连接初始化 | 环境变量连接 + VastbaseVectorStore 实例创建 |
| 2 | 文档向量摄入 (add) | TextNode + embedding + metadata 批量写入 |
| 3 | 向量相似性搜索 (DEFAULT) | 余弦相似度检索 top_k |
| 4 | 元数据过滤搜索 | MetadataFilter + FilterOperator |
| 5 | 全文搜索 (TEXT_SEARCH) | BM25 关键词检索 |
| 6 | 混合搜索 (HYBRID) | dense + sparse → RRF 融合 |
| 7 | MMR 重排 | raise ValueError 验证 |
| 8 | 删除与清空 | delete + delete_nodes + clear |

---

## 11. Differences from Reference (PGVectorStore)

| 维度 | PGVectorStore | VastbaseVectorStore | 影响 |
|------|-------------|-------------------|------|
| **ORM** | 动态 SQLAlchemy 模型 (`get_data_model()`) | `CollectionSchema` + `FieldSchema` | 代码量 -53% |
| **连接** | `create_engine()` + `sessionmaker()` 双引擎 | `connect()` 全局连接 + alias 隔离 | 简化连接管理 |
| **向量扩展** | `CREATE EXTENSION IF NOT EXISTS vector` | 无需 | 减少 setup |
| **向量类型** | pgvector `Vector(N)` / `HalfVec(N)` | `floatvector(N)` / `halfvector(N)` 原生 | 类型名不同 |
| **HNSW** | SQL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef)` | API 封装 |
| **全文搜索** | `to_tsvector` + `ts_rank` + GIN | `IndexParams.fulltext_index(BM25)` | 算法等价 (D-05) |
| **混合搜索** | 两次独立查询 + `_dedup_results()` | `col.hybrid_search(RRFRanker)` | RRF 融合 (D-08) |
| **MMR** | Prefetch → MMR rerank → fallback | raise ValueError | 功能删减 |
| **Session 参数** | `SET hnsw.ef_search` SQL | search `param={"ef": N}` | API 简化 |
| **批量写入** | for `session.add()` → `commit()` | `col.insert(data_list)` 一次请求 | 性能更优 |
| **表名** | `data_{table_name}` (动态前缀) | 直接 `collection_name` | 命名简化 |
| **customize_query_fn** | 支持自定义 SQL 注入 | 不支持 | 功能删减 |
| **依赖数量** | 5 个包 | 2 个包 | 减少 60% |
| **代码行数** | ~1698 行 | 预计 ~800 行 | 维护成本 -53% |

---

## 12. Future Work

- **customize_query_fn 等效支持**: 若 pyvastbase 未来提供查询定制钩子，补充实现
- **metadata_indices 自动化**: 若 pyvastbase 提供标量字段 `create_index` 支持，简化元数据索引
- **分区/多租户**: 若 Vastbase 支持分区策略，对标上游 schema_name 多租户
- **MMR 支持**: 若 pyvastbase 提供 MMR rerank API，补充 MMR 模式
- **连接池调优**: 根据生产负载调优 `pool_size` / `max_overflow`
- **JSONB 操作符服务端化**: 若 pyvastbase 修复 `?` 占位符冲突，将 ANY/ALL/CONTAINS 迁移到服务端
