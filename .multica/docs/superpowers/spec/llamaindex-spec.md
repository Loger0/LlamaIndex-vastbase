# LlamaIndex PGVectorStore → VastbaseVectorStore 适配 — Design Spec

**Date:** 2026-06-16
**Status:** Draft
**Issue:** [DRI-3](mention://issue/af067c60-269e-4e68-aaca-e26c9034d36b)
**Framework:** LlamaIndex v0.14.22
**Integration Mode:** standalone
**Coupling Level:** low

---

## 1. Overview

### 目标

将 LlamaIndex 的 PGVectorStore（`llama-index-vector-stores-postgres` v0.8.1）适配到 Vastbase，用 pyvastbase 的 PyMilvus-style API 替换 SQLAlchemy + psycopg2/asyncpg + pgvector 驱动栈，仅适配 vector-store 层。

### 非目标

- 不修改 llama-index-core 源码
- 不涉及 LlamaIndex 其他组件（如 ingestion pipeline、query engine）
- 不提供 manage.py 等 CLI 工具
- 不实现 PGVectorStore 未定义的额外功能

### 包信息

- **包名**: `llama-index-vector-stores-vastbase`
- **包版本**: 0.1.0
- **类名**: `VastbaseVectorStore(BasePydanticVectorStore)`
- **仓库**: `https://github.com/Loger0/LlamaIndex-vastbase.git`

---

## 2. Package Structure

```
llama-index-vector-stores-vastbase/
├── pyproject.toml
├── llama_index/
│   └── vector_stores/
│       └── vastbase/
│           ├── __init__.py          # 导出 VastbaseVectorStore
│           ├── base.py              # VastbaseVectorStore 核心实现 (~800 行)
│           └── utils.py             # filter 转换、结果转换工具 (~150 行)
├── tests/
│   ├── conftest.py                  # pytest fixtures
│   ├── test_vastbase_vector_store.py  # 单元测试（mock pyvastbase）
│   ├── test_filter_translation.py   # filter 转换专项测试
│   └── test_integration.py          # 集成测试（真实 Vastbase）
├── framework-tests/
│   ├── conftest.py
│   └── test_vector_stores_postgres.py  # LlamaIndex 官方测试对标
└── demo/
    └── llamaindex_vastbase_demo.py  # 8 场景端到端 Demo
```

### pyproject.toml

```toml
[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
requires-python = ">=3.9,<4.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.0",
]
```

---

## 3. Class Design

### 3.1 Pydantic 字段（序列化/反序列化）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stores_text` | `bool` | `True` | 声明向量存储支持文本存储 |
| `flat_metadata` | `bool` | `False` | 是否扁平化元数据 |
| `uri` | `Optional[str]` | `None` | 统一连接字符串（可选） |
| `host` | `str` | `"localhost"` | Vastbase 主机地址 |
| `port` | `int` | `15432` | Vastbase 端口 |
| `database` | `str` | `"vastbase"` | 数据库名 |
| `user` | `str` | `"aidev"` | 用户名 |
| `password` | `str` | `"Vbase_123456"` | 密码 |
| `collection_name` | `str` | `"llamaindex"` | Collection 名 |
| `embed_dim` | `int` | `1536` | 向量维度 |
| `use_halfvec` | `bool` | `False` | 启用半精度向量（需 Vastbase 3.0.9+） |
| `use_jsonb` | `bool` | `False` | 保留参数，Vastbase JSONB 原生支持 |
| `hybrid_search` | `bool` | `False` | 启用混合搜索 |
| `text_search_config` | `str` | `"english"` | 全文分词器 |
| `hnsw_kwargs` | `Optional[Dict]` | `None` | HNSW 索引参数 |
| `perform_setup` | `bool` | `True` | 自动创建 Collection + 索引 |
| `debug` | `bool` | `False` | 调试模式 |
| `indexed_metadata_keys` | `Optional[Set[Tuple[str, str]]]` | `None` | 需索引的元数据键 |
| `initialization_fail_on_error` | `bool` | `False` | 初始化失败时抛出异常 |

### 3.2 私有属性

| 字段 | 类型 | 说明 |
|------|------|------|
| `_connection_alias` | `str` | pyvastbase 连接别名 |
| `_collection` | `Collection` | pyvastbase 同步 Collection 实例 |
| `_async_collection` | `AsyncCollection` | pyvastbase 异步 Collection 实例 |
| `_is_initialized` | `bool` | 初始化标志 |

### 3.3 构造函数

```python
def __init__(
    self,
    uri: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 15432,
    database: str = "vastbase",
    user: str = "aidev",
    password: str = "Vbase_123456",
    collection_name: str = "llamaindex",
    embed_dim: int = 1536,
    use_halfvec: bool = False,
    use_jsonb: bool = False,
    hybrid_search: bool = False,
    text_search_config: str = "english",
    hnsw_kwargs: Optional[Dict[str, Any]] = None,
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None,
    perform_setup: bool = True,
    debug: bool = False,
    initialization_fail_on_error: bool = False,
) -> None:
```

### 3.4 from_params 类方法

与上游 `PGVectorStore.from_params()` 签名一致：

```python
@classmethod
def from_params(
    cls,
    host: Optional[str] = None,
    port: Optional[str] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    collection_name: str = "llamaindex",
    embed_dim: int = 1536,
    use_halfvec: bool = False,
    use_jsonb: bool = False,
    hybrid_search: bool = False,
    text_search_config: str = "english",
    hnsw_kwargs: Optional[Dict[str, Any]] = None,
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None,
    perform_setup: bool = True,
    debug: bool = False,
) -> "VastbaseVectorStore":
```

### 3.5 client 属性

```python
@property
def client(self) -> Any:
    """返回已初始化的 Collection 实例。"""
    if not self._is_initialized:
        return None
    return self._collection
```

---

## 4. Collection Initialization Flow

```
_initialize()
│
├── 1. 生成连接 alias
│   _connection_alias = f"vb_{collection_name}_{id(self)}"
│
├── 2. 建立连接
│   connect(host, port, database, user, password, alias=_connection_alias)
│
├── 3. 版本检测（若 use_halfvec=True）
│   ver = get_vb_version()
│   if not check_vb_version((3, 0, 9)):
│       raise VectorDimensionError(
│           f"use_halfvec requires Vastbase >= 3.0.9, "
│           f"got {ver['major']}.{ver['minor']}.{ver['patch']}"
│       )
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
├── 5. 创建/获取 Collection
│   if has_collection(collection_name):
│       self._collection = Collection(collection_name)
│   else:
│       self._collection = Collection(collection_name, schema=schema)
│       self._collection.create()
│
├── 6. 创建 HNSW 索引（若 hnsw_kwargs 非空）
│   m = hnsw_kwargs.pop("hnsw_m", 16)
│   ef_construction = hnsw_kwargs.pop("hnsw_ef_construction", 64)
│   self._collection.create_index(
│       field_name="embedding",
│       index_params=IndexParams.graph_index(m=m, ef_construction=ef_construction)
│   )
│
├── 7. 创建全文索引（若 hybrid_search=True）
│   self._collection.create_index(
│       field_name="text",
│       index_params=IndexParams.fulltext_index(
│           dictionary=text_search_config,
│           algorithm="BM25"
│       )
│   )
│
├── 8. 创建异步 Collection
│   self._async_collection = AsyncCollection(collection_name)
│
└── 9. 标记完成
    self._is_initialized = True
```

### 与 PGVectorStore 初始化差异

| 步骤 | PGVectorStore | VastbaseVectorStore |
|------|-------------|-------------------|
| 连接 | `create_engine()` + 双 `sessionmaker()` | `connect()` 全局连接（实例 alias 隔离） |
| Schema | `_create_schema_if_not_exists()` 检查 `information_schema.schemata` | pyvastbase `CollectionSchema` 创建即管理 |
| 扩展 | `CREATE EXTENSION IF NOT EXISTS vector` | 无需 — Vastbase 原生 floatvector/halfvector |
| 建表 | `_table_class.__table__.create(checkfirst=True)` | `col.create()` |
| HNSW | 原始 SQL `CREATE INDEX ... USING hnsw ... WITH (...)` | `col.create_index(IndexParams.graph_index(...))` |
| 全文 | tsvector 生成列 `Computed("to_tsvector(...)")` | `IndexParams.fulltext_index(algorithm="BM25")` |
| 异步 | `create_async_engine()` + `AsyncSession` | `AsyncCollection(collection_name)` |

---

## 5. Core Methods

### 5.1 add / async_add — 批量写入节点

```
add(nodes: List[BaseNode]) → List[str]
│
├── self._initialize()
├── 对每个 node 提取:
│   ├── node_id     → node.node_id
│   ├── embedding   → node.get_embedding()
│   ├── text        → node.get_content(metadata_mode=NONE)
│   └── metadata_   → node_to_metadata_dict(node, remove_text=True, flat_metadata=False)
│
├── 构造 data_list = [{"node_id": ..., "text": ..., "metadata_": ..., "embedding": [...]}, ...]
├── result = self._collection.insert(data_list)
└── return [node.node_id for node in nodes]
```

**对应上游**: 逐条 `session.add()` + `session.commit()` → pyvastbase `col.insert()` 一次请求批量写入。

### 5.2 delete / adelete — 按 ref_doc_id 删除

```
delete(ref_doc_id: str) → None
│
├── self._initialize()
├── expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
├── self._collection.delete(expr=expr)
└── return None
```

### 5.3 delete_nodes / adelete_nodes — 按条件删除

```
delete_nodes(node_ids=None, filters=None) → None
│
├── if not node_ids and not filters: return
├── self._initialize()
├── 构建 expr 子句:
│   ├── 若有 node_ids: f"node_id IN ('id1', 'id2', ...)"
│   ├── 若有 filters: _build_expr(filters)  (见 Section 7)
│   └── 两者都有: 用 AND 连接
├── self._collection.delete(expr=combined_expr)
```

### 5.4 get_nodes / aget_nodes — 读取节点

```
get_nodes(node_ids=None, filters=None) → List[BaseNode]
│
├── assert node_ids or filters
├── self._initialize()
├── 构建查询:
│   ├── 若有 node_ids: col.query(expr=f"node_id IN (...)", output_fields=[...])
│   ├── 若有 filters: col.query(expr=_build_expr(filters), output_fields=[...])
│   └── 两者都有: 合并 expr
├── 对每行结果:
│   └── metadata_dict_to_node(metadata) + set_content(text) + embedding
└── return nodes
```

### 5.5 clear / aclear — 清空集合

```
clear() → None
│
├── self._initialize()
├── self._collection.truncate()
└── return None
```

### 5.6 query / aquery — 按模式分发

```
query(query: VectorStoreQuery) → VectorStoreQueryResult
│
├── self._initialize()
├── filter_expr = _build_expr(query.filters) if query.filters else None
├── output_fields = ["node_id", "text", "metadata_"]
│
├── switch query.mode:
│   │
│   ├── DEFAULT:
│   │   results = col.search(
│   │       data=[query.query_embedding],
│   │       anns_field="embedding",
│   │       param={"metric_type": "COSINE"},
│   │       limit=query.similarity_top_k,
│   │       expr=filter_expr,
│   │       output_fields=output_fields,
│   │   )
│   │   similarities = [1 - r.distance for r in results[0]]
│   │
│   ├── SPARSE / TEXT_SEARCH:
│   │   要求 hybrid_search=True 且全文索引已就绪
│   │   results = _fulltext_search(query.query_str, limit, filter_expr)
│   │   similarities = [r.score for r in results]
│   │
│   ├── HYBRID:
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
│   │   similarities = [r.score for r in results[0]]
│   │
│   └── MMR:
│       prefetch_k = max(similarity_top_k * 4, similarity_top_k)
│       candidates = col.search(data=[query.query_embedding],
│           anns_field="embedding", limit=prefetch_k,
│           output_fields=["node_id", "text", "metadata_", "embedding"])
│       提取 embedding 列表
│       mmr_ids, mmr_sims = get_top_k_mmr_embeddings(
│           query_embedding, embeddings, similarity_top_k,
│           node_ids, mmr_threshold
│       )
│
└── return VectorStoreQueryResult(nodes=..., similarities=..., ids=...)
```

### 5.7 close — 清理连接

```python
async def close(self) -> None:
    if not self._is_initialized:
        return
    remove_connection(self._connection_alias)
    self._is_initialized = False
```

---

## 6. Async API

每个同步方法对应一个 `async` 版本，通过 pyvastbase 的 `AsyncCollection` 实现：

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
| MMR | 异步预取 embedding → 同步 MMR 重排 | 相同模式 |
| Session 参数 | `await session.execute(SET hnsw.ef_search)` | 搜索时通过 `param={"ef": N}` 传递 |

---

## 7. Filter Translation

### 7.1 操作符映射表

`_to_vb_operator()` 与上游 `_to_postgres_operator()` 完全一致 — Vastbase 基于 PG 15 内核：

| FilterOperator | SQL 操作符 | PG 原生 | Vastbase |
|---------------|-----------|--------|---------|
| `EQ` | `=` | ✅ | ✅ |
| `GT` | `>` | ✅ | ✅ |
| `LT` | `<` | ✅ | ✅ |
| `NE` | `!=` | ✅ | ✅ |
| `GTE` | `>=` | ✅ | ✅ |
| `LTE` | `<=` | ✅ | ✅ |
| `IN` | `IN` | ✅ | ✅ |
| `NIN` | `NOT IN` | ✅ | ✅ |
| `CONTAINS` | `@>` (JSONB) | ✅ | ✅ |
| `TEXT_MATCH` | `LIKE` | ✅ | ✅ |
| `TEXT_MATCH_INSENSITIVE` | `ILIKE` | ✅ | ✅ |
| `IS_EMPTY` | `IS NULL` | ✅ | ✅ |
| `ANY` | `?\|` (JSONB) | ✅ | 集成测试验证 |
| `ALL` | `?&` (JSONB) | ✅ | 集成测试验证 |

### 7.2 _build_filter_clause() 分派逻辑

```python
def _build_filter_clause(filter_: MetadataFilter) -> str:
    """单个 MetadataFilter → pyvastbase expr 字符串片段。"""
    op = _to_vb_operator(filter_.operator)
    key = filter_.key

    # IN / NOT IN → 值列表
    if filter_.operator in (IN, NIN):
        values = ", ".join(f"'{v}'" for v in filter_.value)
        return f"metadata_->>'{key}' {op} ({values})"

    # ANY / ALL → JSONB text[] 操作符
    if filter_.operator in (ANY, ALL):
        values = ", ".join(f"'{v}'" for v in filter_.value)
        return f"metadata_::jsonb->'{key}' {op} array[{values}]"

    # CONTAINS → JSONB 包含
    if filter_.operator == CONTAINS:
        return f"metadata_::jsonb->'{key}' {op} '[\"{filter_.value}\"]'"

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
    except ValueError:
        return f"metadata_->>'{key}' {op} '{filter_.value}'"
```

### 7.3 _build_expr() — 递归 AND/OR

```python
def _build_expr(filters: MetadataFilters) -> str:
    """递归构建 pyvastbase expr 字符串，支持嵌套 AND/OR。"""
    clauses = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            clauses.append(f"({_build_expr(f)})")
        else:
            clauses.append(_build_filter_clause(f))

    condition = f" {filters.condition.upper()} "  # " AND " 或 " OR "
    return condition.join(clauses)
```

**与上游关键差异**: 返回 `str` 而非 SQLAlchemy `text()` 对象 — pyvastbase 的 `expr` 参数接受原生 SQL 字符串。

---

## 8. Error Handling

### 8.1 异常传播策略

**操作级异常**（add/query/delete/get_nodes）不额外捕获 — pyvastbase 异常直接向上传播：

| pyvastbase 异常 | 触发场景 |
|----------------|---------|
| `CollectionNotExistsError` | 操作不存在的 Collection |
| `VectorDimensionError` | 向量维度不匹配 |
| `ConnectionError` | 网络/认证问题 |
| `SchemaError` | Schema 定义不合法 |
| `DataError` | 插入数据格式错误 |
| `SearchError` | 搜索参数非法 |

### 8.2 初始化错误处理

对标上游 `_initialize()` 模式，由 `initialization_fail_on_error` 控制：

- `True`（严格模式）: 任何初始化失败直接抛出异常
- `False`（宽松模式，默认）: `_logger.warning()` 记录后继续，首次操作时可能失败

### 8.3 TEXT_SEARCH 无索引

当 `query.mode=TEXT_SEARCH` 但 `hybrid_search=False` 或全文索引未就绪时，抛出明确错误：

```python
raise ValueError(
    "TEXT_SEARCH mode requires hybrid_search=True. "
    "Set hybrid_search=True and perform_setup=True to enable fulltext indexing."
)
```

---

## 9. Differences from Reference (PGVectorStore)

| 维度 | PGVectorStore | VastbaseVectorStore | 影响 |
|------|-------------|-------------------|------|
| **ORM** | 动态 SQLAlchemy 模型 (`get_data_model()`) | `CollectionSchema` + `FieldSchema` 声明式 | 代码量 -60% |
| **连接** | `create_engine()` + `sessionmaker()` 双引擎 | `connect()` 全局连接 + alias 隔离 | 连接管理简化 |
| **向量扩展** | `CREATE EXTENSION IF NOT EXISTS vector` | 无需 | 减少 setup 步骤 |
| **向量类型** | pgvector `Vector(N)` / `HALFVEC(N)` | `floatvector(N)` / `halfvector(N)` 原生支持 | 类型名不同 |
| **HNSW 索引** | 原始 SQL `CREATE INDEX ... USING hnsw ... WITH (...)` | `IndexParams.graph_index(m, ef_construction)` | API 封装 |
| **全文搜索** | `to_tsvector()` + `ts_rank()` + GIN 索引 | `IndexParams.fulltext_index(algorithm="BM25")` | 算法等价 |
| **混合搜索** | 两次独立查询 + `_dedup_results()` 去重 | `col.hybrid_search(RRFRanker)` RRF 融合 | RRF 排序更优 |
| **Session 参数** | `SET ivfflat.probes` / `SET hnsw.ef_search` / `SET LOCAL enable_bitmapscan` | pyvastbase search `param={"ef": N}` 传递 | API 简化 |
| **delete** | `session.execute(delete(...))` | `col.delete(expr=...)` | API 简化 |
| **clear** | `session.execute(delete(full_table))` | `col.truncate()` | API 简化 |
| **批量写入** | for 循环 `session.add()` → 统一 `commit()` | `col.insert(data_list)` 一次请求 | 性能更优 |
| **表名** | `data_{table_name}` (动态前缀) | 直接使用 `collection_name` | 命名更简洁 |
| **customize_query_fn** | 支持自定义 SQL 子句注入 | 不支持（pyvastbase 无对应概念） | 功能删减 |
| **metadata_indices** | BTREE + GIN 索引 | 通过 pyvastbase 原始 SQL 创建 | 需额外适配 |
| **代码行数** | ~1698 行 | 预计 ~800 行 | 维护成本 -53% |

---

## 10. Testing Strategy

### 10.1 测试分层

| 层级 | 文件 | 覆盖范围 | 依赖 |
|------|------|---------|------|
| **单元测试** | `test_vastbase_vector_store.py` | 类构造、字段验证、方法签名 | mock pyvastbase |
| **Filter 测试** | `test_filter_translation.py` | 14 种操作符映射、嵌套 AND/OR | 无 |
| **集成测试** | `test_integration.py` | Collection CRUD、向量搜索、元数据过滤 | 真实 Vastbase 连接 |
| **框架测试** | `framework-tests/test_vector_stores_postgres.py` | VectorStoreIndex + VastbaseVectorStore 端到端 | 真实 Vastbase 连接 |

### 10.2 集成测试验证点

1. Collection 创建：`VastbaseVectorStore(perform_setup=True)` 成功创建 Collection + 索引
2. 向量写入：`add(nodes)` 返回正确的 node_id 列表
3. 向量搜索 DEFAULT：`query(embedding)` 返回 top_k 结果，similarity 正确
4. 全文搜索 TEXT_SEARCH：`query(query_str, mode=TEXT_SEARCH)` 返回关键词匹配
5. 混合搜索 HYBRID：`query(embedding + query_str, mode=HYBRID)` RRF 融合
6. MMR 重排：`query(embedding, mode=MMR)` 多样性重排
7. 元数据过滤：14 种操作符逐一验证
8. 删除：`delete(ref_doc_id)` / `delete_nodes()` 正确删除
9. 清空：`clear()` 清空所有数据
10. 异步：所有 async 方法功能等价

### 10.3 Demo 验证场景（8 场景）

```
1. 连接初始化 — 通过环境变量连接 Vastbase 并创建 VastbaseVectorStore 实例
2. 文档向量摄入 (add) — 创建 TextNode（含 embedding + metadata），批量写入
3. 向量相似性搜索 (DEFAULT) — 余弦相似度检索，返回 top_k 结果
4. 元数据过滤搜索 — 用 MetadataFilter + FilterOperator 过滤结果
5. 全文搜索 (TEXT_SEARCH, BM25) — Vastbase 全文索引关键词检索
6. 混合搜索 (HYBRID, RRF 融合) — dense ANN + fulltext BM25，RRF 重排合并
7. MMR 最大边际相关性重排 — query.mode=MMR 多样性重排
8. 删除与清空生命周期 — delete + delete_nodes + clear 完整验证
```

---

## 11. Future Work

- **customize_query_fn 等效支持**: 若 pyvastbase 未来提供查询定制钩子，补充实现
- **metadata_indices 自动化**: 若 pyvastbase 提供 `create_index` 的标量类型支持，简化元数据索引创建
- **分区/多租户**: 若未来 Vastbase 支持表分区策略，考虑对标上游的 schema_name 多租户模式
- **连接池调优**: 根据生产负载调优 pyvastbase `pool_size` / `max_overflow` 参数
