# LlamaIndex Vastbase 适配 — Design Spec

**Date:** 2026-06-17
**Status:** Draft
**Integration Mode:** standalone
**pyvastbase Version:** >=0.2.0
**LlamaIndex Core Version:** >=0.13.0, <0.15

---

## 1. Overview（目标 + 非目标）

### 目标

为 LlamaIndex 开发 `llama-index-vector-stores-vastbase` 独立 pip 包，提供 `VastbaseVectorStore` 类，继承 `BasePydanticVectorStore`，使用 pyvastbase 0.2.0 替代上游 PGVectorStore 的 SQLAlchemy + psycopg2/asyncpg + pgvector 驱动栈，支持 DEFAULT / SPARSE / TEXT_SEARCH / HYBRID 四种查询模式。

### 非目标

- **不实现 MMR 查询模式** — 与上游 PGVectorStore 行为一致，遇 MMR 抛 `ValueError`
- **不修改 LlamaIndex 核心源码** — standalone 模式，独立发布
- **不直接依赖 SQLAlchemy / psycopg2 / asyncpg / pgvector** — 全部由 pyvastbase 替代
- **不保证 Vastbase PG 全文搜索函数（to_tsvector / to_tsquery / ts_rank）可用** — 采用客户端降级策略

### 关键设计决策

| ID | 决策 | 依据 |
|----|------|------|
| D-01 | 同步 + 异步对等实现 | 上游 PGVectorStore 每个方法均有 sync/async 版本 |
| D-02 | 全部替换为 pyvastbase >= 0.2.0 | 移除所有 PG 生态驱动依赖 |
| D-03 | 混合错误处理（fail_on_error 可配置） | 上游通过 add_kwargs 透传 |
| D-04 | MMR 不实现，抛 ValueError | 与上游 PGVectorStore 一致 |
| D-05 | 客户端降级策略 | Vastbase PG 兼容性不足时自动 fallback |
| D-06 | HYBRID 模式使用 pyvastbase 原生 hybrid_search + RRF | 结果质量优于上游的简单去重 |
| D-07 | 单文件实现（base.py） | 与上游 PGVectorStore 结构一致 |

---

## 2. Package Structure（文件树 + pyproject.toml）

### 文件树

```
llama-index-vector-stores-vastbase/
├── pyproject.toml
├── README.md
├── LICENSE
└── llama_index/
    └── vector_stores/
        └── vastbase/
            ├── __init__.py       # 导出 VastbaseVectorStore
            └── base.py           # VastbaseVectorStore 完整实现（~800-1000 行）
```

### pyproject.toml

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

[tool.hatch.build.targets.wheel]
packages = ["llama_index"]
```

---

## 3. pyvastbase API 映射

### 驱动栈替换总览

| 上游 PGVectorStore | pyvastbase 替代 |
|---|---|
| `sqlalchemy.create_engine(url)` | `VastbaseClient(host, port, user, password, database)` |
| `sqlalchemy.create_async_engine(url)` | `AsyncCollection` 上下文管理器 |
| `sessionmaker(engine)` | `VastbaseClient` 内置连接池 |
| `declarative_base()` + 动态 `type()` 建表 | `Collection(collection_name, schema).create()` |
| `Column(Vector(embed_dim))` | `FieldSchema(name, dtype=FLOAT_VECTOR, dim=embed_dim)` |
| `Column(HALFVEC(embed_dim))` | `FieldSchema(name, dtype=FLOAT16_VECTOR, dim=embed_dim)` |
| `CREATE EXTENSION vector` | 无需（Vastbase V3 原生支持） |
| `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef_construction)` + `col.create_index()` |
| `Session.execute(SET ivfflat.probes=...)` | `search(param={"metric_type", "ef"})` API 参数 |
| `pgvector.cosine_distance` | `col.search(param={"metric_type": "COSINE"})` |
| `to_tsvector` / `to_tsquery` / `ts_rank` / `@@` | 客户端 LIKE/ILIKE + RRF 降级（SPARSE）；pyvastbase native hybrid_search（HYBRID） |
| JSONB `->>` / `->` / `?|` / `?&` / `@>` | pyvastbase filter 表达式字符串 + 客户端数组过滤 fallback |
| `text()` SQL 片段 | pyvastbase expr 字符串 |
| `metadata_dict_to_node()` | 保留（llama-index-core 提供） |

### 核心 API 映射表

| LlamaIndex 操作 | pyvastbase API | 说明 |
|---|---|---|
| `add(nodes)` | `col.insert(rows)` | 批量插入节点 |
| `query(DEFAULT)` | `col.search(data, anns_field, limit, expr)` | COSINE 向量搜索 |
| `query(SPARSE/TEXT_SEARCH)` | `col.query(expr="text LIKE '%...%'", limit)` | 客户端 LIKE fallback |
| `query(HYBRID)` | `col.hybrid_search(reqs, rerank=RRFRanker(), limit)` | 原生混合搜索 |
| `query(MMR)` | `raise ValueError` | 与上游一致 |
| `delete(ref_doc_id)` | `col.delete(expr="metadata->>'ref_doc_id' = '...'")` | 按文档 ID 删除 |
| `delete_nodes(node_ids, filters)` | `col.delete(expr)` | 组合条件删除 |
| `get_nodes(node_ids, filters)` | `col.query(expr, output_fields)` | 按条件查询 |
| `clear()` | `col.truncate()` | 清空 collection |

---

## 4. Class Design（字段表 + 连接处理）

### 类定义

```python
class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store — replaces PGVectorStore's SQLAlchemy+pgvector stack."""

    stores_text: bool = True
    flat_metadata: bool = False

    # === Pydantic 公开字段 ===
    connection_params: Dict[str, Any]       # Vastbase 连接参数
    collection_name: str = "llamaindex"     # Collection 名（替代 table_name）
    embed_dim: int = 1536                   # 向量维度
    hnsw_kwargs: Optional[Dict[str, Any]]   # HNSW 参数：m, ef_construction, ef_search
    enable_sparse: bool = False             # 是否启用 SPARSE/TEXT_SEARCH 模式
    enable_hybrid: bool = False             # 是否启用 HYBRID 模式
    debug: bool = False                     # 调试日志
    perform_setup: bool = True              # 是否在首次操作时自动建 Collection

    # === PrivateAttr ===
    _client: Any = PrivateAttr()            # VastbaseClient 实例
    _collection: Any = PrivateAttr()        # Collection 实例
    _async_collection: Any = PrivateAttr()  # AsyncCollection 实例
    _is_initialized: bool = PrivateAttr(default=False)
    _hnsw_ef_search: int = PrivateAttr(default=100)
    _indexed_metadata_keys: Optional[Set[Tuple[str, str]]]  # (key, type) pairs
    _customize_query_fn: Optional[Callable] = PrivateAttr(default=None)

    # === 已废弃字段（保留兼容，不传值） ===
    # connection_string, async_connection_string, table_name, schema_name,
    # text_search_config, cache_ok, use_jsonb, create_engine_kwargs,
    # use_halfvec, engine, async_engine
```

### 连接方式

```python
# 方式 1：connection_params dict
VastbaseVectorStore(
    connection_params={
        "host": "172.16.105.107",
        "port": 15432,
        "database": "vastbase",
        "user": "aidev",
        "password": "Vbase_123456",
    },
    collection_name="my_docs",
    embed_dim=1536,
)

# 方式 2：from_params 工厂方法
VastbaseVectorStore.from_params(
    host="172.16.105.107",
    port=15432,
    database="vastbase",
    user="aidev",
    password="Vbase_123456",
    collection_name="my_docs",
    embed_dim=1536,
)
```

### 与 PGVectorStore 的关键差异

| 项目 | PGVectorStore | VastbaseVectorStore |
|---|---|---|
| 连接对象 | 两个 SQLAlchemy Engine（sync + async） | 一个 VastbaseClient（连接池） |
| 表/集合 | 动态 SQLAlchemy Model（`type()` 生成） | Collection 对象 |
| 向量字段 | `Column(Vector(dim))` | Collection Schema 中 `FLOAT_VECTOR` 字段 |
| 元数据字段 | JSONB `metadata_` 列 | Collection 中 `JSON` 字段 |
| 全文搜索列 | `TSVECTOR GENERATED ALWAYS AS ... STORED` | 不需要（客户端降级） |
| 自增 ID | `BIGINT PRIMARY KEY AUTOINCREMENT` | 可选，构造时不强制 |
| 会话参数 | `SET ivfflat.probes / SET hnsw.ef_search` | search() API 参数 |

---

## 5. Collection Initialization Flow

### 初始化流程

```
VastbaseVectorStore 首次操作
    │
    ▼
_initialize()
    │
    ├── 1. _connect()
    │       ├── VastbaseClient(host, port, user, password, database)
    │       │   → self._client
    │       └── 连接验证：health_check()
    │
    ├── 2. _ensure_collection()
    │       ├── has_collection(self.collection_name)?
    │       │   ├── YES → 加载已有 Collection
    │       │   │       ├── describe_collection() 检查 schema 兼容性
    │       │   │       └── 如果 embed_dim 不匹配 → 警告
    │       │   └── NO  → create_collection() 创建新 Collection
    │       │
    │       └── Collection Schema:
    │           - id: INT64 (is_primary=True, auto_increment)
    │           - embedding: FLOAT_VECTOR(dim=embed_dim)
    │           - node_id: VARCHAR(256)
    │           - ref_doc_id: VARCHAR(256)
    │           - text: TEXT
    │           - metadata_: JSON
    │
    └── 3. _create_indices()  [if hnsw_kwargs is not None]
            ├── IndexParams.graph_index(
            │       m=hnsw_kwargs.get("m", 16),
            │       ef_construction=hnsw_kwargs.get("ef_construction", 128),
            │   )
            ├── col.create_index("embedding", index_params)
            └── 存储 hnsw_ef_search 到 self._hnsw_ef_search
```

### Schema 字段定义

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
    FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text", dtype=DataType.TEXT),
    FieldSchema(name="metadata_", dtype=DataType.JSON),
]

# 稀疏搜索模式额外字段
if enable_sparse:
    fields.append(FieldSchema(name="text_tsv", dtype=DataType.TEXT))
```

### 初始化错误处理

每个步骤包装 try/except，记录警告日志。仅 `perform_setup=True` 时抛出初始化失败异常。

---

## 6. Core Methods（每个方法的数据流和伪代码）

### 6.1 add / async_add

```
add(nodes: List[BaseNode], **add_kwargs) -> List[str]
    │
    ├── _initialize()
    ├── fail_on_error = add_kwargs.get("fail_on_error", True)
    │
    ├── FOR each node in nodes:
    │       ├── 提取 node.embedding, node.node_id, node.ref_doc_id, node.text
    │       ├── 序列化 node.metadata → JSON string
    │       └── 构建 row dict:
    │           {"embedding": [...], "node_id": "...", "ref_doc_id": "...",
    │            "text": "...", "metadata_": {...}}
    │
    ├── 批量插入: col.insert(rows)
    │       ├── 成功 → 返回 [node_id, ...]
    │       └── 失败:
    │           ├── fail_on_error=True  → raise
    │           └── fail_on_error=False → 记录警告，跳过失败行，继续
    │
    └── RETURN List[str] (成功插入的 node_ids)
```

**对应上游方法:** `PGVectorStore.add()` (line 631) — 节点→ORM 对象→session.add_all()→commit

**变更:**
- 移除 SQLAlchemy ORM 对象构建
- 移除 session.add_all + commit 事务模型
- pyvastbase `col.insert()` 内部处理事务

### 6.2 query / aquery — DEFAULT 模式

```
query(query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult
    │
    ├── _initialize()
    │
    ├── SWITCH query.mode:
    │
    ├── case DEFAULT:
    │       │
    │       ├── 构建 filter expr: _build_filter_clause(query.filters)
    │       ├── 调用 _customize_query_fn (if set)
    │       ├── col.search(
    │       │       data=[query.query_embedding],
    │       │       anns_field="embedding",
    │       │       param={
    │       │           "metric_type": "COSINE",
    │       │           "ef": self._hnsw_ef_search,
    │       │       },
    │       │       limit=query.similarity_top_k,
    │       │       expr=filter_expr,
    │       │       output_fields=["node_id", "text", "metadata_"],
    │       │   )
    │       │
    │       ├── FOR each result:
    │       │       ├── distance → similarity: 1.0 - distance (if COSINE)
    │       │       ├── metadata_dict → BaseNode (via metadata_dict_to_node)
    │       │       └── node.embedding = result.get("embedding") (if in output)
    │       │
    │       └── RETURN VectorStoreQueryResult(
    │               nodes=[...], similarities=[...], ids=[...])
```

**对应上游方法:** `PGVectorStore._query_with_score()` (line 814) + `_build_query()` (line 792)

**变更:**
- 移除 `SET ivfflat.probes / SET hnsw.ef_search / SET LOCAL enable_bitmapscan` 会话参数
- `cosine_distance` → `metric_type="COSINE"` API 参数
- 移除 SQLAlchemy statement 构建，改为 pyvastbase search() API

### 6.3 query — SPARSE / TEXT_SEARCH 模式

```
query(query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult
    │
    ├── case SPARSE / TEXT_SEARCH:
    │       │
    │       ├── IF query.query_str is None → raise ValueError
    │       │
    │       ├── 客户端全文搜索降级策略:
    │       │
    │       ├── 清洗 query_str: 移除特殊字符，保留字母数字和空格
    │       ├── 构建关键词列表: query_str.split()
    │       ├── 构建 pyvastbase expr:
    │       │   " OR ".join([f"text ILIKE '%{kw}%'" for kw in keywords])
    │       │
    │       ├── 合并 metadata filters: [text_filter] AND [metadata_filter]
    │       ├── col.query(
    │       │       expr=combined_expr,
    │       │       output_fields=["node_id", "text", "metadata_"],
    │       │       limit=query.similarity_top_k,
    │       │   )
    │       │
    │       ├── 计算简单相似度: (匹配关键词数 / 总关键词数) 作为 score
    │       │
    │       └── RETURN VectorStoreQueryResult(...)
```

**对应上游方法:** `PGVectorStore._build_sparse_query()` (line 903) + `_sparse_query_with_rank()` (line 980)

**变更:**
- 移除 `to_tsvector` / `to_tsquery` / `ts_rank` / `@@` — 改用客户端 ILIKE
- 移除 `REGCONFIG` 类型定义
- 移除 `text_search_tsv` 计算列

### 6.4 query — HYBRID 模式

```
query(query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult
    │
    ├── case HYBRID:
    │       │
    │       ├── IF query.query_embedding is None → raise ValueError
    │       │
    │       ├── dense_req = AnnSearchRequest(
    │       │       data=query.query_embedding,
    │       │       anns_field="embedding",
    │       │       param={"metric_type": "COSINE", "ef": self._hnsw_ef_search},
    │       │       limit=query.similarity_top_k,
    │       │   )
    │       │
    │       ├── IF enable_sparse:
    │       │       ├── 构建 sparse expr (同 SPARSE 模式的 ILIKE 逻辑)
    │       │       ├── sparse_results = col.query(expr=...)  [取 top_k 个]
    │       │       └── 构建 sparse_req = AnnSearchRequest(...)
    │       │   ELSE:
    │       │       └── sparse_req = AnnSearchRequest(
    │       │               data=query.query_embedding,
    │       │               anns_field="embedding",  # fallback: dense only
    │       │               ...)
    │       │
    │       ├── col.hybrid_search(
    │       │       reqs=[dense_req, sparse_req],
    │       │       rerank=RRFRanker(k=60),
    │       │       limit=query.similarity_top_k,
    │       │   )
    │       │
    │       ├── hybrid_top_k → 最终结果截断
    │       │
    │       └── RETURN VectorStoreQueryResult(...)
```

**对应上游方法:** `PGVectorStore._hybrid_query()` (line 1030)

**变更:**
- 移除手动并行 dense + sparse → 拼接 → `_dedup_results()` 去重
- 改为 pyvastbase 原生 `hybrid_search()` + RRF 重排序
- 结果质量：RRF 融合 > 简单 node_id 去重

### 6.5 delete / adelete

```
delete(ref_doc_id: str, **delete_kwargs) -> None
    │
    ├── _initialize()
    ├── expr = f"metadata_->>'ref_doc_id' = '{ref_doc_id}'"
    ├── col.delete(expr=expr)
    └── RETURN None
```

**对应上游方法:** `PGVectorStore.delete()` (line 1131)

### 6.6 delete_nodes / adelete_nodes

```
delete_nodes(node_ids=None, filters=None, **delete_kwargs) -> None
    │
    ├── IF node_ids is None AND filters is None → RETURN
    │
    ├── 构建 expr 列表:
    │       ├── IF node_ids: "node_id IN ('id1', 'id2', ...)"
    │       └── IF filters: _build_filter_clause(filters)
    │
    ├── 合并: " AND ".join(expr_parts)
    ├── col.delete(expr=combined_expr)
    └── RETURN None
```

### 6.7 get_nodes / aget_nodes

```
get_nodes(node_ids=None, filters=None) -> List[BaseNode]
    │
    ├── REQUIRE node_ids 或 filters 至少一个不为 None
    │
    ├── 构建 expr (同 delete_nodes)
    ├── results = col.query(
    │       expr=combined_expr,
    │       output_fields=["node_id", "text", "metadata_", "embedding"],
    │   )
    │
    ├── FOR each row in results:
    │       ├── node = metadata_dict_to_node(row["metadata_"])
    │       │   OR TextNode(text=row["text"], id_=row["node_id"])
    │       ├── node.embedding = row.get("embedding")
    │       └── nodes.append(node)
    │
    └── RETURN nodes
```

### 6.8 clear / aclear

```
clear() -> None
    │
    ├── _initialize()
    ├── col.truncate()
    └── RETURN None
```

### 6.9 close

```
close() -> None
    ├── IF self._client:
    │       self._client.close()
    │       self._client = None
    ├── self._is_initialized = False

async_close() -> None
    ├── 同 close() — pyvastbase AsyncCollection 由 context manager 管理
```

---

## 7. Filter Translation（操作符映射表）

### _build_filter_clause 核心逻辑

```python
def _build_filter_clause(self, filters: Optional[MetadataFilters]) -> Optional[str]:
    """将 LlamaIndex MetadataFilters 转换为 pyvastbase expr 字符串."""
    if filters is None or not filters.filters:
        return None
    return self._recursively_apply_filters(filters)

def _recursively_apply_filters(self, filters: MetadataFilters) -> str:
    clauses = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            clauses.append(self._recursively_apply_filters(f))
        else:
            clauses.append(self._build_single_filter(f))

    joiner = " AND " if filters.condition == FilterCondition.AND else " OR "
    return f"({joiner.join(clauses)})"
```

### FilterOperator → pyvastbase expr 映射

| FilterOperator | pyvastbase expr 模式 | 说明 |
|---|---|---|
| EQ | `metadata_->>'key' = 'value'` | 精确匹配 |
| GT | `(metadata_->>'key')::float > N` | 大于（尝试数值转换） |
| LT | `(metadata_->>'key')::float < N` | 小于 |
| NE | `metadata_->>'key' != 'value'` | 不等于 |
| GTE | `(metadata_->>'key')::float >= N` | 大于等于 |
| LTE | `(metadata_->>'key')::float <= N` | 小于等于 |
| IN | `metadata_->>'key' IN ('v1', 'v2')` | 包含于列表 |
| NIN | `metadata_->>'key' NOT IN ('v1', 'v2')` | 不包含于列表 |
| CONTAINS | `metadata_->>'key' LIKE '%value%'` | 字符串包含（降级） |
| TEXT_MATCH | `metadata_->>'key' LIKE '%value%'` | 模糊匹配 |
| TEXT_MATCH_INSENSITIVE | `metadata_->>'key' ILIKE '%value%'` | 忽略大小写 |
| IS_EMPTY | `metadata_->>'key' IS NULL` | 值为空 |
| ANY | 客户端内存过滤 fallback | Vastbase 可能不支持 `?|` |
| ALL | 客户端内存过滤 fallback | Vastbase 可能不支持 `?&` |

### 客户端 Fallback 层

对于可能不被 Vastbase 支持的操作符（`?|`, `?&`, `@>`），实现一个 `_apply_client_filter()` 方法：

```
_apply_client_filter(results, filter) -> filtered_results
    │
    ├── ANY: 检查 metadata_["key"] ∩ values != ∅
    ├── ALL: 检查 metadata_["key"] ⊇ values
    └── CONTAINS: 检查 metadata_["key"] 包含 value
```

实现方式：
1. 先用宽松的 expr 从 Vastbase 拉取候选集
2. 在 Python 客户端侧对结果做精确过滤
3. 返回过滤后的结果和截断后的 ID 列表

---

## 8. Async API

### 实现策略

```python
# 同步方法：使用 Collection
def add(self, nodes, **kwargs):
    col = self._get_collection()
    return col.insert(rows)

# 异步方法：使用 AsyncCollection
async def async_add(self, nodes, **kwargs):
    async with AsyncCollection(self.collection_name) as col:
        return await col.insert(rows)
```

### 异步方法清单

| 同步方法 | 异步方法 | 实现方式 |
|---|---|---|
| `add` | `async_add` | AsyncCollection.insert() |
| `delete` | `adelete` | AsyncCollection.delete(expr) |
| `query` | `aquery` | AsyncCollection.search() / query() / hybrid_search() |
| `delete_nodes` | `adelete_nodes` | AsyncCollection.delete(expr) |
| `get_nodes` | `aget_nodes` | AsyncCollection.query(expr) |
| `clear` | `aclear` | AsyncCollection.truncate() |
| `close` | `async_close` | client.close() |

### AsyncCollection 连接管理

AsyncCollection 通过 async with 上下文管理器使用，每次操作创建独立的异步连接。连接参数从已有的 `VastbaseClient` 配置中获取。

---

## 9. Error Handling

### 异常传播策略

```
LlamaIndex 调用层
    │
    ▼
VastbaseVectorStore 方法
    │
    ├── pyvastbase 异常 → 直接传播（不包装）
    │       ├── CollectionNotExistsError → 用户需检查 Collection 是否存在
    │       ├── VectorDimensionError → 用户需检查 embed_dim 一致性
    │       ├── ConnectionError → 用户需检查 Vastbase 连接
    │       ├── SearchError → 查询参数或索引问题
    │       └── DataError → 数据格式或类型问题
    │
    ├── LlamaIndex 异常 → 直接传播
    │       └── ValueError → 无效参数（如 MMR mode）
    │
    └── 适配器自身异常 → ValueError / RuntimeError
            ├── 无效的 connection_params → ValueError
            ├── 初始化失败 + perform_setup=True → RuntimeError
            └── node_ids 和 filters 均为 None → AssertionError
```

### fail_on_error 机制

```python
def add(self, nodes, **add_kwargs):
    fail_on_error = add_kwargs.get("fail_on_error", True)
    try:
        return self._collection.insert(rows)
    except DataError as e:
        if fail_on_error:
            raise
        _logger.warning(f"Failed to insert nodes: {e}")
        return []
```

### 初始化错误处理

每个初始化步骤独立 try/except：
1. 连接失败 → 警告，若 perform_setup=True 则抛出
2. Collection 创建失败 → 警告，若 perform_setup=True 则抛出
3. 索引创建失败 → 警告（非致命）

---

## 10. Testing Strategy（三层测试体系）

### Layer 1: 单元测试（Mock pyvastbase）

- 测试 `_build_filter_clause` 所有 14 种 FilterOperator → pyvastbase expr 字符串转换
- 测试 `_recursively_apply_filters` 嵌套 AND/OR 组合
- 测试 `_node_to_fields` 节点序列化
- 测试 `from_params` 工厂方法参数解析
- 测试 MMR mode → ValueError 抛出
- 测试异常传播路径

### Layer 2: pyvastbase 集成测试（真实 Vastbase）

- **连接:** `172.16.105.107:15432` / `vastbase` / `aidev` / `Vbase_123456`
- Collection 创建 / 删除 / 重复创建幂等性
- `add` + `get_nodes` round-trip：插入节点 → 按 node_id 查询 → 验证数据完整性
- `query(DEFAULT)` 向量搜索：插入已知向量 → 搜索 → 验证距离排序
- `query(SPARSE/TEXT_SEARCH)` 文本搜索：插入含特定文本的节点 → ILIKE 搜索 → 验证召回
- `query(HYBRID)` 混合搜索：dense + sparse → RRF 重排
- `delete(ref_doc_id)` + `delete_nodes` 删除验证
- `clear` 清空验证
- HNSW 索引创建 + ef_search 参数验证
- metadata filter 各操作符验证

### Layer 3: LlamaIndex 框架验收测试（全链路）

- 使用 LlamaIndex `VectorStoreIndex` + `VastbaseVectorStore` 端到端测试
- 文档索引 → 检索 → 问答完整链路
- `from_documents` → `as_query_engine` → `query`
- 参数化 sync/async 两套路径

### 测试框架

- `pytest` + `pytest-asyncio`
- Mock: `unittest.mock`（Layer 1）
- 集成测试标记: `@pytest.mark.integration`（Layer 2）
- E2E 标记: `@pytest.mark.e2e`（Layer 3）

---

## 11. Differences from Reference（逐项差异表）

| # | 项目 | PGVectorStore (v0.7.3) | VastbaseVectorStore (v0.1.0) | 原因 |
|---|---|---|---|---|
| 1 | 驱动栈 | SQLAlchemy + psycopg2 + asyncpg + pgvector | pyvastbase >= 0.2.0 | Vastbase 原生向量引擎 |
| 2 | 连接管理 | 两个 Engine（sync + async） + sessionmaker | 一个 VastbaseClient（连接池） | pyvastbase 统一连接 |
| 3 | 建表方式 | `declarative_base()` + 动态 `type()` | `Collection.create()` | pyvastbase Collection API |
| 4 | 向量类型 | pgvector `Vector(dim)` / `HALFVEC(dim)` | `FLOAT_VECTOR(dim)` / `FLOAT16_VECTOR(dim)` | Vastbase V3 原生向量类型 |
| 5 | 扩展安装 | `CREATE EXTENSION vector` | 无需 | Vastbase V3 内置 |
| 6 | HNSW 索引 | DDL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index()` + `col.create_index()` | pyvastbase 索引 API |
| 7 | IVFFlat probes | `SET ivfflat.probes = N` | `search(param={...})` API 参数 | pyvastbase 统一参数传递 |
| 8 | 全文搜索 (SPARSE) | PG `to_tsvector` + `ts_rank` + `@@` | 客户端 ILIKE fallback | Vastbase PG 兼容性未知 |
| 9 | HYBRID 融合 | Dense + Sparse 拼接 + node_id 去重 | pyvastbase `hybrid_search()` + RRF | RRF > 简单去重 |
| 10 | MMR 查询 | `raise ValueError` | `raise ValueError`（一致） | 与上游一致 |
| 11 | JSONB 数组操作符 | `?|` / `?&` / `@>` 原生支持 | 客户端内存过滤 fallback | Vastbase 兼容性未知 |
| 12 | TSVECTOR 计算列 | `GENERATED ALWAYS AS ... STORED` | 不需要 | 无 ts_vector 依赖 |
| 13 | halfvec | `use_halfvec` 参数控制 | 不单独暴露；用户可通过 custom schema 使用 FLOAT16_VECTOR | 简化接口 |
| 14 | `customize_query_fn` | SQLAlchemy statement 回调 | pyvastbase expr 字符串回调 | 保持回调语义，适配新接口 |

---

## 12. Future Work

1. **Vastbase PG 兼容性探测** — 连接时自动检测 `to_tsvector`/`?|`/`?&`/`@>` 是否可用，记录到能力标志位，按需启用原生路径
2. **BM25 全文索引集成** — 如果 Vastbase FULLTEXT 索引性能优于客户端 ILIKE，提供 `text_search_backend="bm25"` 选项
3. **INT8_VECTOR 支持** — 提供 `quantization="int8"` 参数，利用 INT8_VECTOR 节省 75% 向量存储
4. **SPARSE_FLOAT_VECTOR 支持** — 支持稀疏向量类型，适配 SPLADE 等稀疏嵌入模型
5. **Collection 别名 + 多租户** — 通过 `namespace` 参数支持逻辑隔离
6. **Vastbase 连接池高级配置** — 暴露 pool_size / max_overflow / timeout 等参数
7. **性能基准测试** — 与 PGVectorStore 的延迟和吞吐对比
8. **上游 PGVectorStore 功能同步** — 持续跟踪 PGVectorStore 新增功能（如 0.8.x 的新查询模式）
