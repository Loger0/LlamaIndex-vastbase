# LlamaIndex Vastbase 适配 — Design Spec

**Date:** 2026-06-17
**Status:** Draft
**Integration Mode:** standalone
**pyvastbase Version:** >=0.2.7
**LlamaIndex Core:** >=0.13.0,<0.15

## 1. Overview

### 目标

将 LlamaIndex 的 PGVectorStore（PostgreSQL + pgvector 向量存储后端）适配到 Vastbase，用 pyvastbase 的 PyMilvus-style API 替代上游的 SQLAlchemy + psycopg2/asyncpg + pgvector 驱动栈，实现全量功能对等的 `VastbaseVectorStore`。

### 非目标

- 不修改 LlamaIndex 框架源码（standalone 模式）
- 不实现 Vastbase 不支持的功能（如 pgvector 专用的 `ivfflat_probes` / `enable_bitmapscan`）
- 不提供 SQLAlchemy 兼容层

## 2. Package Structure

```
llama-index-vector-stores-vastbase/
├── pyproject.toml
├── README.md
└── llama_index/
    └── vector_stores/
        └── vastbase/
            ├── __init__.py          # 导出 VastbaseVectorStore
            └── base.py              # VastbaseVectorStore 完整实现 (~1200 行)
```

### pyproject.toml

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

## 3. Architecture

### Architecture Pattern: Faithful Adapter

将 PGVectorStore（1698 行）逐模块翻译为 pyvastbase 等价物，保持上游内部结构和命名风格。

```
用户代码
  ↓
VastbaseVectorStore(BasePydanticVectorStore)
  ├── VastbaseClient（连接 + 池管理）
  ├── CollectionSchema + FieldSchema（替代 get_data_model 动态 ORM）
  ├── IndexParams.graph_index()（替代 CREATE INDEX USING hnsw）
  ├── _build_filter_clause()（PG JSON 操作符 → expr 字符串）
  ├── _query_with_score()（client.search 替代 cosine_distance SQL）
  ├── _sparse_query()（to_tsvector/ts_rank expr 替代）
  ├── _hybrid_query()（并发 dense+sparse → 去重合并）
  ├── _mmr_query()（预取 + get_top_k_mmr_embeddings Python 重排）
  └── _db_rows_to_query_result()（结果转换）
    ↓
pyvastbase (VastbaseClient / Collection / AsyncCollection)
    ↓
Vastbase V3
```

### Design Decisions (from clarification phase)

| # | 决策点 | 决议 |
|---|--------|------|
| D-01 | API 风格 | sync + async 双实现 |
| D-02 | 依赖策略 | 全部替换为 pyvastbase |
| D-03 | 错误处理 | 沿用 PGVectorStore 模式 (_logger.warning + fail_on_error) |
| D-04 | 功能裁剪 | 全量移植，功能对等 |
| D-05 | 连接管理 | 单一 VastbaseClient 统一管理 |
| Q1 | 内部 API 风格 | VastbaseClient 单一入口 |
| Q2 | Metadata 过滤 | 纯 expr 字符串（SQL 下推） |
| Q3 | 全文搜索 | PG 兼容路径（to_tsvector/to_tsquery） |
| Q4 | customize_query_fn | 搜索参数回调钩子 customize_search_fn |
| Q5 | 半精度向量 | 支持 FLOAT16_VECTOR（use_halfvec 参数） |

## 4. Class Design

### VastbaseVectorStore

```python
class VastbaseVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = False

    # ===== 连接参数（替代 upstream connection_string + async_connection_string）=====
    host: str
    port: int = 15432
    database: str = "vastbase"
    user: str = "aidev"
    password: str = ""

    # ===== 集合参数（替代 upstream table_name + schema_name）=====
    table_name: str = "llamaindex"
    schema_name: str = "public"

    # ===== 向量配置 =====
    embed_dim: int = 1536
    use_halfvec: bool = False  # True → FLOAT16_VECTOR

    # ===== 全文搜索配置 =====
    hybrid_search: bool = False
    text_search_config: str = "english"

    # ===== Metadata 配置 =====
    use_jsonb: bool = False
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None

    # ===== 索引配置 =====
    hnsw_kwargs: Optional[Dict[str, Any]] = None
    # 必需键: hnsw_m, hnsw_ef_construction, hnsw_ef_search

    # ===== 行为配置 =====
    perform_setup: bool = True
    debug: bool = False
    initialization_fail_on_error: bool = False

    # ===== 私有属性 =====
    _client: VastbaseClient = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _collection_name: str = PrivateAttr(default=None)
    _customize_search_fn: Optional[Callable] = PrivateAttr(default=None)
```

### from_params() 工厂方法

参数签名与上游 `PGVectorStore.from_params()` 保持对齐，但移除 SQLAlchemy 专有参数（`connection_string`、`async_connection_string`、`create_engine_kwargs`、`engine`、`async_engine`、`cache_ok`），新增 `customize_search_fn`。

### client 属性

```python
@property
def client(self) -> Any:
    if not self._is_initialized:
        return None
    return self._client
```

## 5. pyvastbase API Mapping

| 上游 PGVectorStore | VastbaseVectorStore | pyvastbase API |
|---|---|---|
| `_connect()` | `VastbaseClient(...)` 构造函数 | `VastbaseClient(host, port, db, user, password)` |
| `_create_schema_if_not_exists()` | 删除 | — (Vastbase 默认 public schema) |
| `_create_extension()` | 删除 | — (Vastbase 原生向量引擎) |
| `_create_tables_if_not_exists()` | `_create_collection_if_not_exists()` | `client.create_collection()` / `client.has_collection()` |
| `_create_hnsw_index()` | 同名 | `client.create_index(field_name, IndexParams.graph_index())` |
| `add()` | 同名 | `client.insert(collection, rows)` |
| `async_add()` | 同名 | `AsyncCollection.insert(rows)` |
| `delete()` | 同名 | `client.delete(collection, expr=...)` |
| `adelete()` | 同名 | `AsyncCollection.delete(expr=...)` |
| `delete_nodes()` | 同名 | `client.delete(collection, expr=...)` |
| `adelete_nodes()` | 同名 | `AsyncCollection.delete(expr=...)` |
| `clear()` | 同名 | `client.truncate_collection(collection)` |
| `aclear()` | 同名 | `AsyncCollection.truncate()` |
| `get_nodes()` | 同名 | `client.query(collection, expr=..., output_fields=...)` |
| `aget_nodes()` | 同名 | `AsyncCollection.query(expr=..., output_fields=...)` |
| `query()` | 同名 (mode dispatch) | dispatch to `client.search()` / `client.query()` / hybrid |
| `aquery()` | 同名 (mode dispatch) | dispatch to `AsyncCollection.search()` / `AsyncCollection.query()` |
| `_query_with_score()` | 同名 | `client.search(data=[emb], limit=N, expr=..., param={"metric_type":"COSINE"})` |
| `_sparse_query_with_rank()` | 同名 | `client.query(expr="@@ to_tsquery(...)")` |
| `_hybrid_query()` | 同名 | 两次调用 + `_dedup_results()`（Python 侧去重） |
| `_mmr_query()` | 同名 | `_query_with_embedding()` + `get_top_k_mmr_embeddings()`（LlamaIndex core） |
| `close()` | 同名 | `client.close()` |

## 6. Collection Initialization Flow

```
_initialize()
├── VastbaseClient(host, port, database, user, password)    # Step 1: 连接
├── if perform_setup:
│   ├── _create_collection_if_not_exists()                    # Step 2: Collection
│   │   └── CollectionSchema + FieldSchema → client.create_collection_with_schema()
│   ├── if hnsw_kwargs:
│   │   └── _create_hnsw_index()                             # Step 3: HNSW 索引
│   │       └── IndexParams.graph_index(m, ef_construction) → client.create_index()
│   └── if hybrid_search:
│       └── _create_fulltext_index()                          # Step 4: FULLTEXT 索引
│           └── IndexParams.fulltext_index(dictionary, algorithm) → client.create_index()
└── _is_initialized = True
```

### Collection Schema

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
    FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text", dtype=DataType.TEXT),
    FieldSchema(name="metadata_", dtype=DataType.JSON),       # use_jsonb=False → JSON
    FieldSchema(name="embedding", dtype=vector_dtype, dim=embed_dim),
]
# hybrid_search=True 时追加:
#   FieldSchema(name="text_search_tsv", dtype=DataType.TEXT)
```

## 7. Core Methods

### 7.1 add / async_add

**数据流**: `BaseNode` → `_node_to_row_dict()` → `[{dict}, ...]` → `client.insert(collection, rows)` → `List[str]`

**关键点**: 替代上游的 SQLAlchemy session.add + commit 模式，pyvastbase insert 内置事务管理。

### 7.2 delete / adelete

**数据流**: `ref_doc_id` → `expr = "metadata_->>'ref_doc_id' = '{ref_doc_id}'"` → `client.delete(collection, expr)` → None

### 7.3 delete_nodes / adelete_nodes

**数据流**: `node_ids + filters` → 构造 AND expr → `client.delete(collection, expr)` → None

### 7.4 get_nodes / aget_nodes

**数据流**: `node_ids + filters` → 构造 AND expr → `client.query(collection, expr, output_fields=...)` → `[_row_to_node(r) for r in results]` → `List[BaseNode]`

### 7.5 clear / aclear

**数据流**: → `client.truncate_collection(collection)` → None

### 7.6 query / aquery

与上游相同的 mode dispatch：
- **DEFAULT** → `_query_with_score()` → `client.search()`
- **SPARSE / TEXT_SEARCH** → `_sparse_query_with_rank()` → `client.query(expr="@@ to_tsquery(...)")`
- **HYBRID** → `_hybrid_query()` → 并发 dense + sparse → `_dedup_results()`
- **MMR** → `_mmr_query()` → `_query_with_embedding()` + Python 侧 `get_top_k_mmr_embeddings()`

### 7.7 _query_with_score

```python
search_params = {"metric_type": "COSINE"}
if hnsw_ef_search:
    search_params["ef"] = int(hnsw_ef_search)

# apply customize_search_fn if set
if _customize_search_fn:
    params = {"expr": expr, "limit": limit, "param": search_params}
    params = _customize_search_fn(params, **kwargs)

results = _client.search(
    collection_name, data=[embedding], limit=limit,
    expr=expr, param=search_params,
    output_fields=["node_id", "text", "metadata_"],
)
return [_search_result_to_row(r) for r in results[0]]
```

### 7.8 _sparse_query_with_rank

```python
# 清洗查询字符串（与上游相同：去特殊字符、替换空格为 |）
query_str = re.sub(r"(?!\b\.\b)\W+", " ", query_str).strip().replace(" ", "|")
expr = f"text_search_tsv @@ to_tsquery('{text_search_config}', '{query_str}')"
results = _client.query(collection_name, expr=expr, output_fields=[...], limit=limit)
return [_query_result_to_row(r) for r in results]
```

### 7.9 _hybrid_query

```python
dense_results = _query_with_score(embedding, top_k, filters)
sparse_results = _sparse_query_with_rank(query_str, sparse_top_k, filters)
return _dedup_results(dense_results + sparse_results)
```

### 7.10 _mmr_query

```python
prefetch_k, mmr_threshold = _prepare_mmr_query(query, **kwargs)
results_with_embeddings = _query_with_embedding(embedding, prefetch_k, filters)
result = _mmr_rerank_results(query, results_with_embeddings, mmr_threshold)
if result is not None:
    return result
# fallback → _query_with_score()
```

## 8. Filter Translation

### 8.1 Operator Map

| FilterOperator | SQL 操作符 |
|----------------|-----------|
| EQ | `=` |
| GT | `>` |
| LT | `<` |
| NE | `!=` |
| GTE | `>=` |
| LTE | `<=` |
| IN | `IN` |
| NIN | `NOT IN` |
| CONTAINS | `@>` |
| TEXT_MATCH | `LIKE` |
| TEXT_MATCH_INSENSITIVE | `ILIKE` |
| IS_EMPTY | `IS NULL` |
| ANY | `?\|` |
| ALL | `?&` |

### 8.2 _build_filter_clause

每种操作符生成对应的 `metadata_->>'key' operator value` 格式 expr 子串。与上游逻辑完全一致，不依赖 SQLAlchemy `text()`。

### 8.3 _recursively_apply_filters

递归处理 AND/OR 嵌套的 MetadataFilters，输出 `(clause1) AND (clause2) OR (clause3)` 格式字符串。

## 9. Async API

### 策略

sync/async 方法对成对实现（与上游一致）。

### 实现模式

```python
async def async_add(self, nodes, **kwargs):
    self._initialize()
    if self._async_collection is None:
        self._async_collection = AsyncCollection(self._collection_name)
    rows = [self._node_to_row_dict(n) for n in nodes]
    await self._async_collection.insert(rows)
    return [n.node_id for n in nodes]
```

### 异步 HYBRID 并发

```python
async def _async_hybrid_query(self, query, **kwargs):
    dense, sparse = await asyncio.gather(
        self._aquery_with_score(query.query_embedding, query.similarity_top_k, query.filters),
        self._async_sparse_query_with_rank(query.query_str, sparse_top_k, query.filters),
    )
    return _dedup_results(dense + sparse)
```

## 10. Error Handling

沿用上游 PGVectorStore 模式（decisions D-03）：

- `_logger.warning()` 记录非致命错误
- `initialization_fail_on_error: bool = False` 控制是否抛出初始化异常
- `perform_setup: bool = True` 控制是否执行自动建表/建索引
- 捕获 `VastbaseException` 及其子类

### 异常映射

| 上游异常 | Vastbase 异常 |
|----------|--------------|
| `sqlalchemy.exc.OperationalError` | `ConnectionError` |
| `sqlalchemy.exc.ProgrammingError` | `SchemaError` |
| `asyncpg.exceptions.*` | `VastbaseException` |
| `psycopg2.errors.*` | `VastbaseException` |

## 11. Differences from Reference (PGVectorStore)

| 维度 | PGVectorStore | VastbaseVectorStore | 原因 |
|------|--------------|---------------------|------|
| 驱动 | psycopg2 + asyncpg + SQLAlchemy + pgvector | pyvastbase >=0.2.7 | 需求明确 |
| 连接方式 | 双引擎 (sync + async engine) | 单一 VastbaseClient | D-05 决策 |
| 表创建 | SQLAlchemy ORM declarative_base + 动态类 | CollectionSchema + FieldSchema | pyvastbase API |
| 向量类型 | Vector(dim) / HALFVEC(dim) | FLOAT_VECTOR(dim) / FLOAT16_VECTOR(dim) | Vastbase 原生类型 |
| 向量索引 | Raw SQL CREATE INDEX USING hnsw | IndexParams.graph_index() | pyvastbase API |
| 全文搜索 | to_tsvector Computed 列 + GIN 索引 | to_tsvector/ts_query expr + FULLTEXT 索引 | Q3 决策 (PG 兼容) |
| 向量距离 | embedding.cosine_distance() (pgvector) | client.search(param={"metric_type": "COSINE"}) | pyvastbase API |
| Session 参数 | SET hnsw.ef_search / SET ivfflat.probes | search(param={"ef": N}) | pyvastbase API |
| Metadata 索引 | BTREE + GIN (SQLAlchemy Index) | Raw SQL CREATE INDEX (MVP 阶段) | pyvastbase 限制 |
| 扩展点 | customize_query_fn(Select) | customize_search_fn(Dict) | Q4 决策 |
| Schema 创建 | CREATE SCHEMA IF NOT EXISTS | 删除 (默认 public) | Vastbase 简化 |
| CREATE EXTENSION | CREATE EXTENSION vector | 删除 | Vastbase 原生向量 |
| 连接参数 | connection_string + async_connection_string | host/port/database/user/password | pyvastbase API |
| cache_ok 参数 | TypeDecorator 缓存标志 | 删除 | 无自定义 TypeDecorator |
| create_engine_kwargs | SQLAlchemy engine 参数 | 删除 | 无 SQLAlchemy |
| 外部 engine 注入 | engine / async_engine 参数 | 删除 | 无 SQLAlchemy |

## 12. Known Risks & Mitigations

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Vastbase JSONB 操作符 (?\|, ?&, @>) 不完全兼容 | ANY/ALL/CONTAINS 过滤失败 | 集成测试逐条验证，不兼容则降级为客户端后过滤 |
| to_tsvector/to_tsquery 与 FULLTEXT BM25 冲突 | SPARSE 模式异常 | 集成测试验证 Vastbase 全文搜索函数，不支持则回退到 FULLTEXT+BM25 |
| CollectionSchema 不支持 Computed 生成列 | text_search_tsv 列需手动更新 | add() 时在 Python 侧预计算 ts_vector |
| pyvastbase search() expr 参数能力边界未知 | 复杂过滤可能受限 | 逐条验证 14 种 FilterOperator 的 expr 执行 |
| Vastbase 版本 < 3.0.9 不支持 FLOAT16_VECTOR | use_halfvec=True 时报错 | check_vb_version() 运行时检测，降级 + warning |

## 13. Testing Strategy

三层测试体系：

- **Layer 1 — 单元测试**: Filter 转换规则、结果映射、参数校验 (mock pyvastbase)
- **Layer 2 — pyvastbase 集成测试**: 真实 Vastbase 连接，CRUD 全生命周期 + 4 种 query mode
- **Layer 3 — 框架集成验收测试**: 端到端 RAG 场景（文档入库 → 向量检索 → 结果验证）

## 14. Future Work

- BTREE/GIN metadata 索引通过 pyvastbase 原生 API 支持
- 如果 Vastbase 验证不兼容 PG 全文搜索函数，增加 FULLTEXT+BM25 作为 SPARSE 备选路径
- 支持 `hybrid_ann_search` (原生单向量 + 标量过滤) 作为 HYBRID 的 Vastbase 特化路径
- INT8_VECTOR 支持（当 embedding 模型支持 int8 量化时）
- pypi 发布 CI/CD 流程
