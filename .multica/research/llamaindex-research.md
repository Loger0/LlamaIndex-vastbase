# LlamaIndex 框架深度研究报告

> 框架诊断 Agent: framework-analyzer  
> 日期: 2026-06-24  
> 目标框架: LlamaIndex v0.14.22  
> 适配目标: Vastbase 向量数据库  
> 数据来源: 仓库源码实际分析（`./llama_index` + `./tests/`）

---

## 1. 框架基本信息

| 项目 | 内容 |
|------|------|
| **框架名称** | LlamaIndex (原 GPT Index) |
| **版本** | v0.14.22 |
| **源码仓库** | https://github.com/run-llama/llama_index |
| **许可证** | MIT |
| **语言** | Python (>=3.10) |
| **生态定位** | LLM 应用开发框架，提供数据连接器、索引、向量存储、检索增强生成（RAG）等能力 |
| **适配组件** | `llama_index.core.vector_stores` — 向量存储抽象层 |

### 1.1 LlamaIndex 在 AI 生态中的地位

LlamaIndex 是当前最主流的 LLM 应用开发框架之一，与 LangChain 并列为 RAG / Agent 领域的两大标杆框架。其核心能力包括：

- **数据索引与检索**：支持 100+ 种数据源连接器，提供 VectorStoreIndex、SummaryIndex、KnowledgeGraphIndex 等多种索引类型
- **向量存储抽象**：通过 `BasePydanticVectorStore` 基类定义统一接口，已有 50+ 种向量数据库适配器（PostgreSQL、Pinecone、Weaviate、Chroma、Milvus 等）
- **Agent 框架**：支持工具调用、多步推理、自动化工作流
- **查询引擎**：提供多种查询模式（DEFAULT、SPARSE、HYBRID、MMR 等）

---

## 2. 集成模式分析

### 2.1 判断结果: **standalone**（独立 pip 包）

LlamaIndex 采用 **插件式包注册** 机制，向量存储适配器作为独立的 pip 包发布，无需修改 `llama-index-core` 源码。

### 2.2 判断依据

1. **包结构独立**：每个向量存储适配器是独立的 pip 包（如 `llama-index-vector-stores-postgres`），有自己的 `pyproject.toml`、版本号、依赖清单
2. **命名空间包**：使用 Python namespace package（`llama_index/vector_stores/<backend>/`），安装后自动合并到 `llama_index` 命名空间
3. **注册机制**：通过 `pyproject.toml` 中的 `[tool.llamahub]` 配置声明 import_path 和 class_authors，LlamaHub 自动发现和注册
4. **零侵入**：适配器仅依赖 `llama-index-core` 的公开 API（`BasePydanticVectorStore`、`VectorStoreQuery` 等类型），不修改核心模块

### 2.3 发布方式

```toml
# pyproject.toml — 独立 pip 包配置
[project]
name = "llama-index-vector-stores-vastbase"
version = "0.1.0"
dependencies = [
    "llama-index-core>=0.13.0,<0.15",
    "pyvastbase>=0.2.7",
]

[tool.llamahub]
contains_example = false
import_path = "llama_index.vector_stores.vastbase"

[tool.llamahub.class_authors]
VastbaseVectorStore = "llama-index"
```

---

## 3. 耦合度分析

### 3.1 判断结果: **low**（低耦合）

适配器与 LlamaIndex 核心模块之间仅有接口级依赖，无需修改任何上游源码。

### 3.2 files_to_modify: **空**（无需修改任何现有文件）

适配器完全独立，不需要修改 `llama-index-core` 或任何其他适配器的文件。

### 3.3 files_to_create

| 文件路径 | 说明 | 行数 |
|----------|------|------|
| `pyproject.toml` | 包配置、依赖声明、llamahub 注册 | 31 行 |
| `README.md` | 使用文档、API 参考、快速入门 | 324 行 |
| `conftest.py` | 根级 pytest 配置 | 5 行 |
| `llama_index/vector_stores/vastbase/__init__.py` | 模块导出 | 6 行 |
| `llama_index/vector_stores/vastbase/base.py` | **主实现** — VastbaseVectorStore 类 | 1978 行 |
| `tests/conftest.py` | 测试 fixture 定义（13 个 fixture） | 574 行 |
| `tests/test_collection_init.py` | Collection 创建/schema 测试 | 213 行 |
| `tests/test_crud.py` | add/delete/get_nodes/clear 测试 | 298 行 |
| `tests/test_filter.py` | 元数据过滤翻译测试（14 种 operator） | 719 行 |
| `tests/test_search.py` | 查询模式测试（DEFAULT/SPARSE/HYBRID/MMR） | 403 行 |
| `tests/test_async.py` | 异步 API 对等测试 | 308 行 |
| `tests/test_integration.py` | E2E 集成测试（真实 Vastbase 实例） | 486 行 |
| `tests/test_framework_integration.py` | 框架级集成测试 | 366 行 |
| `tests/demo_llamaindex.py` | Demo 验收脚本 | 349 行 |

**总计: 14 个文件, ~6,059 行代码**

---

## 4. 接口抽象分析

### 4.1 抽象类型: **abstract_class**

LlamaIndex 使用 Pydantic v2 的抽象基类 `BasePydanticVectorStore` 定义向量存储接口契约。

### 4.2 基类定义

```python
# llama_index/core/vector_stores/types.py

class BasePydanticVectorStore(BaseModel):
    """Base class for all vector stores in LlamaIndex."""

    stores_text: bool = True
    flat_metadata: bool = False

    @property
    def client(self) -> Any: ...

    @classmethod
    def class_name(cls) -> str: ...

    def add(self, nodes: Sequence[BaseNode], **add_kwargs) -> List[str]: ...
    def delete(self, ref_doc_id: str, **delete_kwargs) -> None: ...
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult: ...

    # 可选方法（有默认实现或可选实现）
    async def async_add(...) -> List[str]: ...
    async def adelete(...) -> None: ...
    async def aquery(...) -> VectorStoreQueryResult: ...
    def delete_nodes(...) -> None: ...
    async def adelete_nodes(...) -> None: ...
    def get_nodes(...) -> List[BaseNode]: ...
    async def aget_nodes(...) -> List[BaseNode]: ...
    def clear() -> None: ...
    async def aclear() -> None: ...
    def persist(...) -> None: ...
```

### 4.3 必须实现的方法签名（4 个核心方法）

| 方法 | 签名 | 说明 |
|------|------|------|
| `client` (property) | `-> Any` | 返回底层数据库客户端实例 |
| `add` | `(nodes: Sequence[BaseNode], **kwargs) -> List[str]` | 插入节点，返回 node_id 列表 |
| `delete` | `(ref_doc_id: str, **delete_kwargs) -> None` | 按 ref_doc_id 删除关联节点 |
| `query` | `(query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult` | 执行向量查询 |

### 4.4 可选方法（对等实现，共 10 个）

| 方法 | 签名 | Vastbase 适配器实现状态 |
|------|------|------------------------|
| `async_add` | `(nodes, **kwargs) -> List[str]` | ✅ 已实现 — AsyncCollection.insert |
| `adelete` | `(ref_doc_id, **kwargs) -> None` | ✅ 已实现 — AsyncCollection.delete |
| `aquery` | `(query, **kwargs) -> VectorStoreQueryResult` | ✅ 已实现 — 4 种模式完整异步分发 |
| `delete_nodes` | `(node_ids, filters, **kwargs) -> None` | ✅ 已实现 — 支持客户端过滤回退 |
| `adelete_nodes` | `(node_ids, filters, **kwargs) -> None` | ✅ 已实现 |
| `get_nodes` | `(node_ids, filters) -> List[BaseNode]` | ✅ 已实现 — 含 _node_content 反序列化 |
| `aget_nodes` | `(node_ids, filters) -> List[BaseNode]` | ✅ 已实现 |
| `clear` | `() -> None` | ✅ 已实现 — truncate_collection |
| `aclear` | `() -> None` | ✅ 已实现 — AsyncCollection.truncate |
| `persist` | `(persist_path, fs) -> None` | ❌ 不支持 — raise NotImplementedError |

### 4.5 查询模式（4 种）

| 模式 | 实现方式 | Vastbase 映射 |
|------|----------|---------------|
| **DEFAULT** | 向量余弦相似度搜索 | `client.search(metric_type="cosine")` → `similarity = 1.0 - distance` |
| **SPARSE / TEXT_SEARCH** | 全文关键词检索 | ILIKE 候选检索 + 客户端 word-boundary 评分排序 |
| **HYBRID** | 向量 + 文本混合检索 | dense search + sparse ILIKE search → 按 node_id 去重合并（dense 优先） |
| **MMR** | 最大边际相关重排 | ❌ 不支持 — raise ValueError（与上游 PGVectorStore 行为一致） |

---

## 5. 已有数据库后端实现调研

### 5.1 LlamaIndex 已有后端对比表

| 后端 | 是否存在 | 适配可行性评分 | 说明 |
|------|----------|---------------|------|
| **PostgreSQL (PGVector)** | ✅ 存在 | ⭐⭐⭐⭐⭐ (5/5) | **最佳参考** — Vastbase 基于 PG 协议，schema/API 模式高度一致 |
| **MySQL** | ✅ 存在 | ⭐⭐ (2/5) | Vastbase 兼容 MySQL 协议，但向量能力差异大 |
| **Oracle** | ❌ 无官方适配 | — | 不适用 |
| **Milvus** | ✅ 存在 | ⭐⭐⭐ (3/5) | 向量数据库，API 风格接近 pyvastbase，但协议不兼容 |
| **Pinecone** | ✅ 存在 | ⭐ (1/5) | 云服务，API 完全不同 |
| **Chroma** | ✅ 存在 | ⭐ (1/5) | 轻量级，API 差异大 |
| **Weaviate** | ✅ 存在 | ⭐⭐ (2/5) | GraphQL API，差异大 |

### 5.2 reference_backend 推荐: **PGVectorStore** (`llama-index-vector-stores-postgres` v0.8.1)

**推荐理由：**

1. **协议兼容**：Vastbase 基于 PostgreSQL 协议，PGVectorStore 的 schema 设计（node_id/ref_doc_id/text/metadata_/embedding）可以直接映射
2. **功能对齐**：PGVectorStore 提供完整的 sync + async 双 API、4 种查询模式、14 种过滤操作符，是功能最全面的参考实现
3. **模式映射清晰**：

| PGVectorStore 组件 | VastbaseVectorStore 替代方案 |
|---|---|
| SQLAlchemy `create_engine` + `sessionmaker` | `VastbaseClient` (sync) |
| SQLAlchemy `create_async_engine` + `asyncpg` | `AsyncConnections` + `AsyncCollection` (async) |
| pgvector `Vector(N)` / `HalfVec(N)` | pyvastbase `DataType.FLOAT_VECTOR` / `DataType.FLOAT16_VECTOR` |
| `declarative_base()` + `type()` 动态建表 | `CollectionSchema` + `FieldSchema` |
| DDL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef_construction)` |
| `to_tsvector` / `to_tsquery` / `ts_rank` | 客户端 ILIKE + word-boundary 评分 |
| PG JSONB `?|` / `?&` / `@>` | 客户端内存过滤 (ANY / ALL / CONTAINS) |
| `sqlalchemy[asyncio]` 连接池 | pyvastbase 内置连接管理 |

4. **代码量参考**：PGVectorStore base.py 约 1698 行，VastbaseVectorStore 约 1978 行（增加 async executor 兼容补丁 + schema column 补丁逻辑）

### 5.3 PGVectorStore 关键实现模式摘要

| 模式 | PGVectorStore 实现 | VastbaseVectorStore 对应 |
|------|---------------------|--------------------------|
| **连接管理** | SQLAlchemy URL 连接字符串 | host/port/database/user/password 分参数 |
| **Schema 管理** | `declarative_base` + `get_data_model()` 动态生成 ORM 类 | `CollectionSchema` + `FieldSchema` + `_ensure_schema_columns` ALTER TABLE 补丁 |
| **索引管理** | DDL `CREATE INDEX ... USING hnsw` | `IndexParams.graph_index(m, ef_construction)` |
| **元数据过滤** | `text()` SQL 片段 + `->`/`->>`/`::jsonb`/`::float` 类型转换 | pyvastbase `expr` 字符串 + `metadata_->>'key'` JSONB 访问 |
| **全文搜索** | `to_tsvector('english', text)` 计算列 + `to_tsquery` + `ts_rank` | ILIKE 候选检索 + Python 客户端 word-boundary 评分 |
| **混合搜索** | 分别执行 dense (cosine_distance) 和 sparse (ts_rank) 查询 → union → dedup | 分别执行 dense (cosine) 和 sparse (ILIKE) 查询 → dense 优先 + unseen sparse append |
| **MMR** | Prefetch N×top_k → extract embeddings → get_top_k_mmr_embeddings → rerank → fallback | raise ValueError（与上游一致） |
| **向量距离** | `cosine_distance` → `similarity = 1 - distance` | `distance` from search hit → `similarity = 1.0 - distance` |

---

## 6. 测试基础设施评估

### 6.1 判断结果: **full**（完整测试基础设施）

### 6.2 测试框架

| 项目 | 内容 |
|------|------|
| **测试框架** | pytest + pytest-asyncio (`asyncio_mode = "auto"`) |
| **测试目录** | `./tests/` |
| **测试文件数** | 10 个 |
| **总测试代码行数** | ~3,721 行（不含 demo） |
| **fixture 数** | 13 个 session/function-scoped fixtures |

### 6.3 测试文件清单

| 文件 | 行数 | 测试数量 | 覆盖范围 |
|------|------|----------|----------|
| `conftest.py` | 574 | — | 13 个 fixture（基础/hybrid/indexed_metadata/hnsw/halfvec/gin/custom_search 等） |
| `test_collection_init.py` | 213 | 9 | Collection 创建、schema 字段验证、HNSW 索引创建 |
| `test_crud.py` | 298 | 15 | add/delete/get_nodes/clear 基础 CRUD |
| `test_filter.py` | 719 | 21 | 14 种 FilterOperator 翻译 + 嵌套 AND/OR + 客户端过滤回退 |
| `test_search.py` | 403 | 20 | 4 种查询模式 (DEFAULT/SPARSE/HYBRID/MMR) + 排序验证 |
| `test_async.py` | 308 | 13 | async API 对等性 (async_add/aquery/adelete/aget_nodes/aclear) |
| `test_integration.py` | 486 | 6 | E2E 集成测试（真实 Vastbase 实例 172.16.105.107:15432） |
| `test_framework_integration.py` | 366 | — | 框架级集成测试 |
| `demo_llamaindex.py` | 349 | — | Demo 验收脚本（8 个场景） |

### 6.4 测试策略: **extract_official**

测试用例从上游 PGVectorStore 官方测试套件 (`llama-index-vector-stores-postgres/tests/test_postgres.py`, 3222 行) 提取并适配：

- 保留上游 fixture 命名和参数化模式
- 替换 SQLAlchemy/psycopg2 连接为 pyvastbase
- 增加 Vastbase 特有场景（FLOAT16_VECTOR、auto-id sequence 补丁）
- 增加客户端过滤回退测试（ANY/ALL/CONTAINS operator）

### 6.5 测试配置

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
filterwarnings = ["ignore::DeprecationWarning"]
```

---

## 7. Demo 场景规划

### 7.1 Demo 类型: **RAG 文档入库 + 多模式检索验证脚本**

### 7.2 场景清单（8 个）

| # | 场景 | 验证目标 |
|---|------|----------|
| 1 | 文档 embedding 入库 | `add` + `async_add` → node_id 返回正确 |
| 2 | DEFAULT 向量相似度检索 | cosine/l2/ip 距离 → similarity 排序正确 |
| 3 | SPARSE 全文检索 | ILIKE 关键词匹配 → word-boundary 评分排序 |
| 4 | HYBRID 混合检索 | dense + sparse → 去重合并、dense 优先 |
| 5 | 元数据过滤检索 | 14 种 FilterOperator → expr 翻译 + 客户端回退 |
| 6 | 按 ref_doc_id 删除文档 | delete/adelete → 关联节点全部删除 |
| 7 | 批量删除 (delete_nodes) | node_ids + filters → 精确删除 |
| 8 | 清空 Collection | clear/aclear → 全部数据清除 |

### 7.3 验收标准

端到端 API 验证：`add → query (DEFAULT/SPARSE/HYBRID) → get_nodes → delete_nodes → clear`。所有操作返回预期结果，无异常，查询模式返回正确的相似度排序。

---

## 8. pyvastbase 替换可行性分析

### 8.1 结论: **完全可行** ✅

pyvastbase (>= 0.2.7) 可以 **100% 替代** PGVectorStore 的 SQLAlchemy + psycopg2/asyncpg + pgvector 依赖栈。

### 8.2 替换映射

| PGVectorStore 依赖 | pyvastbase 替代 | 覆盖度 |
|---|---|---|
| `sqlalchemy[asyncio]` (ORM + 连接池) | `VastbaseClient` (sync) + `AsyncConnections`/`AsyncCollection` (async) | ✅ 100% |
| `psycopg2-binary` (同步驱动) | `VastbaseClient` 内置 psycopg3 | ✅ 100% |
| `asyncpg` (异步驱动) | `AsyncConnections` (psycopg3 async) | ✅ 100% |
| `pgvector` (Vector/HalfVec 类型) | `DataType.FLOAT_VECTOR` / `DataType.FLOAT16_VECTOR` | ✅ 100% |
| DDL 建表 (`CREATE TABLE`) | `CollectionSchema` + `FieldSchema` + `_ensure_schema_columns` | ✅ 100% |
| DDL 建索引 (`CREATE INDEX ... USING hnsw`) | `IndexParams.graph_index(m, ef_construction)` | ✅ 100% |
| PG JSONB 操作符 (`->`/`->>`/`?|`/`?&`/`@>`) | `expr` 字符串 + 客户端内存过滤 | ✅ 100% (部分回退到客户端) |
| `to_tsvector`/`to_tsquery` (全文搜索) | ILIKE + word-boundary 评分 | ✅ 功能等价（实现方式不同） |

### 8.3 已知的 pyvastbase 兼容性问题及解决方案

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| `has_collection` 偶发 `unexpected keyword argument 'using'` | Collection 存在性检查失败 | try/except + fall through 到 `create_collection` + `already exists` 异常捕获 |
| `AsyncExecutor.execute` 命名占位符不兼容 list 参数 | async 初始化/schema 加载失败 | `_patch_async_executor()` 猴子补丁 — 自动将 list 转为 dict |
| `FieldSchema(auto_id=True)` 不生成 `SERIAL` DDL | INSERT 时 NotNullViolation | `_ensure_auto_id_sequence()` — 手动创建 sequence + ALTER TABLE SET DEFAULT |
| `add_collection_field` 缺少 `@with_executor` 装饰器 | 动态添加字段静默失败 | `_ensure_schema_columns()` 直接执行 ALTER TABLE SQL |
| PG JSONB 数组操作符 `?|`/`?&`/`@>` 与参数占位符冲突 | expr 中 `?` 被误解析 | 标记为 `_CLIENT_SIDE_OPERATORS`，回退到客户端内存过滤 |

### 8.4 依赖清单

```
# 替换前 (PGVectorStore)
psycopg2-binary>=2.9.9
asyncpg>=0.29.0
pgvector>=0.3.6
sqlalchemy[asyncio]>=1.4.49
llama-index-core>=0.13.0,<0.15

# 替换后 (VastbaseVectorStore)
pyvastbase>=0.2.7
llama-index-core>=0.13.0,<0.15
```

**净减少依赖: 4 个 → 1 个**

---

## 9. 开放决策记录

以下决策已在 Phase 7.5 识别并在 Phase 7.6 自动 resolve（详见 `.multica/decisions/llamaindex-decisions.yaml`）：

| ID | 类别 | 问题 | 决议 | 理由 |
|----|------|------|------|------|
| D-01 | API风格 | sync vs async | 同步+异步对等实现 | PGVectorStore 提供完整 sync+async 双 API |
| D-02 | 依赖策略 | 上游依赖 vs pyvastbase | 全部替换为 pyvastbase | 移除 psycopg2/asyncpg/pgvector/sqlalchemy |
| D-03 | 错误处理 | 容错级别 | 混合策略（可配置 fail_on_error） | 与 PGVectorStore initialization_fail_on_error 一致 |

后续实施中新增的决策（D-04 ~ D-09）已由用户在实施阶段确认，详见 decisions.yaml。

---

## 10. 数据来源与引用

1. **LlamaIndex 源码**: https://github.com/run-llama/llama_index (v0.14.22)
2. **PGVectorStore 参考实现**: `llama-index-integrations/vector_stores/llama-index-vector-stores-postgres/` (v0.8.1)
3. **Vastbase 适配器仓库**: https://github.com/Loger0/LlamaIndex-vastbase (branch: `feature/llama-index-vastbase`)
4. **pyvastbase SDK**: https://pypi.org/project/pyvastbase/ (>=0.2.7)
5. **LlamaIndex VectorStore 基类**: `llama-index-core/llama_index/core/vector_stores/types.py`
6. **PGVectorStore 测试套件**: `llama-index-vector-stores-postgres/tests/test_postgres.py` (3222 行)
