# LlamaIndex Vastbase 适配 — 测试交付报告 (TEST PLAN)

> **状态**: RED Phase — 测试已编写，待执行后全部预期 FAIL（实现尚未开始）
> **日期**: 2026-06-17
> **test-scout**: 5c83ae03-ca33-4931-b29f-5f4452049ee2

---

## 一、测试概览

| 指标 | 数值 |
|------|------|
| 测试文件数 | 6 (`conftest.py` 除外) |
| 测试函数总数 (含参数化展开) | 83 |
| 覆盖的 required_methods | 13/13 (100%) |
| 覆盖的 demo 场景 | 9/9 (100%) |
| 覆盖的 FilterOperator | 14/14 (100%) |
| 覆盖的 QueryMode | 4/4 (DEFAULT/SPARSE/HYBRID/MMR) |

### 文件清单

| 文件 | 描述 | 测试数 |
|------|------|--------|
| `tests/conftest.py` | pytest fixtures — Vastbase 连接、store 实例、节点数据 | — |
| `tests/test_collection_init.py` | Collection 创建、HNSW/全文索引、半精度 schema | 9 |
| `tests/test_crud.py` | add / delete / delete_nodes / get_nodes / clear | 15 |
| `tests/test_search.py` | DEFAULT / SPARSE / HYBRID / MMR 四种查询模式 | 20 |
| `tests/test_filter.py` | 14 种 FilterOperator + AND/OR + GIN 数组操作 | 21 |
| `tests/test_async.py` | async_add / adelete / adelete_nodes / aget_nodes / aclear / aquery | 13 |
| `tests/test_integration.py` | E2E 生命周期、Hybrid 端到端、IndexNode、多实例隔离 | 5 |

---

## 二、上游测试场景映射

| # | 上游测试 (test_postgres.py) | Vastbase 适配测试 | 变更说明 |
|---|---------------------------|------------------|---------|
| 1 | `test_instance_creation` | `test_vastbase_instance_creation` | PGVectorStore → VastbaseVectorStore；移除 `_engine` 断言 |
| 2 | `test_add_to_db_and_query` | `test_search_default` | `async_add` / `add` → `add`；`aquery` / `query` → `query` |
| 3 | `test_query_hnsw` | `test_search_hnsw` | HNSW 参数映射至 IndexParams.graph_index() |
| 4 | `test_add_to_db_and_query_with_metadata_filters` | `test_search_default_with_metadata_filter` | ExactMatchFilter → MetadataFilter(operator=EQ) |
| 5 | `test_*_in_operator` | `test_filter_in` / `test_filter_in_single` | 无变更，expr 字符串下推 |
| 6 | `test_*_any_operator` | `test_filter_any` | ?\| 操作符映射，下推到 expr 字符串 |
| 7 | `test_*_all_operator` | `test_filter_all` | ?& 操作符映射，下推到 expr 字符串 |
| 8 | `test_*_contains_operator` | `test_filter_contains` | @> 操作符映射，下推到 expr 字符串 |
| 9 | `test_*_is_empty` | `test_filter_is_empty` | IS NULL 操作符映射 |
| 10 | `test_add_to_db_query_and_delete` | `test_delete_by_ref_doc_id` | DELETE SQL → client.delete(expr=...) |
| 11 | `test_sparse_query` | `test_sparse_query` | to_tsquery SQL → ILIKE fallback |
| 12 | `test_hybrid_query` | `test_hybrid_query` / `test_hybrid_query_hnsw` | pyvastbase hybrid_search + RRFRanker |
| 13 | `test_delete_nodes` | `test_delete_nodes_by_ids` | DELETE SQL → client.delete(expr=...) |
| 14 | `test_delete_nodes_metadata` | `test_delete_nodes_by_ids_and_filters` / `test_delete_nodes_by_filters` | 组合 expr |
| 15 | `test_get_nodes_parametrized` | `test_get_nodes_parametrized` | SELECT → client.query(expr=...) |
| 16 | `test_clear` | `test_clear_collection` | DELETE FROM → col.truncate() |
| 17 | `test_add_to_db_and_query_index_nodes` | `test_add_index_nodes` | Node 序列化方式不变 |
| 18 | `test_custom_query` | `test_customize_search_fn_integration` | SQLAlchemy Select → dict 参数回调 |
| 19 | `test_gin_index_query_with_contains` | `test_gin_contains` | GIN 索引通过 expr 字符串下推 |
| 20 | `test_gin_index_query_with_any_operator` | `test_gin_any` | 同上 |
| 21 | `test_gin_index_query_with_all_operator` | `test_gin_all` | 同上 |
| 22 | `test_mixed_btree_and_gin_indices` | `test_mixed_btree_and_gin` | BTREE + GIN 组合 |
| 23-33 | MMR mock tests (11 tests) | `TestMMRQuery` (7 tests) | Mock 测试，无需 Vastbase，直接验证算法行为 |
| 34-35 | `test_custom_engines` / `test_custom_*_only` | **已移除** — Vastbase 不支持外部 engine 注入 | — |
| 36 | `test_hnsw_index_creation` | `test_hnsw_collection_initialization` | 验证 hnsw_kwargs 配置 |
| 37 | `test_indexed_metadata` | `test_indexed_metadata_initialization` | 验证 indexed_metadata_keys |

### 已移除的上游测试

| 上游测试 | 移除原因 |
|---------|---------|
| `test_custom_engines` | VastbaseVectorStore 不接受外部 SQLAlchemy engine |
| `test_custom_sync_engine_only` | 同上 |
| `test_custom_async_engine_only` | 同上 |
| `test_sparse_query_special_character_parsing` | 合并为 `test_sparse_query_string_cleaning`（直接测 regex），无需 DB |
| `test_gin_index_creation_in_database` | 索引验证需实施后通过 pyvastbase API 完成，RED 阶段暂为配置验证 |

### 新增的 Vastbase 特有测试

| 测试 | 目的 |
|------|------|
| `test_from_params_defaults` | 验证 from_params 默认值 |
| `test_from_params_custom` | 验证 from_params 全部自定义参数 |
| `test_collection_name_format` | 验证 _collection_name 格式 |
| `test_multiple_stores_isolation` | 验证多实例之间的数据隔离 |
| `test_full_crud_lifecycle` | 完整生命周期（创建→增→查→读→删→清空） |
| `test_hybrid_search_e2e` | 混合搜索端到端 |
| `test_sync_add_async_query` | 同步写 + 异步读兼容性 |

---

## 三、执行预期

### RED Phase（当前阶段）

```
$ python -m pytest tests/ --collect-only -v
# 期望结果：全部 83 个测试成功收集，无 import 错误

$ python -m pytest tests/ -v
# 期望结果：
# - 7 个 MMR mock 测试 PASS（纯逻辑，无需 DB）
# - 76 个 DB 测试 SKIP（Vastbase 不可达时）或 FAIL（Vastbase 可达但 adapter 未实现）
```

### GREEN Phase（实施完成后）

所有测试应 PASS，验证 VastbaseVectorStore 实现了以下功能：
- 13 个 required_methods
- 4 种 QueryMode
- 14 种 FilterOperator
- Sync + Async 双通道
- 多实例隔离
- HNSW 索引
- 全文搜索（ILIKE fallback）
- Halfvec（FLOAT16_VECTOR）

---

## 四、依赖与运行

### 安装依赖
```bash
pip install llama-index-core>=0.13.0,<0.15 pyvastbase>=0.2.7 pytest pytest-asyncio numpy
```

### 运行环境
- Vastbase V3 (>=3.0.8)
- Python >=3.9

### 运行命令
```bash
# 收集测试（验证 import 无误）
python -m pytest tests/ --collect-only -v

# 运行全部测试
python -m pytest tests/ -v

# 按模式筛选
python -m pytest tests/test_search.py -v
python -m pytest tests/test_filter.py -v -k "gin"
```

---

## 五、关键适配规则遵守情况

| 规则 | 状态 |
|------|------|
| ❌ 不 import sqlalchemy | ✅ 已遵守（零引用） |
| ❌ 不 import psycopg2 | ✅ 已遵守（零引用） |
| ❌ 不写原始 SQL (CREATE TABLE/INDEX/INSERT) | ✅ 已遵守 |
| ❌ 不复制框架测试的 import 和 fixture | ✅ 已遵守 — 使用 pyvastbase 原生 API |
| ✅ 使用 pyvastbase Client / Collection / Schema API | ✅ 已遵守 |
| ✅ 测试逻辑保留框架原始意图 | ✅ 已遵守 |
| ✅ 使用 pytest 标准写法 | ✅ 已遵守 |
| ✅ 测试文件命名符合规范 | ✅ 已遵守 |
