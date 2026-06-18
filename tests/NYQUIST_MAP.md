# Nyquist 验证映射表 — LlamaIndex Vastbase 适配

> 状态：✅ 已通过 | ⏳ 待实现 | ❌ 无覆盖
> 生成时间：2026-06-17 (RED Phase)
> 验证时间：2026-06-18 (Phase 5 逐行验收)

## 方法级映射

| 映射 ID | 需求来源 | 方法 | 测试命令 | 类型 | 状态 |
| :-----: | -------- | ---- | -------- | ---- | :--: |
|  M-01   | required_methods | `client` (property) | `pytest tests/test_collection_init.py::test_vastbase_instance_creation -v` | 单元 |  ✅  |
|  M-02   | required_methods | `add` | `pytest tests/test_crud.py::test_add_nodes -v` | 集成 |  ✅  |
|  M-03   | required_methods | `async_add` | `pytest tests/test_async.py::test_async_add -v` | 集成 |  ✅  |
|  M-04   | required_methods | `delete` | `pytest tests/test_crud.py::test_delete_by_ref_doc_id -v` | 集成 |  ✅  |
|  M-05   | required_methods | `adelete` | `pytest tests/test_async.py::test_async_delete -v` | 集成 |  ✅  |
|  M-06   | required_methods | `delete_nodes` | `pytest tests/test_crud.py::test_delete_nodes_by_ids -v` | 集成 |  ✅  |
|  M-07   | required_methods | `adelete_nodes` | `pytest tests/test_async.py::test_async_delete_nodes -v` | 集成 |  ✅  |
|  M-08   | required_methods | `clear` | `pytest tests/test_crud.py::test_clear_collection -v` | 集成 |  ✅  |
|  M-09   | required_methods | `aclear` | `pytest tests/test_async.py::test_async_clear -v` | 集成 |  ✅  |
|  M-10   | required_methods | `get_nodes` | `pytest tests/test_crud.py::test_get_nodes_parametrized -v` | 集成 |  ✅  |
|  M-11   | required_methods | `aget_nodes` | `pytest tests/test_async.py::test_async_get_nodes -v` | 集成 |  ✅  |
|  M-12   | required_methods | `query` (sync) | `pytest tests/test_search.py::test_search_default -v` | 集成 |  ✅  |
|  M-13   | required_methods | `aquery` (async) | `pytest tests/test_async.py::test_async_query_default -v` | 集成 |  ✅  |

## Demo 场景映射

| 映射 ID | 需求来源 | 场景 | 测试命令 | 类型 | 状态 |
| :-----: | -------- | ---- | -------- | ---- | :--: |
|  D-01   | demo.scenarios | VastbaseVectorStore 实例创建与连接 | `pytest tests/test_collection_init.py::test_vastbase_instance_creation -v` | E2E |  ✅  |
|  D-02   | demo.scenarios | 批量节点写入 (add / async_add) | `pytest tests/test_crud.py::test_add_nodes -v && pytest tests/test_async.py::test_async_add -v` | E2E |  ✅  |
|  D-03   | demo.scenarios | 向量检索 — DEFAULT 模式 | `pytest tests/test_search.py::test_search_default -v` | E2E |  ✅  |
|  D-04   | demo.scenarios | metadata 过滤检索 (14 种 FilterOperator) | `pytest tests/test_filter.py -v` | E2E |  ✅  |
|  D-05   | demo.scenarios | AND/OR 逻辑组合过滤 | `pytest tests/test_filter.py::test_filter_and_combination -v && pytest tests/test_filter.py::test_filter_or_combination -v` | E2E |  ✅  |
|  D-06   | demo.scenarios | 批量删除 (delete_nodes by node_ids + filters) | `pytest tests/test_crud.py::test_delete_nodes_by_ids_and_filters -v` | E2E |  ✅  |
|  D-07   | demo.scenarios | 清空表 (clear / aclear) | `pytest tests/test_crud.py::test_clear_collection -v && pytest tests/test_async.py::test_async_clear -v` | E2E |  ✅  |
|  D-08   | demo.scenarios | 节点读取 (get_nodes by node_ids + filters) | `pytest tests/test_crud.py::test_get_nodes_parametrized -v` | E2E |  ✅  |
|  D-09   | demo.scenarios | 连接关闭 (close) | `pytest tests/test_async.py::test_close_connection -v` | E2E |  ✅  |

## 跨方法集成场景

| 映射 ID | 场景 | 测试命令 | 类型 | 状态 |
| :-----: | ---- | -------- | ---- | :--: |
|  I-01   | 完整 CRUD 流程 (create → add → search → get → delete → clear) | `pytest tests/test_integration.py::test_full_crud_lifecycle -v` | 集成 |  ✅  |
|  I-02   | Hybrid 搜索端到端 (add → hybrid search → verify) | `pytest tests/test_integration.py::test_hybrid_search_e2e -v` | 集成 |  ✅  |
|  I-03   | IndexNode 往返 (add IndexNode → query → verify index_id) | `pytest tests/test_integration.py::test_index_node_roundtrip -v` | 集成 |  ✅  |
|  I-04   | 多实例数据隔离 (store_a add → store_b add → verify isolation) | `pytest tests/test_integration.py::test_multiple_stores_isolation -v` | 集成 |  ✅  |
|  I-05   | Sync 写 + Async 读兼容性 | `pytest tests/test_async.py::test_sync_add_async_query -v` | 集成 |  ✅  |
|  I-06   | HNSW 索引 + 默认搜索 | `pytest tests/test_search.py::test_search_hnsw -v` | 集成 |  ✅  |
|  I-07   | SPARSE 全文搜索 | `pytest tests/test_search.py::test_sparse_query -v` | 集成 |  ✅  |
|  I-08   | HYBRID 混合搜索 + metadata filter | `pytest tests/test_search.py::test_hybrid_query_with_metadata_filters -v` | 集成 |  ✅  |
|  I-09   | GIN 数组索引 + ANY/ALL/CONTAINS | `pytest tests/test_filter.py::test_gin_any -v && pytest tests/test_filter.py::test_gin_all -v && pytest tests/test_filter.py::test_gin_contains -v` | 集成 |  ✅  |
|  I-10   | customize_search_fn 回调集成 | `pytest tests/test_integration.py::test_customize_search_fn_integration -v` | 集成 |  ✅  |
|  I-11   | Halfvec (FLOAT16_VECTOR) 初始化验证 | `pytest tests/test_collection_init.py::test_halfvec_collection_initialization -v` | 单元 |  ✅  |
|  I-12   | MMR 查询 — ValueError 拒绝 | `pytest tests/test_search.py::test_mmr_query_raises_value_error tests/test_search.py::test_mmr_aquery_raises_value_error tests/test_search.py::test_mmr_diverse_selection_utility -v` | 单元 |  ✅  |

## 统计

- **总需求数**: 34 (13 methods + 9 demo + 12 integration)
- **已通过**: 34
- **待实现**: 0
- **无覆盖**: 0
- **覆盖率**: 100%

### 覆盖明细

| 类别 | 总数 | ✅ 已通过 | ⏳ 待实现 | ❌ 无覆盖 | 覆盖率 |
|------|------|-----------|-----------|-----------|--------|
| required_methods | 13 | 13 | 0 | 0 | 100% |
| demo.scenarios | 9 | 9 | 0 | 0 | 100% |
| 跨方法集成 | 12 | 12 | 0 | 0 | 100% |
| **合计** | **34** | **34** | **0** | **0** | **100%** |
