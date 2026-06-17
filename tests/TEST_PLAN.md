# Test Scout 测试交付报告 — LlamaIndex ChatStore Vastbase 适配

> 日期: 2026-06-17 | 状态: RED Phase (all tests expected to fail)
> 目标框架: llama-index-storage-chat-store-postgres v0.4.0
> 目标仓库: https://github.com/Loger0/LlamaIndex-vastbase.git

---

## 1. 测试覆盖总览

| 维度 | 数量 | 说明 |
|------|------|------|
| 总测试函数 | 37 | 覆盖 14 方法 (7 sync + 7 async) + 初始化 + 集成 |
| 同步方法测试 | 19 | set_messages, get_messages, add_message, delete_messages, delete_message, delete_last_message, get_keys |
| 异步方法测试 | 14 | 全部 7 个 async 方法的原生异步测试 |
| 初始化测试 | 9 | from_params, from_uri, 表名, legacy compat, schema 自动创建 |
| 集成测试 | 8 | 完整 CRUD 流程, 多 key, 复杂消息, 覆盖写, 幂等性 |
| 多模态测试 | 1 | TextBlock + ImageBlock 异步往返 |

---

## 2. 测试文件结构

```
tests/
├── conftest.py                  # Vastbase 连接 fixture + 工具函数
├── test_chat_store_sync.py      # 同步 CRUD 测试 (19 tests)
├── test_chat_store_async.py     # 异步 CRUD + 多模态测试 (16 tests)
├── test_chat_store_init.py      # 初始化 + 表名管理测试 (9 tests)
├── test_chat_store_integration.py # 端到端集成测试 (8 tests)
├── TEST_PLAN.md                 # 本文件
└── NYQUIST_MAP.md               # Nyquist 验证映射表
```

---

## 3. 上游测试场景提取与适配

### 来源: `llama-index-storage-chat-store-postgres/tests/test_chat_store_postgres_chat_store.py`

| # | 上游测试 | 适配后测试 | 适配文件 | 关键变化 |
|---|---------|-----------|---------|---------|
| 1 | `test_class` | `test_vastbase_chat_store_inherits_base_chat_store` | test_chat_store_sync.py | 验证继承关系 |
| 2 | `test_postgres_add_message` | `test_add_message` | test_chat_store_sync.py | pyvastbase Collection API 替代 SQLAlchemy |
| 3 | `test_set_and_retrieve_messages` | `test_set_and_retrieve_messages` | test_chat_store_sync.py | 同逻辑, 不同底层 |
| 4 | `test_delete_messages` | `test_delete_messages` | test_chat_store_sync.py | Collection delete() 替代 SQL DELETE |
| 5 | `test_delete_specific_message` | `test_delete_specific_message` | test_chat_store_sync.py | Python 层数组操作替代 PG array_cat + 切片 |
| 6 | `test_get_keys` | `test_get_keys` | test_chat_store_sync.py | Collection query() 替代 SELECT key |
| 7 | `test_delete_last_message` | `test_delete_last_message` | test_chat_store_sync.py | Python pop() 替代 PG array_length + 切片 |
| 8 | `test_async_postgres_add_message` | `test_async_add_message` | test_chat_store_async.py | AsyncCollection 替代 asyncpg |
| 9 | `test_async_set_and_retrieve_messages` | `test_async_set_and_retrieve_messages` | test_chat_store_async.py | 同上 |
| 10 | `test_adelete_messages` | `test_async_delete_messages` | test_chat_store_async.py | 同上 |
| 11 | `test_async_delete_specific_message` | `test_async_delete_specific_message` | test_chat_store_async.py | 同上 |
| 12 | `test_async_get_keys` | `test_async_get_keys` | test_chat_store_async.py | 同上 |
| 13 | `test_async_delete_last_message` | `test_async_delete_last_message` | test_chat_store_async.py | 同上 |
| 14 | `test_async_multimodal_messages` | `test_async_multimodal_messages` | test_chat_store_async.py | AsyncCollection 替代 asyncpg |
| 15 | `test_table_name_without_prefix` | `test_custom_table_name` | test_chat_store_init.py | Collection name 替代 SQLAlchemy tablename |
| 16 | `test_legacy_table_name_detection` | `test_legacy_table_name_detection` | test_chat_store_init.py | has_collection() 替代 SQL inspect |
| 17 | `test_empty_table_name_defaults_to_chatstore` | `test_empty_table_name_defaults_to_chatstore` | test_chat_store_init.py | 同上 |

### 新增测试 (上游无对应, 基于边界条件分析)

| # | 测试 | 文件 | 测试场景 |
|---|------|------|---------|
| 18 | `test_set_messages_overwrites_existing` | test_chat_store_sync.py | ON CONFLICT 覆盖写入 |
| 19 | `test_get_messages_nonexistent_key` | test_chat_store_sync.py | 不存在 key 返回 [] |
| 20 | `test_set_messages_empty_list` | test_chat_store_sync.py | 空消息列表存储 |
| 21 | `test_add_message_to_new_key` | test_chat_store_sync.py | 不存在的 key 自动创建 |
| 22 | `test_add_message_appends_to_end` | test_chat_store_sync.py | 追加顺序验证 |
| 23 | `test_delete_messages_nonexistent_key` | test_chat_store_sync.py | 删除不存在的 key |
| 24 | `test_delete_message_first_index` | test_chat_store_sync.py | 删除索引 0 |
| 25 | `test_delete_message_last_index` | test_chat_store_sync.py | 删除显式最后索引 |
| 26 | `test_delete_message_returns_deleted` | test_chat_store_sync.py | 返回被删消息 |
| 27 | `test_delete_message_nonexistent_key` | test_chat_store_sync.py | idx 越界返回 None |
| 28 | `test_delete_message_out_of_bounds` | test_chat_store_sync.py | 负索引 + 大索引 |
| 29 | `test_delete_last_message_single_entry` | test_chat_store_sync.py | 仅一条消息 |
| 30 | `test_delete_last_message_nonexistent_key` | test_chat_store_sync.py | 不存在 key |
| 31 | `test_delete_last_message_empty_array` | test_chat_store_sync.py | 空数组 |
| 32 | `test_get_keys_empty_store` | test_chat_store_sync.py | 空 store |
| 33 | `test_full_crud_lifecycle` | test_chat_store_integration.py | 完整增删改查链路 |
| 34 | `test_multiple_keys_independent` | test_chat_store_integration.py | 多 key 隔离 |
| 35 | `test_chat_message_additional_kwargs` | test_chat_store_integration.py | metadata 保留 |
| 36 | `test_message_with_all_roles` | test_chat_store_integration.py | 全部角色类型 |
| 37 | `test_set_messages_idempotency` | test_chat_store_integration.py | 幂等性 |

---

## 4. 适配规则验证

| 规则 | 验证 | 说明 |
|------|------|------|
| ❌ 不使用 SQLAlchemy | ✅ | 所有测试仅 import pyvastbase 相关 |
| ❌ 不使用 psycopg2 | ✅ | 无原始连接 |
| ❌ 不使用原始 SQL | ✅ | 无 CREATE TABLE/INSERT/DELETE 等 |
| ❌ 不复制框架 fixture | ✅ | 使用 Vastbase 自定义 fixture |
| ✅ 使用 pyvastbase API | ✅ | connect, Collection, AsyncCollection |
| ✅ 保留框架原始测试逻辑 | ✅ | 断言逻辑与上游对齐 |
| ✅ 使用 pytest 标准语法 | ✅ | 标准 fixtures + marks |

---

## 5. 环境信息

| 参数 | 值 |
|------|-----|
| Vastbase Host | 172.16.105.107 |
| Vastbase Port | 15432 |
| Database | vastbase |
| Python | >= 3.9 |
| pyvastbase | >= 0.2.7 |
| llama-index-core | >= 0.13.0, < 0.15 |
| pytest | >= 7.0 |
| pytest-asyncio | >= 0.21 |

---

## 6. 运行命令

```bash
# 安装依赖
pip install pyvastbase pytest pytest-asyncio llama-index-core

# 仅收集测试 (验证语法 + import)
python -m pytest tests/ --collect-only -v

# 运行全部测试 (RED phase — 预期全部 FAIL)
python -m pytest tests/ -v

# 运行特定文件
python -m pytest tests/test_chat_store_sync.py -v
python -m pytest tests/test_chat_store_async.py -v
python -m pytest tests/test_chat_store_init.py -v
python -m pytest tests/test_chat_store_integration.py -v
```

---

## 7. 预期结果 (RED Phase)

- 包 stub 存在: `llama_index/storage/chat_store/vastbase/` (VastbaseChatStore 抛出 NotImplementedError)
- `--collect-only`: ✅ 通过 (37 tests collected)
- `-v`: ❌ 全部 FAIL (NotImplementedError) — 等待 adapter-dev 实现 VastbaseChatStore
- 实现完成后: 测试验证 pyvastbase Collection API 行为与上游 PostgresChatStore 对等
