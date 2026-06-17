# Nyquist 验证映射表 — LlamaIndex ChatStore Vastbase 适配

> 状态: ✅ 已通过 | ⏳ 待实现 | ❌ 无覆盖

## 方法级映射

| 映射 ID | 需求来源 | 方法 | 测试命令 | 类型 | 状态 |
| :-----: | -------- | ---- | -------- | ---- | :--: |
| M-01 | required_methods | set_messages | pytest tests/test_chat_store_sync.py::test_set_and_retrieve_messages -v | 集成 | ⏳ |
| M-02 | required_methods | set_messages (覆盖写) | pytest tests/test_chat_store_sync.py::test_set_messages_overwrites_existing -v | 集成 | ⏳ |
| M-03 | required_methods | set_messages (空列表) | pytest tests/test_chat_store_sync.py::test_set_messages_empty_list -v | 集成 | ⏳ |
| M-04 | required_methods | get_messages | pytest tests/test_chat_store_sync.py::test_get_messages_nonexistent_key -v | 集成 | ⏳ |
| M-05 | required_methods | add_message | pytest tests/test_chat_store_sync.py::test_add_message -v | 集成 | ⏳ |
| M-06 | required_methods | add_message (新建 key) | pytest tests/test_chat_store_sync.py::test_add_message_to_new_key -v | 集成 | ⏳ |
| M-07 | required_methods | add_message (追加顺序) | pytest tests/test_chat_store_sync.py::test_add_message_appends_to_end -v | 集成 | ⏳ |
| M-08 | required_methods | delete_messages | pytest tests/test_chat_store_sync.py::test_delete_messages -v | 集成 | ⏳ |
| M-09 | required_methods | delete_messages (不存在) | pytest tests/test_chat_store_sync.py::test_delete_messages_nonexistent_key -v | 集成 | ⏳ |
| M-10 | required_methods | delete_message | pytest tests/test_chat_store_sync.py::test_delete_specific_message -v | 集成 | ⏳ |
| M-11 | required_methods | delete_message (返回值) | pytest tests/test_chat_store_sync.py::test_delete_message_returns_deleted -v | 集成 | ⏳ |
| M-12 | required_methods | delete_message (越界) | pytest tests/test_chat_store_sync.py::test_delete_message_out_of_bounds -v | 集成 | ⏳ |
| M-13 | required_methods | delete_last_message | pytest tests/test_chat_store_sync.py::test_delete_last_message -v | 集成 | ⏳ |
| M-14 | required_methods | delete_last_message (单条) | pytest tests/test_chat_store_sync.py::test_delete_last_message_single_entry -v | 集成 | ⏳ |
| M-15 | required_methods | delete_last_message (空) | pytest tests/test_chat_store_sync.py::test_delete_last_message_empty_array -v | 集成 | ⏳ |
| M-16 | required_methods | get_keys | pytest tests/test_chat_store_sync.py::test_get_keys -v | 集成 | ⏳ |
| M-17 | required_methods | get_keys (空 store) | pytest tests/test_chat_store_sync.py::test_get_keys_empty_store -v | 集成 | ⏳ |
| M-18 | required_methods | aset_messages | pytest tests/test_chat_store_async.py::test_async_set_and_retrieve_messages -v | 集成 | ⏳ |
| M-19 | required_methods | aget_messages | pytest tests/test_chat_store_async.py::test_async_get_messages_nonexistent_key -v | 集成 | ⏳ |
| M-20 | required_methods | async_add_message | pytest tests/test_chat_store_async.py::test_async_add_message -v | 集成 | ⏳ |
| M-21 | required_methods | adelete_messages | pytest tests/test_chat_store_async.py::test_async_delete_messages -v | 集成 | ⏳ |
| M-22 | required_methods | adelete_message | pytest tests/test_chat_store_async.py::test_async_delete_specific_message -v | 集成 | ⏳ |
| M-23 | required_methods | adelete_last_message | pytest tests/test_chat_store_async.py::test_async_delete_last_message -v | 集成 | ⏳ |
| M-24 | required_methods | aget_keys | pytest tests/test_chat_store_async.py::test_async_get_keys -v | 集成 | ⏳ |
| M-25 | required_methods | 继承 BaseChatStore | pytest tests/test_chat_store_sync.py::test_vastbase_chat_store_inherits_base_chat_store -v | 单元 | ⏳ |
| M-26 | required_methods | from_params | pytest tests/test_chat_store_init.py::test_from_params_creates_instance -v | 集成 | ⏳ |
| M-27 | required_methods | from_uri | pytest tests/test_chat_store_init.py::test_from_uri_parses_vastbase_uri -v | 集成 | ⏳ |
| M-28 | required_methods | 自定义表名 | pytest tests/test_chat_store_init.py::test_custom_table_name -v | 集成 | ⏳ |
| M-29 | required_methods | 空表名默认值 | pytest tests/test_chat_store_init.py::test_empty_table_name_defaults_to_chatstore -v | 集成 | ⏳ |
| M-30 | required_methods | 旧表兼容 | pytest tests/test_chat_store_init.py::test_legacy_table_name_detection -v | 集成 | ⏳ |
| M-31 | required_methods | Collection 自动创建 | pytest tests/test_chat_store_init.py::test_collection_auto_created_on_init -v | 集成 | ⏳ |

## Demo 场景映射

| 映射 ID | 需求来源 | 场景 | 测试命令 | 类型 | 状态 |
| :-----: | -------- | ---- | -------- | ---- | :--: |
| D-01 | demo.scenarios | 多模态消息 (TextBlock+ImageBlock) | pytest tests/test_chat_store_async.py::test_async_multimodal_messages -v | E2E | ⏳ |
| D-02 | demo.scenarios | additional_kwargs 保留 | pytest tests/test_chat_store_integration.py::test_chat_message_additional_kwargs -v | E2E | ⏳ |
| D-03 | demo.scenarios | 全部 MessageRole 类型 | pytest tests/test_chat_store_integration.py::test_message_with_all_roles -v | E2E | ⏳ |

## 跨方法集成场景

| 映射 ID | 场景 | 测试命令 | 类型 | 状态 |
| :-----: | ---- | -------- | ---- | :--: |
| I-01 | 完整 CRUD 流程 (set→get→add→delete_msg→delete_last→delete_all) | pytest tests/test_chat_store_integration.py::test_full_crud_lifecycle -v | 集成 | ⏳ |
| I-02 | 多 key 操作隔离 | pytest tests/test_chat_store_integration.py::test_multiple_keys_independent -v | 集成 | ⏳ |
| I-03 | key 持久化 | pytest tests/test_chat_store_integration.py::test_keys_persistence_across_get_keys -v | 集成 | ⏳ |
| I-04 | 幂等性验证 | pytest tests/test_chat_store_integration.py::test_set_messages_idempotency -v | 集成 | ⏳ |
| I-05 | 覆盖写入缩短数组 | pytest tests/test_chat_store_integration.py::test_set_messages_overwrite_shorter -v | 集成 | ⏳ |

## 统计

- 总需求数: 39 | 已通过: 0 | 待实现: 39 | 无覆盖: 0 | 覆盖率: 100%

> 注: RED Phase — 全部测试预期 FAIL (NotImplementedError)。adapter-dev 实现 VastbaseChatStore 后测试应全部切换为 ✅。
