# Multica Workflow State — LlamaIndex ChatStore Vastbase 适配

> Last updated: 2026-06-18

## Project

- **Framework**: llama-index-storage-chat-store-postgres
- **Version**: 0.2.0（注：Issue 描述中提及 v0.4.0，但上游 pyproject.toml 实际为 v0.2.0）
- **integration_mode**: plugin（LlamaHub plugin，通过 `tool.llamahub.import_path` 发现）
- **coupling_level**: low（仅从 core 导入 BaseChatStore + ChatMessage）
- **Target Repo**: https://github.com/Loger0/LlamaIndex-vastbase.git
- **Upstream**: https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/storage/chat_store/llama-index-storage-chat-store-postgres
- **feature_branch**: feature/llamaindex-chat-store-vastbase-adapter

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 框架诊断 (Phase 0) | ✅ 完成 — 2026-06-18 | Framework Analyzer 独立诊断完成 |
| 需求分析 | ⏳ | 待分配 |
| 方案设计 | ⏳ | 待分配 |
| 代码实现 | ⏳ | 待分配 |
| 验证 | ⏳ | 待分配 |

## Key Decisions

| ID | Category | Decision | Status |
|----|----------|----------|--------|
| D-01 | API 风格 | sync + native async（14 方法全部显式覆写） | resolved |
| D-02 | 依赖策略 | 完全替换为 pyvastbase，移除 sqlalchemy+psycopg+asyncpg | resolved |
| D-03 | 测试环境 | 使用已有 Vastbase 实例（172.16.105.107:15432），不搭建 Docker | resolved |
| D-04 | 错误处理 | Silent None/empty list，与上游一致 | resolved |

## Artifacts

| Type | Path | Agent |
|------|------|-------|
| Framework Profile | `.multica/profiles/llamaindex-chat-store-postgres-profile.json` | framework-analyzer |
| Decisions YAML | `.multica/decisions/llamaindex-chat-store-postgres-decisions.yaml` | framework-analyzer |
| STATE.md | `.multica/STATE.md` | framework-analyzer |

## Diagnosis Summary

- **abstraction_type**: abstract_class（BaseChatStore，7 个 @abstractmethod）
- **base_class**: `BaseChatStore(BaseComponent)` → pydantic BaseModel → ABC
- **reference_backend**: PostgresChatStore（正在被替换的目标后端）
- **required_methods**: 7 sync + 7 async（async 可选覆写，基类提供 asyncio.to_thread 默认实现）
- **test_infrastructure**: partial（13 个 Docker 集成测试，基于真实 PostgreSQL 容器）
- **test_strategy**: extend_partial（提取上游测试结构，适配为 Vastbase 集成测试）
- **integration_mode**: plugin — LlamaHub 插件机制，独立 pip 包，通过 `tool.llamahub.import_path` 在 `pyproject.toml` 中注册
- **coupling_level**: low — 仅依赖 BaseChatStore 抽象接口，无框架核心服务注入
