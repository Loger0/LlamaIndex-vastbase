# Multica 工作流状态

## 项目 A: LlamaIndex VectorStore 适配

- **Issue**: [TES-1] 适配 LlamaIndex Vector Store 到 Vastbase（pyvastbase 驱动）
- **目标框架**: LlamaIndex v0.14.22
- **集成模式**: standalone
- **Feature 分支**: `feature/llamaindex-vastbase-adapter`
- **工作区仓库**: https://github.com/Loger0/LlamaIndex-vastbase.git

### Phase 0: 框架诊断 ✅ 完成 — 2026-06-17T09:30:00Z

- 集成模式：standalone（独立 pip 包，不修改框架源码）
- 耦合度：low（仅新增 4 个文件）
- 基类：BasePydanticVectorStore（abstract_class）
- 参考后端：PGVectorStore v0.8.1
- 必须实现方法：13 个（含 sync/async 对）
- 测试基础设施：partial（有官方测试但需扩展）
- 开放决策：5 条（已全部自动 resolve）

### Phase 1: eco-issue-analyst — 需求分析与方案设计 ✅ 完成 — 2026-06-17T10:36:00Z

- 方案架构：Faithful Adapter（忠实适配层）
- 澄清决策：5 条（Q1-Q5，全部确认为 A）
- 产出：Design Spec (14 节, 419 行) + Implementation Plan (16 TDD 任务)
- Plan-Check: pending

### 关键决策 (TES-1)

| ID | 决策类别 | 问题 | 决议 | 状态 |
| -- | -------- | ---- | ---- | ---- |
| D-01 | API 风格 | 同步还是异步？ | sync + async 双实现 | resolved |
| D-02 | 依赖策略 | 保留上游依赖或全替换？ | 全部替换为 pyvastbase | resolved |
| D-03 | 错误处理 | 错误处理策略？ | 沿用 PGVectorStore 模式 | resolved |
| D-04 | 功能裁剪 | 高级 PG 特性是否全移植？ | 全量移植，分阶段实现 | resolved |
| D-05 | 连接管理 | 双连接还是统一管理？ | 单一 VastbaseClient 统一管理 | resolved |

---

## 项目 B: LlamaIndex ChatStore 适配

- **Issue**: [TES-2] LlamaIndex Chat-Store Vastbase 适配
- **目标框架**: llama-index-storage-chat-store-postgres v0.4.0
- **集成模式**: standalone
- **Feature 分支**: `feature/llamaindex-vastbase-adapter`
- **工作区仓库**: https://github.com/Loger0/LlamaIndex-vastbase.git

### Phase 0: 框架诊断 ✅ 完成 — 2026-06-17T03:00:00Z

- 执行 Agent: Framework Analyzer (f84497a4-57e4-4b14-b164-512d06318e79)
- 集成模式：standalone（独立 pip 包，不修改框架源码）
- 耦合度：low（仅新增 2 个文件）
- 基类：BaseChatStore（abstract_class）
- 参考后端：PostgresChatStore v0.4.0（PG 后端，与 Vastbase 最接近）
- 必须实现方法：14 个（7 同步 + 7 异步）
- 测试基础设施：full（官方 Docker PG 测试套件可直接提取适配）
- 开放决策：3 条（已全部自动 resolve）

### 关键决策 (TES-2)

| ID | 决策类别 | 问题 | 决议 | 状态 |
| -- | -------- | ---- | ---- | ---- |
| D-01 | API 风格 | 同步还是异步？ | sync + async 双实现（7+7 方法） | resolved |
| D-02 | 依赖策略 | SQLAlchemy 还是 pyvastbase？ | 完全替换为 pyvastbase Collection API | resolved |
| D-03 | 错误处理 | 错误处理策略？ | 与上游一致：静默 None/空列表 | resolved |

## 产出物索引

### TES-1 (VectorStore)
- `.multica/profiles/llamaindex-profile.json` — Framework Profile
- `.multica/decisions/llamaindex-decisions.yaml` — 开放决策
- `.multica/specs/llamaindex-spec.md` — Design Spec
- `.multica/plans/llamaindex-plan.md` — Implementation Plan

### TES-2 (ChatStore)
- `.multica/profiles/llamaindex-chat-store-postgres-profile.json` — Framework Profile
- `.multica/decisions/llamaindex-chat-store-postgres-decisions.yaml` — 开放决策
- `.multica/STATE.md` — 本文件

- **最后更新**: 2026-06-17T03:00:00Z
