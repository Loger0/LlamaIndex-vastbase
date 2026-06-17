# Multica Workspace STATE

> 最后更新：2026-06-17T11:20:00Z

---

## 项目 A: KV-Store Vastbase 适配

| 字段 | 值 |
| --- | --- |
| 框架 | llama-index-storage-kvstore-postgres |
| 版本 | 0.4.0 |
| 集成模式 | standalone |
| Feature 分支 | feature/llamaindex-kvstore-vastbase-adapter |
| Phase 0 (框架诊断) | ✅ 完成 — 2026-06-17 |

## 项目 B: Chat-Store Vastbase 适配

| 字段 | 值 |
| --- | --- |
| 框架 | llama-index-storage-chat-store-postgres |
| 版本 | 0.4.0 |
| 集成模式 | standalone |
| Feature 分支 | feature/llamaindex-chat-store-vastbase-adapter |
| Phase 0 (框架诊断) | ✅ 完成 — 2026-06-17 |

### 关键决策

| ID | 类别 | 决议 |
| --- | --- | --- |
| D-01 | API 风格 | 同步+原生异步双实现（7+7=14 方法） |
| D-02 | 依赖策略 | 完全替换为 pyvastbase >= 0.2.7 |
| D-03 | 错误处理 | 与上游一致：静默 None/空列表 |

---

## 产出物索引

| 文件 | 关联项目 | 描述 |
| --- | --- | --- |
| .multica/profiles/llamaindex-chat-store-postgres-profile.json | 项目 B | ChatStore Framework Profile |
| .multica/decisions/llamaindex-chat-store-postgres-decisions.yaml | 项目 B | ChatStore 开放决策 |
