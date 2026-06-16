# STATE.md — LlamaIndex Vastbase Adapter

## Phase Progress

| Phase | 名称 | 状态 |
|-------|------|------|
| 0 | Feature 分支确认 | ✅ 完成 |
| 1 | 框架源码获取 | ✅ 完成 |
| 2 | 集成模式诊断 | ✅ 完成 |
| 3 | 耦合度分析 | ✅ 完成 |
| 4 | 接口抽象分析 | ✅ 完成 |
| 5 | 测试基础评估 | ✅ 完成 |
| 6 | Demo 规划 | ✅ 完成 |
| 7 | Profile 发布 | ✅ 完成 |
| 7.5 | 开放决策识别 | ✅ 完成 |
| 7.6 | 决策落盘 + 状态汇报 | ✅ 完成 |
| 8 | eco-issue-analyst 需求分析与方案设计 | ✅ 完成 |

## 关键决策

| 决策ID | 类别 | 问题 | 决议 |
|--------|------|------|------|
| D-01 | API风格 | sync vs async | 全量实现 13 个方法，对标 PGVectorStore |
| D-02 | 依赖策略 | 驱动栈选择 | pyvastbase >= 0.2.0 |
| D-03 | 错误处理 | 容错级别 | 默认宽松，支持严格模式切换 |
| D-04 | 连接管理 | 连接生命周期 | 实例级连接，alias 隔离 |
| D-05 | 全文搜索 | TEXT_SEARCH 实现 | 仅全文索引路径，hybrid_search=True 前提 |
| D-06 | Filter操作符 | ?\| / ?& 兼容 | 假设兼容，集成测试验证 |
| D-07 | 向量类型 | use_halfvec 策略 | 保持上游参数 + check_vb_version 检测 |
| D-08 | 混合搜索 | HYBRID 实现 | pyvastbase native hybrid_search + RRFRanker |
| D-09 | 架构方案 | 方案选择 | Collection-Centric Thin Wrapper（方案 1） |

## 产出物索引

| 产出物 | 路径 | 说明 |
|--------|------|------|
| Framework Profile | `.multica/profiles/llamaindex-profile.json` | 框架诊断报告 |
| Open Decisions | `.multica/decisions/decisions.yaml` | 技术决策记录 |
| Design Spec | `.multica/docs/superpowers/spec/llamaindex-spec.md` | 需求规格与方案设计 |

## 框架诊断摘要

- **框架**: LlamaIndex v0.14.22
- **集成模式**: standalone（独立 pip 包，继承 BasePydanticVectorStore）
- **耦合度**: low（无需修改 llama-index-core 源码）
- **抽象类型**: abstract_class
- **参考后端**: PGVectorStore (llama-index-vector-stores-postgres v0.8.1)
- **测试基础**: full（已有完整适配测试套件）
- **Demo 类型**: rag_vector_store_script（8 个场景）

## 下一步

标签切换为「方案待审核」，等待用户审核 Spec。审核通过后进入实施计划阶段。
