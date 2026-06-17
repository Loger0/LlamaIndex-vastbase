# Multica 工作流状态

## 项目 A: LlamaIndex VectorStore 适配

- **Issue**: [TES-24] LlamaIndex Vector Store 适配 Vastbase — VastbaseVectorStore 开发
- **Issue ID**: 204b117e-92aa-4f90-8c09-112316096004
- **目标框架**: LlamaIndex v0.14.22
- **集成模式**: standalone
- **Feature 分支**: `feature/llamaindex-vastbase-adapter`
- **工作区仓库**: https://github.com/Loger0/LlamaIndex-vastbase.git

### Phase 0: 框架诊断 ✅ 完成

- 集成模式：standalone（独立 pip 包，不修改框架源码）
- 耦合度：low（仅新增 4 个文件）
- 基类：BasePydanticVectorStore（abstract_class）
- 参考后端：PGVectorStore v0.8.1（1698 行）
- 必须实现方法：13 个（含 sync/async 对）
- 测试基础设施：full（3222 行 pytest 套件）

### Phase 1: 需求分析与方案设计 ✅ 完成

- 方案架构：Faithful Adapter（忠实适配层）
- 澄清决策：7 条（D-01~D-07，全部确认）
- Plan-Check: ✅ 通过（5 维验证）
- Convention Spec: ✅ 已提取

### Phase 2: test-scout — 测试侦察与适配 ✅ 完成 — 2026-06-17

- 目标框架测试分析：37 个上游测试场景（test_postgres.py 3222 行）
- Vastbase 适配测试产出：83 个测试（6 个测试文件 + conftest.py）
- 测试收集验证：✅ `pytest --collect-only` 成功（83 tests collected）
- RED Phase 状态：测试全部预期 FAIL（VastbaseVectorStore 尚未实现）
- Nyquist 覆盖率：100%（13 methods + 9 demo + 12 integration）
- 产出物：
  - `tests/conftest.py` — Vastbase 连接 fixtures + 节点数据 fixtures
  - `tests/test_collection_init.py` — 9 tests
  - `tests/test_crud.py` — 15 tests
  - `tests/test_search.py` — 20 tests
  - `tests/test_filter.py` — 21 tests
  - `tests/test_async.py` — 13 tests
  - `tests/test_integration.py` — 5 tests
  - `tests/TEST_PLAN.md` — 测试交付报告
  - `tests/NYQUIST_MAP.md` — Nyquist 验证映射表

### 关键决策

| ID | 决策类别 | 问题 | 决议 | 状态 |
| -- | -------- | ---- | ---- | ---- |
| D-01 | API 风格 | 同步还是异步？ | sync + async 双实现 | resolved |
| D-02 | 依赖策略 | 保留上游依赖或全替换？ | 全部替换为 pyvastbase | resolved |
| D-03 | 错误处理 | 错误处理策略？ | 混合策略（可配置 fail_on_error） | resolved |
| D-04 | MMR | 是否实现 MMR 模式？ | 与 PG 一致，不实现（抛 ValueError） | resolved |
| D-05 | 降级策略 | PG 特性不可用时如何降级？ | 客户端降级（ILIKE + 内存过滤） | resolved |
| D-06 | HYBRID | 混合搜索实现方式？ | pyvastbase 原生 hybrid_search + RRF | resolved |
| D-07 | 文件结构 | 单文件还是多文件？ | 单文件 base.py（~800-1000 行） | resolved |

- **最后更新**: 2026-06-17 (test-scout 完成)
