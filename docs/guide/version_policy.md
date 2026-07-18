# CANN-Bench 版本管理策略

> 版本管理策略文档，记录版本规则、里程碑触发条件、发布流程。

## 1. 双层版本体系

CANN-Bench 采用 **框架版本** + **评测集版本** 双层版本体系：

| VERSION 文件 | 管理范围 | 当前值 |
|-------------|---------|--------|
| `VERSION`（项目根） | 框架核心包（`kernel_eval` + `auto_pipeline`） | 1.0.0 |
| `tasks/VERSION` | 评测集数据（`tasks/` 目录下的算子规格、用例、golden） | 1.0.0 |

两层版本 **独立演进**：
- 框架 bug 修复 → 框架版本 PATCH 升级，评测集不动
- 新增算子到 tasks → 评测集版本 MINOR 升级，框架不动
- 里程碑版本（如评分公式改版 + 新增10个算子） → 两者同时升级

## 2. 版本真相源

**唯一真相源**：项目根 `VERSION` 文件和 `tasks/VERSION` 文件。

所有代码通过 `src/kernel_eval/_version.py` 动态读取：

```python
from kernel_eval._version import FRAMEWORK_VERSION, TASKS_VERSION
# 或
import kernel_eval
kernel_eval.__version__       # 框架版本
kernel_eval.TASKS_VERSION     # 评测集版本
```

**禁止在代码中硬编码版本号**。版本一致性由单元测试 `tests/ut/test_version_consistency.py` 自动校验。

## 3. 语义化版本规范（SemVer）

采用 `MAJOR.MINOR.PATCH` 格式：

### 框架版本升级时机

| 版本段 | 含义 | 升级时机 |
|--------|------|----------|
| **MAJOR** | 架构重大变更 | 评分公式根本性改变、框架不兼容重构 |
| **MINOR** | 功能新增 | 新增评测维度、新增 bench 注册、评分公式兼容性调整 |
| **PATCH** | 问题修复 | Bug 修复、阈值微调、文档修正 |

### 评测集版本升级时机

| 版本段 | 含义 | 升级时机 |
|--------|------|----------|
| **MAJOR** | 用例规范重大变更 | 用例规范根本性改变、难度等级体系重构 |
| **MINOR** | 新增算子 | 新增算子评测集 |
| **PATCH** | 用例修正 | 用例参数修正、golden 修复 |

### Pre-release 标记

开发期可用以下标记：
- `0.3.1-dev` — 开发中版本
- `0.4.0-alpha.1` — 内部测试版本
- `0.4.0-beta.1` — 外部测试版本

## 4. 里程碑驱动的版本发布

### 里程碑触发条件

| 里程碑 | 框架版本升级 | 评测集版本升级 |
|--------|-------------|-------------|
| 新增 ≥5 个算子 | 不动或 PATCH | MINOR |
| 评分公式改版 | MINOR | 不动或 PATCH |
| 框架重大重构 | MAJOR | 不动或 PATCH |
| 两者同时升级（联合里程碑） | MINOR/MAJOR | MINOR/MAJOR |
| Bug 修复（随时） | PATCH | PATCH |

### 版本发布流程

```bash
# 1. 判断是否到达里程碑触发条件
# 2. 修改 VERSION 文件
echo "1.0.0" > VERSION

# 3. 修改 tasks/VERSION（如果评测集也升级）
echo "1.0.0" > tasks/VERSION
echo "# requires-framework: >=1.0.0" >> tasks/VERSION

# 4. 更新 docs/changelog.md（新增版本条目，标注里程碑）

# 5. 运行版本一致性校验
pytest tests/ut/test_version_consistency.py

# 6. 提交
git add VERSION tasks/VERSION docs/changelog.md
git commit -m "release: V1.0.0 / tasks-v1.0.0"

# 7. 打 tag
git tag -a v1.0.0 -m "V1.0.0: <里程碑描述>"
git tag -a tasks-v1.0.0 -m "tasks-v1.0.0: <里程碑描述>"  # 如评测集也升级

# 8. 推送
git push origin master --follow-tags
```

## 5. Git Tag 规范

| Tag 格式 | 含义 | 示例 |
|----------|------|------|
| `vX.Y.Z` | 框架正式发布 | `v0.4.0` |
| `tasks-vX.Y.Z` | 评测集正式发布 | `tasks-v0.4.0` |
| `vX.Y.Z-alpha.N` | 框架内部测试 | `v0.4.0-alpha.1` |
| `vX.Y.Z-beta.N` | 框架外部测试 | `v0.4.0-beta.1` |

里程碑发布时打 **双 tag**：`v1.0.0` + `tasks-v1.0.0`（如果两者同时升级）。

## 6. Changelog 维护规范

保留 `docs/changelog.md` 手写格式，增加以下规范：

1. **框架条目**：标注 `V{version}`，附 git tag
2. **评测集条目**：标注 `tasks-v{version}`，列出新增/修改的算子清单
3. **每个版本条目下方标注 git tag**：便于追溯
4. **PR 中检查版本更新**：是否需要升级 VERSION / tasks/VERSION

Changelog 条目格式示例：

```markdown
## V1.0.0 (2026-07-XX) · git tag: v1.0.0

**里程碑：新增5个L2算子 + 性能评测流程优化**

### 框架变更
- 性能评测增加 InputPool 防缓存攻击机制
- 修复 case_loader 参数解析 bug

### 评测集变更 (tasks-v1.0.0)
- 新增 L2 算子：ApplyRotaryPosEmb、CrossEntropyLoss、DynamicQuant、Gather、GroupNorm
- 修正 L1 Exp 用例 value_range 参数
```

## 7. 评测集兼容性

`tasks/VERSION` 文件中标注兼容的框架版本范围：

```
1.0.0
# requires-framework: >=1.0.0
```

框架加载评测集时应校验兼容性，不兼容时给出警告。

## 8. 不纳入版本管理的目录

| 目录 | 原因 |
|------|------|
| `bench_lab/` | 孵化区，算子经测试成熟后迁移到 `tasks/`。迁移时触发评测集版本升级 |
| `examples/` | 独立的 whl 包示例（cann_bench），有自己的版本号，与框架版本无关 |
| `nouse/` | 已废弃代码，不纳入版本管理 |

## 9. 版本一致性校验

版本一致性由单元测试 `tests/ut/test_version_consistency.py` 自动校验，每次 pytest 运行时检查：

1. VERSION 文件存在性
2. VERSION 文件格式（X.Y.Z）
3. `kernel_eval.__version__` 与 VERSION 文件一致
4. `kernel_eval.TASKS_VERSION` 与 tasks/VERSION 文件一致
5. src/kernel_eval/ 中无残留硬编码版本字符串
6. changelog 最新版本号与 VERSION 文件一致

**修改版本时只需编辑 VERSION 和 tasks/VERSION 文件，所有引用自动跟随变化。**