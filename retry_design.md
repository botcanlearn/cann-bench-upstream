# TaskUnit 重试机制设计

## 目标
当 TaskUnit 中某个 case 因硬件问题卡死时，重新下发 TaskUnit，跳过失败用例，重跑剩余用例。

## 核心特性

### 1. TaskUnit级重试
- 自动恢复已完成的用例（利用现有的 `_try_recover_partial_results`）
- 重新下发包含失败用例的TaskUnit
- 自动选择不同的NPU卡，避开有问题的硬件

### 2. Case级故障隔离
**关键问题**：有问题的case会反复卡死进程，浪费资源

**解决方案**：
- 追踪每个case的失败次数
- 识别多次失败（≥2次）的case
- 后续重试时自动排除这些有问题的case
- 剩余健康的case打包成一个TaskUnit继续执行

**工作流程**：
```
初始:    TaskUnit(用例1-10, Card0) → 用例6卡死
         成功: 1-5, 失败: 6-10

重试1:   TaskUnit(用例6-10, Card1) → 用例6再次卡死
         成功: 7-10, 失败: 6 (失败计数=2)

重试2:   检测到用例6已失败2次，排除用例6
         只重试剩余失败的用例（本例中已无需重试）

最终:    ✅ 用例1-5, 7-10成功  ❌ 用例6失败
```

### 3. 设备可见性映射（修复评审问题）

**问题**：task.device_id是逻辑索引（0, 1, 2...），直接写入`ASCEND_RT_VISIBLE_DEVICES`会导致受限可见性场景下指向错卡。

**场景示例**：
```
父进程: ASCEND_RT_VISIBLE_DEVICES=6,7
task.device_id=0 （逻辑索引）

错误做法: 子进程 ASCEND_RT_VISIBLE_DEVICES=0 → 实际跑到物理卡0 ❌
正确做法: 子进程 ASCEND_RT_VISIBLE_DEVICES=6 → 实际跑到物理卡6 ✅
```

**解决方案**：
- 从父进程环境变量解析物理设备列表
- 将task.device_id（逻辑索引）映射为物理设备号
- 子进程使用物理设备号，确保运行在正确的卡上

**实现**：
```python
# 解析父进程可见设备: "6,7" → ['6', '7']
visible_tokens = _visible_device_tokens_from_env(env)

# 逻辑索引映射物理设备: 逻辑0 → 物理'6', 逻辑1 → 物理'7'
physical_device = visible_tokens[task.device_id]

# 子进程使用物理设备号
env['ASCEND_RT_VISIBLE_DEVICES'] = physical_device
```

## 配置选项

```python
@dataclass
class ProcessConfig:
    max_retries: int = 1                         # 最大重试次数（0=禁用）
    retry_on_timeout: bool = True                # 超时是否重试
    retry_on_oom: bool = True                    # OOM 是否重试
    retry_on_failure: bool = True                # 子进程异常是否重试
    exclude_repeatedly_failed_cases: bool = True # 排除多次失败的case
```

**使用示例**：
```python
# 默认配置（推荐）
config = ProcessConfig(max_retries=1)

# 完全禁用重试
config = ProcessConfig(max_retries=0)

# 选择性重试
config = ProcessConfig(
    max_retries=2,
    retry_on_timeout=True,
    retry_on_oom=True,
    retry_on_failure=False,  # 不重试进程崩溃
)

# 禁用case隔离（所有失败的case都重试）
config = ProcessConfig(
    max_retries=2,
    exclude_repeatedly_failed_cases=False,
)
```

## 数据结构

```python
@dataclass
class TaskUnit:
    operator: str
    rel_path: str
    cases: List[CaseSpec]
    device_id: int                    # 逻辑设备索引（0, 1, 2...）
    retry_count: int = 0              # 当前重试次数
    excluded_devices: set = None      # 排除的设备ID集合
    parent_task_id: str = None        # 父任务ID（追踪用）
```

## 核心逻辑

### 设备可见性映射
```python
def _visible_device_tokens_from_env(env):
    """解析父进程可见的物理设备列表"""
    for var in _DEVICE_VISIBILITY_ENV_VARS:
        raw = env.get(var, "")
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if tokens and not any(t.lower() in {"all", "none"} for t in tokens):
            return tokens
    return []

def _physical_device_token_for_task(task, env):
    """将逻辑device_id映射为物理设备token"""
    visible_tokens = _visible_device_tokens_from_env(env)
    if visible_tokens and 0 <= task.device_id < len(visible_tokens):
        return visible_tokens[task.device_id]
    return str(task.device_id)
```

### 重试逻辑
```python
# 1. 维护case失败计数
case_failure_count: Dict[str, int] = defaultdict(int)

# 2. 任务执行失败后，判断是否重试
if should_retry and failed_cases:
    # 更新失败计数
    for case in failed_cases:
        case_failure_count[case.get_case_id_str()] += 1
    
    # 过滤多次失败的case
    cases_to_retry = [
        c for c in failed_cases 
        if case_failure_count[c.get_case_id_str()] < 2
    ]
    
    # 创建重试任务
    if cases_to_retry:
        retry_task = TaskUnit(
            operator=task.operator,
            cases=cases_to_retry,
            device_id=new_device,
            retry_count=task.retry_count + 1,
            excluded_devices=task.excluded_devices | {task.device_id},
        )
        retry_queue.append(retry_task)
```

### _select_device_for_retry()
```python
def _select_device_for_retry(original_device, excluded_devices, healthy_cards):
    """选择设备，优先选择未失败的卡"""
    available = [d for d in healthy_cards if d not in excluded_devices]
    return available[0] if available else original_device
```

## 日志输出

```
[INFO] 重试策略: 最大重试 1 次 (timeout=True, oom=True, failure=True)

[WARN] TaskUnit attention@Card2 超时 (600s)
[INFO] attention: 超时后恢复 5 个已完成用例，合成 5 个超时失败用例
[INFO] 重试任务已加入队列: attention (retry 1/1, 5 个用例, Card2→Card3, 原因: timeout)

[WARN] 用例 attention_case_6 已连续失败 2 次，不再重试，标记为最终失败
[INFO] 重试任务已加入队列: attention (retry 2/2, 4 个用例, 排除 1 个多次失败的用例, Card3→Card4)

[INFO] [重试 1/2] Card 4: attention ✅ (4/4) (retry 2)
```

## 设计要点

### 可重试的失败类型
- ✅ timeout：超时（硬件卡死、负载过高）
- ✅ oom_killed：OOM Kill（内存不足）
- ✅ subprocess_failure：子进程异常（硬件故障、环境问题）
- ❌ 代码错误：不应该重试

### Case隔离阈值
- **第1次失败**：可能是偶发问题，允许重试
- **第2次失败**：很可能是case本身问题，停止重试
- 阈值=2：平衡误判率和效率

### 设备选择策略
1. 优先选择未失败的健康卡
2. 所有卡都失败时，回退到原设备（可能是偶发问题）

### 设备映射策略
1. 解析父进程的`ASCEND_RT_VISIBLE_DEVICES`等环境变量
2. 逻辑索引映射为物理设备号
3. 子进程使用物理设备号，确保运行在正确的卡上

## 优势

1. **提高成功率**：硬件偶发问题不会导致整个用例组失败
2. **智能隔离**：自动识别并排除有问题的case
3. **资源优化**：避免有问题的case反复卡死进程
4. **批量执行**：保持执行效率，不是单case执行
5. **可配置**：灵活适应不同场景
6. **向后兼容**：默认启用，也可禁用
7. **正确映射**：修复受限可见性场景的设备映射问题

## 风险控制

1. **重试次数限制**：避免无限重试
2. **超时计算**：重试任务的超时时间重新计算
3. **日志记录**：清晰记录重试历史，便于诊断
4. **Case隔离**：防止坏case浪费重试机会
5. **设备映射**：确保子进程运行在正确的物理卡上

## 测试

- `tests/ut/test_retry_mechanism.py`：重试机制测试（6个测试）
- `tests/ut/test_device_visibility.py`：设备可见性映射测试（6个测试）
  - 测试父进程可见性为`6,7`时的映射
  - 测试单卡模式`--device-id 6`的映射
  - 测试逻辑索引到物理设备的转换

## 实现文件

- `src/kernel_eval/eval/process_pool.py`
  - `ProcessConfig`：新增重试配置字段
  - `TaskUnit`：新增重试追踪字段
  - `_visible_device_tokens_from_env()`：解析物理设备列表
  - `_physical_device_token_for_task()`：逻辑→物理映射
  - `_should_narrow_child_visibility()`：判断是否收窄可见性
  - `_child_device_id()`：计算子进程逻辑device_id
  - `_build_env_for_task()`：构建子进程环境变量（含物理设备映射）
  - `_select_device_for_retry()`：设备选择
  - `_process_retry_queue()`：处理重试队列
  - `evaluate_task_units()`：主调度逻辑

