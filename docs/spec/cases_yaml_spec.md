# cases.yaml 规范文档

## 1. input_shape 顺序规范

**核心规则**：`input_shape` 必须与 `proto.yaml` 的 `inputs` **顺序严格一致**。

省略的 optional 参数必须用 `null` 占位，不能跳过。

### 示例

假设 `proto.yaml` 定义：
```yaml
inputs:
- name: x                 # 必需
- name: weight_ih         # 必需
- name: weight_hh         # 必需
- name: bias_ih           # optional
- name: bias_hh           # optional
- name: h0                # optional
```

**正确写法**（省略 h0）：
```yaml
input_shape:
- [10, 4, 64]      # x
- [[96, 64]]       # weight_ih
- [[96, 32]]       # weight_hh
- [[96]]           # bias_ih
- [[96]]           # bias_hh
- null             # h0 省略，用 null 占位
```

**错误写法**（跳过 h0）：
```yaml
input_shape:
- [10, 4, 64]      # x
- [[96, 64]]       # weight_ih
- [[96, 32]]       # weight_hh
- [[96]]           # bias_ih
- [[96]]           # bias_hh
# 错误！缺少 h0 的 null 占位，后端无法判断是省略 h0 还是省略 bias_hh
```

## 2. TensorList 格式规范

TensorList（如 weights）用**嵌套列表**表示：

```yaml
# 单个 TensorList（包含多个 tensor）
- [[96, 64], [96, 32]]  # TensorList of 2 tensors

# 或使用 YAML anchor 引用
- &weight_shape
  - [96, 64]
- *weight_shape        # 引用上面定义的 shape
```

## 3. dtype 对应规范

`dtype` 字段也必须与 `input_shape` 顺序一致：

```yaml
dtype:
- float32           # x 的 dtype
- [float32]         # weight_ih (TensorList，每个 tensor 的 dtype)
- [float32]         # weight_hh
- [float32]         # bias_ih
- [float32]         # bias_hh
- null              # h0 省略时 dtype 也用 null（可选，也可省略该项）
```

## 4. value_range 对应规范

`value_range` 同样需要与 `input_shape` 一致：

```yaml
value_range:
- [-0.1, 0.1]       # x
- [[-0.1, 0.1]]     # weight_ih
- [[-0.1, 0.1]]     # weight_hh
- [[-0.01, 0.01]]   # bias_ih
- [[-0.01, 0.01]]   # bias_hh
- null              # h0 省略
```

## 5. 后端简化处理逻辑

采用本规范后，后端无需复杂分支判断：

```python
# 简化后的参数映射逻辑
params = {}
tensor_idx = 0
for i, input_info in enumerate(proto_inputs):
    shape = input_shapes[i]
    if shape is None:
        # optional 参数省略
        params[input_info.name] = None
    else:
        # 按 shape 生成 tensor
        tensor = generate_tensor(shape, dtypes[i], value_ranges[i])
        params[input_info.name] = tensor
```

## 6. 检查清单

编写 cases.yaml 时请检查：

| 检查项 | 说明 |
|---|---|
| ✅ 顺序一致 | `input_shape` 长度 == `proto.yaml inputs` 长度 |
| ✅ null 占位 | 每个省略的 optional 参数都有 `null` |
| ✅ dtype 对应 | `dtype` 长度与 `input_shape` 一致 |
| ✅ value_range 对应 | `value_range` 长度与 `input_shape` 一致 |
| ✅ TensorList 格式 | 嵌套列表表示多个 tensor |

## 7. 迁移指南

现有 cases.yaml 迁移步骤：

1. 查看 `proto.yaml` 的 `inputs` 定义和顺序
2. 对照现有 `input_shape`，找出缺失的 optional 参数
3. 在正确位置插入 `null`
4. 同步更新 `dtype` 和 `value_range`