# EmbeddingHashTableLookupOrInsert 算子 API 描述

## 1. 算子简介

哈希表动态查找/插入算子，根据 key 值在哈希表中查找，存在则返回对应 value，不存在则插入并返回默认值。

**主要应用场景**：
- 推荐系统中的动态特征嵌入
- 大规模稀疏特征的在线学习
- 哈希表形式的 embedding 管理

**算子特征**：
- 难度等级：L3（HashTable）
- 支持动态哈希表查找与插入
- 仅支持 Ascend 950PR/Ascend 950DT 产品

## 2. 算子定义

### 功能说明

根据 key 值查看 table 中是否存在 key：
- **存在**：不插入 value，返回当前位置上的值
- **不存在**：对 key 进行 hash，找到位置后插入 value（默认值）

### 数学公式

$$
\text{hash\_idx} = \text{hash}(key) \mod \text{bucket\_size}
$$

$$
\text{values}[i] = \begin{cases}
  \text{table}[\text{hash\_idx}] & \text{if key exists in table} \\
  \text{default\_value} & \text{if key not exists}
\end{cases}
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.embedding_hash_table_lookup_or_insert(
    Tensor table_handle, 
    Tensor keys, 
    int bucket_size, 
    int embedding_dim, 
    str filter_mode="no_filter", 
    int filter_freq=0, 
    bool default_key_or_value=False, 
    int default_key=0, 
    float default_value=0.0, 
    bool filter_key_flag=False, 
    int filter_key=-1
) -> Tensor values
```

### 输入参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| table_handle | Tensor | 哈希表句柄，包含表头地址等信息，shape 为 [5]，dtype 为 int64 |
| keys | Tensor | 要查找/插入的 key 序列，dtype 为 int64 |

### 必选属性

| 参数 | 类型 | 描述 |
|------|------|------|
| bucket_size | int | 哈希表容量 |
| embedding_dim | int | value 的维度 |

### 可选属性

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| filter_mode | str | "no_filter" | 过滤模式，可选 "no_filter" 或 "counter" |
| filter_freq | int | 0 | 过滤阈值，启用 counter 模式时使用 |
| default_key_or_value | bool | False | True 返回 default_key 的值，False 返回 default_value |
| default_key | int | 0 | 用户设置的默认 key |
| default_value | float | 0.0 | 用户设置的默认 value |
| filter_key_flag | bool | False | True 启用 filter_key 功能 |
| filter_key | int | -1 | 过滤输入 key，启用时返回 default_value |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| values | (N, embedding_dim) | float32 | 查找到或插入后的值 |

### 数据类型

| 输入 (table_handle) | 输入 (keys) | 输出 (values) |
|---------------------|-------------|---------------|
| int64 | int64 | float32 |

### 规则与约束

- 仅支持 Ascend 950PR/Ascend 950DT 产品
- `table_handle` shape 必须为 [5]
- `keys` 为任意 shape 的 int64 张量
- `bucket_size` 和 `embedding_dim` 为必选属性
- 输出 `values` shape 为 (keys_numel, embedding_dim)

## 4. 精度要求

计算结果与参考实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float32 | 相对误差 | 1e-4 | 1e-4 |

## 5. 标准 Golden 代码

```python
import torch
import numpy as np

"""
EmbeddingHashTableLookupOrInsert 算子 Torch Golden 参考实现

哈希表动态查找/插入算子：
- 根据 key 查看 table 中是否存在
- 存在：返回对应位置的值
- 不存在：插入并返回默认值

对标 ops-nn EmbeddingHashTableLookupOrInsert
"""
class HashTable:
    """简单的哈希表实现，用于 Golden 参考"""
    def __init__(self, bucket_size: int, embedding_dim: int):
        self.bucket_size = bucket_size
        self.embedding_dim = embedding_dim
        # 哈希表存储：key -> value
        self.table = {}  # 使用 dict 模拟哈希表
        # 模拟 table_handle (5个元素)
        self.table_handle = np.zeros(5, dtype=np.int64)
    
    def hash_func(self, key: int) -> int:
        """简单 hash 函数"""
        return key % self.bucket_size
    
    def lookup_or_insert(self, key: int, default_value: float = 0.0) -> np.ndarray:
        """查找或插入"""
        hash_idx = self.hash_func(key)
        if hash_idx in self.table:
            return self.table[hash_idx]
        else:
            # 不存在，插入默认值
            value = np.full(self.embedding_dim, default_value, dtype=np.float32)
            self.table[hash_idx] = value
            return value

def embedding_hash_table_lookup_or_insert(
    table_handle: torch.Tensor,
    keys: torch.Tensor,
    bucket_size: int,
    embedding_dim: int,
    filter_mode: str = "no_filter",
    filter_freq: int = 0,
    default_key_or_value: bool = False,
    default_key: int = 0,
    default_value: float = 0.0,
    filter_key_flag: bool = False,
    filter_key: int = -1
) -> torch.Tensor:
    """
    EmbeddingHashTableLookupOrInsert 算子
    
    Args:
        table_handle: 哈希表句柄，shape [5]
        keys: 要查找/插入的 key 序列
        bucket_size: 哈希表容量
        embedding_dim: value 维度
        filter_mode: 过滤模式
        filter_freq: 过滤阈值
        default_key_or_value: 返回 default_key 或 default_value
        default_key: 默认 key
        default_value: 默认 value
        filter_key_flag: 启用 filter_key
        filter_key: 过滤 key
    
    Returns:
        values: 查找到或插入后的值，shape (N, embedding_dim)
    """
    # 创建哈希表
    hash_table = HashTable(bucket_size, embedding_dim)
    
    # 处理 keys
    keys_flat = keys.flatten()
    num_keys = keys_flat.numel()
    
    # 输出 values
    values = np.zeros((num_keys, embedding_dim), dtype=np.float32)
    
    for i, key in enumerate(keys_flat):
        key_val = int(key.item())
        
        # filter_key 功能
        if filter_key_flag and key_val == filter_key:
            values[i] = np.full(embedding_dim, default_value, dtype=np.float32)
            continue
        
        # 查找或插入
        result = hash_table.lookup_or_insert(key_val, default_value)
        values[i] = result
    
    return torch.from_numpy(values)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 创建哈希表句柄
table_handle = torch.zeros(5, dtype=torch.int64, device="npu")

# 要查找的 keys
keys = torch.tensor([1, 2, 3, 1, 5], dtype=torch.int64, device="npu")

# 调用算子
values = cann_bench.embedding_hash_table_lookup_or_insert(
    table_handle, 
    keys, 
    bucket_size=1024, 
    embedding_dim=128
)
# 输出 shape: (5, 128)
```

### 性能基线参考

测试用例覆盖不同 bucket_size、embedding_dim、keys 数量的组合。

### 相关算子

- **EmbeddingHashTableImport**: 哈希表导入算子
- **EmbeddingHashTableApplyAdamW**: 哈希表 AdamW 优化器更新算子