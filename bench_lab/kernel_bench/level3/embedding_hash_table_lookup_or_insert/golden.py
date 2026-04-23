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