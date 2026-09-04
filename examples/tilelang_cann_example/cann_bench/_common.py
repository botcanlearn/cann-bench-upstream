__version__ = "1.0.0"

import torch
import tilelang
from tilelang import language as T

# TL_ASCEND_* 配置项仅由 TileLang-Ascend（昇腾适配版）提供；
# PyPI 上游 tilelang（0.1.x）不含 Ascend 后端。缺失时给出明确指引，
# 避免评测时裸 AttributeError。
_REQUIRED_PASS_CONFIG_KEYS = (
    "TL_ASCEND_AUTO_SYNC",
    "TL_ASCEND_MEMORY_PLANNING",
    "TL_ASCEND_AUTO_CV_COMBINE",
)

_missing_keys = [
    key
    for key in _REQUIRED_PASS_CONFIG_KEYS
    if not hasattr(tilelang.PassConfigKey, key)
]
if _missing_keys:
    raise ImportError(
        "当前安装的 tilelang 缺少 Ascend 专用配置项: "
        + ", ".join(_missing_keys)
        + "。\n本示例依赖 TileLang-Ascend（昇腾适配版，"
        "https://github.com/tile-ai/tilelang-ascend）；"
        "PyPI 上的上游 tilelang（0.1.x）不包含 Ascend 后端，无法运行本示例。\n"
        "请按 examples/tilelang_cann_example/README.md「前置条件」"
        "安装 TileLang-Ascend 后重试。"
    )


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# 手动同步变体（kernel 内自行插入 T.barrier_all()/set_flag，关闭自动同步）
PASS_CONFIGS_MANUAL_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


def torch_dtype_to_tl(dtype):
    if dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
