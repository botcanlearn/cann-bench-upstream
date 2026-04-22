"""
参数构建器

职责：
1. 解析golden函数签名
2. 根据用例数据构建调用参数
"""

from typing import Dict, List, Any, Callable
from inspect import signature, Parameter


class ParamBuilder:
    """参数构建器"""

    def __init__(self, importer=None):
        self.importer = importer

    def build_call_params(self, golden_func: Callable, case: Any, input_tensors: List) -> Dict[str, Any]:
        """构建golden函数调用参数"""
        sig = signature(golden_func)
        params = {}

        # 分类参数
        tensor_params = []
        tensor_list_params = []
        attr_params = []

        for name, param in sig.parameters.items():
            annotation = str(param.annotation) if param.annotation != Parameter.empty else ""
            if 'List[' in annotation and 'Tensor' in annotation:
                tensor_list_params.append(name)
            elif 'Tensor' in annotation:
                tensor_params.append(name)
            else:
                attr_params.append(name)

        # 匹配张量参数
        tensor_idx = 0
        for name in tensor_params:
            if tensor_idx < len(input_tensors):
                val = input_tensors[tensor_idx]
                params[name] = val[0] if isinstance(val, list) else val
                tensor_idx += 1

        for name in tensor_list_params:
            if tensor_idx < len(input_tensors):
                val = input_tensors[tensor_idx]
                params[name] = val if isinstance(val, list) else [val]
                tensor_idx += 1

        # 处理属性参数
        attrs = case.attrs or {}
        for name in attr_params:
            if name in attrs:
                params[name] = self._convert_value(attrs[name])
            elif sig.parameters[name].default != Parameter.empty:
                params[name] = sig.parameters[name].default

        return params

    def _convert_value(self, value: Any) -> Any:
        """转换特殊值"""
        if isinstance(value, str):
            if value == 'inf':
                return float('inf')
            elif value == '-inf':
                return float('-inf')
            elif value == 'nan':
                return float('nan')
            try:
                return float(value)
            except ValueError:
                pass
        return value