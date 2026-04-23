"""
kernel_eval 评测工程包

用于验证AI生成的Ascend C算子代码（通过whl包传递）
涵盖精度验证和性能评测两个核心维度

主要模块：
- data: 数据层（算子/用例/Golden加载、数据生成）
- eval: 评测层（精度评测、性能评测、算子执行器）
- report: 报告层（JSON+Markdown报告、评分计算）
- utils: 工具层（设备管理、类型映射、精度验证）

使用方法：
    import kernel_eval

    # 加载算子信息
    from kernel_eval.data.operator_loader import OperatorLoader
    loader = OperatorLoader()
    op_info = loader.get_operator("Exp", level=1)

    # 执行评测
    from kernel_eval.eval.evaluator import Evaluator
    evaluator = Evaluator()
    results = evaluator.evaluate_operator("Exp", level=1)
"""

__version__ = "0.1.0"

from .config import Config, get_config

__all__ = ["Config", "get_config", "__version__"]