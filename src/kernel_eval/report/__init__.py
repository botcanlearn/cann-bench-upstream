"""
报告层模块

职责：
1. 评测报告生成（JSON + Markdown）
2. Summary生成（几何平均加速比）
3. 评分计算（功能得分 + 性能得分）
"""

from .report_generator import ReportGenerator, EvalResult
from .scoring import ScoringCalculator, ScoreInfo
from .summary_generator import (
    EvaluationSummary, OperatorSummary,
    calculate_geometric_mean, generate_summary, render_summary_markdown, save_summary,
)

__all__ = [
    "ReportGenerator", "EvalResult",
    "ScoringCalculator", "ScoreInfo",
    "EvaluationSummary", "OperatorSummary",
    "calculate_geometric_mean", "generate_summary", "render_summary_markdown", "save_summary",
]