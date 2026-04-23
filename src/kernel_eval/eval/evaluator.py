"""
综合评测调度器

职责：
1. 协调精度评测和性能评测执行顺序
2. 支持源码目录扫描、编译、安装
3. 实现评测任务筛选（level/operator/case_id）
4. 生成评测结果
"""

import gc
import traceback
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass

from ..config import Config, get_config
from ..data.case_loader import CaseLoader, CaseInfo
from ..data.golden_loader import GoldenLoader
from ..data.data_generator import DataGenerator
from ..data.operator_loader import OperatorLoader, OperatorInfo
from ..data.package_manager import PackageManager, PackageInfo
from ..utils.device_manager import DeviceManager, DeviceConfig
from ..utils.param_builder import ParamBuilder
from ..eval.op_runner import OpRunner, OpRunResult
from ..eval.accuracy_eval import AccuracyEvaluator, AccuracyResult
from ..eval.perf_eval import PerfEvaluator, PerfResult


@dataclass
class EvalCaseResult:
    """单用例评测结果"""
    case_id: str
    level: int
    operator: str
    case_num: int
    success: bool
    accuracy_result: Optional[AccuracyResult] = None
    perf_result: Optional[PerfResult] = None
    golden_run_result: Optional[OpRunResult] = None
    ai_run_result: Optional[OpRunResult] = None
    error_msg: Optional[str] = None
    baseline_perf_us: float = 0.0

    def get_speedup(self) -> float:
        """计算加速比"""
        if self.perf_result and self.baseline_perf_us > 0:
            return self.baseline_perf_us / self.perf_result.elapsed_us if self.perf_result.elapsed_us > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'level': self.level,
            'operator': self.operator,
            'case_num': self.case_num,
            'success': self.success,
            'accuracy': self.accuracy_result.to_dict() if self.accuracy_result else None,
            'perf': {
                'elapsed_us': self.perf_result.elapsed_us if self.perf_result else 0,
                'speedup': self.get_speedup(),
                'op_times': self.perf_result.op_times if self.perf_result else {},
            } if self.perf_result else None,
            'golden_elapsed_us': self.golden_run_result.elapsed_us if self.golden_run_result else 0,
            'ai_elapsed_us': self.ai_run_result.elapsed_us if self.ai_run_result else 0,
            'error_msg': self.error_msg,
            'baseline_perf_us': self.baseline_perf_us,
        }


@dataclass
class EvalOperatorResult:
    """算子评测结果"""
    operator: str
    level: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    results: List[EvalCaseResult]
    pass_rate: float
    avg_speedup: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operator': self.operator,
            'level': self.level,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'skipped_cases': self.skipped_cases,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'results': [r.to_dict() for r in self.results],
        }


@dataclass
class EvalSessionResult:
    """评测会话结果"""
    operators: List[EvalOperatorResult]
    package_info: Optional[PackageInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operators': [op.to_dict() for op in self.operators],
            'package_info': {
                'source_dir': self.package_info.source_dir if self.package_info else '',
                'whl_path': self.package_info.whl_path if self.package_info else '',
                'run_path': self.package_info.run_path if self.package_info else '',
            } if self.package_info else None,
        }


class Evaluator:
    """综合评测调度器"""

    def __init__(self, config: Config = None):
        self.config = config or get_config()

        # 初始化设备管理器
        device_config = DeviceConfig(
            type=self.config.device_type,
            device_id=self.config.device_id,
            auto_fallback=self.config.auto_fallback,
        )
        self.device_manager = DeviceManager(device_config)

        # 初始化性能评测器
        self.perf_evaluator = PerfEvaluator(
            enabled=self.config.enable_profiler and self.device_manager.is_npu_mode(),
            device_manager=self.device_manager,
            warmup=self.config.warmup,
            repeat=self.config.repeat,
        )

        # 初始化算子执行器
        self.op_runner = OpRunner(self.device_manager, self.perf_evaluator)

        # 初始化精度评测器
        self.accuracy_evaluator = AccuracyEvaluator(self.config.precision_thresholds)

        # 初始化数据层组件
        self.case_loader = CaseLoader(self.config.kernel_bench_root)
        self.golden_loader = GoldenLoader(self.config.kernel_bench_root)
        self.operator_loader = OperatorLoader(self.config.kernel_bench_root)
        self.data_generator = DataGenerator()
        self.param_builder = ParamBuilder(self.golden_loader)

        # 初始化包管理器
        self.package_manager = PackageManager()

        # AI算子模块缓存
        self._ai_op_cache: Dict[str, Callable] = {}

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数"""
        cache_key = operator_name.lower()
        if cache_key in self._ai_op_cache:
            return self._ai_op_cache[cache_key]

        try:
            import cann_bench
            # CamelCase → snake_case（如 MaskedScale → masked_scale）
            def to_snake_case(name: str) -> str:
                import re
                s = re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')
                return re.sub(r'_{2,}', '_', s)

            candidates = [
                to_snake_case(operator_name),
                operator_name.lower(),
                operator_name,
            ]
            for name in candidates:
                if hasattr(cann_bench, name):
                    func = getattr(cann_bench, name)
                    self._ai_op_cache[cache_key] = func
                    return func

            # 尝试 torch.ops.cann_bench
            try:
                import torch
                for name in candidates:
                    if hasattr(torch.ops.cann_bench, name):
                        func = getattr(torch.ops.cann_bench, name)
                        self._ai_op_cache[cache_key] = func
                        return func
            except Exception:
                pass

            raise AttributeError(f"无法找到算子 {operator_name} 在 cann_bench 模块中")

        except ImportError as e:
            raise ImportError(f"无法导入 cann_bench 模块: {e}")

    def evaluate_case(self, case: CaseInfo, ai_op_func: Callable = None) -> EvalCaseResult:
        """评测单个用例"""
        case_id_str = case.get_case_id_str()

        try:
            # 1. 获取golden函数
            golden_func = self.golden_loader.get_golden_function(case.level, case.operator)

            # 2. 生成输入数据
            input_tensors = self.data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
            )

            # 2.5 调用 get_input 预处理（如果存在）
            get_input_func = self.golden_loader.get_input_function(case.level, case.operator)
            if get_input_func is not None:
                params_for_get_input = self.param_builder.build_call_params(golden_func, case, input_tensors)
                input_tensors = get_input_func(**params_for_get_input)
                if isinstance(input_tensors, tuple):
                    input_tensors = list(input_tensors)

            # 3. 构建调用参数
            params = self.param_builder.build_call_params(golden_func, case, input_tensors)

            # 4. 执行Golden函数获取参考结果
            golden_result = self.op_runner.run_golden(golden_func, params, case_id_str, input_tensors)
            if not golden_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    error_msg=f"Golden执行失败: {golden_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 5. 如果没有AI算子，只返回Golden结果（用于测试Golden正确性）
            if ai_op_func is None:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=True,
                    golden_run_result=golden_result,
                    accuracy_result=AccuracyResult(passed=True, dtype=case.dtypes[0] if case.dtypes else 'float32', threshold=0, mere=0, mare=0),
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 6. 执行AI算子
            ai_result = self.op_runner.run_ai_op(ai_op_func, params, case_id_str, input_tensors, enable_perf=True)
            if not ai_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    ai_run_result=ai_result,
                    error_msg=f"AI算子执行失败: {ai_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 7. 精度验证
            dtype = case.dtypes[0] if case.dtypes else 'float32'
            accuracy_result = self.accuracy_evaluator.evaluate(
                ai_output=ai_result.outputs,
                golden_output=golden_result.outputs,
                dtype=dtype,
            )

            # 8. 性能评测结果：profiler路径取perf_result，非profiler路径用simple timing包装
            perf_result = ai_result.perf_result
            if perf_result is None and ai_result.elapsed_us > 0:
                perf_result = PerfResult(
                    case_id=case_id_str,
                    elapsed_us=ai_result.elapsed_us,
                )

            # 清理内存
            self._cleanup_memory()

            return EvalCaseResult(
                case_id=case_id_str,
                level=case.level,
                operator=case.operator,
                case_num=case.case_id,
                success=accuracy_result.passed,
                accuracy_result=accuracy_result,
                perf_result=perf_result,
                golden_run_result=golden_result,
                ai_run_result=ai_result,
                baseline_perf_us=case.baseline_perf_us,
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return EvalCaseResult(
                case_id=case_id_str,
                level=case.level,
                operator=case.operator,
                case_num=case.case_id,
                success=False,
                error_msg=f"评测异常: {e}",
            )

    def evaluate_operator(
        self,
        operator: str,
        level: int,
        ai_op_func: Callable = None,
        case_filter: Dict = None,
    ) -> EvalOperatorResult:
        """
        评测单个算子

        Args:
            operator: 算子名称
            level: 难度级别
            ai_op_func: AI算子函数（可选，如果不提供则只测试Golden）
            case_filter: 用例筛选条件（可选）

        Returns:
            EvalOperatorResult: 算子评测结果
        """
        # 加载用例
        cases = self.case_loader.scan_by_operator(level, operator)

        # 应用筛选条件
        if case_filter:
            cases = self._filter_cases(cases, case_filter)

        if not cases:
            return EvalOperatorResult(
                operator=operator,
                level=level,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        # 清空AI算子缓存（确保使用最新加载的函数）
        self._ai_op_cache.clear()

        # 逐个评测
        results = []
        print(f"[INFO] 评测算子 {operator} (L{level}), 用例数: {len(cases)}")
        for i, case in enumerate(cases, 1):
            case_id_str = case.get_case_id_str()
            result = self.evaluate_case(case, ai_op_func)
            results.append(result)

            # 打印进度
            status_icon = "✅" if result.success else "❌"
            elapsed_str = f"{result.ai_run_result.elapsed_us:.2f}μs" if result.ai_run_result else "N/A"
            speedup_str = f"{result.get_speedup():.2f}x" if result.get_speedup() > 0 else "N/A"
            print(f"[{i}/{len(cases)}] {case_id_str}: {status_icon} (耗时: {elapsed_str}, 加速比: {speedup_str})")

        # 等待性能解析完成
        self.perf_evaluator.wait_all()

        # 计算统计
        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success and r.accuracy_result is not None)
        skipped = sum(1 for r in results if not r.success and r.accuracy_result is None)

        # 计算平均加速比（只考虑通过的用例）
        speedups = [r.get_speedup() for r in results if r.success and r.get_speedup() > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return EvalOperatorResult(
            operator=operator,
            level=level,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped,
            results=results,
            pass_rate=passed / len(cases) if len(cases) > 0 else 0.0,
            avg_speedup=avg_speedup,
        )

    def evaluate_from_source(
        self,
        source_dir: str,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
        verbose: bool = False,
    ) -> EvalSessionResult:
        """
        从源码目录执行完整评测

        流程：
        1. 扫描源码目录
        2. 检查/编译包
        3. 安装包
        4. 扫描接口
        5. 逐个评测算子
        6. 返回评测结果

        Args:
            source_dir: 源码目录路径
            operator_filter: 算子筛选列表
            case_filter: 用例筛选条件
            verbose: 详细输出

        Returns:
            EvalSessionResult: 评测会话结果
        """
        print("")
        print("=" * 60)
        print("开始评测")
        print("=" * 60)

        # 1. 准备环境（编译安装）
        matched_operators, package_info = self.package_manager.prepare_from_source(
            source_dir,
            verbose=verbose,
        )

        if not matched_operators:
            return EvalSessionResult(
                operators=[],
                package_info=package_info,
            )

        # 2. 应用算子筛选
        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 3. 逐个评测算子
        results = []
        for operator_name in matched_operators:
            # 获取算子信息（确定level）
            op_info = self._find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            # 加载AI算子函数
            try:
                ai_op_func = self.load_ai_operator(operator_name)
            except Exception as e:
                print(f"[ERROR] 加载算子 {operator_name} 失败: {e}")
                continue

            # 执行评测
            result = self.evaluate_operator(
                operator=operator_name,
                level=op_info.level,
                ai_op_func=ai_op_func,
                case_filter=case_filter,
            )
            results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(
            operators=results,
            package_info=package_info,
        )

    def evaluate_skip_build(
        self,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
    ) -> EvalSessionResult:
        """
        跳过编译安装，直接评测已安装的cann_bench

        Args:
            operator_filter: 算子筛选列表
            case_filter: 用例筛选条件

        Returns:
            EvalSessionResult: 评测会话结果
        """
        print("")
        print("=" * 60)
        print("开始评测（跳过编译安装）")
        print("=" * 60)

        # 扫描已安装的cann_bench接口
        matched_operators = self.package_manager.prepare_skip_build()

        if not matched_operators:
            return EvalSessionResult(operators=[])

        # 应用算子筛选
        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 逐个评测算子
        results = []
        for operator_name in matched_operators:
            # 获取算子信息（确定level）
            op_info = self._find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            # 加载AI算子函数
            try:
                ai_op_func = self.load_ai_operator(operator_name)
            except Exception as e:
                print(f"[ERROR] 加载算子 {operator_name} 失败: {e}")
                continue

            # 执行评测
            result = self.evaluate_operator(
                operator=operator_name,
                level=op_info.level,
                ai_op_func=ai_op_func,
                case_filter=case_filter,
            )
            results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(operators=results)

    def evaluate_golden_only(
        self,
        operator: str,
        level: int,
        case_filter: Dict = None,
    ) -> EvalOperatorResult:
        """
        仅执行Golden验证（不安装whl包）

        Args:
            operator: 算子名称
            level: 难度级别
            case_filter: 用例筛选条件

        Returns:
            EvalOperatorResult: 算子评测结果
        """
        return self.evaluate_operator(
            operator=operator,
            level=level,
            ai_op_func=None,  # 不加载AI算子
            case_filter=case_filter,
        )

    def _find_operator_info(self, operator_name: str) -> Optional[OperatorInfo]:
        """查找算子定义信息"""
        operators = self.operator_loader.list_operators()
        for op_info in operators:
            if op_info.name == operator_name:
                return op_info
        return None

    def _filter_cases(self, cases: List[CaseInfo], filter_dict: Dict) -> List[CaseInfo]:
        """筛选用例"""
        result = cases
        if 'case_id' in filter_dict:
            result = [c for c in result if c.case_id == filter_dict['case_id']]
        if 'dtype' in filter_dict:
            result = [c for c in result if filter_dict['dtype'].lower() in [d.lower() for d in c.dtypes]]
        return result

    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                import torch_npu
                if hasattr(torch_npu, 'npu'):
                    torch_npu.npu.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass

    def shutdown(self):
        """关闭评测器，释放资源"""
        self.perf_evaluator.shutdown()