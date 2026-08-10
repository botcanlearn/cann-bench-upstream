#!/usr/bin/env python3
# coding=utf-8
"""pypto_pro AI 算子子进程执行器。

在独立子进程中执行 pypto_pro JIT kernel，确保每个 case 的 pypto_pro runtime
全局状态完全隔离，避免不同 class kernel 连续执行时的 AICore 异常。

由 evaluator._run_ai_op_isolated() 通过 subprocess.Popen 启动。

两种模式：
  --no-perf (prof_dir 为空):   简单执行一次 fn()，返回 outputs
  profiler  (prof_dir 非空):   启动 torch_npu.profiler 采集性能数据，
                               trace 写入 prof_dir，返回 outputs
"""

import os
import sys
import json
import pickle
import traceback
import tempfile
import shutil
import logging

import torch
import torch_npu


class _NpuProfileConditioner:
    """Stabilize NPU frequency and L2 state for PyPTO-Pro profiling."""

    def __init__(self, device):
        from cann_bench_utils import cann_bench_cache_clean, cann_bench_warmup

        self._warmup_op = cann_bench_warmup
        self._cache_clean_op = cann_bench_cache_clean
        self._mm1 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
        self._mm2 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
        self._reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).to(device)

    def stabilize(self):
        """Boost NPU frequency and establish a clean initial L2 state."""
        self._warmup_op(self._mm1, self._mm2)
        torch.npu.synchronize(self._mm1.device)
        self._cache_clean_op(self._reduce_input)
        torch.npu.synchronize(self._reduce_input.device)

    def clear_cache(self):
        """Clear L2 immediately before one active profiler repeat."""
        self._cache_clean_op(self._reduce_input)
        torch.npu.synchronize(self._reduce_input.device)


def _resolve_profiler_level(profiler_level):
    level_map = {
        "Level1": torch_npu.profiler.ProfilerLevel.Level1,
        "Level2": torch_npu.profiler.ProfilerLevel.Level2,
    }
    return level_map.get(profiler_level, torch_npu.profiler.ProfilerLevel.Level1)


def _profiler_activities():
    """Match the standard single-device evaluator: collect NPU events only."""
    return [torch_npu.profiler.ProfilerActivity.NPU]


def _run_preflight(fn, conditioner=None):
    if conditioner is not None:
        conditioner.stabilize()
    fn()
    torch.npu.synchronize()


def _run_profile_steps(fn, prof, warmup, repeat, conditioner=None):
    fn_exc = None
    for i in range(warmup + repeat):
        if conditioner is not None and i >= warmup:
            conditioner.clear_cache()
        try:
            fn()
            torch.npu.synchronize()
        except BaseException as exc:
            fn_exc = exc
            prof.step()
            break
        prof.step()
    return fn_exc


def _run_with_profiler(fn, prof_dir, warmup, repeat, profiler_level="Level1",
                       freq_boost=True, device="npu"):
    """在 torch_npu.profiler 下执行 fn，trace 输出到 prof_dir。

    与普通单卡评测保持相同的升频、L2 cache 和 profiler 配置；仍保留
    PyPTO-Pro 每 case 独立子进程及独立 JIT 目录。
    """
    # 抑制 profiler parser 日志
    og_basicConfig = logging.basicConfig
    logging.basicConfig = lambda **kw: og_basicConfig(**{**kw, "level": logging.ERROR, "force": True})
    try:
        for name in ['', 'torch', 'torch_npu', 'torch_npu.profiler', 'ascend', 'profiler']:
            lg = logging.getLogger(name)
            lg.setLevel(logging.ERROR)
            lg.handlers = []
            lg.addHandler(logging.NullHandler())
    finally:
        logging.basicConfig = og_basicConfig

    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'

    conditioner = _NpuProfileConditioner(device) if freq_boost else None

    # 先升频并清理初始 L2，再执行不带 profiler 的 pre-flight。
    _run_preflight(fn, conditioner)

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    sink_file = tempfile.NamedTemporaryFile(
        mode='w+', prefix='trace_profiler_', suffix='.log', delete=False
    )
    sink_fd = sink_file.fileno()

    try:
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=_resolve_profiler_level(profiler_level),
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        )

        with torch_npu.profiler.profile(
            activities=_profiler_activities(),
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as prof:
            fn_exc = _run_profile_steps(
                fn, prof, warmup, repeat, conditioner=conditioner
            )
            if fn_exc is not None:
                raise fn_exc

        # 等待 profiler 异步解析完成
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            pool = MultiProcessPool()
            pool.close_pool(wait=True)
        except Exception:
            pass

    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        logging.basicConfig = og_basicConfig
        sink_file.close()
        try:
            os.unlink(sink_file.name)
        except OSError:
            pass


def main():
    work_dir = sys.argv[1]
    func_name = sys.argv[2]        # cann_bench 中的函数名，如 "foreach_addcdiv_scalar"
    device_id = int(sys.argv[3])   # NPU 设备 ID
    prof_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else ""  # profiler 输出目录
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 3
    repeat = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else 5
    profiler_level = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else "Level1"
    freq_boost = sys.argv[8].lower() not in ("0", "false", "no") if len(sys.argv) > 8 else True

    # 1. 读取输入数据
    with open(os.path.join(work_dir, "input.pkl"), "rb") as f:
        input_tensors = pickle.load(f)
    with open(os.path.join(work_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    # 2. 初始化 NPU
    # 隔离模式下（只可见 1 张卡）一律用 0，与 cli.py cmd_eval_child 保持一致。
    if torch.npu.device_count() == 1:
        device_id = 0
    torch_npu.npu.set_device(device_id)
    torch.npu.set_compile_mode(jit_compile=False)

    # 3. 导入 cann_bench 并找到算子函数
    import cann_bench
    func = getattr(cann_bench, func_name, None)
    if func is None:
        raise AttributeError(f"cann_bench 中未找到算子函数 {func_name}")

    # 4. 把 cann_bench 包下所有 class 子目录加入 sys.path，
    #    使 wrapper 中的 `from <op>_golden import _get_device` 能找到模块
    #    （dispatcher._load 在加载后移除 sys.path，但 wrapper 函数调用时
    #     可能需要 golden 模块，且不同 class 的 golden 文件名相同）
    pkg_dir = os.path.dirname(cann_bench.__file__)
    for name in sorted(os.listdir(pkg_dir)):
        cpath = os.path.join(pkg_dir, name)
        if os.path.isdir(cpath) and name.startswith("c"):
            sys.path.insert(0, cpath)

    # 5. chdir 到独立临时目录（pypto_pro JIT build_dir 隔离）
    jit_dir = tempfile.mkdtemp(prefix="jit_iso_")
    os.chdir(jit_dir)

    # 6. 迁移输入到设备
    dev = f"npu:{device_id}"

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(dev)
        if isinstance(value, list):
            return [to_device(v) for v in value]
        return value

    updated_params = {key: to_device(val) for key, val in params.items()}

    # 7. 执行算子
    try:
        if prof_dir:
            # profiler 模式: 在 torch_npu.profiler 下执行 warmup+repeat 次
            os.makedirs(prof_dir, exist_ok=True)
            # 清理上次评测遗留的时间戳子目录
            for entry in os.listdir(prof_dir):
                entry_path = os.path.join(prof_dir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)

            last_outputs = [None]

            def _fn():
                last_outputs[0] = func(**updated_params)

            _run_with_profiler(
                _fn,
                prof_dir,
                warmup,
                repeat,
                profiler_level=profiler_level,
                freq_boost=freq_boost,
                device=dev,
            )
            outputs = last_outputs[0]
        else:
            # --no-perf 模式: 简单执行一次
            outputs = func(**updated_params)
            torch.npu.synchronize()

        # 输出转到 CPU 再保存（避免 NPU tensor pickle 警告）
        def to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.cpu()
            if isinstance(value, list):
                return [to_cpu(v) for v in value]
            return value

        with open(os.path.join(work_dir, "output.pkl"), "wb") as f:
            pickle.dump(to_cpu(outputs), f)

        result = {"success": True, "error": None}
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # 8. 清理 JIT 临时目录
    shutil.rmtree(jit_dir, ignore_errors=True)

    with open(os.path.join(work_dir, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
