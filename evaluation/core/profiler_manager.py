"""Kernel-only timing via torch_npu.profiler — restored from the cann-bench
original. Using chrome-trace parsing (cat-less events = device kernels) we get
the NPU-side wall time, stripping Python dispatch + OpCommand overhead so
aclnn/direct submissions can be compared on kernel time alone.

Flow:
  1. Run warmup + active steps inside torch_npu.profiler.profile() with a
     schedule, after a MatMul+ReduceMax pair that boosts NPU frequency and
     clears the L2 cache (same pattern used in the original harness).
  2. on_trace_ready dumps trace_view.json into a temp dir.
  3. Parse the trace: sum `dur` on device-kernel events (those without a `cat`
     field), filtering out profiler/HostToDevice/Free/Computing and the
     freq-boost kernels (MatMul, ReduceMax, ReduceD).
  4. Return the total kernel time averaged over `repeat` steps.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


_WARMUP_KERNEL_KEYWORDS = ("MatMul", "ReduceMax", "ReduceD")
_SKIP_EVENT_PREFIXES = ("empty_tensor", "profiler", "torch_to_npu",
                        "HostToDevice", "Free", "Computing")


@dataclass
class ProfilerResult:
    elapsed_us: float = 0.0
    error: Optional[str] = None


def _prepare_warmup_tensors():
    """Matrix + reduce tensors sized to cover typical AI-core/L2 footprints."""
    import torch
    mm1 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    mm2 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).npu()
    return mm1, mm2, reduce_input


def _boost_freq_and_clear_cache(mm1, mm2, reduce_input):
    import torch
    torch.matmul(mm1, mm2)
    torch.npu.synchronize()
    torch.max(reduce_input)
    torch.npu.synchronize()


def _parse_trace(trace_file: str) -> float:
    """Sum device-kernel durations from a chrome trace JSON."""
    if not trace_file or not os.path.exists(trace_file):
        return 0.0
    total = 0.0
    with open(trace_file) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = ev.get("dur", 0)
        if dur <= 0:
            continue
        name = ev.get("name", "")
        if not name or name.startswith(_SKIP_EVENT_PREFIXES):
            continue
        if any(kw in name for kw in _WARMUP_KERNEL_KEYWORDS):
            continue
        # Device kernel events have no `cat` field; host-side events do.
        if "cat" in ev:
            continue
        total += dur
    return total


def measure_kernel_us(fn: Callable[[], None], warmup: int = 3, repeat: int = 5,
                      freq_boost: bool = True) -> ProfilerResult:
    """Return the average NPU kernel time of ``fn`` across ``repeat`` iterations."""
    import torch_npu  # noqa

    prof_dir = tempfile.mkdtemp(prefix="cann_prof_")
    warmup_tensors = _prepare_warmup_tensors() if freq_boost else None
    result = ProfilerResult()

    try:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(warmup + repeat):
                if warmup_tensors is not None:
                    _boost_freq_and_clear_cache(*warmup_tensors)
                fn()
                prof.step()
    except Exception as e:
        result.error = str(e)
        _cleanup(prof_dir)
        return result
    finally:
        del warmup_tensors

    trace_file = _find_trace(prof_dir)
    if not trace_file:
        result.error = "no trace_view.json produced"
        _cleanup(prof_dir)
        return result

    try:
        total_us = _parse_trace(trace_file)
        result.elapsed_us = round(total_us / max(repeat, 1), 2)
    except Exception as e:
        result.error = f"trace parse failed: {e}"
    finally:
        _cleanup(prof_dir)
    return result


def _find_trace(prof_dir: str) -> Optional[str]:
    for root, _, files in os.walk(prof_dir):
        for f in files:
            if f == "trace_view.json":
                return os.path.join(root, f)
    return None


def _cleanup(prof_dir: str) -> None:
    if prof_dir and os.path.isdir(prof_dir):
        shutil.rmtree(prof_dir, ignore_errors=True)
