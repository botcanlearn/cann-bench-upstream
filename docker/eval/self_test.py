#!/usr/bin/env python3
"""Post-build smoke for the cann-bench-eval image (`docker run <image> --self-test`).

Required checks (any failure -> non-zero exit):
  [1] python / torch / torch_npu importable
  [2] torch_npu sees a device AND a bare H2D copy works -- the copy half is the real gate on a
      950-class SoC, where it routes through aclnnInplaceCopy and so needs OPS_MODE=refonly
      (910B does a plain aclrtMemcpy and is happy with OPS_MODE=none)
  [3] CANN compiler version.info readable
  [4] cann_bench_utils importable -- the V3 anti-cheat warmup/cache-clean provider, a hard
      dependency of every evaluation, baked in at build time
  [5] the frozen harness imports and can enumerate tasks (kernel_eval + tasks/ present)

Informational probes (never fail the run, but decide which OPS_MODE an image needs):
  [6] can a builtin aclnn op actually launch? 0-ops images say no -- that is the anti-cheat
      working as designed, not a defect.
  [7] optional Triton-Ascend backend compiles and runs a vector add
"""

import os
import subprocess
import sys
from pathlib import Path

CANN_BENCH_DIR = Path(os.environ.get("CANN_BENCH_DIR", "/opt/cann-bench"))
failed = []

# [1] versions
try:
    import platform

    import torch
    import torch_npu

    py = ".".join(str(v) for v in sys.version_info[:3])
    print(
        f"[OK]   [1] python {py}, torch {torch.__version__}, torch_npu {torch_npu.__version__}"
        f" ({platform.machine()})"
    )
except Exception as e:
    print(f"[FAIL] [1] import/version: {e}")
    failed.append(1)

# [2] device visible AND usable
try:
    import torch
    import torch_npu

    count = torch_npu.npu.device_count()
    assert count > 0, f"device_count = {count}"
    name = torch.npu.get_device_name(0)
    got = torch.arange(4, dtype=torch.float32).npu().cpu().tolist()
    assert got == [0.0, 1.0, 2.0, 3.0], got
    print(f"[OK]   [2] {count} x {name}; bare H2D copy works")
except Exception as e:
    print(f"[FAIL] [2] device / H2D copy: {e}")
    if "ERR01007" in str(e) or "aclnnInplaceCopy" in str(e):
        print("           ^ this SoC routes the copy through aclnn -- rebuild with OPS_MODE=refonly")
    failed.append(2)

# [3] CANN intact
try:
    vfile = Path(os.environ["ASCEND_HOME_PATH"]) / "compiler" / "version.info"
    line = vfile.read_text().strip().splitlines()[0]
    print(f"[OK]   [3] CANN compiler {line}")
except Exception as e:
    print(f"[FAIL] [3] CANN compiler version.info: {e}")
    failed.append(3)

# [4] warmup provider
try:
    from cann_bench_utils import cann_bench_cache_clean, cann_bench_warmup  # noqa: F401

    print(f"[OK]   [4] cann_bench_utils importable (NPU_ARCH={os.environ.get('NPU_ARCH', '?')})")
except Exception as e:
    print(f"[FAIL] [4] cann_bench_utils: {e}")
    failed.append(4)

# [5] frozen harness. Enumerating operators exercises PYTHONPATH, get_project_root()'s walk for
# tasks/, and the task registry in one shot -- an import alone would not prove tasks/ came along.
try:
    out = subprocess.run(
        ["bash", str(CANN_BENCH_DIR / "scripts" / "run_evaluation.sh"), "-a", "list"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-500:] or out.stdout.strip()[-500:])
    levels = sorted({p.name for p in (CANN_BENCH_DIR / "tasks").glob("level*")})
    n_ops = sum(1 for p in (CANN_BENCH_DIR / "tasks").glob("level*/*") if p.is_dir())
    print(f"[OK]   [5] harness lists operators; tasks/ = {n_ops} ops across {levels}")
except Exception as e:
    print(f"[FAIL] [5] harness / tasks enumeration: {e}")
    failed.append(5)

# [6] builtin availability -- diagnostic only. matmul is the canonical builtin the framework's own
# warmup used to call before cann_bench_utils replaced it, so it is the right probe. Expect it to FAIL:
# that is the anti-cheat working. The error differs by posture -- 500001 LazyInitAclops when libopapi
# is absent (none), 561103 "Parse dynamic kernel config fail" when it is present but the kernel
# binaries were stripped (refonly).
posture = f"OPS_MODE={os.environ.get('OPS_MODE', '?')} NPU_ARCH={os.environ.get('NPU_ARCH', '?')}"
try:
    import torch

    a = torch.randn(64, 64, device="npu:0")
    (a @ a).cpu()
    print(f"[WARN] [6] builtin aclnn ops CAN launch ({posture}) -- submissions could cheat by calling them")
except Exception as e:
    print(f"[INFO] [6] builtin aclnn ops blocked ({posture}): {type(e).__name__}")
    print(f"           ^ expected -- submissions must ship their own kernel. ({str(e)[:120]})")

# [7] optional Triton-Ascend
triton_ascend_version = os.environ.get("TRITON_ASCEND_VERSION", "").strip()
if triton_ascend_version:
    try:
        import importlib.metadata

        import torch
        import triton
        import triton.language as tl
        from triton.runtime import driver

        @triton.jit
        def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, x + y, mask=mask)

        installed_version = importlib.metadata.version("triton-ascend")
        assert installed_version == triton_ascend_version, (
            f"triton-ascend {installed_version}, expected {triton_ascend_version}"
        )
        target = driver.active.get_current_target()
        assert target.backend == "npu", f"active Triton backend = {target.backend}"

        x = torch.arange(1024, dtype=torch.float32, device="npu:0")
        y = torch.full_like(x, 2.0)
        output = torch.empty_like(x)
        _vector_add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, output, x.numel(), BLOCK_SIZE=256)
        torch.npu.synchronize()
        torch.testing.assert_close(output, x + y, rtol=0, atol=0)
        print(f"[OK]   [7] triton-ascend {installed_version}, target={target.arch}, vector add passed")
    except Exception as e:
        print(f"[FAIL] [7] Triton-Ascend JIT/vector add: {e}")
        failed.append(7)
else:
    print("[SKIP] [7] Triton-Ascend not requested for this image")

if failed:
    sys.exit(f"\nFAILED: {failed}")
print("\nALL CHECKS PASSED")
