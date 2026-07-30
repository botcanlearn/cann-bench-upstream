#!/usr/bin/env python3
"""Post-build smoke for cann-bench:cann9.0.0-* execution image.

Verifies in order:
  [1] python / torch / torch_npu importable, log versions
  [2] torch_npu can see at least one NPU device
  [3] npu-smi info works (driver/runtime loaded)
  [4] CANN compiler version.info readable (proves CANN install intact)
  [5] optional Triton-Ascend backend compiles and runs vector add

Exits 0 with "ALL CHECKS PASSED" only if all required checks pass.
"""

import os
import subprocess
import sys

failed = []

# [1] python/torch/torch_npu versions
try:
    import torch
    import torch_npu

    py = ".".join(str(v) for v in sys.version_info[:3])
    print(f"[OK]   [1] python {py}, torch {torch.__version__}, torch_npu {torch_npu.__version__}")
except Exception as e:
    print(f"[FAIL] [1] import/version: {e}")
    failed.append(1)

# [2] torch_npu device visible
try:
    import torch_npu

    count = torch_npu.npu.device_count()
    assert count > 0, f"device_count = {count}"
    print(f"[OK]   [2] torch_npu.npu.device_count() = {count}")
except Exception as e:
    print(f"[FAIL] [2] torch_npu device_count: {e}")
    failed.append(2)

# [3] npu-smi info exits 0
try:
    subprocess.check_call(
        ["npu-smi", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[OK]   [3] npu-smi reachable")
except Exception as e:
    print(f"[FAIL] [3] npu-smi info: {e}")
    failed.append(3)

# [4] CANN version readable. ascendhub base lays version.info per
# component, no aggregate at toolkit root — compiler/ is the canonical core.
try:
    vfile = os.path.join(os.environ["ASCEND_HOME_PATH"], "compiler", "version.info")
    with open(vfile) as f:
        line = f.read().strip().splitlines()[0]
    print(f"[OK]   [4] CANN compiler {line}")
except Exception as e:
    print(f"[FAIL] [4] CANN compiler version.info: {e}")
    failed.append(4)

# A non-empty build arg enables a real JIT smoke. This catches wrong Triton
# packages and compiler/CANN mismatches that import-only checks cannot detect.
triton_ascend_version = os.environ.get("TRITON_ASCEND_VERSION", "").strip()
if triton_ascend_version:
    try:
        import importlib.metadata

        import triton
        import triton.language as tl
        from triton.runtime import driver

        @triton.jit
        def _vector_add_kernel(
            x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
        ):
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
        grid = (triton.cdiv(x.numel(), 256),)
        _vector_add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=256)
        torch.npu.synchronize()
        torch.testing.assert_close(output, x + y, rtol=0, atol=0)
        print(
            f"[OK]   [5] triton-ascend {installed_version}, "
            f"target={target.arch}, vector add passed"
        )
    except Exception as e:
        print(f"[FAIL] [5] Triton-Ascend JIT/vector add: {e}")
        failed.append(5)
else:
    print("[SKIP] [5] Triton-Ascend not requested for this image")

if failed:
    sys.exit(f"\nFAILED: {failed}")
print("\nALL CHECKS PASSED")
