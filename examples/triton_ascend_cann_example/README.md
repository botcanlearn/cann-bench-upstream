# Triton Ascend CANN Submission Example

This is a submit-ready `cann_bench` wheel package containing Triton-Ascend
implementations of the Level 1 `Exp`, `MaskedScale`, `Mish`, `Sigmoid`, and
`SwiGlu` interfaces. CANN Bench treats a submitted
function as a normal Python callable, so no Triton-specific runner is needed.
The package exports all five callables from `cann_bench`; the runner moves inputs
to NPU, invokes the selected callable, synchronizes its stream, and profiles the
kernels it launches.

## Layout

```text
triton_ascend_cann_example/
├── build.sh
├── setup.py
└── cann_bench/
    ├── __init__.py
    ├── exp.py
    ├── masked_scale.py
    ├── mish.py
    ├── sigmoid.py
    └── swi_glu.py
```

Each operator module has two deliberate layers:

- A private Triton JIT kernel. All numerical work is performed with
  Triton language operations.
- A public package interface. Its name, parameter order, defaults, return
  type, and tensor device must match its `tasks/level1/<operator>/proto.yaml`.

## Prerequisites

The evaluation environment must already contain a compatible CANN, `torch`,
`torch_npu`, and Triton-Ascend runtime. The benchmark installs the submission
wheel with `--no-deps`, so package installation does not install Triton for you.

For the repository's CANN 9.0.0 environment, run this from the repository root:

```bash
pip install -r requirements-triton.txt
```

See `docs/guide/triton_ascend_quick_start.md` for the Docker smoke and a
step-by-step end-to-end check.

Check the active interpreter before building:

```bash
python -c 'import torch, torch_npu, triton; print(torch.__version__, triton.__version__)'
```

An import is not sufficient validation. Run `docker/test_env.py` in a
Triton-enabled image to compile and execute vector add before evaluating a
submission. Triton-Ascend bundles an NPU-IR compiler and must match the installed
CANN version; a beta CANN toolkit may require its matching compiler build.

## Build

```bash
cd examples/triton_ascend_cann_example
bash build.sh
```

The resulting `dist/cann_bench-*.whl` is the standard CANN Bench submission
artifact. The source directory itself can also be passed to the evaluator; it
will call `build.sh`, install the wheel, and discover public functions exported
by `cann_bench`.

## Run

Run a correctness pass first:

```bash
./scripts/run_evaluation.sh \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --device-id 0 \
  --no-perf
```

Then enable profiler-based scoring:

```bash
./scripts/run_evaluation.sh \
  --source-dir examples/triton_ascend_cann_example \
  --task-dir tasks/level1/exp \
  --operator Exp \
  --device-id 0
```

Replace the task directory and operator name with `masked_scale`/
`MaskedScale`, `mish`/`Mish`, `sigmoid`/`Sigmoid`, or `swi_glu`/`SwiGlu` to
evaluate the other exported kernels.

## Adding Operators

For every additional task, add a module that exposes the schema function from
`cann_bench/__init__.py`. The wrapper may use tensor allocation, shape handling,
and layout preparation, but it must not dispatch the task's core computation to
the matching PyTorch or `torch_npu` builtin. CANN Bench's execution guards reject
those calls. Keep kernel execution on NPU and return NPU tensors directly.
