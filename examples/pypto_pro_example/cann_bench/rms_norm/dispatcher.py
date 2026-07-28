"""Auto-generated dim/dtype dispatcher for `rms_norm`. Do not edit by hand."""

import importlib.util
from pathlib import Path

_OP_NAME = 'rms_norm'
_CLASSES = [{"subdir": "c1", "signature": [[3, "float16"], [1, "float16"]]}, {"subdir": "c2", "signature": [[3, "float32"], [1, "float32"]]}, {"subdir": "c3", "signature": [[3, "bfloat16"], [1, "bfloat16"]]}, {"subdir": "c4", "signature": [[2, "bfloat16"], [1, "bfloat16"]]}, {"subdir": "c5", "signature": [[4, "float16"], [1, "float16"]]}, {"subdir": "c6", "signature": [[4, "float32"], [1, "float32"]]}, {"subdir": "c7", "signature": [[5, "float32"], [1, "float32"]]}]
_BASE = Path(__file__).resolve().parent
_impl_cache = {}


def _dtype_name(value):
    return str(value).rsplit(".", 1)[-1]


def _signature(args):
    sig = []
    for a in args:
        if isinstance(a, (list, tuple)) and a and hasattr(a[0], "dim") and hasattr(a[0], "dtype"):
            sig.append([int(a[0].dim()), _dtype_name(a[0].dtype)])
        elif hasattr(a, "dim") and hasattr(a, "dtype"):
            sig.append([int(a.dim()), _dtype_name(a.dtype)])
    return sig


def _load(subdir):
    if subdir not in _impl_cache:
        path = _BASE / subdir / f"test_{_OP_NAME}.py"
        spec = importlib.util.spec_from_file_location(f"{_OP_NAME}_{subdir}_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = next((getattr(module, n) for n in (_OP_NAME, f"{_OP_NAME}_wrapper") if callable(getattr(module, n, None))), None)
        if fn is None:
            raise AttributeError(f"{path} exposes no '{_OP_NAME}' or '{_OP_NAME}_wrapper' entry")
        _impl_cache[subdir] = fn
    return _impl_cache[subdir]


def rms_norm(*args, **kwargs):
    sig = _signature(args)
    for entry in _CLASSES:
        n = len(entry["signature"])
        if sig[:n] == entry["signature"]:
            return _load(entry["subdir"])(*args, **kwargs)
    raise ValueError(f"no {_OP_NAME} class for signature {sig}; classes={_CLASSES}")
