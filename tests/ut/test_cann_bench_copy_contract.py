import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_IMPL = REPO_ROOT / "src/cann_bench_utils/cann_bench_utils/__init__.py"
CPP_IMPL = REPO_ROOT / "src/cann_bench_utils/csrc/ops/device_memcpy/op_plugin/device_memcpy_plugin.cpp"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_cann_bench_copy_declares_mutation_and_wrapper_return():
    tree = ast.parse(PY_IMPL.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "define"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "cann_bench_utils::cann_bench_copy"
    ]
    assert len(definitions) == 1
    assert definitions[0].args[1].value == "(Tensor src, Tensor(a!) dst) -> ()"

    for name in ("_copy_npu", "_copy_meta"):
        function = _function(tree, name)
        assert not any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(function))

    wrapper = _function(tree, "cann_bench_copy")
    returns = [node for node in ast.walk(wrapper) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Name)
    assert returns[0].value.id == "dst"


def test_cann_bench_copy_cpp_bindings_are_void():
    cpp = CPP_IMPL.read_text(encoding="utf-8")
    assert "void device_memcpy_meta(" in cpp
    assert "void device_memcpy_npu(" in cpp
    assert "return dst;" not in cpp
