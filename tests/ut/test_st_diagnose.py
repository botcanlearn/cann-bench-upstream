import importlib.util
import sys
import types
from pathlib import Path


def _load_diagnose():
    package_name = "st_harness_for_diagnose_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    eval_run = types.ModuleType(f"{package_name}.eval_run")
    eval_run.EVAL_LOG_NAME = "eval_cli.log"
    sys.modules[package_name] = package
    sys.modules[eval_run.__name__] = eval_run

    path = Path(__file__).resolve().parents[1] / "st" / "harness" / "diagnose.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.diagnose", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnose = _load_diagnose()


def _report(case):
    return {
        "device": "npu",
        "eval_code": "eval_probe",
        "summary": {
            "total_operators": 1,
            "total_cases": 1,
            "passed_cases": int(case["accuracy"]["passed"]),
            "failed_cases": int(not case["accuracy"]["passed"]),
            "overall_score": 0,
        },
        "operators": [{
            "operator": "Cummin",
            "rel_path": "level2/cummin",
            "passed_cases": int(case["accuracy"]["passed"]),
            "total_cases": 1,
            "score": 0,
            "cases": [case],
        }],
    }


def test_digest_reports_accuracy_failure_even_when_perf_is_present(capsys):
    diagnose.digest_report(_report({
        "case_id": "level2/cummin_1",
        "status": "failed",
        "failure_type": "accuracy",
        "error_msg": "",
        "accuracy": {"passed": False, "error_msg": "mismatch"},
        "elapsed_us": 123,
    }))

    output = capsys.readouterr().out
    assert "x1 cases [1]" in output
    assert "accuracy.error_msg: mismatch" in output
    assert "perf=ok" in output


def test_digest_still_omits_fully_healthy_case(capsys):
    diagnose.digest_report(_report({
        "case_id": "level2/cummin_1",
        "status": "passed",
        "failure_type": None,
        "error_msg": "",
        "accuracy": {"passed": True, "error_msg": ""},
        "elapsed_us": 123,
    }))

    output = capsys.readouterr().out
    assert "x1 cases" not in output
    assert "accuracy.error_msg:" not in output
