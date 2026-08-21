from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_pipeline.converter.submission import (
    SUBMISSION_RULE_IDS,
    collect_submission_issues,
    is_submission_dir,
    validate_submission,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _case(tmp_path: Path, *, bench_name: str = "cann", task: Path | None = None):
    files = {"task": task} if task is not None else {}
    return SimpleNamespace(
        bench_name=bench_name,
        task_dir=tmp_path,
        operator="Example",
        rel_path="level1/Example",
        files=files,
        metadata={},
    )


def test_cann_validation_reports_all_structural_issues(tmp_path):
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("notes.txt").write_text("incomplete", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_submission("cann", _case(tmp_path), source_dir, label="test-agent")

    message = str(exc_info.value)
    assert "[AUTO-CANN-001]" in message
    assert "[AUTO-CANN-002]" in message
    assert "notes.txt" in message
    assert "expected:" in message
    assert "actual:" in message
    assert "fix:" in message
    assert "docs/spec/submission_spec.md" in message


def test_cann_source_dir_accepts_package_sources(tmp_path):
    source_dir = tmp_path / "submission"
    source_dir.joinpath("cann_bench").mkdir(parents=True)
    source_dir.joinpath("build.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    assert collect_submission_issues("cann", source_dir) == []
    assert is_submission_dir("cann", source_dir)


def test_stanford_validation_reports_missing_ai_op(tmp_path):
    source_dir = tmp_path / "submission"
    source_dir.mkdir()

    with pytest.raises(ValueError, match="AUTO-STANFORD-001"):
        validate_submission("stanford", _case(tmp_path, bench_name="stanford"), source_dir, label="test-agent")


def test_stanford_validation_reports_relative_import_location(tmp_path):
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("ai_op.py").write_text("from .helper import kernel\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        validate_submission("stanford", _case(tmp_path, bench_name="stanford"), source_dir, label="test-agent")

    message = str(exc_info.value)
    assert "[AUTO-STANFORD-003]" in message
    assert "line 1" in message
    assert "_sys.path.insert" not in source_dir.joinpath("ai_op.py").read_text(encoding="utf-8")


def test_stanford_validation_aggregates_signature_mismatches(tmp_path):
    task_path = tmp_path / "task.py"
    task_path.write_text(
        "class Model:\n"
        "    def __init__(self, size=1):\n"
        "        pass\n"
        "    def forward(self, x, scale=1):\n"
        "        return x\n"
        "    def state_dict(self):\n"
        "        return {}\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("ai_op.py").write_text(
        "class ModelNew:\n"
        "    def __init__(self, size=2):\n"
        "        pass\n"
        "    def forward(self, x, scale=2):\n"
        "        return x\n"
        "    def state_dict(self):\n"
        "        return {}\n",
        encoding="utf-8",
    )

    case = _case(tmp_path, bench_name="stanford", task=task_path)
    with pytest.raises(ValueError) as exc_info:
        validate_submission("stanford", case, source_dir, label="test-agent")

    message = str(exc_info.value)
    assert "[AUTO-STANFORD-007]" in message
    assert "[AUTO-STANFORD-008]" in message


def test_all_auto_pipeline_rule_ids_are_documented():
    spec = REPO_ROOT.joinpath("docs/spec/submission_spec.md").read_text(encoding="utf-8")

    missing = [rule_id for rule_id in SUBMISSION_RULE_IDS if f"`{rule_id}`" not in spec]
    assert missing == []
