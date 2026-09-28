import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_evaluation.sh"


def _run_with_python_stub(tmp_path, args):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "python-argv.txt"
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == --version ]]; then echo 'Python stub 3.12'; exit 0; fi\n"
        "if [[ \"$1\" == -c ]]; then exit 0; fi\n"
        "printf '%s\\n' --CALL-- \"$@\" >> \"$CAPTURE_FILE\"\n"
    )
    python_stub.chmod(0o755)
    pip_stub = bin_dir / "pip"
    pip_stub.write_text("#!/bin/bash\nexit 1\n")
    pip_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CAPTURE_FILE"] = str(capture)
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=SCRIPT.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = capture.read_text().splitlines()
    marker = lines.index("--CALL--")
    return lines[marker + 1 :], result.stdout


def test_list_keeps_task_dir_as_one_argument(tmp_path):
    argv, _ = _run_with_python_stub(
        tmp_path, ["--action", "list", "--task-dir", "/tmp/task path"]
    )
    assert argv[0:2] == ["-m", "kernel_eval.cli"]
    assert argv[argv.index("--task-dir") + 1] == "/tmp/task path"


def test_eval_keeps_all_directory_arguments_as_one_argument(tmp_path):
    argv, _ = _run_with_python_stub(
        tmp_path,
        [
            "--action",
            "eval",
            "--device",
            "cpu",
            "--source-dir",
            "/tmp/source path",
            "--task-dir",
            "/tmp/task path",
            "--reports-dir",
            str(tmp_path / "report path"),
        ],
    )
    assert argv[0:2] == ["-m", "kernel_eval.cli"]
    for option, value in (
        ("--source-dir", "/tmp/source path"),
        ("--task-dir", "/tmp/task path"),
        ("--reports-dir", str(tmp_path / "report path")),
    ):
        assert argv[argv.index(option) + 1] == value


def test_staged_eval_keeps_task_dir_as_one_argument(tmp_path):
    argv, _ = _run_with_python_stub(
        tmp_path,
        [
            "--action",
            "eval",
            "--task-dir",
            "/tmp/task path",
            "--reports-dir",
            str(tmp_path / "report path"),
        ],
    )
    assert argv[0:2] == ["-m", "kernel_eval.staged_eval"]
    assert argv[argv.index("--task-dir") + 1] == "/tmp/task path"
