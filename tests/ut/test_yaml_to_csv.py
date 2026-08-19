import csv

import pytest

from scripts.utils.yaml_to_csv import yaml_to_csv


@pytest.mark.parametrize(
    ("content", "expected_error"),
    [
        ("", "YAML 文件 {path} 为空"),
        ("- item\n", "顶层结构应为字典，实际为 list"),
        ("plain text\n", "顶层结构应为字典，实际为 str"),
    ],
)
def test_yaml_to_csv_rejects_invalid_top_level_without_traceback(
    tmp_path, capsys, content, expected_error
):
    input_file = tmp_path / "cases.yaml"
    output_file = tmp_path / "cases.csv"
    input_file.write_text(content, encoding="utf-8")

    assert yaml_to_csv(str(input_file), str(output_file)) is False

    stderr = capsys.readouterr().err
    assert expected_error.format(path=input_file) in stderr
    assert "Traceback" not in stderr
    assert not output_file.exists()


@pytest.mark.parametrize(
    ("content", "expected_error"),
    [
        ("cases: invalid\n", "cases 字段应为列表，实际为 str"),
        ("cases: {}\n", "cases 字段应为列表，实际为 dict"),
        ("cases: [null]\n", "cases 第 1 个元素应为字典，实际为 NoneType"),
    ],
)
def test_yaml_to_csv_rejects_invalid_cases_structure_without_traceback(
    tmp_path, capsys, content, expected_error
):
    input_file = tmp_path / "cases.yaml"
    output_file = tmp_path / "cases.csv"
    input_file.write_text(content, encoding="utf-8")

    assert yaml_to_csv(str(input_file), str(output_file)) is False

    stderr = capsys.readouterr().err
    assert expected_error in stderr
    assert "Traceback" not in stderr
    assert not output_file.exists()


def test_yaml_to_csv_preserves_valid_conversion(tmp_path):
    input_file = tmp_path / "cases.yaml"
    output_file = tmp_path / "cases.csv"
    input_file.write_text(
        """cases:
  - operator: Add
    case_id: case_1
    input_shape: [2, 3]
    custom_field: value
""",
        encoding="utf-8",
    )

    assert yaml_to_csv(str(input_file), str(output_file)) is True

    with output_file.open(newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))
    assert rows == [
        {
            "operator": "Add",
            "case_id": "case_1",
            "input_shape": "[2, 3]",
            "custom_field": "value",
        }
    ]
