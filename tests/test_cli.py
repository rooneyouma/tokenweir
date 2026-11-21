from pathlib import Path

import pytest

from tokenweir.cli import main


def test_cli_accepts_positional_inline_task(capsys) -> None:
    assert main(["Add a feature"]) == 0
    assert "Loaded task from inline input" in capsys.readouterr().out


def test_cli_accepts_positional_task_file(tmp_path: Path, capsys) -> None:
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a feature\n", encoding="utf-8")

    assert main([str(task_file)]) == 0
    output = capsys.readouterr().out
    assert f"Loaded task from {task_file}" in output
    assert "(13 characters)" in output


def test_cli_accepts_explicit_inline_task(capsys) -> None:
    assert main(["--task", "Add a feature"]) == 0
    assert "Loaded task from inline input" in capsys.readouterr().out


def test_cli_accepts_explicit_task_file(tmp_path: Path, capsys) -> None:
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a feature\n", encoding="utf-8")

    assert main(["--task-file", str(task_file)]) == 0
    assert "Loaded task from" in capsys.readouterr().out


def test_cli_rejects_empty_task(capsys) -> None:
    assert main(["--task", "  "]) == 2
    assert "task must not be empty" in capsys.readouterr().err


def test_cli_rejects_missing_task_source(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2
    assert "provide a task" in capsys.readouterr().err


def test_cli_rejects_multiple_task_sources(tmp_path: Path, capsys) -> None:
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a feature\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["inline task", "--task-file", str(task_file)])

    assert error.value.code == 2
    assert "only one task source" in capsys.readouterr().err


def test_cli_reports_unreadable_task_file(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.md"

    assert main(["--task-file", str(missing)]) == 2
    assert "cannot read task file" in capsys.readouterr().err
