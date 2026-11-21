from pathlib import Path

from tokenweir.cli import main


def test_cli_accepts_inline_task(capsys) -> None:
    assert main(["--task", "Add a feature"]) == 0
    assert "Loaded task from inline input" in capsys.readouterr().out


def test_cli_accepts_task_file(tmp_path: Path, capsys) -> None:
    task_file = tmp_path / "task.md"
    task_file.write_text("Add a feature\n", encoding="utf-8")

    assert main(["--task-file", str(task_file)]) == 0
    assert "Loaded task from" in capsys.readouterr().out


def test_cli_rejects_empty_task(capsys) -> None:
    assert main(["--task", "  "]) == 2
    assert "task must not be empty" in capsys.readouterr().err
