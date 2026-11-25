import json
import sys
from pathlib import Path
from typing import Literal

import pytest

import tokenweir.runner as runner_module
from tokenweir.budget import BudgetExceededError
from tokenweir.cli import _prompt_for_new_cap, main
from tokenweir.runner import RunResult
from tokenweir.schemas import Plan, Subtask, TestFile as GradingTestFile


def _make_tests(parent: Path) -> Path:
    tests = parent / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    return tests


def _make_task_dir(parent: Path) -> Path:
    task_dir = parent / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text("Add a feature\n", encoding="utf-8")
    _make_tests(task_dir)
    return task_dir


def _fake_run(
    calls: list,
    status: Literal["passed", "failed", "paused"] = "passed",
    kwargs_calls: list | None = None,
):
    def fake(
        task: str,
        task_source: str,
        tests_dir: Path | None,
        **kwargs,
    ) -> RunResult:
        calls.append((task, task_source, tests_dir))
        if kwargs_calls is not None:
            kwargs_calls.append(kwargs)
        return RunResult(
            status=status,
            workspace=Path("workspace/run"),
            attempts=1,
            record_path=Path("traces/run.json"),
        )

    return fake


def test_cli_accepts_positional_inline_task(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tests = _make_tests(tmp_path)
    calls: list = []
    monkeypatch.setattr(runner_module, "run_task", _fake_run(calls))

    assert main(["Add a feature", "--tests", str(tests)]) == 0

    assert calls == [("Add a feature", "inline input", tests)]
    assert "Loaded task from inline input" in capsys.readouterr().out


def test_cli_accepts_positional_task_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    task_dir = _make_task_dir(tmp_path)
    calls: list = []
    monkeypatch.setattr(runner_module, "run_task", _fake_run(calls))

    assert main([str(task_dir / "task.md")]) == 0

    assert calls[0][0] == "Add a feature"
    assert calls[0][1] == str(task_dir / "task.md")
    assert calls[0][2] == task_dir / "tests"
    assert "Loaded task from" in capsys.readouterr().out


def test_cli_accepts_task_directory(tmp_path: Path, monkeypatch) -> None:
    task_dir = _make_task_dir(tmp_path)
    calls: list = []
    monkeypatch.setattr(runner_module, "run_task", _fake_run(calls))

    assert main([str(task_dir)]) == 0

    assert calls[0][0] == "Add a feature"
    assert calls[0][1] == str(task_dir / "task.md")
    assert calls[0][2] == task_dir / "tests"


def test_cli_accepts_explicit_inline_task(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tests = _make_tests(tmp_path)
    monkeypatch.setattr(runner_module, "run_task", _fake_run([]))

    assert main(["--task", "Add a feature", "--tests", str(tests)]) == 0
    assert "Loaded task from inline input" in capsys.readouterr().out


def test_cli_accepts_explicit_task_file(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    task_dir = _make_task_dir(tmp_path)
    monkeypatch.setattr(runner_module, "run_task", _fake_run([]))

    assert main(["--task-file", str(task_dir / "task.md")]) == 0
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
    task_dir = _make_task_dir(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(["inline task", "--task-file", str(task_dir / "task.md")])

    assert error.value.code == 2
    assert "only one task source" in capsys.readouterr().err


def test_cli_reports_unreadable_task_file(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.md"

    assert main(["--task-file", str(missing)]) == 2
    assert "cannot read task file" in capsys.readouterr().err


def test_cli_reports_directory_without_task_md(tmp_path: Path, capsys) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    assert main([str(task_dir)]) == 2
    assert "no task.md found" in capsys.readouterr().err


def test_cli_uses_architect_mode_without_tests(
    monkeypatch, capsys
) -> None:
    calls: list = []
    monkeypatch.setattr(runner_module, "run_task", _fake_run(calls))

    assert main(["Add a feature"]) == 0

    assert calls == [("Add a feature", "inline input", None)]
    out = capsys.readouterr().out
    assert "asking the architect" in out


def test_cli_plan_only_prints_plan(monkeypatch, capsys) -> None:
    plan = Plan(
        task="Add a feature",
        subtasks=[
            Subtask(
                id="impl",
                title="Implement feature",
                type="function",
                difficulty="easy",
                recommended_tier="local_small",
                files=["feature.py"],
                spec="Write feature.py.",
                tests=["tests/test_feature.py"],
            )
        ],
        test_files=[
            GradingTestFile(
                path="tests/test_feature.py",
                content="def test_feature() -> None:\n    assert True\n",
            )
        ],
    )
    calls: list = []

    def fake_plan(task: str, **kwargs) -> Plan:
        calls.append(task)
        return plan

    monkeypatch.setattr(runner_module, "plan_only", fake_plan)
    monkeypatch.setattr(
        runner_module,
        "run_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run_task must not be called in plan-only mode")
        ),
    )

    assert main(["Add a feature", "--plan-only"]) == 0

    assert calls == ["Add a feature"]
    out = capsys.readouterr().out
    assert '"subtasks"' in out
    assert '"impl"' in out


def test_cli_plan_only_reports_runner_errors(monkeypatch, capsys) -> None:
    def fail_plan(task: str, **kwargs) -> Plan:
        raise runner_module.RunnerError("architect failed")

    monkeypatch.setattr(runner_module, "plan_only", fail_plan)

    assert main(["Add a feature", "--plan-only"]) == 2
    assert "architect failed" in capsys.readouterr().err


def test_cli_rejects_empty_tests_directory(tmp_path: Path, capsys) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()

    assert main(["Add a feature", "--tests", str(tests)]) == 2
    assert "no test_*.py files" in capsys.readouterr().err


def test_cli_rejects_tests_path_that_is_a_file(tmp_path: Path, capsys) -> None:
    tests_file = tmp_path / "tests"
    tests_file.write_text("not a directory", encoding="utf-8")

    assert main(["Add a feature", "--tests", str(tests_file)]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_returns_one_when_run_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tests = _make_tests(tmp_path)
    monkeypatch.setattr(
        runner_module, "run_task", _fake_run([], status="failed")
    )

    assert main(["Add a feature", "--tests", str(tests)]) == 1


def test_cli_reports_runner_errors(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tests = _make_tests(tmp_path)

    def fail(*args, **kwargs):
        raise runner_module.RunnerError("builder provider is not configured")

    monkeypatch.setattr(runner_module, "run_task", fail)

    assert main(["Add a feature", "--tests", str(tests)]) == 2
    assert "builder provider is not configured" in capsys.readouterr().err


def test_cli_passes_project_and_cap_to_runner(
    tmp_path: Path, monkeypatch
) -> None:
    tests = _make_tests(tmp_path)
    calls: list = []
    kwargs_calls: list = []
    monkeypatch.setattr(
        runner_module,
        "run_task",
        _fake_run(calls, kwargs_calls=kwargs_calls),
    )
    state_dir = tmp_path / "projects"

    assert (
        main(
            [
                "Add a feature",
                "--tests",
                str(tests),
                "--project",
                "demo",
                "--cap",
                "3.5",
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )

    assert calls == [("Add a feature", "inline input", tests)]
    kwargs = kwargs_calls[0]
    assert kwargs["project"] == "demo"
    assert kwargs["use_router"] is True
    assert kwargs["budget_policy"].budget.project == "demo"
    assert kwargs["budget_policy"].budget.cap_usd == 3.5
    assert (state_dir / "demo" / "budget.json").exists()


def test_cli_can_disable_the_router(
    tmp_path: Path, monkeypatch
) -> None:
    tests = _make_tests(tmp_path)
    kwargs_calls: list = []
    monkeypatch.setattr(
        runner_module,
        "run_task",
        _fake_run([], kwargs_calls=kwargs_calls),
    )

    assert main(["Add a feature", "--tests", str(tests), "--no-router"]) == 0

    assert kwargs_calls[0]["use_router"] is False


def test_cli_prints_budget_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tests = _make_tests(tmp_path)
    monkeypatch.setattr(runner_module, "run_task", _fake_run([]))

    assert main(["Add a feature", "--tests", str(tests)]) == 0

    out = capsys.readouterr().out
    assert "Budget: project 'default' cap $2.00" in out


def test_cli_returns_three_when_run_pauses(
    tmp_path: Path, monkeypatch
) -> None:
    tests = _make_tests(tmp_path)
    monkeypatch.setattr(
        runner_module, "run_task", _fake_run([], status="paused")
    )

    assert main(["Add a feature", "--tests", str(tests)]) == 3


def test_cli_returns_three_when_plan_budget_is_exceeded(
    monkeypatch, capsys
) -> None:
    def fail_plan(task: str, **kwargs) -> Plan:
        raise BudgetExceededError(
            project="default",
            cap_usd=0.0,
            spent_usd=0.0,
            needed_usd=0.001,
        )

    monkeypatch.setattr(runner_module, "plan_only", fail_plan)

    assert main(["Add a feature", "--plan-only"]) == 3
    assert "cap" in capsys.readouterr().err


def test_cli_rejects_negative_cap(tmp_path: Path, capsys) -> None:
    tests = _make_tests(tmp_path)

    assert main(["Add a feature", "--tests", str(tests), "--cap", "-1"]) == 2
    assert "finite non-negative" in capsys.readouterr().err


def test_prompt_for_new_cap_returns_none_without_a_tty(
    monkeypatch, capsys
) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    exc = BudgetExceededError(
        project="demo", cap_usd=1.0, spent_usd=1.0
    )

    assert _prompt_for_new_cap(exc) is None
    err = capsys.readouterr().err
    assert "rerun with --cap" in err


def test_prompt_for_new_cap_retries_until_valid(monkeypatch) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    answers = iter(["not-a-number", "0.5", "2"])
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    exc = BudgetExceededError(
        project="demo", cap_usd=1.0, spent_usd=1.0
    )

    assert _prompt_for_new_cap(exc) == 2.0


def test_prompt_for_new_cap_returns_none_on_blank(monkeypatch) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    exc = BudgetExceededError(
        project="demo", cap_usd=1.0, spent_usd=1.0
    )

    assert _prompt_for_new_cap(exc) is None


def test_cli_resumes_record_and_uses_recorded_project(
    tmp_path: Path, monkeypatch
) -> None:
    record_path = tmp_path / "traces" / "old.json"
    record_path.parent.mkdir()
    record_path.write_text(
        json.dumps(
            {
                "run_id": "old",
                "mode": "direct",
                "status": "paused",
                "project": "demo",
                "workspace": str(tmp_path / "workspace" / "old"),
                "task": "Create impl.py",
                "task_source": "task.md",
            }
        ),
        encoding="utf-8",
    )
    calls: list = []

    def fake_resume(path: Path, **kwargs) -> RunResult:
        calls.append((path, kwargs))
        return RunResult(
            status="passed",
            workspace=tmp_path / "workspace" / "old",
            attempts=1,
            record_path=record_path,
        )

    monkeypatch.setattr(runner_module, "resume_run", fake_resume)
    state_dir = tmp_path / "projects"

    assert (
        main(
            [
                "--resume",
                str(record_path),
                "--state-dir",
                str(state_dir),
            ]
        )
        == 0
    )

    path, kwargs = calls[0]
    assert path == record_path
    assert kwargs["project"] == "demo"
    assert kwargs["budget_policy"].budget.project == "demo"
    assert kwargs["use_router"] is True
