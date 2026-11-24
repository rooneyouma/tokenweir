import json
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import tokenweir.runner as runner_module
from tokenweir.architect import ArchitectResponseError
from tokenweir.builder import BuilderResponseError
from tokenweir.runner import (
    RunnerError,
    plan_only,
    run_task,
    select_architect_provider,
    select_builder_provider,
)
from tokenweir.sandbox import SandboxResult
from tokenweir.schemas import (
    BuilderFile,
    BuilderOutput,
    Plan,
    Subtask,
    TestFile as GradingTestFile,
)
from tokenweir.settings import (
    ArchitectDefinition,
    BuilderDefinition,
    ConfiguredProvider,
    LoadedSettings,
)

GOOD_CODE = (
    "def is_palindrome(s: str) -> bool:\n"
    '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
    "    return cleaned == cleaned[::-1]\n"
)
BAD_CODE = "def is_palindrome(s: str) -> bool:\n    return False\n"


def _tests_dir(tmp_path: Path) -> Path:
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_palindrome.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    return tests


def _settings(
    *,
    max_attempts: int = 3,
    provider_name: str = "fake",
    with_key: bool = False,
    architect_provider: str | None = None,
) -> LoadedSettings:
    provider = ConfiguredProvider(
        name=provider_name,
        model="fake/model",
        timeout_s=5,
        api_key=SecretStr("secret-key") if with_key else None,
    )
    return LoadedSettings(
        providers=[provider],
        unconfigured=[],
        errors=[],
        builder=BuilderDefinition(
            provider=provider_name,
            max_attempts=max_attempts,
            temperature=0.0,
        ),
        architect=ArchitectDefinition(
            provider=architect_provider or provider_name,
            temperature=0.0,
        ),
    )


def _plan() -> Plan:
    return Plan(
        task="Create palindrome.py",
        subtasks=[
            Subtask(
                id="impl",
                title="Implement palindrome",
                type="function",
                difficulty="easy",
                recommended_tier="local_small",
                files=["palindrome.py"],
                spec="Implement is_palindrome in palindrome.py.",
                tests=["tests/test_palindrome.py"],
            )
        ],
        test_files=[
            GradingTestFile(
                path="tests/test_palindrome.py",
                content=(
                    "from palindrome import is_palindrome\n\n"
                    "def test_palindrome() -> None:\n"
                    "    assert is_palindrome('racecar')\n"
                ),
            )
        ],
    )


def _output(
    path: str = "palindrome.py", content: str = GOOD_CODE
) -> BuilderOutput:
    return BuilderOutput(files=[BuilderFile(path=path, content=content)])


def _run(
    tmp_path: Path,
    settings: LoadedSettings,
):
    return run_task(
        "Create palindrome.py",
        "test task",
        _tests_dir(tmp_path),
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
    )


def _run_architect(tmp_path: Path, settings: LoadedSettings):
    return run_task(
        "Create palindrome.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
    )


def test_run_task_passes_and_commits(tmp_path: Path, monkeypatch) -> None:
    calls: list = []

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        calls.append(feedback)
        return _output()

    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            True, "passed", "All checks passed."
        ),
    )

    result = _run(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 1
    assert (result.workspace / "palindrome.py").exists()
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=result.workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Pass task: Create palindrome.py" in log
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["builder"]["provider"] == "fake"
    assert calls == [None]


def test_run_task_retries_with_feedback_and_removes_stale_files(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []
    outputs = [
        BuilderOutput(
            files=[
                BuilderFile(path="old.py", content="old = True\n"),
                BuilderFile(path="palindrome.py", content=BAD_CODE),
            ]
        ),
        _output(),
    ]
    check_results = [
        SandboxResult(False, "pytest", "AssertionError: nope"),
        SandboxResult(True, "passed", "All checks passed."),
    ]

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        calls.append(feedback)
        return outputs.pop(0)

    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: check_results.pop(0),
    )

    result = _run(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 2
    assert calls[0] is None
    assert "AssertionError: nope" in calls[1]
    assert not (result.workspace / "old.py").exists()
    assert (result.workspace / "palindrome.py").read_text(
        encoding="utf-8"
    ) == GOOD_CODE


def test_run_task_stops_after_bounded_attempts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_files",
        lambda task, provider, **kwargs: _output(content=BAD_CODE),
    )
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            False, "pytest", "AssertionError"
        ),
    )

    result = _run(tmp_path, _settings(max_attempts=2, with_key=True))

    assert result.status == "failed"
    assert result.attempts == 2
    assert result.workspace.exists()
    record_text = result.record_path.read_text(encoding="utf-8")
    record = json.loads(record_text)
    assert record["status"] == "failed"
    assert len(record["attempts"]) == 2
    assert "secret-key" not in record_text


def test_run_task_retries_malformed_builder_output(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        calls.append(feedback)
        if len(calls) == 1:
            raise BuilderResponseError("builder response was not valid JSON")
        return _output()

    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            True, "passed", "All checks passed."
        ),
    )

    result = _run(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 2
    assert "not valid JSON" in calls[1]


def test_run_architect_passes_and_records_plan(tmp_path: Path, monkeypatch) -> None:
    built_tasks: list[str] = []
    check_calls: list[tuple] = []

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        built_tasks.append(task)
        return _output()

    def fake_checks(workspace, *args, **kwargs):
        check_calls.append((args, kwargs))
        return SandboxResult(True, "passed", "All checks passed.")

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(runner_module, "run_checks", fake_checks)

    result = _run_architect(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 1
    assert (result.workspace / "palindrome.py").exists()
    assert (result.workspace / "tests/test_palindrome.py").exists()
    assert "Subtask impl: Implement palindrome" in built_tasks[0]
    assert "Do not write tests" in built_tasks[0]
    assert check_calls[0][0] == (["tests/test_palindrome.py"], ["palindrome.py"])
    assert check_calls[1][0] == (None, ["palindrome.py"])
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=result.workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Pass impl: Implement palindrome" in log
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["mode"] == "architect"
    assert record["plan"]["subtasks"][0]["id"] == "impl"
    assert record["subtasks"][0]["status"] == "passed"
    assert record["final_checks"]["passed"] is True


def test_run_architect_retries_subtask_with_feedback(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []
    outputs = [_output(content=BAD_CODE), _output()]
    check_results = [
        SandboxResult(False, "pytest", "AssertionError"),
        SandboxResult(True, "passed", "All checks passed."),
        SandboxResult(True, "passed", "All checks passed."),
    ]

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        calls.append(feedback)
        return outputs.pop(0)

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: check_results.pop(0),
    )

    result = _run_architect(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 2
    assert calls[0] is None
    assert "AssertionError" in calls[1]
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert len(record["subtasks"][0]["attempts"]) == 2
    assert record["subtasks"][0]["attempts"][0]["passed"] is False
    assert record["subtasks"][0]["attempts"][1]["passed"] is True


def test_run_architect_rejects_builder_files_outside_subtask(
    tmp_path: Path, monkeypatch
) -> None:
    check_calls: list[tuple] = []

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(
        runner_module,
        "build_files",
        lambda task, provider, **kwargs: _output(path="other.py"),
    )

    def fake_checks(*args, **kwargs):
        check_calls.append(args)
        return SandboxResult(True, "passed", "All checks passed.")

    monkeypatch.setattr(
        runner_module,
        "run_checks",
        fake_checks,
    )

    result = _run_architect(tmp_path, _settings(max_attempts=2))

    assert result.status == "failed"
    assert result.attempts == 2
    assert check_calls == []
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    attempts = record["subtasks"][0]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["stage"] == "builder"
    assert "declared files" in attempts[0]["output"]
    assert record["final_checks"] is None


def test_run_architect_stops_after_failed_subtask(
    tmp_path: Path, monkeypatch
) -> None:
    plan = Plan(
        task="Create two modules",
        subtasks=[
            Subtask(
                id="one",
                title="First module",
                type="module",
                difficulty="easy",
                recommended_tier="local_small",
                files=["one.py"],
                spec="Write one.py.",
                tests=["tests/test_one.py"],
            ),
            Subtask(
                id="two",
                title="Second module",
                type="module",
                difficulty="easy",
                recommended_tier="local_small",
                files=["two.py"],
                spec="Write two.py.",
                tests=["tests/test_two.py"],
            ),
        ],
        test_files=[
            GradingTestFile(
                path="tests/test_one.py",
                content="def test_one() -> None:\n    assert True\n",
            ),
            GradingTestFile(
                path="tests/test_two.py",
                content="def test_two() -> None:\n    assert True\n",
            ),
        ],
    )
    built_paths: list[str] = []

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        path = "one.py" if "Subtask one" in task else "two.py"
        built_paths.append(path)
        return _output(path=path)

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: plan)
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            False, "pytest", "AssertionError"
        ),
    )

    result = _run_architect(tmp_path, _settings(max_attempts=1))

    assert result.status == "failed"
    assert result.attempts == 1
    assert built_paths == ["one.py"]
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert [subtask["id"] for subtask in record["subtasks"]] == ["one"]
    assert record["subtasks"][0]["status"] == "failed"


def test_run_architect_invalid_plan_creates_no_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_plan(*args, **kwargs):
        raise ArchitectResponseError("plan was malformed")

    monkeypatch.setattr(runner_module, "plan_task", fail_plan)

    with pytest.raises(RunnerError, match="architect failed"):
        _run_architect(tmp_path, _settings())

    assert not (tmp_path / "workspace").exists()


def test_plan_only_returns_validated_plan(monkeypatch) -> None:
    captured: dict = {}

    def fake_plan(task, provider, *, prompt, brief, temperature=0.0):
        captured.update(
            {
                "task": task,
                "provider": provider,
                "prompt": prompt,
                "brief": brief,
                "temperature": temperature,
            }
        )
        return _plan()

    monkeypatch.setattr(runner_module, "plan_task", fake_plan)

    plan = plan_only("Create palindrome.py", settings=_settings())

    assert plan.task == "Create palindrome.py"
    assert captured["task"] == "Create palindrome.py"
    assert captured["provider"].name == "fake"
    assert "TokenWeir architect" in captured["prompt"]
    assert "TokenWeir project brief" in captured["brief"]


def test_run_task_requires_existing_tests_directory(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="does not exist"):
        run_task(
            "Create palindrome.py",
            "test task",
            tmp_path / "missing",
            workspace_root=tmp_path / "workspace",
            traces_dir=tmp_path / "traces",
            settings=_settings(),
        )


def test_run_task_requires_test_files(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()

    with pytest.raises(RunnerError, match="no test_"):
        run_task(
            "Create palindrome.py",
            "test task",
            tests,
            workspace_root=tmp_path / "workspace",
            traces_dir=tmp_path / "traces",
            settings=_settings(),
        )


def test_select_builder_provider_reports_missing_configuration() -> None:
    settings = LoadedSettings(
        providers=[],
        unconfigured=["fake: set FAKE_MODEL to a LiteLLM model ID"],
        errors=[],
        builder=BuilderDefinition(provider="fake"),
    )

    with pytest.raises(RunnerError, match="not configured") as error:
        select_builder_provider(settings)

    assert "FAKE_MODEL" in str(error.value)


def test_select_builder_provider_requires_a_provider_name() -> None:
    settings = LoadedSettings(
        providers=[],
        unconfigured=[],
        errors=[],
        builder=BuilderDefinition(provider=None),
    )

    with pytest.raises(RunnerError, match="no builder provider"):
        select_builder_provider(settings)


def test_select_architect_provider_reports_missing_configuration() -> None:
    settings = LoadedSettings(
        providers=[],
        unconfigured=["fake: set FAKE_MODEL to a LiteLLM model ID"],
        errors=[],
        builder=BuilderDefinition(provider="fake"),
        architect=ArchitectDefinition(provider="fake"),
    )

    with pytest.raises(RunnerError, match="not configured") as error:
        select_architect_provider(settings)

    assert "FAKE_MODEL" in str(error.value)


def test_select_architect_provider_requires_a_provider_name() -> None:
    settings = LoadedSettings(
        providers=[],
        unconfigured=[],
        errors=[],
        builder=BuilderDefinition(provider="fake"),
        architect=ArchitectDefinition(provider=None),
    )

    with pytest.raises(RunnerError, match="no architect provider"):
        select_architect_provider(settings)


def test_run_task_integration_with_real_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    tests_dir = (
        Path(__file__).resolve().parents[1]
        / "bench/tasks/palindrome/tests"
    )
    monkeypatch.setattr(
        runner_module,
        "build_files",
        lambda task, provider, **kwargs: _output(),
    )

    result = run_task(
        "Create palindrome.py",
        "palindrome task",
        tests_dir,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=_settings(),
    )

    assert result.status == "passed"
    assert result.attempts == 1
    assert (result.workspace / "palindrome.py").exists()
