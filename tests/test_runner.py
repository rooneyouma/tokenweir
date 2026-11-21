import json
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

import tokenweir.runner as runner_module
from tokenweir.builder import BuilderResponseError
from tokenweir.runner import RunnerError, run_task, select_builder_provider
from tokenweir.sandbox import SandboxResult
from tokenweir.schemas import BuilderFile, BuilderOutput
from tokenweir.settings import (
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


def test_run_task_passes_and_commits(tmp_path: Path, monkeypatch) -> None:
    calls: list = []

    def fake_build(task, provider, *, prompt, feedback=None, temperature=0.0):
        calls.append(feedback)
        return _output()

    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace: SandboxResult(True, "passed", "All checks passed."),
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
        runner_module, "run_checks", lambda workspace: check_results.pop(0)
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
        lambda workspace: SandboxResult(False, "pytest", "AssertionError"),
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
        lambda workspace: SandboxResult(True, "passed", "All checks passed."),
    )

    result = _run(tmp_path, _settings())

    assert result.status == "passed"
    assert result.attempts == 2
    assert "not valid JSON" in calls[1]


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
