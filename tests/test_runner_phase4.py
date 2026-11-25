import json
from pathlib import Path
from typing import Literal

import pytest

import tokenweir.runner as runner_module
from tokenweir.budget import (
    BudgetExceededError,
    BudgetPolicy,
    load_project_budget,
)
from tokenweir.runner import RunnerError, run_task
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
    RouterDefinition,
)

GOOD_CODE = "def value() -> int:\n    return 1\n"

Tier = Literal["local", "free_cloud", "paid"]


def _provider(name: str, tier: Tier = "local") -> ConfiguredProvider:
    return ConfiguredProvider(
        name=name,
        model=f"model/{name}",
        timeout_s=5,
        tier=tier,
        free=tier != "paid",
    )


def _settings(
    providers: list[ConfiguredProvider],
    *,
    attempts_per_provider: int = 2,
    max_total_attempts: int = 6,
) -> LoadedSettings:
    return LoadedSettings(
        providers=providers,
        unconfigured=[],
        errors=[],
        builder=BuilderDefinition(
            provider=providers[0].name if providers else None,
            max_attempts=3,
            temperature=0.0,
        ),
        architect=ArchitectDefinition(
            provider=providers[0].name if providers else None,
            temperature=0.0,
        ),
        router=RouterDefinition(
            attempts_per_provider=attempts_per_provider,
            max_total_attempts=max_total_attempts,
        ),
    )


def _plan() -> Plan:
    return Plan(
        task="Create impl.py",
        subtasks=[
            Subtask(
                id="impl",
                title="Implement value",
                type="function",
                difficulty="easy",
                recommended_tier="local_small",
                files=["impl.py"],
                spec="Implement value in impl.py.",
                tests=["tests/test_impl.py"],
            )
        ],
        test_files=[
            GradingTestFile(
                path="tests/test_impl.py",
                content=(
                    "from impl import value\n\n"
                    "def test_value() -> None:\n"
                    "    assert value() == 1\n"
                ),
            )
        ],
    )


def _output() -> BuilderOutput:
    return BuilderOutput(
        files=[BuilderFile(path="impl.py", content=GOOD_CODE)]
    )


def _tests_dir(tmp_path: Path) -> Path:
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_impl.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    return tests


def _policy(tmp_path: Path, *, cap_usd: float) -> BudgetPolicy:
    budget = load_project_budget(
        "phase4", state_dir=tmp_path / "projects", cap_usd=cap_usd
    )
    return BudgetPolicy(budget)


def test_router_escalates_across_providers_and_records_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(
        [_provider("local-a"), _provider("local-b")],
        attempts_per_provider=1,
        max_total_attempts=2,
    )
    seen: list[str] = []
    checks = [
        SandboxResult(False, "pytest", "AssertionError"),
        SandboxResult(True, "passed", "All checks passed."),
        SandboxResult(True, "passed", "All checks passed."),
    ]

    def fake_build(task, provider, **kwargs):
        seen.append(provider.name)
        return _output()

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: checks.pop(0),
    )

    result = run_task(
        "Create impl.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        use_router=True,
    )

    assert result.status == "passed"
    assert result.attempts == 2
    assert seen == ["local-a", "local-b"]
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    attempts = record["subtasks"][0]["attempts"]
    assert [attempt["provider"] for attempt in attempts] == [
        "local-a",
        "local-b",
    ]
    assert record["router"] == {
        "attempts_per_provider": 1,
        "max_total_attempts": 2,
    }


def test_router_does_not_require_the_default_builder_provider(
    tmp_path: Path, monkeypatch
) -> None:
    settings = LoadedSettings(
        providers=[_provider("local-a"), _provider("local-b")],
        unconfigured=[],
        errors=[],
        builder=BuilderDefinition(provider="missing", max_attempts=3),
        architect=ArchitectDefinition(provider="local-a"),
        router=RouterDefinition(
            attempts_per_provider=1, max_total_attempts=2
        ),
    )
    seen: list[str] = []
    checks = [
        SandboxResult(True, "passed", "All checks passed."),
        SandboxResult(True, "passed", "All checks passed."),
    ]

    def fake_build(task, provider, **kwargs):
        seen.append(provider.name)
        return _output()

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: checks.pop(0),
    )

    result = run_task(
        "Create impl.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        use_router=True,
    )

    assert result.status == "passed"
    assert seen == ["local-a"]
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["builder"] is None


def test_router_stops_at_the_total_attempt_bound(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(
        [_provider("local-a"), _provider("local-b")],
        attempts_per_provider=2,
        max_total_attempts=3,
    )
    seen: list[str] = []

    def fake_build(task, provider, **kwargs):
        seen.append(provider.name)
        return _output()

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            False, "pytest", "AssertionError"
        ),
    )

    result = run_task(
        "Create impl.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        use_router=True,
    )

    assert result.status == "failed"
    assert result.attempts == 3
    assert seen == ["local-a", "local-a", "local-b"]


def test_router_excludes_paid_providers_when_the_budget_is_exhausted(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(
        [_provider("paid-a", "paid"), _provider("local-a", "local")],
        attempts_per_provider=1,
        max_total_attempts=2,
    )
    policy = _policy(tmp_path, cap_usd=0.0)
    seen: list[str] = []
    checks = [
        SandboxResult(True, "passed", "All checks passed."),
        SandboxResult(True, "passed", "All checks passed."),
    ]

    def fake_build(task, provider, **kwargs):
        seen.append(provider.name)
        return _output()

    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: checks.pop(0),
    )

    result = run_task(
        "Create impl.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        budget_policy=policy,
        use_router=True,
    )

    assert result.status == "passed"
    assert seen == ["local-a"]


def test_router_reports_when_no_provider_fits_the_budget(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(
        [_provider("paid-a", "paid")],
        attempts_per_provider=1,
        max_total_attempts=1,
    )
    policy = _policy(tmp_path, cap_usd=0.0)
    monkeypatch.setattr(runner_module, "plan_task", lambda *args, **kwargs: _plan())

    with pytest.raises(RunnerError, match="within the current budget"):
        run_task(
            "Create impl.py",
            "test task",
            None,
            workspace_root=tmp_path / "workspace",
            traces_dir=tmp_path / "traces",
            settings=settings,
            budget_policy=policy,
            use_router=True,
        )


def test_direct_builder_budget_pause_preserves_the_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings([_provider("local-a")])
    policy = _policy(tmp_path, cap_usd=0.0)

    def fake_build(task, provider, **kwargs):
        raise BudgetExceededError(
            project="phase4",
            cap_usd=0.0,
            spent_usd=0.0,
            needed_usd=0.001,
            reason="paid builder call",
        )

    monkeypatch.setattr(runner_module, "build_files", fake_build)

    result = run_task(
        "Create impl.py",
        "test task",
        _tests_dir(tmp_path),
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        project="phase4",
        budget_policy=policy,
    )

    assert result.status == "paused"
    assert result.attempts == 1
    assert result.workspace.exists()
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["status"] == "paused"
    assert record["project"] == "phase4"
    assert record["budget"]["cap_usd"] == 0.0
    assert record["attempts"][0]["stage"] == "budget"
    assert "budget_error" in record


def test_architect_plan_budget_pause_creates_no_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings([_provider("local-a")])
    policy = _policy(tmp_path, cap_usd=0.0)

    def fail_plan(*args, **kwargs):
        raise BudgetExceededError(
            project="phase4",
            cap_usd=0.0,
            spent_usd=0.0,
            needed_usd=0.001,
            reason="paid architect call",
        )

    monkeypatch.setattr(runner_module, "plan_task", fail_plan)

    result = run_task(
        "Create impl.py",
        "test task",
        None,
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        project="phase4",
        budget_policy=policy,
        use_router=True,
    )

    assert result.status == "paused"
    assert result.attempts == 0
    assert not result.workspace.exists()
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["status"] == "paused"
    assert record["workspace_created"] is False
    assert record["budget"]["cap_usd"] == 0.0
    assert "budget_error" in record


def test_direct_builder_receives_budget_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings([_provider("local-a")])
    policy = _policy(tmp_path, cap_usd=1.0)
    captured: dict = {}

    def fake_build(task, provider, **kwargs):
        captured.update(kwargs)
        return _output()

    monkeypatch.setattr(runner_module, "build_files", fake_build)
    monkeypatch.setattr(
        runner_module,
        "run_checks",
        lambda workspace, *args, **kwargs: SandboxResult(
            True, "passed", "All checks passed."
        ),
    )

    result = run_task(
        "Create impl.py",
        "test task",
        _tests_dir(tmp_path),
        workspace_root=tmp_path / "workspace",
        traces_dir=tmp_path / "traces",
        settings=settings,
        project="my-project",
        budget_policy=policy,
    )

    assert result.status == "passed"
    assert captured["budget_policy"] is policy
    assert captured["subtask"] is None
    assert captured["attempt"] == 1
    assert captured["run_id"]
    record = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert record["project"] == "my-project"
    assert record["budget"]["project"] == "phase4"
