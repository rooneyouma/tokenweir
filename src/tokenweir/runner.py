from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tokenweir.architect import (
    ArchitectError,
    load_architect_prompt,
    load_project_brief,
    plan_task,
)
from tokenweir.builder import (
    BuilderError,
    BuilderResponseError,
    build_files,
    load_builder_prompt,
)
from tokenweir.sandbox import commit_step, init_workspace, run_checks, write_files
from tokenweir.schemas import BuilderOutput, Plan, Subtask
from tokenweir.settings import (
    ConfiguredProvider,
    LoadedSettings,
    SettingsError,
    load_settings,
)


class RunnerError(RuntimeError):
    """Raised when a run cannot start or finish safely."""


@dataclass
class RunResult:
    status: Literal["passed", "failed"]
    workspace: Path
    attempts: int
    record_path: Path


@dataclass
class BuildStepResult:
    passed: bool
    attempts: list[dict[str, Any]]
    written: list[Path]


def _select_provider(
    settings: LoadedSettings, role: str, name: str | None
) -> ConfiguredProvider:
    if name is None:
        raise RunnerError(f"no {role} provider is configured in registry.json")
    for provider in settings.providers:
        if provider.name == name:
            return provider

    details = [
        message
        for message in settings.unconfigured + settings.errors
        if message.startswith(f"{name}:")
    ]
    suffix = f" ({'; '.join(details)})" if details else ""
    raise RunnerError(f"{role} provider '{name}' is not configured{suffix}")


def select_builder_provider(settings: LoadedSettings) -> ConfiguredProvider:
    return _select_provider(settings, "builder", settings.builder.provider)


def select_architect_provider(settings: LoadedSettings) -> ConfiguredProvider:
    return _select_provider(settings, "architect", settings.architect.provider)


def _load_settings(settings: LoadedSettings | None) -> LoadedSettings:
    try:
        return settings if settings is not None else load_settings()
    except SettingsError as exc:
        raise RunnerError(str(exc)) from exc


def _task_title(task: str) -> str:
    for line in task.splitlines():
        title = line.strip()
        if title:
            return title[:72]
    return "task"


def _attempt(
    number: int, stage: str, passed: bool, output: str
) -> dict[str, Any]:
    return {
        "attempt": number,
        "stage": stage,
        "passed": passed,
        "output": output[-2000:],
    }


def _feedback(stage: str, output: str) -> str:
    return f"Stage: {stage}\n{output[-2000:]}"


def _new_run_id() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def _prepare_workspace(
    workspace_root: Path, run_id: str, tests_dir: Path | None
) -> Path:
    workspace = workspace_root / run_id
    try:
        init_workspace(workspace, tests_dir, workspace_root=workspace_root)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"cannot prepare the workspace: {exc}") from exc
    return workspace


def _write_record(record_path: Path, record: dict[str, Any]) -> None:
    try:
        record_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise RunnerError(f"cannot write the run record {record_path}: {exc}") from exc


def _validate_expected_files(
    expected: set[str], output: BuilderOutput
) -> None:
    actual = {file.path for file in output.files}
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if extra:
        details.append("unexpected: " + ", ".join(extra))
    raise BuilderResponseError(
        "subtask output must contain exactly the declared files ("
        + "; ".join(details)
        + ")"
    )


def _run_build_step(
    *,
    task_text: str,
    provider: ConfiguredProvider,
    prompt: str,
    max_attempts: int,
    temperature: float,
    workspace: Path,
    commit_message: str,
    expected_files: set[str] | None = None,
    test_paths: list[str] | None = None,
    lint_paths: list[str] | None = None,
) -> BuildStepResult:
    attempts: list[dict[str, Any]] = []
    written: list[Path] = []
    feedback: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        print(
            f"Attempt {attempt_number}/{max_attempts}: "
            f"{provider.name} ({provider.model})"
        )
        try:
            output = build_files(
                task_text,
                provider,
                prompt=prompt,
                feedback=feedback,
                temperature=temperature,
            )
            if expected_files is not None:
                _validate_expected_files(expected_files, output)
        except BuilderError as exc:
            detail = str(exc)
            attempts.append(_attempt(attempt_number, "builder", False, detail))
            feedback = _feedback("builder", detail)
            print(f"Attempt {attempt_number} failed at builder: {detail}")
            continue

        for path in written:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            written = write_files(
                workspace,
                [(file.path, file.content) for file in output.files],
            )
        except (OSError, ValueError) as exc:
            written = []
            detail = str(exc)
            attempts.append(_attempt(attempt_number, "write", False, detail))
            feedback = _feedback("write", detail)
            print(f"Attempt {attempt_number} failed at write: {detail}")
            continue

        result = run_checks(workspace, test_paths, lint_paths)
        attempts.append(
            _attempt(attempt_number, result.stage, result.passed, result.output)
        )
        if result.passed:
            try:
                commit_step(workspace, commit_message)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RunnerError(
                    f"checks passed but the workspace commit failed: {exc}"
                ) from exc
            print(f"Attempt {attempt_number} passed; committed the workspace.")
            return BuildStepResult(True, attempts, written)

        feedback = _feedback(result.stage, result.output)
        print(f"Attempt {attempt_number} failed at {result.stage}.")
        print(result.output)

    return BuildStepResult(False, attempts, written)


def _run_direct(
    task: str,
    task_source: str,
    tests_dir: Path,
    *,
    workspace_root: Path,
    traces_dir: Path,
    loaded: LoadedSettings,
) -> RunResult:
    if not tests_dir.is_dir():
        raise RunnerError(f"grading tests directory does not exist: {tests_dir}")
    if not any(tests_dir.rglob("test_*.py")):
        raise RunnerError(f"grading tests directory has no test_*.py files: {tests_dir}")

    provider = select_builder_provider(loaded)
    try:
        prompt = load_builder_prompt()
    except BuilderError as exc:
        raise RunnerError(str(exc)) from exc

    run_id = _new_run_id()
    workspace = _prepare_workspace(workspace_root, run_id, tests_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    record_path = traces_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc)

    step = _run_build_step(
        task_text=task,
        provider=provider,
        prompt=prompt,
        max_attempts=loaded.builder.max_attempts,
        temperature=loaded.builder.temperature,
        workspace=workspace,
        commit_message=f"Pass task: {_task_title(task)}",
    )
    status: Literal["passed", "failed"] = "passed" if step.passed else "failed"

    record: dict[str, Any] = {
        "run_id": run_id,
        "mode": "direct",
        "task_source": task_source,
        "task": task,
        "workspace": str(workspace),
        "builder": {
            "provider": provider.name,
            "model": provider.model,
            "max_attempts": loaded.builder.max_attempts,
            "temperature": loaded.builder.temperature,
        },
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempts": step.attempts,
    }
    _write_record(record_path, record)

    if status == "passed":
        print(f"Run passed after {len(step.attempts)} attempt(s).")
    else:
        print(
            f"Run failed after {len(step.attempts)} attempt(s). "
            f"Workspace preserved at {workspace}."
        )
    print(f"Run record: {record_path}")

    return RunResult(
        status=status,
        workspace=workspace,
        attempts=len(step.attempts),
        record_path=record_path,
    )


def _subtask_prompt(task: str, subtask: Subtask) -> str:
    lines = [
        f"Overall task:\n{task}",
        f"Subtask {subtask.id}: {subtask.title}",
        f"Type: {subtask.type}",
        f"Difficulty: {subtask.difficulty}",
        f"Spec:\n{subtask.spec}",
    ]
    if subtask.signatures:
        lines.append(
            "Required signatures:\n"
            + "\n".join(f"- {signature}" for signature in subtask.signatures)
        )
    lines.append(
        "Files to return (exactly these paths):\n"
        + "\n".join(f"- {path}" for path in subtask.files)
    )
    if subtask.depends_on:
        lines.append("Completed dependencies: " + ", ".join(subtask.depends_on))
    lines.append("Do not write tests. Grading tests are supplied separately.")
    lines.append("Return the complete contents of every listed file as JSON.")
    return "\n\n".join(lines)


def _run_architect(
    task: str,
    task_source: str,
    *,
    workspace_root: Path,
    traces_dir: Path,
    loaded: LoadedSettings,
) -> RunResult:
    architect_provider = select_architect_provider(loaded)
    builder_provider = select_builder_provider(loaded)
    try:
        architect_prompt = load_architect_prompt()
        brief = load_project_brief()
        builder_prompt = load_builder_prompt()
    except (ArchitectError, BuilderError) as exc:
        raise RunnerError(str(exc)) from exc

    try:
        plan = plan_task(
            task,
            architect_provider,
            prompt=architect_prompt,
            brief=brief,
            temperature=loaded.architect.temperature,
        )
    except ArchitectError as exc:
        raise RunnerError(f"architect failed: {exc}") from exc

    ordered = plan.ordered_subtasks()
    test_contents = {test_file.path: test_file.content for test_file in plan.test_files}

    run_id = _new_run_id()
    workspace = _prepare_workspace(workspace_root, run_id, None)
    traces_dir.mkdir(parents=True, exist_ok=True)
    record_path = traces_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc)
    subtask_records: list[dict[str, Any]] = []
    total_attempts = 0
    final_checks: dict[str, Any] | None = None
    status: Literal["passed", "failed"] = "failed"

    for subtask in ordered:
        print(f"Subtask {subtask.id}: {subtask.title}")
        try:
            write_files(
                workspace,
                [(path, test_contents[path]) for path in subtask.tests],
                allow_tests=True,
            )
        except (OSError, ValueError) as exc:
            raise RunnerError(
                f"cannot write grading tests for subtask {subtask.id}: {exc}"
            ) from exc

        step = _run_build_step(
            task_text=_subtask_prompt(task, subtask),
            provider=builder_provider,
            prompt=builder_prompt,
            max_attempts=loaded.builder.max_attempts,
            temperature=loaded.builder.temperature,
            workspace=workspace,
            commit_message=f"Pass {subtask.id}: {subtask.title}",
            expected_files=set(subtask.files),
            test_paths=subtask.tests,
            lint_paths=subtask.files,
        )
        total_attempts += len(step.attempts)
        subtask_records.append(
            {
                "id": subtask.id,
                "title": subtask.title,
                "status": "passed" if step.passed else "failed",
                "files": subtask.files,
                "tests": subtask.tests,
                "attempts": step.attempts,
            }
        )
        if not step.passed:
            break

    all_passed = len(subtask_records) == len(ordered) and all(
        record["status"] == "passed" for record in subtask_records
    )
    if all_passed:
        generated_files = sorted(
            {path for subtask in ordered for path in subtask.files}
        )
        final = run_checks(workspace, None, generated_files)
        final_checks = {
            "stage": final.stage,
            "passed": final.passed,
            "output": final.output[-2000:],
        }
        status = "passed" if final.passed else "failed"
        if final.passed:
            print("Final full test suite passed.")
        else:
            print(f"Final full test suite failed at {final.stage}.")
            print(final.output)

    record: dict[str, Any] = {
        "run_id": run_id,
        "mode": "architect",
        "task_source": task_source,
        "task": task,
        "workspace": str(workspace),
        "architect": {
            "provider": architect_provider.name,
            "model": architect_provider.model,
            "temperature": loaded.architect.temperature,
        },
        "builder": {
            "provider": builder_provider.name,
            "model": builder_provider.model,
            "max_attempts": loaded.builder.max_attempts,
            "temperature": loaded.builder.temperature,
        },
        "plan": plan.model_dump(),
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_attempts": total_attempts,
        "subtasks": subtask_records,
        "final_checks": final_checks,
    }
    _write_record(record_path, record)

    if status == "passed":
        print(
            f"Run passed after {total_attempts} attempt(s) across "
            f"{len(ordered)} subtask(s)."
        )
    else:
        print(
            f"Run failed after {total_attempts} attempt(s). "
            f"Workspace preserved at {workspace}."
        )
    print(f"Run record: {record_path}")

    return RunResult(
        status=status,
        workspace=workspace,
        attempts=total_attempts,
        record_path=record_path,
    )


def run_task(
    task: str,
    task_source: str,
    tests_dir: Path | None,
    *,
    workspace_root: Path = Path("workspace"),
    traces_dir: Path = Path("traces"),
    settings: LoadedSettings | None = None,
) -> RunResult:
    """Run one task directly with pre-supplied tests or via architect planning."""
    loaded = _load_settings(settings)
    if tests_dir is not None:
        return _run_direct(
            task,
            task_source,
            tests_dir,
            workspace_root=workspace_root,
            traces_dir=traces_dir,
            loaded=loaded,
        )
    return _run_architect(
        task,
        task_source,
        workspace_root=workspace_root,
        traces_dir=traces_dir,
        loaded=loaded,
    )


def plan_only(
    task: str,
    *,
    settings: LoadedSettings | None = None,
) -> Plan:
    """Make one architect call and return the validated plan without executing it."""
    loaded = _load_settings(settings)
    provider = select_architect_provider(loaded)
    try:
        prompt = load_architect_prompt()
        brief = load_project_brief()
    except ArchitectError as exc:
        raise RunnerError(str(exc)) from exc
    try:
        return plan_task(
            task,
            provider,
            prompt=prompt,
            brief=brief,
            temperature=loaded.architect.temperature,
        )
    except ArchitectError as exc:
        raise RunnerError(f"architect failed: {exc}") from exc
