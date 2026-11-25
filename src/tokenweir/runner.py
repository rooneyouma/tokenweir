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
from tokenweir.budget import BudgetError, BudgetExceededError, BudgetPolicy
from tokenweir.builder import (
    BuilderError,
    BuilderResponseError,
    build_files,
    load_builder_prompt,
)
from tokenweir.router import RoutingError
from tokenweir.router import candidates as route_candidates
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
    status: Literal["passed", "failed", "paused"]
    workspace: Path
    attempts: int
    record_path: Path


@dataclass
class BuildStepResult:
    passed: bool
    attempts: list[dict[str, Any]]
    written: list[Path]
    paused: bool = False
    provider: ConfiguredProvider | None = None
    budget_error: BudgetExceededError | None = None


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
    number: int,
    stage: str,
    passed: bool,
    output: str,
    *,
    provider: ConfiguredProvider | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "attempt": number,
        "stage": stage,
        "passed": passed,
        "output": output[-2000:],
    }
    if provider is not None:
        data["provider"] = provider.name
        data["model"] = provider.model
    return data


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


def _budget_summary(policy: BudgetPolicy | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    budget = policy.budget
    return {
        "project": budget.project,
        "cap_usd": round(budget.cap_usd, 8),
        "spent_usd": round(budget.spent_usd, 8),
        "remaining_usd": round(budget.remaining_usd, 8),
        "ledger": str(budget.ledger_path),
    }


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
    providers: list[ConfiguredProvider],
    prompt: str,
    attempts_per_provider: int,
    max_total_attempts: int,
    temperature: float,
    workspace: Path,
    commit_message: str,
    budget_policy: BudgetPolicy | None = None,
    run_id: str = "",
    subtask: str | None = None,
    expected_files: set[str] | None = None,
    test_paths: list[str] | None = None,
    lint_paths: list[str] | None = None,
) -> BuildStepResult:
    if not providers:
        raise RunnerError("no provider is available for this step")
    attempts: list[dict[str, Any]] = []
    written: list[Path] = []
    feedback: str | None = None
    total_attempts = 0
    last_provider: ConfiguredProvider | None = None

    for provider in providers:
        last_provider = provider
        for _ in range(1, attempts_per_provider + 1):
            if total_attempts >= max_total_attempts:
                break
            total_attempts += 1
            print(
                f"Attempt {total_attempts}/{max_total_attempts}: "
                f"{provider.name} ({provider.model})"
            )
            extra: dict[str, Any] = {}
            if budget_policy is not None:
                extra = {
                    "budget_policy": budget_policy,
                    "subtask": subtask,
                    "attempt": total_attempts,
                    "run_id": run_id,
                }
            try:
                output = build_files(
                    task_text,
                    provider,
                    prompt=prompt,
                    feedback=feedback,
                    temperature=temperature,
                    **extra,
                )
                if expected_files is not None:
                    _validate_expected_files(expected_files, output)
            except BudgetExceededError as exc:
                attempts.append(
                    _attempt(
                        total_attempts,
                        "budget",
                        False,
                        str(exc),
                        provider=provider,
                    )
                )
                return BuildStepResult(
                    False,
                    attempts,
                    written,
                    paused=True,
                    provider=provider,
                    budget_error=exc,
                )
            except BudgetError as exc:
                raise RunnerError(f"budget configuration error: {exc}") from exc
            except BuilderError as exc:
                detail = str(exc)
                attempts.append(
                    _attempt(
                        total_attempts,
                        "builder",
                        False,
                        detail,
                        provider=provider,
                    )
                )
                feedback = _feedback("builder", detail)
                print(
                    f"Attempt {total_attempts} failed at builder: {detail}"
                )
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
                attempts.append(
                    _attempt(
                        total_attempts,
                        "write",
                        False,
                        detail,
                        provider=provider,
                    )
                )
                feedback = _feedback("write", detail)
                print(f"Attempt {total_attempts} failed at write: {detail}")
                continue

            result = run_checks(workspace, test_paths, lint_paths)
            attempts.append(
                _attempt(
                    total_attempts,
                    result.stage,
                    result.passed,
                    result.output,
                    provider=provider,
                )
            )
            if result.passed:
                try:
                    commit_step(workspace, commit_message)
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise RunnerError(
                        f"checks passed but the workspace commit failed: {exc}"
                    ) from exc
                print(
                    f"Attempt {total_attempts} passed; committed the workspace."
                )
                return BuildStepResult(
                    True, attempts, written, provider=provider
                )

            feedback = _feedback(result.stage, result.output)
            print(f"Attempt {total_attempts} failed at {result.stage}.")
            print(result.output)

        if total_attempts >= max_total_attempts:
            break
        print(f"Escalating from {provider.name}.")

    return BuildStepResult(
        False, attempts, written, provider=last_provider
    )


def _run_direct(
    task: str,
    task_source: str,
    tests_dir: Path,
    *,
    workspace_root: Path,
    traces_dir: Path,
    loaded: LoadedSettings,
    project: str,
    budget_policy: BudgetPolicy | None,
) -> RunResult:
    if not tests_dir.is_dir():
        raise RunnerError(f"grading tests directory does not exist: {tests_dir}")
    if not any(tests_dir.rglob("test_*.py")):
        raise RunnerError(
            f"grading tests directory has no test_*.py files: {tests_dir}"
        )

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
        providers=[provider],
        prompt=prompt,
        attempts_per_provider=loaded.builder.max_attempts,
        max_total_attempts=loaded.builder.max_attempts,
        temperature=loaded.builder.temperature,
        workspace=workspace,
        commit_message=f"Pass task: {_task_title(task)}",
        budget_policy=budget_policy,
        run_id=run_id,
    )
    if step.paused:
        status: Literal["passed", "failed", "paused"] = "paused"
    else:
        status = "passed" if step.passed else "failed"

    record: dict[str, Any] = {
        "run_id": run_id,
        "mode": "direct",
        "project": project,
        "task_source": task_source,
        "task": task,
        "workspace": str(workspace),
        "builder": {
            "provider": provider.name,
            "model": provider.model,
            "max_attempts": loaded.builder.max_attempts,
            "temperature": loaded.builder.temperature,
        },
        "budget": _budget_summary(budget_policy),
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempts": step.attempts,
    }
    if step.budget_error is not None:
        record["budget_error"] = str(step.budget_error)
    _write_record(record_path, record)

    if status == "passed":
        print(f"Run passed after {len(step.attempts)} attempt(s).")
    elif status == "paused":
        print(f"Run paused: {step.budget_error}")
        print(f"Workspace preserved at {workspace}.")
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
    project: str,
    budget_policy: BudgetPolicy | None,
    use_router: bool,
) -> RunResult:
    architect_provider = select_architect_provider(loaded)
    builder_provider: ConfiguredProvider | None = None
    if use_router:
        try:
            builder_provider = select_builder_provider(loaded)
        except RunnerError:
            builder_provider = None
    else:
        builder_provider = select_builder_provider(loaded)
    try:
        architect_prompt = load_architect_prompt()
        brief = load_project_brief()
        builder_prompt = load_builder_prompt()
    except (ArchitectError, BuilderError) as exc:
        raise RunnerError(str(exc)) from exc

    run_id = _new_run_id()
    workspace = workspace_root / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    record_path = traces_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc)

    try:
        plan_extra: dict[str, Any] = {}
        if budget_policy is not None:
            plan_extra = {"budget_policy": budget_policy, "run_id": run_id}
        plan = plan_task(
            task,
            architect_provider,
            prompt=architect_prompt,
            brief=brief,
            temperature=loaded.architect.temperature,
            **plan_extra,
        )
    except BudgetExceededError as exc:
        record: dict[str, Any] = {
            "run_id": run_id,
            "mode": "architect",
            "project": project,
            "task_source": task_source,
            "task": task,
            "workspace": str(workspace),
            "workspace_created": False,
            "architect": {
                "provider": architect_provider.name,
                "model": architect_provider.model,
                "temperature": loaded.architect.temperature,
            },
            "budget": _budget_summary(budget_policy),
            "budget_error": str(exc),
            "status": "paused",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_attempts": 0,
            "subtasks": [],
            "final_checks": None,
        }
        _write_record(record_path, record)
        print(f"Run paused before workspace creation: {exc}")
        print(f"Run record: {record_path}")
        return RunResult(
            status="paused",
            workspace=workspace,
            attempts=0,
            record_path=record_path,
        )
    except BudgetError as exc:
        raise RunnerError(str(exc)) from exc
    except ArchitectError as exc:
        raise RunnerError(f"architect failed: {exc}") from exc

    ordered = plan.ordered_subtasks()
    test_contents = {test_file.path: test_file.content for test_file in plan.test_files}

    workspace = _prepare_workspace(workspace_root, run_id, None)
    subtask_records: list[dict[str, Any]] = []
    total_attempts = 0
    final_checks: dict[str, Any] | None = None
    status: Literal["passed", "failed", "paused"] = "failed"
    budget_error: BudgetExceededError | None = None

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

        if use_router:
            budget_available = (
                budget_policy is None
                or budget_policy.budget.remaining_usd > 0
            )
            try:
                providers = route_candidates(
                    loaded, subtask, budget_available=budget_available
                )
            except RoutingError as exc:
                raise RunnerError(str(exc)) from exc
            attempts_per_provider = loaded.router.attempts_per_provider
            max_total_attempts = loaded.router.max_total_attempts
        else:
            if builder_provider is None:
                raise RunnerError(
                    "no builder provider is configured in registry.json"
                )
            providers = [builder_provider]
            attempts_per_provider = loaded.builder.max_attempts
            max_total_attempts = loaded.builder.max_attempts

        step = _run_build_step(
            task_text=_subtask_prompt(task, subtask),
            providers=providers,
            prompt=builder_prompt,
            attempts_per_provider=attempts_per_provider,
            max_total_attempts=max_total_attempts,
            temperature=loaded.builder.temperature,
            workspace=workspace,
            commit_message=f"Pass {subtask.id}: {subtask.title}",
            budget_policy=budget_policy,
            run_id=run_id,
            subtask=subtask.id,
            expected_files=set(subtask.files),
            test_paths=subtask.tests,
            lint_paths=subtask.files,
        )
        total_attempts += len(step.attempts)
        subtask_records.append(
            {
                "id": subtask.id,
                "title": subtask.title,
                "status": (
                    "paused"
                    if step.paused
                    else "passed" if step.passed else "failed"
                ),
                "files": subtask.files,
                "tests": subtask.tests,
                "attempts": step.attempts,
            }
        )
        if step.paused:
            status = "paused"
            budget_error = step.budget_error
            break
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

    record = {
        "run_id": run_id,
        "mode": "architect",
        "project": project,
        "task_source": task_source,
        "task": task,
        "workspace": str(workspace),
        "architect": {
            "provider": architect_provider.name,
            "model": architect_provider.model,
            "temperature": loaded.architect.temperature,
        },
        "builder": (
            {
                "provider": builder_provider.name,
                "model": builder_provider.model,
                "max_attempts": loaded.builder.max_attempts,
                "temperature": loaded.builder.temperature,
            }
            if builder_provider is not None
            else None
        ),
        "router": (
            {
                "attempts_per_provider": loaded.router.attempts_per_provider,
                "max_total_attempts": loaded.router.max_total_attempts,
            }
            if use_router
            else None
        ),
        "budget": _budget_summary(budget_policy),
        "plan": plan.model_dump(),
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_attempts": total_attempts,
        "subtasks": subtask_records,
        "final_checks": final_checks,
    }
    if budget_error is not None:
        record["budget_error"] = str(budget_error)
    _write_record(record_path, record)

    if status == "passed":
        print(
            f"Run passed after {total_attempts} attempt(s) across "
            f"{len(ordered)} subtask(s)."
        )
    elif status == "paused":
        print(f"Run paused: {budget_error}")
        print(f"Workspace preserved at {workspace}.")
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
    project: str = "default",
    budget_policy: BudgetPolicy | None = None,
    use_router: bool = False,
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
            project=project,
            budget_policy=budget_policy,
        )
    return _run_architect(
        task,
        task_source,
        workspace_root=workspace_root,
        traces_dir=traces_dir,
        loaded=loaded,
        project=project,
        budget_policy=budget_policy,
        use_router=use_router,
    )


def plan_only(
    task: str,
    *,
    settings: LoadedSettings | None = None,
    budget_policy: BudgetPolicy | None = None,
    run_id: str = "",
) -> Plan:
    """Make one architect call and return the validated plan without executing it."""
    loaded = _load_settings(settings)
    provider = select_architect_provider(loaded)
    try:
        prompt = load_architect_prompt()
        brief = load_project_brief()
    except ArchitectError as exc:
        raise RunnerError(str(exc)) from exc
    extra: dict[str, Any] = {}
    if budget_policy is not None:
        extra = {
            "budget_policy": budget_policy,
            "run_id": run_id or _new_run_id(),
        }
    try:
        return plan_task(
            task,
            provider,
            prompt=prompt,
            brief=brief,
            temperature=loaded.architect.temperature,
            **extra,
        )
    except BudgetExceededError:
        raise
    except BudgetError as exc:
        raise RunnerError(str(exc)) from exc
    except ArchitectError as exc:
        raise RunnerError(f"architect failed: {exc}") from exc
