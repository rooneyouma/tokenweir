from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tokenweir.architect import (
    ArchitectError,
    ArchitectProviderUnavailableError,
    load_architect_prompt,
    load_project_brief,
    plan_task,
)
from tokenweir.budget import BudgetError, BudgetExceededError, BudgetPolicy
from tokenweir.builder import (
    BuilderError,
    BuilderProviderUnavailableError,
    BuilderResponseError,
    build_files,
    load_builder_prompt,
)
from tokenweir.router import RoutingError
from tokenweir.router import candidates as route_candidates
from tokenweir.sandbox import (
    commit_step,
    init_workspace,
    reset_workspace,
    run_checks,
    write_files,
)
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
    unavailable_providers: list[str] = field(default_factory=list)


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
    unavailable: list[str] = []

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
                    unavailable_providers=unavailable,
                )
            except BudgetError as exc:
                raise RunnerError(f"budget configuration error: {exc}") from exc
            except BuilderProviderUnavailableError as exc:
                detail = str(exc)
                attempts.append(
                    _attempt(
                        total_attempts,
                        "provider",
                        False,
                        detail,
                        provider=provider,
                    )
                )
                unavailable.append(provider.name)
                print(
                    f"Provider {provider.name} is unavailable: {detail}"
                )
                break
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
                    True,
                    attempts,
                    written,
                    provider=provider,
                    unavailable_providers=unavailable,
                )

            feedback = _feedback(result.stage, result.output)
            print(f"Attempt {total_attempts} failed at {result.stage}.")
            print(result.output)

        if total_attempts >= max_total_attempts:
            break
        print(f"Escalating from {provider.name}.")

    return BuildStepResult(
        False,
        attempts,
        written,
        provider=last_provider,
        unavailable_providers=unavailable,
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


def _free_planning_candidates(
    loaded: LoadedSettings, primary: ConfiguredProvider
) -> list[ConfiguredProvider]:
    candidates = [
        provider
        for provider in loaded.providers
        if provider.name != primary.name
        and (provider.free or provider.tier != "paid")
    ]
    return sorted(
        candidates,
        key=lambda provider: (
            0 if provider.tier == "local" else 1,
            provider.name,
        ),
    )


def _plan_with_fallback(
    task: str,
    loaded: LoadedSettings,
    *,
    prompt: str,
    brief: str,
    budget_policy: BudgetPolicy | None,
    run_id: str,
    fallbacks: list[dict[str, str]],
) -> tuple[Plan, ConfiguredProvider]:
    primary = select_architect_provider(loaded)
    candidates = [primary] + _free_planning_candidates(loaded, primary)
    errors: list[str] = []

    for index, provider in enumerate(candidates):
        extra: dict[str, Any] = {}
        if budget_policy is not None:
            extra = {"budget_policy": budget_policy, "run_id": run_id}
        try:
            plan = plan_task(
                task,
                provider,
                prompt=prompt,
                brief=brief,
                temperature=loaded.architect.temperature,
                **extra,
            )
            return plan, provider
        except ArchitectProviderUnavailableError as exc:
            detail = str(exc)
            fallbacks.append(
                {
                    "provider": provider.name,
                    "kind": "unavailable",
                    "reason": detail,
                }
            )
            errors.append(f"{provider.name}: {detail}")
            if index + 1 < len(candidates):
                print(
                    f"Architect provider {provider.name} is unavailable; "
                    f"trying {candidates[index + 1].name}."
                )
            continue
        except BudgetExceededError as exc:
            free_fallbacks = [
                candidate
                for candidate in candidates[index + 1:]
                if candidate.free or candidate.tier != "paid"
            ]
            if provider.free or provider.tier != "paid" or not free_fallbacks:
                raise
            detail = str(exc)
            fallbacks.append(
                {
                    "provider": provider.name,
                    "kind": "budget",
                    "reason": detail,
                }
            )
            errors.append(f"{provider.name}: {detail}")
            print(
                f"Architect provider {provider.name} cannot fit the budget; "
                f"trying free fallback {free_fallbacks[0].name}."
            )
            continue
        except BudgetError as exc:
            raise RunnerError(str(exc)) from exc
        except ArchitectError as exc:
            raise RunnerError(f"architect failed: {exc}") from exc

    raise RunnerError(
        "architect unavailable and no free fallback could plan: "
        + "; ".join(errors)
    )


def _validate_completed_records(
    plan: Plan, completed_records: list[dict[str, Any]]
) -> None:
    ordered = plan.ordered_subtasks()
    for index, record in enumerate(completed_records):
        if (
            index >= len(ordered)
            or record.get("id") != ordered[index].id
            or record.get("status") != "passed"
        ):
            raise RunnerError(
                "resume state does not match the plan's completed subtasks"
            )


def _execute_subtasks(
    *,
    task: str,
    plan: Plan,
    workspace: Path,
    loaded: LoadedSettings,
    builder_provider: ConfiguredProvider | None,
    budget_policy: BudgetPolicy | None,
    use_router: bool,
    run_id: str,
    builder_prompt: str,
    completed_records: list[dict[str, Any]] | None = None,
) -> tuple[
    Literal["passed", "failed", "paused"],
    list[dict[str, Any]],
    int,
    dict[str, Any] | None,
    BudgetExceededError | None,
    list[str],
]:
    ordered = plan.ordered_subtasks()
    test_contents = {
        test_file.path: test_file.content for test_file in plan.test_files
    }
    subtask_records = list(completed_records or [])
    _validate_completed_records(plan, subtask_records)

    unavailable: set[str] = set()
    new_attempts = 0
    final_checks: dict[str, Any] | None = None
    status: Literal["passed", "failed", "paused"] = "failed"
    budget_error: BudgetExceededError | None = None

    for subtask in ordered[len(subtask_records):]:
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
            providers = [
                provider
                for provider in providers
                if provider.name not in unavailable
            ]
            attempts_per_provider = loaded.router.attempts_per_provider
            max_total_attempts = loaded.router.max_total_attempts
        else:
            if builder_provider is None:
                raise RunnerError(
                    "no builder provider is configured in registry.json"
                )
            providers = (
                []
                if builder_provider.name in unavailable
                else [builder_provider]
            )
            attempts_per_provider = loaded.builder.max_attempts
            max_total_attempts = loaded.builder.max_attempts

        if not providers:
            detail = (
                "all candidate providers are rate-limited or out of quota"
            )
            print(detail)
            subtask_records.append(
                {
                    "id": subtask.id,
                    "title": subtask.title,
                    "status": "failed",
                    "files": subtask.files,
                    "tests": subtask.tests,
                    "attempts": [],
                    "provider_error": detail,
                }
            )
            break

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
        new_attempts += len(step.attempts)
        unavailable.update(step.unavailable_providers)
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

    return (
        status,
        subtask_records,
        new_attempts,
        final_checks,
        budget_error,
        sorted(unavailable),
    )


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

    fallbacks: list[dict[str, str]] = []
    try:
        plan, architect_provider = _plan_with_fallback(
            task,
            loaded,
            prompt=architect_prompt,
            brief=brief,
            budget_policy=budget_policy,
            run_id=run_id,
            fallbacks=fallbacks,
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
            "architect_fallbacks": fallbacks,
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

    workspace = _prepare_workspace(workspace_root, run_id, None)
    (
        status,
        subtask_records,
        total_attempts,
        final_checks,
        budget_error,
        unavailable,
    ) = _execute_subtasks(
        task=task,
        plan=plan,
        workspace=workspace,
        loaded=loaded,
        builder_provider=builder_provider,
        budget_policy=budget_policy,
        use_router=use_router,
        run_id=run_id,
        builder_prompt=builder_prompt,
    )

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
        "architect_fallbacks": fallbacks,
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
        "unavailable_providers": unavailable,
        "subtasks": subtask_records,
        "final_checks": final_checks,
    }
    if budget_error is not None:
        record["budget_error"] = str(budget_error)
    _write_record(record_path, record)

    if status == "passed":
        print(
            f"Run passed after {total_attempts} attempt(s) across "
            f"{len(plan.subtasks)} subtask(s)."
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


def load_run_record(record_path: Path) -> dict[str, Any]:
    """Load a JSON run record for inspection or resume."""
    try:
        raw = record_path.read_text(encoding="utf-8")
        record = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read run record {record_path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RunnerError(f"run record is not a JSON object: {record_path}")
    return record


def _resolve_record_workspace(
    record: dict[str, Any], workspace_root: Path
) -> Path:
    raw = record.get("workspace")
    if not isinstance(raw, str) or not raw:
        raise RunnerError("run record does not name a workspace")
    workspace = Path(raw)
    try:
        root = workspace_root.resolve()
        resolved = workspace.resolve()
    except OSError as exc:
        raise RunnerError(f"cannot resolve recorded workspace: {exc}") from exc
    if resolved == root or not resolved.is_relative_to(root):
        raise RunnerError(
            f"recorded workspace is outside {root}: {workspace}"
        )
    if not resolved.is_dir():
        raise RunnerError(f"recorded workspace does not exist: {workspace}")
    if not (resolved / ".git").exists():
        raise RunnerError(
            f"recorded workspace is not a git repository: {workspace}"
        )
    return resolved


def _attempt_count(record: dict[str, Any]) -> int:
    total = record.get("total_attempts")
    if isinstance(total, int):
        return total
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        return len(attempts)
    return 0


def resume_run(
    record_path: Path,
    *,
    workspace_root: Path = Path("workspace"),
    traces_dir: Path = Path("traces"),
    settings: LoadedSettings | None = None,
    project: str | None = None,
    budget_policy: BudgetPolicy | None = None,
    use_router: bool = False,
) -> RunResult:
    """Resume a paused or failed run from its preserved workspace."""
    loaded = _load_settings(settings)
    previous = load_run_record(record_path)
    mode = previous.get("mode")
    if mode not in {"architect", "direct"}:
        raise RunnerError(f"cannot resume run with unknown mode: {mode!r}")
    workspace = _resolve_record_workspace(previous, workspace_root)
    resolved_project = project or str(previous.get("project") or "default")
    previous_status = previous.get("status")
    if previous_status == "passed":
        print("Run already passed; nothing to resume.")
        return RunResult(
            status="passed",
            workspace=workspace,
            attempts=0,
            record_path=record_path,
        )
    if previous_status not in {"paused", "failed"}:
        raise RunnerError(
            f"cannot resume run with status {previous_status!r}"
        )

    task = previous.get("task")
    if not isinstance(task, str) or not task:
        raise RunnerError("run record does not contain a task")
    task_source = previous.get("task_source", "task.md")
    previous_attempts = _attempt_count(previous)
    run_id = _new_run_id()
    traces_dir.mkdir(parents=True, exist_ok=True)
    new_record_path = traces_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc)
    resumed_from = previous.get("run_id", str(record_path))

    if mode == "direct":
        provider = select_builder_provider(loaded)
        try:
            prompt = load_builder_prompt()
        except BuilderError as exc:
            raise RunnerError(str(exc)) from exc
        try:
            reset_workspace(workspace)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RunnerError(
                f"cannot reset workspace {workspace}: {exc}"
            ) from exc
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
        new_attempts = len(step.attempts)
        record: dict[str, Any] = {
            "run_id": run_id,
            "mode": "direct",
            "project": resolved_project,
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
            "resumed_from": resumed_from,
            "previous_attempts": previous_attempts,
            "new_attempts": new_attempts,
            "total_attempts": previous_attempts + new_attempts,
            "attempts": step.attempts,
            "unavailable_providers": step.unavailable_providers,
        }
        if step.budget_error is not None:
            record["budget_error"] = str(step.budget_error)
        _write_record(new_record_path, record)
        if status == "passed":
            print(
                f"Resumed run passed after {new_attempts} new attempt(s)."
            )
        elif status == "paused":
            print(f"Resumed run paused: {step.budget_error}")
            print(f"Workspace preserved at {workspace}.")
        else:
            print(
                f"Resumed run failed after {new_attempts} new attempt(s). "
                f"Workspace preserved at {workspace}."
            )
        print(f"Run record: {new_record_path}")
        return RunResult(
            status=status,
            workspace=workspace,
            attempts=new_attempts,
            record_path=new_record_path,
        )

    plan_raw = previous.get("plan")
    if not isinstance(plan_raw, dict):
        raise RunnerError("cannot resume a run that paused before planning")
    try:
        plan = Plan.model_validate(plan_raw)
    except ValueError as exc:
        raise RunnerError(
            f"run record contains an invalid plan: {exc}"
        ) from exc
    raw_subtasks = previous.get("subtasks")
    if not isinstance(raw_subtasks, list):
        raise RunnerError("run record does not contain subtask state")
    completed_records = [
        record
        for record in raw_subtasks
        if isinstance(record, dict) and record.get("status") == "passed"
    ]
    _validate_completed_records(plan, completed_records)

    builder_provider: ConfiguredProvider | None = None
    if use_router:
        try:
            builder_provider = select_builder_provider(loaded)
        except RunnerError:
            builder_provider = None
    else:
        builder_provider = select_builder_provider(loaded)
    try:
        builder_prompt = load_builder_prompt()
    except BuilderError as exc:
        raise RunnerError(str(exc)) from exc
    try:
        reset_workspace(workspace)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(
            f"cannot reset workspace {workspace}: {exc}"
        ) from exc

    (
        status,
        subtask_records,
        new_attempts,
        final_checks,
        budget_error,
        unavailable,
    ) = _execute_subtasks(
        task=task,
        plan=plan,
        workspace=workspace,
        loaded=loaded,
        builder_provider=builder_provider,
        budget_policy=budget_policy,
        use_router=use_router,
        run_id=run_id,
        builder_prompt=builder_prompt,
        completed_records=completed_records,
    )
    record = {
        "run_id": run_id,
        "mode": "architect",
        "project": resolved_project,
        "task_source": task_source,
        "task": task,
        "workspace": str(workspace),
        "architect": previous.get("architect"),
        "architect_fallbacks": previous.get("architect_fallbacks", []),
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
        "resumed_from": resumed_from,
        "previous_attempts": previous_attempts,
        "new_attempts": new_attempts,
        "total_attempts": previous_attempts + new_attempts,
        "unavailable_providers": unavailable,
        "subtasks": subtask_records,
        "final_checks": final_checks,
    }
    if budget_error is not None:
        record["budget_error"] = str(budget_error)
    _write_record(new_record_path, record)
    if status == "passed":
        print(
            f"Resumed run passed after {new_attempts} new attempt(s) "
            f"across {len(plan.subtasks)} subtask(s)."
        )
    elif status == "paused":
        print(f"Resumed run paused: {budget_error}")
        print(f"Workspace preserved at {workspace}.")
    else:
        print(
            f"Resumed run failed after {new_attempts} new attempt(s). "
            f"Workspace preserved at {workspace}."
        )
    print(f"Run record: {new_record_path}")
    return RunResult(
        status=status,
        workspace=workspace,
        attempts=new_attempts,
        record_path=new_record_path,
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
    try:
        prompt = load_architect_prompt()
        brief = load_project_brief()
    except ArchitectError as exc:
        raise RunnerError(str(exc)) from exc
    fallbacks: list[dict[str, str]] = []
    plan, _ = _plan_with_fallback(
        task,
        loaded,
        prompt=prompt,
        brief=brief,
        budget_policy=budget_policy,
        run_id=run_id or _new_run_id(),
        fallbacks=fallbacks,
    )
    return plan
