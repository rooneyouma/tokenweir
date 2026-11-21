from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tokenweir.builder import BuilderError, build_files, load_builder_prompt
from tokenweir.sandbox import commit_step, init_workspace, run_checks, write_files
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


def select_builder_provider(settings: LoadedSettings) -> ConfiguredProvider:
    name = settings.builder.provider
    if name is None:
        raise RunnerError("no builder provider is configured in registry.json")
    for provider in settings.providers:
        if provider.name == name:
            return provider

    details = [
        message
        for message in settings.unconfigured + settings.errors
        if message.startswith(f"{name}:")
    ]
    suffix = f" ({'; '.join(details)})" if details else ""
    raise RunnerError(f"builder provider '{name}' is not configured{suffix}")


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


def run_task(
    task: str,
    task_source: str,
    tests_dir: Path,
    *,
    workspace_root: Path = Path("workspace"),
    traces_dir: Path = Path("traces"),
    settings: LoadedSettings | None = None,
) -> RunResult:
    """Run one task through one configured builder model, with bounded retries."""
    if not tests_dir.is_dir():
        raise RunnerError(f"grading tests directory does not exist: {tests_dir}")
    if not any(tests_dir.rglob("test_*.py")):
        raise RunnerError(f"grading tests directory has no test_*.py files: {tests_dir}")

    try:
        loaded = settings if settings is not None else load_settings()
    except SettingsError as exc:
        raise RunnerError(str(exc)) from exc

    provider = select_builder_provider(loaded)
    try:
        prompt = load_builder_prompt()
    except BuilderError as exc:
        raise RunnerError(str(exc)) from exc

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    workspace = workspace_root / run_id
    try:
        init_workspace(workspace, tests_dir, workspace_root=workspace_root)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"cannot prepare the workspace: {exc}") from exc

    traces_dir.mkdir(parents=True, exist_ok=True)
    record_path = traces_dir / f"{run_id}.json"
    started_at = datetime.now(timezone.utc)
    attempts: list[dict[str, Any]] = []
    written: list[Path] = []
    feedback: str | None = None
    status: Literal["passed", "failed"] = "failed"

    for attempt_number in range(1, loaded.builder.max_attempts + 1):
        print(
            f"Attempt {attempt_number}/{loaded.builder.max_attempts}: "
            f"{provider.name} ({provider.model})"
        )
        try:
            output = build_files(
                task,
                provider,
                prompt=prompt,
                feedback=feedback,
                temperature=loaded.builder.temperature,
            )
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

        result = run_checks(workspace)
        attempts.append(
            _attempt(attempt_number, result.stage, result.passed, result.output)
        )
        if result.passed:
            try:
                commit_step(workspace, f"Pass task: {_task_title(task)}")
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RunnerError(
                    f"checks passed but the workspace commit failed: {exc}"
                ) from exc
            status = "passed"
            print(f"Attempt {attempt_number} passed; committed the workspace.")
            break

        feedback = _feedback(result.stage, result.output)
        print(f"Attempt {attempt_number} failed at {result.stage}.")
        print(result.output)

    finished_at = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "run_id": run_id,
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
        "finished_at": finished_at.isoformat(),
        "attempts": attempts,
    }
    try:
        record_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise RunnerError(f"cannot write the run record {record_path}: {exc}") from exc

    if status == "passed":
        print(f"Run passed after {len(attempts)} attempt(s).")
    else:
        print(
            f"Run failed after {len(attempts)} attempt(s). "
            f"Workspace preserved at {workspace}."
        )
    print(f"Run record: {record_path}")

    return RunResult(
        status=status,
        workspace=workspace,
        attempts=len(attempts),
        record_path=record_path,
    )
