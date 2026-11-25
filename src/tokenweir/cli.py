from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from tokenweir.budget import BudgetExceededError, BudgetPolicy
    from tokenweir.settings import LoadedSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenweir",
        description="Plan a coding task and build it with a configured builder model.",
    )
    parser.add_argument(
        "task_source",
        nargs="?",
        metavar="TASK",
        help="Inline task text, a task file, or a directory containing task.md.",
    )
    parser.add_argument(
        "--task",
        dest="inline_task",
        help="Task description supplied inline (explicit form).",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        metavar="PATH",
        help="Path to a Markdown or text file containing the task (explicit form).",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        metavar="PATH",
        help=(
            "Directory containing grading test_*.py files; skips architect "
            "planning and uses direct builder mode."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Ask the architect for a validated plan and print it without building.",
    )
    parser.add_argument(
        "--project",
        default="default",
        metavar="NAME",
        help="Project budget and ledger group to use (default: default).",
    )
    parser.add_argument(
        "--cap",
        type=float,
        metavar="USD",
        help=(
            "Set or replace the project's paid-spend cap before running. "
            "Paid calls are blocked at the cap; free/local calls remain allowed."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        metavar="PATH",
        help="Directory for project budgets and ledgers (default from registry).",
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        help=(
            "Use only the configured builder provider in architect mode "
            "instead of the deterministic router."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN",
        help=(
            "Resume a paused or failed run from its JSON run record, "
            "for example traces/<run_id>.json."
        ),
    )
    return parser


def _read_task_file(task_path: Path) -> str | None:
    try:
        return task_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"tokenweir: cannot read task file {task_path}: {exc}", file=sys.stderr)
        return None


def _resolve_tests_dir(
    explicit: Path | None, task_path: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit
    if task_path is not None:
        sibling = task_path.parent / "tests"
        if sibling.is_dir():
            return sibling
    return None


def _prompt_for_new_cap(exc: "BudgetExceededError") -> float | None:
    if not sys.stdin.isatty():
        print(
            "tokenweir: rerun with --cap USD to extend the project budget.",
            file=sys.stderr,
        )
        return None

    minimum = max(exc.cap_usd, exc.spent_usd)
    while True:
        try:
            raw = input(
                f"{exc}\nEnter a new cap in USD (blank to pause): "
            ).strip()
        except EOFError:
            return None
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("tokenweir: enter a number.", file=sys.stderr)
            continue
        if value <= minimum:
            print(
                f"tokenweir: new cap must be greater than ${minimum:.2f}.",
                file=sys.stderr,
            )
            continue
        return value


def _print_budget(policy: "BudgetPolicy") -> None:
    budget = policy.budget
    print(
        f"Budget: project '{budget.project}' cap ${budget.cap_usd:.2f}, "
        f"spent ${budget.spent_usd:.2f}, "
        f"remaining ${budget.remaining_usd:.2f}.",
        flush=True,
    )


def _load_budget_policy(
    args: argparse.Namespace,
    loaded: "LoadedSettings",
    project: str | None = None,
) -> "BudgetPolicy":
    from tokenweir.budget import BudgetPolicy, load_project_budget

    state_dir = args.state_dir or Path(loaded.budget.state_dir)
    budget = load_project_budget(
        project or args.project,
        state_dir=state_dir,
        cap_usd=args.cap,
        default_cap_usd=loaded.budget.default_cap_usd,
        warn_fraction=loaded.budget.warn_fraction,
    )
    return BudgetPolicy(budget, prompt=_prompt_for_new_cap)


def _resolve_task(args: argparse.Namespace) -> tuple[str, str, Path | None] | None:
    provided = [
        value
        for value in (args.task_source, args.inline_task, args.task_file)
        if value is not None
    ]
    if not provided:
        raise ValueError("provide a task as text or a task file path")
    if len(provided) > 1:
        raise ValueError("provide only one task source")

    task_path: Path | None = None
    if args.inline_task is not None:
        task = args.inline_task
        source = "inline input"
    elif args.task_file is not None:
        task_path = args.task_file
        loaded = _read_task_file(task_path)
        if loaded is None:
            return None
        task = loaded
        source = str(task_path)
    else:
        assert args.task_source is not None
        candidate = Path(args.task_source)
        if candidate.is_dir():
            task_file = candidate / "task.md"
            if not task_file.is_file():
                print(
                    f"tokenweir: no task.md found in {candidate}", file=sys.stderr
                )
                return None
            task_path = task_file
            loaded = _read_task_file(task_file)
            if loaded is None:
                return None
            task = loaded
            source = str(task_file)
        elif candidate.is_file():
            task_path = candidate
            loaded = _read_task_file(candidate)
            if loaded is None:
                return None
            task = loaded
            source = str(candidate)
        else:
            task = args.task_source
            source = "inline input"

    task = task.strip()
    if not task:
        print("tokenweir: task must not be empty", file=sys.stderr)
        return None
    return task, source, task_path


def _resume(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from tokenweir.budget import BudgetError, BudgetExceededError
    from tokenweir.runner import RunnerError, load_run_record, resume_run
    from tokenweir.settings import SettingsError, load_settings

    try:
        loaded = load_settings()
    except SettingsError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    try:
        previous = load_run_record(args.resume)
    except RunnerError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    project = args.project
    if project == parser.get_default("project"):
        recorded_project = previous.get("project")
        if isinstance(recorded_project, str) and recorded_project:
            project = recorded_project
    try:
        policy = _load_budget_policy(args, loaded, project=project)
    except BudgetError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    _print_budget(policy)
    try:
        result = resume_run(
            args.resume,
            settings=loaded,
            project=project,
            budget_policy=policy,
            use_router=not args.no_router,
        )
    except BudgetExceededError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 3
    except BudgetError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    except RunnerError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    if result.status == "passed":
        return 0
    if result.status == "paused":
        return 3
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None:
        if any(
            value is not None
            for value in (
                args.task_source,
                args.inline_task,
                args.task_file,
                args.tests,
            )
        ) or args.plan_only:
            parser.error(
                "--resume cannot be combined with a task, --tests, or --plan-only"
            )
        return _resume(args, parser)
    try:
        resolved = _resolve_task(args)
    except ValueError as exc:
        parser.error(str(exc))
    if resolved is None:
        return 2
    task, source, task_path = resolved

    print(f"Loaded task from {source} ({len(task)} characters).", flush=True)

    from tokenweir.budget import BudgetError, BudgetExceededError
    from tokenweir.runner import RunnerError
    from tokenweir.settings import SettingsError, load_settings

    try:
        loaded = load_settings()
    except SettingsError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2

    try:
        policy = _load_budget_policy(args, loaded)
    except BudgetError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    _print_budget(policy)

    if args.plan_only:
        from tokenweir.runner import plan_only

        try:
            plan = plan_only(task, settings=loaded, budget_policy=policy)
        except BudgetExceededError as exc:
            print(f"tokenweir: {exc}", file=sys.stderr)
            return 3
        except RunnerError as exc:
            print(f"tokenweir: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(plan.model_dump(), indent=2))
        return 0

    tests_dir = _resolve_tests_dir(args.tests, task_path)
    if tests_dir is not None:
        if not tests_dir.is_dir():
            print(
                f"tokenweir: tests path is not a directory: {tests_dir}",
                file=sys.stderr,
            )
            return 2
        if not any(tests_dir.rglob("test_*.py")):
            print(
                f"tokenweir: no test_*.py files found in {tests_dir}",
                file=sys.stderr,
            )
            return 2
        print(f"Using supplied grading tests from {tests_dir}.", flush=True)
    else:
        print(
            "No grading tests supplied; asking the architect to plan and "
            "generate them.",
            flush=True,
        )

    from tokenweir.runner import run_task

    try:
        result = run_task(
            task,
            source,
            tests_dir,
            settings=loaded,
            project=args.project,
            budget_policy=policy,
            use_router=not args.no_router,
        )
    except BudgetExceededError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 3
    except BudgetError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2
    except RunnerError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2

    if result.status == "passed":
        return 0
    if result.status == "paused":
        return 3
    return 1
