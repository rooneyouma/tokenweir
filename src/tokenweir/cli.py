from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provided = [
        value
        for value in (args.task_source, args.inline_task, args.task_file)
        if value is not None
    ]
    if not provided:
        parser.error("provide a task as text or a task file path")
    if len(provided) > 1:
        parser.error("provide only one task source")

    task_path: Path | None = None
    if args.inline_task is not None:
        task = args.inline_task
        source = "inline input"
    elif args.task_file is not None:
        task_path = args.task_file
        loaded = _read_task_file(task_path)
        if loaded is None:
            return 2
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
                return 2
            task_path = task_file
            loaded = _read_task_file(task_file)
            if loaded is None:
                return 2
            task = loaded
            source = str(task_file)
        elif candidate.is_file():
            task_path = candidate
            loaded = _read_task_file(candidate)
            if loaded is None:
                return 2
            task = loaded
            source = str(candidate)
        else:
            task = args.task_source
            source = "inline input"

    task = task.strip()
    if not task:
        print("tokenweir: task must not be empty", file=sys.stderr)
        return 2

    print(f"Loaded task from {source} ({len(task)} characters).", flush=True)

    if args.plan_only:
        from tokenweir.runner import RunnerError, plan_only

        try:
            plan = plan_only(task)
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

    from tokenweir.runner import RunnerError, run_task

    try:
        result = run_task(task, source, tests_dir)
    except RunnerError as exc:
        print(f"tokenweir: {exc}", file=sys.stderr)
        return 2

    return 0 if result.status == "passed" else 1
