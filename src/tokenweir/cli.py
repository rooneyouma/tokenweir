from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenweir",
        description="Load a coding task for TokenWeir.",
    )
    parser.add_argument(
        "task_source",
        nargs="?",
        metavar="TASK",
        help="Inline task text, or the path to a Markdown/text task file.",
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
    return parser


def _read_task_file(task_path: Path) -> str | None:
    try:
        return task_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"tokenweir: cannot read task file {task_path}: {exc}", file=sys.stderr)
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

    if args.inline_task is not None:
        task = args.inline_task
        source = "inline input"
    elif args.task_file is not None:
        loaded = _read_task_file(args.task_file)
        if loaded is None:
            return 2
        task = loaded
        source = str(args.task_file)
    else:
        assert args.task_source is not None
        candidate = Path(args.task_source)
        if candidate.is_file():
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

    print(f"Loaded task from {source} ({len(task)} characters).")
    print("Task execution is not implemented yet.")
    return 0
