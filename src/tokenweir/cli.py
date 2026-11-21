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
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--task", help="Task description supplied inline.")
    task_source.add_argument(
        "--task-file",
        type=Path,
        metavar="PATH",
        help="Path to a Markdown or text file containing the task.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task is not None:
        task = args.task.strip()
        source = "inline input"
    else:
        task_path: Path = args.task_file
        try:
            task = task_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"tokenweir: cannot read task file {task_path}: {exc}", file=sys.stderr)
            return 2
        source = str(task_path)

    if not task:
        print("tokenweir: task must not be empty", file=sys.stderr)
        return 2

    print(f"Loaded task from {source} ({len(task)} characters).")
    print("Task execution is not implemented yet.")
    return 0
