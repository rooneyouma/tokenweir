from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenweir.paths import resolve_workspace_path

_WORKSPACE_GITIGNORE = """\
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
"""


@dataclass
class SandboxResult:
    passed: bool
    stage: str  # "pytest" | "flake8" | "mypy" | "passed"
    output: str


def init_workspace(
    workspace_dir: Path,
    task_tests_dir: Path | None,
    *,
    workspace_root: Path,
) -> None:
    """Create a fresh disposable git repo, optionally copying in task tests."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    resolved_root = workspace_root.resolve()
    resolved_workspace = workspace_dir.resolve()
    if resolved_workspace == resolved_root or not resolved_workspace.is_relative_to(
        resolved_root
    ):
        raise ValueError(f"Workspace must be a child of the owned root {resolved_root}")
    if workspace_dir.exists() or workspace_dir.is_symlink():
        raise FileExistsError(f"Workspace already exists: {workspace_dir}")

    workspace_dir.mkdir(parents=True)

    # Initialize git and set repo-local identity so commits never fail
    subprocess.run(["git", "init"], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "bot@tokenweir.local"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "TokenWeir"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )

    # Keep check caches out of the generated commits
    (workspace_dir / ".gitignore").write_text(_WORKSPACE_GITIGNORE, encoding="utf-8")

    # Copy grading tests into workspace/tests/
    dest_tests = workspace_dir / "tests"
    if task_tests_dir is not None and task_tests_dir.exists():
        shutil.copytree(
            task_tests_dir,
            dest_tests,
            ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
        )

    subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initialize workspace", "--allow-empty"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )


def write_files(
    workspace_dir: Path,
    files: Iterable[tuple[str, str]],
    *,
    allow_tests: bool = False,
) -> list[Path]:
    """Write files inside the workspace, rejecting unsafe paths."""
    targets = [
        (
            resolve_workspace_path(
                workspace_dir, relative_path, allow_tests=allow_tests
            ),
            content,
        )
        for relative_path, content in files
    ]
    written: list[Path] = []
    for target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def commit_step(workspace_dir: Path, message: str) -> None:
    """Stage and commit all changes after a passing subtask."""
    subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )


def _run_cmd(cmd: list[str], cwd: Path, timeout: float = 30) -> tuple[bool, str]:
    """Run a command in cwd and return (success, combined_output)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode == 0, combined
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s: {' '.join(cmd)}"


def run_checks(
    workspace_dir: Path,
    test_paths: Sequence[str] | None = None,
    lint_paths: Sequence[str] | None = None,
) -> SandboxResult:
    """Run pytest, flake8, and mypy in order inside workspace_dir.

    test_paths limits pytest to specific files; None runs the full suite.
    lint_paths limits flake8/mypy to specific paths; None lints the workspace.
    """
    pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    if test_paths:
        pytest_cmd.extend(test_paths)
    lint_targets = list(lint_paths) if lint_paths else ["."]
    checks = [
        ("pytest", pytest_cmd),
        (
            "flake8",
            [sys.executable, "-m", "flake8", "--max-line-length=100", *lint_targets],
        ),
        (
            "mypy",
            [sys.executable, "-m", "mypy", "--ignore-missing-imports", *lint_targets],
        ),
    ]

    for stage, cmd in checks:
        ok, out = _run_cmd(cmd, cwd=workspace_dir)
        if not ok:
            # Keep only the tail of long tracebacks so retry prompts stay compact
            trimmed = out[-2000:] if len(out) > 2000 else out
            return SandboxResult(passed=False, stage=stage, output=trimmed)

    return SandboxResult(passed=True, stage="passed", output="All checks passed.")
