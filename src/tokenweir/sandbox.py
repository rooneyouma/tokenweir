from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    passed: bool
    stage: str  # "pytest" | "flake8" | "mypy" | "passed"
    output: str


def init_workspace(
    workspace_dir: Path,
    task_tests_dir: Path,
    *,
    workspace_root: Path,
) -> None:
    """Create a fresh disposable git repo in workspace_dir with task tests copied in."""
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

    # Copy grading tests into workspace/tests/
    dest_tests = workspace_dir / "tests"
    if task_tests_dir.exists():
        shutil.copytree(
            task_tests_dir,
            dest_tests,
            ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
        )

    subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initialize workspace with tests", "--allow-empty"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )


def commit_step(workspace_dir: Path, message: str) -> None:
    """Stage and commit all changes after a passing subtask."""
    subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )


def _run_cmd(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[bool, str]:
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


def run_checks(workspace_dir: Path) -> SandboxResult:
    """Run pytest, flake8, and mypy in order inside workspace_dir."""
    checks = [
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("flake8", [sys.executable, "-m", "flake8", "--max-line-length=100", "."]),
        ("mypy", [sys.executable, "-m", "mypy", "--ignore-missing-imports", "."]),
    ]

    for stage, cmd in checks:
        ok, out = _run_cmd(cmd, cwd=workspace_dir)
        if not ok:
            # Keep only the tail of long tracebacks so retry prompts stay compact
            trimmed = out[-2000:] if len(out) > 2000 else out
            return SandboxResult(passed=False, stage=stage, output=trimmed)

    return SandboxResult(passed=True, stage="passed", output="All checks passed.")
