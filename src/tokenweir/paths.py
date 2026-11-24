from __future__ import annotations

import re
from pathlib import Path

_BLOCKED_ROOT_FILES = {
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "sitecustomize.py",
    "tox.ini",
    "usercustomize.py",
}


class UnsafePathError(ValueError):
    """Raised when a generated file path could escape or tamper with the workspace."""


def _normalize(value: str, *, allow_tests: bool) -> str:
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise UnsafePathError("file path must not be empty")
    if "\x00" in raw:
        raise UnsafePathError("file path must not contain null bytes")
    if re.match(r"^[A-Za-z]:", raw):
        raise UnsafePathError("file path must not include a drive letter")

    path = Path(raw)
    if path.is_absolute():
        raise UnsafePathError("file path must be relative to the workspace")

    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError("file path must not contain '.', '..', or empty components")

    lowered = [part.lower() for part in parts]
    if ".git" in lowered:
        raise UnsafePathError("file path must not modify Git metadata")
    if lowered[0] == "tests" and not allow_tests:
        raise UnsafePathError("file path must not modify the grading tests")
    if len(parts) == 1 and lowered[0] in _BLOCKED_ROOT_FILES:
        raise UnsafePathError(f"file path must not modify test configuration: {parts[0]}")

    return path.as_posix()


def normalize_relative_path(value: str) -> str:
    """Validate a generated builder file path and return it in POSIX form."""
    return _normalize(value, allow_tests=False)


def normalize_test_path(value: str) -> str:
    """Validate an architect-supplied grading test path."""
    normalized = _normalize(value, allow_tests=True)
    path = Path(normalized)
    if not path.parts or path.parts[0] != "tests":
        raise UnsafePathError("test path must be under tests/")
    if not path.name.startswith("test_") or path.suffix != ".py":
        raise UnsafePathError("test path must name a test_*.py file")
    return normalized


def resolve_workspace_path(
    workspace_dir: Path, relative_path: str, *, allow_tests: bool = False
) -> Path:
    """Resolve a validated relative path and ensure it stays inside the workspace."""
    normalized = (
        normalize_test_path(relative_path)
        if allow_tests
        else normalize_relative_path(relative_path)
    )
    root = workspace_dir.resolve()
    target = (root / normalized).resolve()
    if target == root or not target.is_relative_to(root):
        raise UnsafePathError("file path resolves outside the workspace")
    if target.is_symlink():
        raise UnsafePathError("file path must not target a symbolic link")
    return target
