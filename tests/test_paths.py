from pathlib import Path

import pytest

from tokenweir.paths import (
    UnsafePathError,
    normalize_relative_path,
    resolve_workspace_path,
)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "../evil.py",
        "foo/../../evil.py",
        "/tmp/evil.py",
        "C:\\evil.py",
        "..\\evil.py",
        "tests/test_task.py",
        "tests/nested/test_task.py",
        ".git/config",
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "sitecustomize.py",
        "tox.ini",
        "usercustomize.py",
    ],
)
def test_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(UnsafePathError):
        normalize_relative_path(value)


def test_normalizes_nested_paths() -> None:
    assert normalize_relative_path("src\\pkg\\module.py") == "src/pkg/module.py"
    assert normalize_relative_path("./src/pkg/module.py") == "src/pkg/module.py"


def test_resolve_keeps_path_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = resolve_workspace_path(workspace, "src/module.py")

    assert target == (workspace / "src/module.py").resolve()


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafePathError):
        resolve_workspace_path(workspace, "link/evil.py")
