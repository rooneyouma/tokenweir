from pathlib import Path

import pytest

from tokenweir.paths import (
    UnsafePathError,
    normalize_relative_path,
    normalize_test_path,
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


def test_normalize_test_path_accepts_tests_directory() -> None:
    assert normalize_test_path("tests/test_feature.py") == "tests/test_feature.py"
    assert (
        normalize_test_path("tests/nested/test_feature.py")
        == "tests/nested/test_feature.py"
    )


@pytest.mark.parametrize(
    "value",
    [
        "test_feature.py",
        "src/tests/test_feature.py",
        "tests/helper.py",
        "tests/test_feature.txt",
        "tests/__init__.py",
        "../tests/test_feature.py",
    ],
)
def test_normalize_test_path_rejects_invalid_paths(value: str) -> None:
    with pytest.raises(UnsafePathError):
        normalize_test_path(value)


def test_resolve_workspace_path_allows_tests_only_when_requested(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = resolve_workspace_path(
        workspace, "tests/test_feature.py", allow_tests=True
    )

    assert target == (workspace / "tests/test_feature.py").resolve()
    with pytest.raises(UnsafePathError):
        resolve_workspace_path(workspace, "tests/test_feature.py")
