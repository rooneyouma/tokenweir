import sys
from pathlib import Path

import pytest

from tokenweir.paths import UnsafePathError
from tokenweir.sandbox import (
    _run_cmd,
    commit_step,
    init_workspace,
    run_checks,
    write_files,
)


def test_sandbox_passes_valid_code(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_src = Path(__file__).resolve().parents[1] / "bench/tasks/palindrome/tests"

    init_workspace(workspace, tests_src, workspace_root=tmp_path)

    # Write a clean, typed, passing implementation
    (workspace / "palindrome.py").write_text(
        "def is_palindrome(s: str) -> bool:\n"
        '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
        "    return cleaned == cleaned[::-1]\n",
        encoding="utf-8",
    )

    result = run_checks(workspace)
    assert result.passed is True
    assert result.stage == "passed"

    commit_step(workspace, "Pass palindrome")


def test_sandbox_catches_failing_pytest(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_src = Path(__file__).resolve().parents[1] / "bench/tasks/palindrome/tests"

    init_workspace(workspace, tests_src, workspace_root=tmp_path)

    # Write a buggy implementation
    (workspace / "palindrome.py").write_text(
        "def is_palindrome(s: str) -> bool:\n"
        "    return False\n",
        encoding="utf-8",
    )

    result = run_checks(workspace)
    assert result.passed is False
    assert result.stage == "pytest"
    assert "AssertionError" in result.output


def test_sandbox_catches_mypy_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_src = Path(__file__).resolve().parents[1] / "bench/tasks/palindrome/tests"

    init_workspace(workspace, tests_src, workspace_root=tmp_path)

    # Logic passes pytest, but violates type annotations for mypy
    (workspace / "palindrome.py").write_text(
        "def is_palindrome(s: str) -> bool:\n"
        '    x: int = "wrong_type"\n'
        "    assert x\n"
        '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
        "    return cleaned == cleaned[::-1]\n",
        encoding="utf-8",
    )

    result = run_checks(workspace)
    assert result.passed is False
    assert result.stage == "mypy"


def test_workspace_must_be_inside_owned_root(tmp_path: Path) -> None:
    outside_workspace = tmp_path / "outside" / "workspace"

    try:
        init_workspace(
            outside_workspace,
            tmp_path / "missing-tests",
            workspace_root=tmp_path / "owned",
        )
    except ValueError as exc:
        assert "owned root" in str(exc)
    else:
        raise AssertionError("Expected workspace containment validation to fail")

    assert not outside_workspace.exists()


def test_existing_workspace_is_not_deleted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sentinel = workspace / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    try:
        init_workspace(
            workspace,
            tmp_path / "missing-tests",
            workspace_root=tmp_path,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected an existing workspace to be rejected")

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_workspace_copy_ignores_python_cache_files(tmp_path: Path) -> None:
    source_tests = tmp_path / "source-tests"
    cache_dir = source_tests / "__pycache__"
    cache_dir.mkdir(parents=True)
    (source_tests / "test_task.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (cache_dir / "test_task.pyc").write_bytes(b"cache")

    workspace = tmp_path / "workspace"
    init_workspace(workspace, source_tests, workspace_root=tmp_path)

    assert (workspace / "tests" / "test_task.py").exists()
    assert not (workspace / "tests" / "__pycache__").exists()


def test_init_workspace_without_tests_skips_tests_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    init_workspace(workspace, None, workspace_root=tmp_path)

    assert not (workspace / "tests").exists()


def test_write_files_writes_nested_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    written = write_files(
        workspace,
        [("src/pkg/module.py", "x = 1\n")],
    )

    assert written == [(workspace / "src/pkg/module.py").resolve()]
    assert (workspace / "src/pkg/module.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_files_rejects_unsafe_paths_before_writing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(UnsafePathError):
        write_files(
            workspace,
            [("safe.py", "x = 1\n"), ("../evil.py", "x = 2\n")],
        )

    assert not (workspace / "safe.py").exists()
    assert not (tmp_path / "evil.py").exists()


def test_write_files_rejects_tests_and_git_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for path in ("tests/test_task.py", ".git/config", "conftest.py"):
        with pytest.raises(UnsafePathError):
            write_files(workspace, [(path, "x = 1\n")])


def test_write_files_allows_tests_when_requested(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    written = write_files(
        workspace,
        [("tests/test_feature.py", "def test_feature(): pass\n")],
        allow_tests=True,
    )

    assert written == [(workspace / "tests/test_feature.py").resolve()]
    assert (workspace / "tests/test_feature.py").exists()
    with pytest.raises(UnsafePathError):
        write_files(
            workspace,
            [("tests/test_other.py", "def test_other(): pass\n")],
        )


def test_run_checks_can_target_specific_tests_and_lint_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "good.py").write_text(
        "def answer() -> int:\n    return 42\n", encoding="utf-8"
    )
    tests = workspace / "tests"
    tests.mkdir()
    (tests / "test_good.py").write_text(
        "from good import answer\n\n"
        "def test_answer() -> None:\n"
        "    assert answer() == 42\n",
        encoding="utf-8",
    )
    (tests / "test_bad.py").write_text(
        "def test_bad() -> None:\n    assert False\n", encoding="utf-8"
    )

    passed = run_checks(
        workspace,
        test_paths=["tests/test_good.py"],
        lint_paths=["good.py"],
    )
    assert passed.passed is True
    assert passed.stage == "passed"

    failed = run_checks(
        workspace,
        test_paths=["tests/test_bad.py"],
        lint_paths=["good.py"],
    )
    assert failed.passed is False
    assert failed.stage == "pytest"


def test_init_workspace_writes_cache_gitignore(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    init_workspace(
        workspace,
        tmp_path / "missing-tests",
        workspace_root=tmp_path,
    )

    ignore = (workspace / ".gitignore").read_text(encoding="utf-8")
    assert ".pytest_cache/" in ignore
    assert ".mypy_cache/" in ignore


def test_run_cmd_reports_timeout(tmp_path: Path) -> None:
    ok, output = _run_cmd(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        timeout=0.1,
    )

    assert ok is False
    assert "timed out" in output
