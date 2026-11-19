from pathlib import Path

from tokenweir.sandbox import commit_step, init_workspace, run_checks


def test_sandbox_passes_valid_code(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_src = Path("bench/tasks/palindrome/tests")

    init_workspace(workspace, tests_src)

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
    tests_src = Path("bench/tasks/palindrome/tests")

    init_workspace(workspace, tests_src)

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
    tests_src = Path("bench/tasks/palindrome/tests")

    init_workspace(workspace, tests_src)

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
