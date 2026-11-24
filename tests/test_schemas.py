import pytest
from pydantic import ValidationError

from tokenweir.schemas import Plan, Subtask, TestFile as GradingTestFile


def _subtask(
    subtask_id: str,
    *,
    files: list[str] | None = None,
    tests: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> Subtask:
    return Subtask(
        id=subtask_id,
        title=f"Subtask {subtask_id}",
        type="module",
        difficulty="easy",
        recommended_tier="local_small",
        files=files or [f"{subtask_id}.py"],
        depends_on=depends_on or [],
        spec="Do the work.",
        tests=tests or [f"tests/test_{subtask_id}.py"],
    )


def _test_file(subtask_id: str, content: str | None = None) -> GradingTestFile:
    return GradingTestFile(
        path=f"tests/test_{subtask_id}.py",
        content=(
            content
            if content is not None
            else "def test_ok() -> None:\n    assert True\n"
        ),
    )


def _plan(
    subtasks: list[Subtask],
    test_files: list[GradingTestFile] | None = None,
) -> Plan:
    if test_files is None:
        test_files = [_test_file(subtask.id) for subtask in subtasks]
    return Plan(task="Task", subtasks=subtasks, test_files=test_files)


def test_plan_orders_subtasks_by_dependencies() -> None:
    first = _subtask("first", depends_on=["second"])
    second = _subtask("second")

    plan = _plan([first, second])

    assert [subtask.id for subtask in plan.ordered_subtasks()] == [
        "second",
        "first",
    ]


def test_plan_rejects_dependency_cycles() -> None:
    first = _subtask("first", depends_on=["second"])
    second = _subtask("second", depends_on=["first"])

    with pytest.raises(ValidationError, match="cycle"):
        _plan([first, second])


def test_plan_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        _plan([_subtask("first", depends_on=["first"])])


def test_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown subtask"):
        _plan([_subtask("first", depends_on=["missing"])])


def test_plan_rejects_duplicate_subtask_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        _plan([_subtask("first"), _subtask("first")])


def test_plan_rejects_missing_test_file() -> None:
    subtask = _subtask("first", tests=["tests/test_missing.py"])

    with pytest.raises(ValidationError, match="missing test files"):
        _plan([subtask], test_files=[_test_file("first")])


def test_plan_rejects_unused_test_file() -> None:
    with pytest.raises(ValidationError, match="not referenced"):
        _plan(
            [_subtask("first")],
            test_files=[_test_file("first"), _test_file("extra")],
        )


def test_plan_rejects_invalid_test_syntax() -> None:
    with pytest.raises(ValidationError, match="not valid Python"):
        _plan(
            [_subtask("first")],
            test_files=[_test_file("first", content="def broken(:\n")],
        )


def test_test_file_rejects_empty_content() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _test_file("first", content="   ")


def test_subtask_rejects_builder_files_under_tests() -> None:
    with pytest.raises(ValidationError):
        _subtask("first", files=["tests/test_first.py"])


def test_subtask_normalizes_paths() -> None:
    subtask = Subtask(
        id="first",
        title="First",
        type="module",
        difficulty="easy",
        recommended_tier="local_small",
        files=["src\\pkg\\module.py"],
        spec="Write the module.",
        tests=["tests\\test_module.py"],
    )

    assert subtask.files == ["src/pkg/module.py"]
    assert subtask.tests == ["tests/test_module.py"]
