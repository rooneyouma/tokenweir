from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tokenweir.paths import normalize_relative_path, normalize_test_path

SubtaskType = Literal["module", "function", "script", "config", "bugfix"]
Difficulty = Literal["easy", "medium", "hard"]
Tier = Literal["local_small", "local_medium", "free_cloud"]

_SUBTASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _require_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must not be empty")
    return stripped


class BuilderFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class BuilderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[BuilderFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> "BuilderOutput":
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("file paths must be unique")
        return self


class TestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_test_path(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("test content must not be empty")
        return value


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    type: SubtaskType
    difficulty: Difficulty
    recommended_tier: Tier
    files: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    spec: str
    signatures: list[str] = Field(default_factory=list)
    tests: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if _SUBTASK_ID.fullmatch(value) is None:
            raise ValueError(
                "subtask id must start with a letter and contain only "
                "letters, digits, '_' or '-'"
            )
        return value

    @field_validator("title", "spec")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_text(value)

    @field_validator("files")
    @classmethod
    def validate_files(cls, values: list[str]) -> list[str]:
        normalized = [normalize_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("file paths must be unique within a subtask")
        return normalized

    @field_validator("tests")
    @classmethod
    def validate_tests(cls, values: list[str]) -> list[str]:
        normalized = [normalize_test_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("test paths must be unique within a subtask")
        return normalized

    @field_validator("depends_on")
    @classmethod
    def validate_depends_on(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("depends_on must not contain duplicates")
        for value in normalized:
            if _SUBTASK_ID.fullmatch(value) is None:
                raise ValueError(f"invalid dependency id: {value!r}")
        return normalized

    @field_validator("signatures")
    @classmethod
    def validate_signatures(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    subtasks: list[Subtask] = Field(min_length=1)
    test_files: list[TestFile] = Field(min_length=1)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        return _require_text(value)

    @model_validator(mode="after")
    def validate_plan(self) -> "Plan":
        ids = [subtask.id for subtask in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")
        known = set(ids)

        for subtask in self.subtasks:
            for dependency in subtask.depends_on:
                if dependency == subtask.id:
                    raise ValueError(f"subtask {subtask.id} cannot depend on itself")
                if dependency not in known:
                    raise ValueError(
                        f"subtask {subtask.id} depends on unknown subtask {dependency}"
                    )

        self.ordered_subtasks()

        test_paths = [test_file.path for test_file in self.test_files]
        if len(test_paths) != len(set(test_paths)):
            raise ValueError("test file paths must be unique")

        for test_file in self.test_files:
            try:
                compile(test_file.content, test_file.path, "exec")
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"test file {test_file.path} is not valid Python: {exc}"
                ) from exc

        referenced = {
            path for subtask in self.subtasks for path in subtask.tests
        }
        missing = referenced - set(test_paths)
        if missing:
            raise ValueError(
                "subtasks reference missing test files: " + ", ".join(sorted(missing))
            )
        unused = set(test_paths) - referenced
        if unused:
            raise ValueError(
                "test files are not referenced by any subtask: "
                + ", ".join(sorted(unused))
            )

        builder_paths = {
            path for subtask in self.subtasks for path in subtask.files
        }
        collisions = builder_paths & set(test_paths)
        if collisions:
            raise ValueError(
                "builder files must not overwrite grading tests: "
                + ", ".join(sorted(collisions))
            )
        return self

    def ordered_subtasks(self) -> list[Subtask]:
        """Return subtasks in dependency order, preserving plan order for ties."""
        remaining = {subtask.id: subtask for subtask in self.subtasks}
        completed: set[str] = set()
        ordered: list[Subtask] = []
        while remaining:
            ready = [
                subtask
                for subtask in self.subtasks
                if subtask.id in remaining
                and all(
                    dependency in completed
                    for dependency in subtask.depends_on
                )
            ]
            if not ready:
                raise ValueError("subtask dependencies contain a cycle")
            for subtask in ready:
                ordered.append(subtask)
                completed.add(subtask.id)
                del remaining[subtask.id]
        return ordered
