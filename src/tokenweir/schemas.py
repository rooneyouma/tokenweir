from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tokenweir.paths import normalize_relative_path


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
