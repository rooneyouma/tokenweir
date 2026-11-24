from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)


class SettingsError(ValueError):
    """Raised when registry configuration cannot be loaded or validated."""


class ProviderDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model_env: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    timeout_s: int = Field(default=120, ge=1, le=600)

    @field_validator("model_env", "api_key_env", "base_url_env")
    @classmethod
    def validate_environment_variable(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Z][A-Z0-9_]*", value) is None:
            raise ValueError("environment variable names must use uppercase letters")
        return value


class BuilderDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=8)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ArchitectDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ProviderRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderDefinition]
    builder: BuilderDefinition = Field(default_factory=BuilderDefinition)
    architect: ArchitectDefinition = Field(default_factory=ArchitectDefinition)

    @field_validator("providers")
    @classmethod
    def validate_unique_provider_names(
        cls, providers: list[ProviderDefinition]
    ) -> list[ProviderDefinition]:
        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        return providers

    @model_validator(mode="after")
    def validate_role_providers(self) -> "ProviderRegistry":
        names = [provider.name for provider in self.providers]
        if not names:
            raise ValueError("at least one provider must be defined")
        if self.builder.provider is None:
            self.builder.provider = names[0]
        elif self.builder.provider not in names:
            raise ValueError(
                "builder provider must name a configured provider: "
                + ", ".join(names)
            )
        if self.architect.provider is None:
            self.architect.provider = self.builder.provider
        elif self.architect.provider not in names:
            raise ValueError(
                "architect provider must name a configured provider: "
                + ", ".join(names)
            )
        return self


class ConfiguredProvider(BaseModel):
    name: str
    model: str
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_s: int = 120


class LoadedSettings(BaseModel):
    providers: list[ConfiguredProvider]
    unconfigured: list[str]
    errors: list[str]
    builder: BuilderDefinition = Field(default_factory=BuilderDefinition)
    architect: ArchitectDefinition = Field(default_factory=ArchitectDefinition)


def load_settings(
    registry_path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LoadedSettings:
    """Load and validate provider definitions and their environment settings."""
    if environ is None:
        load_dotenv(override=False)
        environ = os.environ

    path = registry_path or Path(__file__).with_name("registry.json")
    try:
        raw_registry = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SettingsError(f"Cannot read provider registry {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid JSON in provider registry {path}: {exc}") from exc

    try:
        registry = ProviderRegistry.model_validate(raw_registry)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise SettingsError(f"Invalid provider registry {path}: {details}") from exc

    providers: list[ConfiguredProvider] = []
    unconfigured: list[str] = []
    errors: list[str] = []
    for definition in registry.providers:
        model = environ.get(definition.model_env, "").strip()
        if not model:
            unconfigured.append(
                f"{definition.name}: set {definition.model_env} to a LiteLLM model ID"
            )
            continue

        api_key: SecretStr | None = None
        if definition.api_key_env:
            key = environ.get(definition.api_key_env, "").strip()
            if not key:
                errors.append(
                    f"{definition.name}: set {definition.api_key_env} before use"
                )
                continue
            api_key = SecretStr(key)

        base_url = (
            environ.get(definition.base_url_env, "").strip()
            if definition.base_url_env
            else ""
        ) or definition.default_base_url
        providers.append(
            ConfiguredProvider(
                name=definition.name,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=definition.timeout_s,
            )
        )

    return LoadedSettings(
        providers=providers,
        unconfigured=unconfigured,
        errors=errors,
        builder=registry.builder,
        architect=registry.architect,
    )
