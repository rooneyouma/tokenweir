import json
from pathlib import Path

import pytest

from tokenweir.settings import SettingsError, load_settings


def test_load_settings_resolves_configured_providers(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "local",
                        "model_env": "LOCAL_MODEL",
                        "base_url_env": "LOCAL_URL",
                        "default_base_url": "http://localhost:11434",
                    },
                    {
                        "name": "cloud",
                        "model_env": "CLOUD_MODEL",
                        "api_key_env": "CLOUD_KEY",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry,
        environ={
            "LOCAL_MODEL": "ollama_chat/example",
            "LOCAL_URL": "http://127.0.0.1:11434",
            "CLOUD_MODEL": "provider/model",
            "CLOUD_KEY": "secret-value",
        },
    )

    assert [provider.name for provider in settings.providers] == ["local", "cloud"]
    assert settings.providers[0].base_url == "http://127.0.0.1:11434"
    assert settings.providers[1].api_key is not None
    assert settings.providers[1].api_key.get_secret_value() == "secret-value"
    assert settings.unconfigured == []
    assert settings.errors == []
    assert "secret-value" not in repr(settings)


def test_load_settings_reports_missing_model_and_key(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                    {
                        "name": "cloud",
                        "model_env": "CLOUD_MODEL",
                        "api_key_env": "CLOUD_KEY",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(registry, environ={"CLOUD_MODEL": "provider/model"})

    assert settings.providers == []
    assert settings.unconfigured == [
        "local: set LOCAL_MODEL to a LiteLLM model ID"
    ]
    assert settings.errors == ["cloud: set CLOUD_KEY before use"]


def test_load_settings_rejects_invalid_registry(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text('{"providers": [{"name": "bad", "model_env": "bad-name"}]}')

    with pytest.raises(SettingsError, match="Invalid provider registry"):
        load_settings(registry, environ={})


def test_invalid_registry_error_does_not_echo_secret_values(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "cloud",
                        "model_env": "CLOUD_MODEL",
                        "api_key": "do-not-print-this",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError) as error:
        load_settings(registry, environ={})

    assert "do-not-print-this" not in str(error.value)


def test_load_settings_applies_builder_config_and_timeouts(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "builder": {
                    "provider": "cloud",
                    "max_attempts": 5,
                    "temperature": 0.2,
                },
                "providers": [
                    {
                        "name": "local",
                        "model_env": "LOCAL_MODEL",
                        "timeout_s": 30,
                    },
                    {
                        "name": "cloud",
                        "model_env": "CLOUD_MODEL",
                        "api_key_env": "CLOUD_KEY",
                        "timeout_s": 90,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry,
        environ={
            "LOCAL_MODEL": "local/model",
            "CLOUD_MODEL": "cloud/model",
            "CLOUD_KEY": "secret-value",
        },
    )

    assert settings.builder.provider == "cloud"
    assert settings.builder.max_attempts == 5
    assert settings.builder.temperature == 0.2
    assert [provider.timeout_s for provider in settings.providers] == [30, 90]


def test_load_settings_defaults_builder_to_first_provider(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(registry, environ={"LOCAL_MODEL": "local/model"})

    assert settings.builder.provider == "local"
    assert settings.builder.max_attempts == 3


def test_load_settings_rejects_unknown_builder_provider(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "builder": {"provider": "missing"},
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="builder provider"):
        load_settings(registry, environ={"LOCAL_MODEL": "local/model"})


def test_load_settings_rejects_out_of_range_attempts(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "builder": {"max_attempts": 99},
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="less than or equal to 8"):
        load_settings(registry, environ={"LOCAL_MODEL": "local/model"})


def test_load_settings_defaults_architect_to_builder_provider(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "builder": {"provider": "cloud"},
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                    {"name": "cloud", "model_env": "CLOUD_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry,
        environ={
            "LOCAL_MODEL": "local/model",
            "CLOUD_MODEL": "cloud/model",
        },
    )

    assert settings.architect.provider == "cloud"
    assert settings.architect.temperature == 0.0


def test_load_settings_applies_architect_config(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "architect": {"provider": "local", "temperature": 0.3},
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry,
        environ={"LOCAL_MODEL": "local/model"},
    )

    assert settings.architect.provider == "local"
    assert settings.architect.temperature == 0.3


def test_load_settings_rejects_unknown_architect_provider(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "architect": {"provider": "missing"},
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="architect provider"):
        load_settings(registry, environ={"LOCAL_MODEL": "local/model"})


def test_load_settings_applies_router_and_budget_config(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "router": {
                    "attempts_per_provider": 1,
                    "max_total_attempts": 4,
                },
                "budget": {
                    "default_cap_usd": 5.0,
                    "warn_fraction": 0.5,
                    "state_dir": "state",
                },
                "providers": [
                    {"name": "local", "model_env": "LOCAL_MODEL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry, environ={"LOCAL_MODEL": "local/model"}
    )

    assert settings.router.attempts_per_provider == 1
    assert settings.router.max_total_attempts == 4
    assert settings.budget.default_cap_usd == 5.0
    assert settings.budget.warn_fraction == 0.5
    assert settings.budget.state_dir == "state"


def test_load_settings_accepts_provider_tiers_and_costs(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "paid",
                        "model_env": "PAID_MODEL",
                        "api_key_env": "PAID_KEY",
                        "tier": "paid",
                        "free": False,
                        "max_output_tokens": 1024,
                        "input_cost_per_million": 1.0,
                        "output_cost_per_million": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(
        registry,
        environ={
            "PAID_MODEL": "paid/model",
            "PAID_KEY": "secret-value",
        },
    )

    provider = settings.providers[0]
    assert provider.tier == "paid"
    assert provider.free is False
    assert provider.max_output_tokens == 1024
    assert provider.input_cost_per_million == 1.0
    assert provider.output_cost_per_million == 2.0


def test_load_settings_rejects_partial_provider_costs(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "paid",
                        "model_env": "PAID_MODEL",
                        "input_cost_per_million": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="set both"):
        load_settings(registry, environ={"PAID_MODEL": "paid/model"})
