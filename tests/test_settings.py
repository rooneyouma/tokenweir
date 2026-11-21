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
    assert settings.errors == ["cloud: set CLOUD_KEY before pinging"]


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
