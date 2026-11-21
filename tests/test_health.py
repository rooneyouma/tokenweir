from litellm.exceptions import AuthenticationError
from pydantic import SecretStr

from tokenweir.health import main, ping_provider
from tokenweir.settings import ConfiguredProvider, LoadedSettings


def test_ping_provider_uses_short_timeout_and_does_not_log_key(monkeypatch, capsys) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("tokenweir.health.completion", fake_completion)
    provider = ConfiguredProvider(
        name="cloud",
        model="provider/model",
        api_key=SecretStr("secret-value"),
    )

    passed, message = ping_provider(provider)

    assert passed is True
    assert message == "reachable"
    assert captured["timeout"] == 2
    assert captured["num_retries"] == 0
    assert captured["api_key"] == "secret-value"
    assert "secret-value" not in capsys.readouterr().out


def test_ping_provider_reports_authentication_failure_without_secret(monkeypatch) -> None:
    def fail_completion(**kwargs):
        raise AuthenticationError(
            message="unauthorized",
            llm_provider="test",
            model="provider/model",
        )

    monkeypatch.setattr("tokenweir.health.completion", fail_completion)
    provider = ConfiguredProvider(
        name="cloud",
        model="provider/model",
        api_key=SecretStr("secret-value"),
    )

    passed, message = ping_provider(provider)

    assert passed is False
    assert "verify the configured API key" in message
    assert "secret-value" not in message


def test_health_command_reports_when_no_providers_are_configured(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "tokenweir.health.load_settings",
        lambda registry: LoadedSettings(providers=[], unconfigured=[], errors=[]),
    )

    assert main([]) == 1
    assert "No providers are configured" in capsys.readouterr().out
