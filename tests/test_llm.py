from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from tokenweir.llm import (
    JsonObjectError,
    ModelCallError,
    call_json_model,
    extract_json_object,
)
from tokenweir.settings import ConfiguredProvider


def _response(content: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _provider(
    *,
    api_key: SecretStr | None = SecretStr("secret-value"),
    base_url: str | None = "http://example.test",
) -> ConfiguredProvider:
    return ConfiguredProvider(
        name="cloud",
        model="provider/model",
        api_key=api_key,
        base_url=base_url,
        timeout_s=45,
    )


def test_extract_json_object_accepts_plain_json() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_strips_markdown_fences() -> None:
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_object_extracts_from_prose() -> None:
    assert extract_json_object('Here you go:\n{"a": 1}\nDone.') == {"a": 1}


def test_extract_json_object_rejects_non_object() -> None:
    with pytest.raises(JsonObjectError, match="must be a JSON object"):
        extract_json_object("[1, 2]")


def test_extract_json_object_rejects_invalid_json() -> None:
    with pytest.raises(JsonObjectError, match="not valid JSON"):
        extract_json_object("not json")


def test_call_json_model_returns_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.llm.completion", lambda **kwargs: _response("{}")
    )

    content = call_json_model(
        _provider(),
        system_prompt="system",
        user_prompt="user",
        role="architect",
    )

    assert content == "{}"


def test_call_json_model_omits_missing_credentials(monkeypatch) -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _response("{}")

    monkeypatch.setattr("tokenweir.llm.completion", fake_completion)

    call_json_model(
        _provider(api_key=None, base_url=None),
        system_prompt="system",
        user_prompt="user",
    )

    assert "api_key" not in captured
    assert "api_base" not in captured


def test_call_json_model_rejects_empty_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.llm.completion", lambda **kwargs: _response(None)
    )

    with pytest.raises(ModelCallError, match="empty response"):
        call_json_model(
            _provider(),
            system_prompt="system",
            user_prompt="user",
        )


def test_call_json_model_rejects_unexpected_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.llm.completion", lambda **kwargs: SimpleNamespace()
    )

    with pytest.raises(ModelCallError, match="unexpected response shape"):
        call_json_model(
            _provider(),
            system_prompt="system",
            user_prompt="user",
        )
