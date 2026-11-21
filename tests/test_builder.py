from types import SimpleNamespace

import pytest
from litellm.exceptions import AuthenticationError
from pydantic import SecretStr

from tokenweir.builder import (
    BuilderError,
    BuilderResponseError,
    build_files,
    load_builder_prompt,
    parse_builder_output,
)
from tokenweir.settings import ConfiguredProvider


def _response(content: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _provider() -> ConfiguredProvider:
    return ConfiguredProvider(
        name="cloud",
        model="provider/model",
        api_key=SecretStr("secret-value"),
        base_url="http://example.test",
        timeout_s=45,
    )


def test_parse_accepts_plain_json() -> None:
    output = parse_builder_output(
        '{"files": [{"path": "solution.py", "content": "x = 1\\n"}]}'
    )

    assert output.files[0].path == "solution.py"
    assert output.files[0].content == "x = 1\n"


def test_parse_strips_markdown_fences() -> None:
    output = parse_builder_output(
        '```json\n{"files": [{"path": "solution.py", "content": "x = 1\\n"}]}\n```'
    )

    assert output.files[0].path == "solution.py"


def test_parse_extracts_object_from_surrounding_text() -> None:
    output = parse_builder_output(
        'Here you go:\n{"files": [{"path": "solution.py", "content": ""}]}\nDone.'
    )

    assert output.files[0].path == "solution.py"


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(BuilderResponseError, match="not valid JSON"):
        parse_builder_output("not json")


def test_parse_rejects_unsafe_path() -> None:
    with pytest.raises(BuilderResponseError, match="file contract"):
        parse_builder_output(
            '{"files": [{"path": "../evil.py", "content": "x = 1\\n"}]}'
        )


def test_parse_rejects_duplicate_paths() -> None:
    with pytest.raises(BuilderResponseError, match="file contract"):
        parse_builder_output(
            '{"files": ['
            '{"path": "solution.py", "content": "x = 1\\n"},'
            '{"path": "solution.py", "content": "x = 2\\n"}'
            "]}"
        )


def test_load_builder_prompt_is_not_empty() -> None:
    assert "JSON" in load_builder_prompt()


def test_build_files_sends_expected_request(monkeypatch, capsys) -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _response('{"files": [{"path": "solution.py", "content": "x = 1\\n"}]}')

    monkeypatch.setattr("tokenweir.builder.completion", fake_completion)

    output = build_files("Add a feature", _provider(), prompt="system prompt")

    assert output.files[0].path == "solution.py"
    assert captured["model"] == "provider/model"
    assert captured["messages"][0] == {
        "role": "system",
        "content": "system prompt",
    }
    assert "Add a feature" in captured["messages"][1]["content"]
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.0
    assert captured["timeout"] == 45
    assert captured["num_retries"] == 0
    assert captured["api_key"] == "secret-value"
    assert captured["api_base"] == "http://example.test"
    assert "secret-value" not in capsys.readouterr().out


def test_build_files_includes_feedback(monkeypatch) -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _response('{"files": [{"path": "solution.py", "content": ""}]}')

    monkeypatch.setattr("tokenweir.builder.completion", fake_completion)

    build_files(
        "Add a feature",
        _provider(),
        prompt="system prompt",
        feedback="Stage: pytest\nAssertionError",
    )

    assert "Stage: pytest" in captured["messages"][1]["content"]
    assert "AssertionError" in captured["messages"][1]["content"]


def test_build_files_wraps_authentication_error(monkeypatch) -> None:
    def fail_completion(**kwargs):
        raise AuthenticationError(
            message="unauthorized",
            llm_provider="test",
            model="provider/model",
        )

    monkeypatch.setattr("tokenweir.builder.completion", fail_completion)

    with pytest.raises(BuilderError, match="authentication failed") as error:
        build_files("Add a feature", _provider(), prompt="system prompt")

    assert "secret-value" not in str(error.value)


def test_build_files_rejects_empty_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.builder.completion",
        lambda **kwargs: _response(None),
    )

    with pytest.raises(BuilderResponseError, match="empty response"):
        build_files("Add a feature", _provider(), prompt="system prompt")


def test_build_files_rejects_unexpected_response_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.builder.completion",
        lambda **kwargs: SimpleNamespace(),
    )

    with pytest.raises(BuilderResponseError, match="unexpected response shape"):
        build_files("Add a feature", _provider(), prompt="system prompt")


def test_build_files_wraps_unexpected_error(monkeypatch) -> None:
    def fail_completion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokenweir.builder.completion", fail_completion)

    with pytest.raises(BuilderError, match="RuntimeError") as error:
        build_files("Add a feature", _provider(), prompt="system prompt")

    assert "boom" not in str(error.value)
