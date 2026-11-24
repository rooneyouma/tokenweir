import json
from types import SimpleNamespace

import pytest
from litellm.exceptions import AuthenticationError
from pydantic import SecretStr

from tokenweir.architect import (
    ArchitectError,
    ArchitectResponseError,
    load_architect_prompt,
    load_project_brief,
    parse_plan,
    plan_task,
)
from tokenweir.settings import ConfiguredProvider

PLAN_JSON = json.dumps(
    {
        "task": "Create palindrome.py",
        "subtasks": [
            {
                "id": "impl",
                "title": "Implement palindrome",
                "type": "function",
                "difficulty": "easy",
                "recommended_tier": "local_small",
                "files": ["palindrome.py"],
                "depends_on": [],
                "spec": "Implement is_palindrome.",
                "signatures": ["def is_palindrome(s: str) -> bool"],
                "tests": ["tests/test_palindrome.py"],
            }
        ],
        "test_files": [
            {
                "path": "tests/test_palindrome.py",
                "content": (
                    "from palindrome import is_palindrome\n\n"
                    "def test_palindrome() -> None:\n"
                    "    assert is_palindrome('racecar')\n"
                ),
            }
        ],
    }
)


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


def test_parse_plan_accepts_valid_json() -> None:
    plan = parse_plan(PLAN_JSON)

    assert plan.task == "Create palindrome.py"
    assert plan.subtasks[0].id == "impl"
    assert plan.test_files[0].path == "tests/test_palindrome.py"


def test_parse_plan_strips_markdown_fences() -> None:
    plan = parse_plan(f"```json\n{PLAN_JSON}\n```")

    assert plan.subtasks[0].id == "impl"


def test_parse_plan_rejects_invalid_json() -> None:
    with pytest.raises(ArchitectResponseError, match="not valid JSON"):
        parse_plan("not json")


def test_parse_plan_rejects_contract_violation() -> None:
    with pytest.raises(ArchitectResponseError, match="did not match the contract"):
        parse_plan('{"task": "x", "subtasks": [], "test_files": []}')


def test_plan_task_sends_expected_request(monkeypatch, capsys) -> None:
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _response(PLAN_JSON)

    monkeypatch.setattr("tokenweir.llm.completion", fake_completion)

    plan = plan_task(
        "Create palindrome.py",
        _provider(),
        prompt="system prompt",
        brief="brief text",
        temperature=0.2,
    )

    assert plan.subtasks[0].id == "impl"
    assert captured["model"] == "provider/model"
    assert captured["messages"][0] == {
        "role": "system",
        "content": "system prompt",
    }
    assert "brief text" in captured["messages"][1]["content"]
    assert "Create palindrome.py" in captured["messages"][1]["content"]
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 45
    assert captured["num_retries"] == 0
    assert captured["api_key"] == "secret-value"
    assert captured["api_base"] == "http://example.test"
    assert "secret-value" not in capsys.readouterr().out


def test_plan_task_wraps_authentication_error(monkeypatch) -> None:
    def fail_completion(**kwargs):
        raise AuthenticationError(
            message="unauthorized",
            llm_provider="test",
            model="provider/model",
        )

    monkeypatch.setattr("tokenweir.llm.completion", fail_completion)

    with pytest.raises(ArchitectError, match="authentication failed") as error:
        plan_task(
            "Create palindrome.py",
            _provider(),
            prompt="system prompt",
            brief="brief text",
        )

    assert "secret-value" not in str(error.value)


def test_prompt_and_brief_are_not_empty() -> None:
    assert "architect" in load_architect_prompt().lower()
    assert "project brief" in load_project_brief().lower()
