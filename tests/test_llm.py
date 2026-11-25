import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import SecretStr

from tokenweir.budget import (
    BudgetExceededError,
    BudgetPolicy,
    load_project_budget,
)
from tokenweir.llm import (
    JsonObjectError,
    ModelCallError,
    call_json_model,
    extract_json_object,
)
from tokenweir.settings import ConfiguredProvider


def _response(
    content: object,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _provider(
    *,
    api_key: SecretStr | None = SecretStr("secret-value"),
    base_url: str | None = "http://example.test",
    free: bool = False,
    tier: Literal["local", "free_cloud", "paid"] = "paid",
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> ConfiguredProvider:
    return ConfiguredProvider(
        name="cloud",
        model="provider/model",
        api_key=api_key,
        base_url=base_url,
        timeout_s=45,
        free=free,
        tier=tier,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )


def _policy(
    tmp_path: Path, *, cap_usd: float = 1.0
) -> BudgetPolicy:
    budget = load_project_budget(
        "test-project", state_dir=tmp_path / "projects", cap_usd=cap_usd
    )
    return BudgetPolicy(budget)


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

    result = call_json_model(
        _provider(),
        system_prompt="system",
        user_prompt="user",
        role="architect",
    )

    assert result.content == "{}"


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


def test_empty_paid_response_is_still_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "tokenweir.llm.completion",
        lambda **kwargs: _response(
            None, prompt_tokens=10, completion_tokens=0
        ),
    )
    monkeypatch.setattr(
        "tokenweir.llm.litellm.completion_cost", lambda **kwargs: 0.0
    )
    policy = _policy(tmp_path, cap_usd=1.0)

    with pytest.raises(ModelCallError, match="empty response"):
        call_json_model(
            _provider(
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
            system_prompt="system",
            user_prompt="user",
            budget_policy=policy,
        )

    entry = json.loads(
        policy.budget.ledger_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 0
    assert entry["cost_usd"] == pytest.approx(10 * 1e-6)


def test_paid_call_records_usage_and_cost(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tokenweir.llm.completion",
        lambda **kwargs: _response(
            "{}", prompt_tokens=10, completion_tokens=5
        ),
    )
    monkeypatch.setattr(
        "tokenweir.llm.litellm.completion_cost", lambda **kwargs: 0.0
    )
    policy = _policy(tmp_path, cap_usd=1.0)

    result = call_json_model(
        _provider(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        ),
        system_prompt="system",
        user_prompt="user",
        role="builder",
        budget_policy=policy,
        task="Build a thing",
        subtask="impl",
        attempt=2,
        run_id="run-1",
    )

    expected = 10 * 1e-6 + 5 * 2e-6
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.cost_usd == pytest.approx(expected)
    assert result.cost_estimated is False
    assert result.free is False
    assert policy.budget.spent_usd == pytest.approx(expected)

    entry = json.loads(
        policy.budget.ledger_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert entry["project"] == "test-project"
    assert entry["run_id"] == "run-1"
    assert entry["role"] == "builder"
    assert entry["provider"] == "cloud"
    assert entry["task"] == "Build a thing"
    assert entry["subtask"] == "impl"
    assert entry["attempt"] == 2
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 5
    assert entry["cost_usd"] == pytest.approx(expected)
    assert entry["free"] is False


def test_free_call_bypasses_depleted_paid_cap(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _response("{}", prompt_tokens=3, completion_tokens=2)

    monkeypatch.setattr("tokenweir.llm.completion", fake_completion)
    policy = _policy(tmp_path, cap_usd=0.0)

    result = call_json_model(
        _provider(
            api_key=None,
            base_url=None,
            free=True,
            tier="local",
        ),
        system_prompt="system",
        user_prompt="user",
        budget_policy=policy,
    )

    assert len(calls) == 1
    assert result.free is True
    assert result.cost_usd == 0.0
    assert policy.budget.spent_usd == 0.0
    entry = json.loads(
        policy.budget.ledger_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert entry["free"] is True
    assert entry["cost_usd"] == 0.0


def test_paid_call_is_blocked_before_completion_at_cap(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _response("{}")

    monkeypatch.setattr("tokenweir.llm.completion", fake_completion)
    policy = _policy(tmp_path, cap_usd=0.0)

    with pytest.raises(BudgetExceededError, match="cap"):
        call_json_model(
            _provider(
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
            system_prompt="system",
            user_prompt="user",
            budget_policy=policy,
        )

    assert calls == []
    assert not policy.budget.ledger_path.exists()


def test_budget_policy_can_extend_the_cap_before_a_paid_call(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _response("{}")

    monkeypatch.setattr("tokenweir.llm.completion", fake_completion)
    monkeypatch.setattr(
        "tokenweir.llm.litellm.completion_cost", lambda **kwargs: 0.0
    )
    budget = load_project_budget(
        "extendable", state_dir=tmp_path / "projects", cap_usd=0.0
    )
    policy = BudgetPolicy(budget, prompt=lambda exc: 1.0)

    call_json_model(
        _provider(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        ),
        system_prompt="system",
        user_prompt="user",
        budget_policy=policy,
    )

    assert len(calls) == 1
    assert policy.budget.cap_usd == 1.0
    assert json.loads(budget.budget_path.read_text(encoding="utf-8"))[
        "cap_usd"
    ] == 1.0
